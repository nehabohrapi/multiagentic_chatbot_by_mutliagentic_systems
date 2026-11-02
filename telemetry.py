"""OpenTelemetry setup and per-query trace/metric helpers."""

from __future__ import annotations

import os
from collections import deque
from contextlib import contextmanager
from time import perf_counter
from threading import Lock
from typing import Any, Iterator

from opentelemetry import metrics, trace
from opentelemetry.trace import Span, Status, StatusCode
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader, MetricReader

_SPAN_CAPACITY = 20_000


class _LocalSpanExporter(SpanExporter):
    """Keep a bounded local trace buffer for the workflow inspector."""

    def __init__(self) -> None:
        self._spans: deque[ReadableSpan] = deque(maxlen=_SPAN_CAPACITY)
        self._lock = Lock()

    def export(self, spans: list[ReadableSpan]) -> SpanExportResult:
        with self._lock:
            self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True

    def spans_for_trace(self, trace_id: int) -> list[ReadableSpan]:
        with self._lock:
            return [
                span for span in self._spans
                if span.context is not None
                and span.context.trace_id == trace_id
            ]


_LOCAL_EXPORTER = _LocalSpanExporter()
_RESOURCE = Resource.create({"service.name": "nexus-multi-agent-research"})
_TRACER_PROVIDER = TracerProvider(resource=_RESOURCE)
_TRACER_PROVIDER.add_span_processor(SimpleSpanProcessor(_LOCAL_EXPORTER))

_OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
if _OTLP_ENDPOINT:
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

    _TRACER_PROVIDER.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=_OTLP_ENDPOINT))
    )
    _OTLP_METRIC_READER: MetricReader | None = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=_OTLP_ENDPOINT)
    )
else:
    _OTLP_METRIC_READER = None

_IN_MEMORY_METRIC_READER = InMemoryMetricReader()
_METRIC_READERS = [_IN_MEMORY_METRIC_READER]
if _OTLP_METRIC_READER is not None:
    _METRIC_READERS.append(_OTLP_METRIC_READER)
_METER_PROVIDER = MeterProvider(
    resource=_RESOURCE,
    metric_readers=_METRIC_READERS,
)

trace.set_tracer_provider(_TRACER_PROVIDER)
metrics.set_meter_provider(_METER_PROVIDER)

_TRACER = _TRACER_PROVIDER.get_tracer("nexus.telemetry")
_METER = _METER_PROVIDER.get_meter("nexus.telemetry")
_SPAN_DURATION = _METER.create_histogram(
    "nexus.span.duration",
    unit="s",
    description="Elapsed wall-clock duration of query and stage spans.",
)
_MODEL_CALLS = _METER.create_counter(
    "gen_ai.client.operation.count",
    unit="{operation}",
    description="Model API operations by provider and operation.",
)
_MODEL_TOKENS = _METER.create_counter(
    "gen_ai.client.token.usage",
    unit="{token}",
    description="Provider-reported or provider-estimated model token usage.",
)
_MODEL_RETRIES = _METER.create_counter(
    "gen_ai.client.retry.count",
    unit="{retry}",
    description="Retries made for model requests.",
)


def _span_attributes(
    span: ReadableSpan,
    *,
    duration_seconds: float,
) -> dict[str, Any]:
    attributes = dict(span.attributes or {})
    label = str(attributes.get("nexus.label") or span.name)
    category = str(attributes.get("nexus.category") or "MODEL")
    record: dict[str, Any] = {
        "label": label,
        "category": category,
        "level": int(attributes.get("nexus.level", 2)),
        "elapsed_seconds": round(duration_seconds, 4),
        "details": str(attributes.get("nexus.details") or ""),
        "status": span.status.status_code.name,
        "trace_id": f"{span.context.trace_id:032x}" if span.context else "",
    }
    provider = attributes.get("gen_ai.system")
    model = attributes.get("gen_ai.request.model")
    operation = attributes.get("gen_ai.operation.name")
    if provider:
        record["provider"] = provider
    if model:
        record["model"] = model
    if operation:
        record["operation"] = operation
    if "nexus.stage" in attributes:
        record["stage"] = attributes["nexus.stage"]
    for source, target in (
        ("gen_ai.usage.input_tokens", "input_tokens"),
        ("gen_ai.usage.output_tokens", "output_tokens"),
        ("gen_ai.usage.total_tokens", "total_tokens"),
    ):
        if source in attributes:
            record[target] = int(attributes[source])
    if operation == "generate":
        record["model_call"] = True
    return record


def _trace_records(trace_id: int) -> list[dict[str, Any]]:
    spans = _LOCAL_EXPORTER.spans_for_trace(trace_id)
    records: list[dict[str, Any]] = []
    for span in spans:
        if span.end_time is None:
            continue
        duration = max(0, span.end_time - span.start_time) / 1_000_000_000
        records.append(_span_attributes(span, duration_seconds=duration))
    return sorted(records, key=lambda record: record.get("elapsed_seconds", 0), reverse=True)


def recent_model_spans(
    *,
    provider: str | None = None,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """Return local model-request spans for diagnostics and tests."""
    with _LOCAL_EXPORTER._lock:
        spans = list(_LOCAL_EXPORTER._spans)
    records: list[dict[str, Any]] = []
    for span in spans:
        attributes = dict(span.attributes or {})
        if not attributes.get("gen_ai.system"):
            continue
        if provider and attributes.get("gen_ai.system") != provider:
            continue
        if model and attributes.get("gen_ai.request.model") != model:
            continue
        if span.end_time is None:
            continue
        duration = max(0, span.end_time - span.start_time) / 1_000_000_000
        records.append(_span_attributes(span, duration_seconds=duration))
    return records


def aggregate_trace_usage(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate provider token usage from generation spans in one query trace."""
    prompt_tokens = 0
    completion_tokens = 0
    call_details: list[dict[str, Any]] = []
    for record in records:
        if not record.get("model_call"):
            continue
        input_count = int(record.get("input_tokens", 0))
        output_count = int(record.get("output_tokens", 0))
        prompt_tokens += input_count
        completion_tokens += output_count
        call_details.append({
            "provider": record.get("provider", "unknown"),
            "model": record.get("model", "unknown"),
            "agent": record.get("stage", "model"),
            "prompt_tokens": input_count,
            "completion_tokens": output_count,
            "total_tokens": int(record.get("total_tokens", input_count + output_count)),
            "elapsed_seconds": record.get("elapsed_seconds", 0.0),
        })
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "call_details": call_details,
        "model_calls": len(call_details),
    }


@contextmanager
def capture_timings() -> Iterator[list[dict[str, Any]]]:
    """Capture one OpenTelemetry query trace and expose its finished spans."""
    records: list[dict[str, Any]] = []
    trace_id: int | None = None
    root_started = perf_counter()
    try:
        with _TRACER.start_as_current_span("query") as active_root_span:
            active_root_span.set_attribute("nexus.label", "Query total")
            active_root_span.set_attribute("nexus.category", "QUERY")
            active_root_span.set_attribute("nexus.level", 0)
            trace_id = active_root_span.get_span_context().trace_id
            yield records
    finally:
        if trace_id is not None:
            records.extend(_trace_records(trace_id))
        _SPAN_DURATION.record(
            perf_counter() - root_started,
            {"span": "Query total", "category": "QUERY", "level": 0},
        )


@contextmanager
def time_block(
    label: str,
    *,
    category: str = "PIPELINE",
    level: int = 1,
) -> Iterator[dict[str, str]]:
    """Record a pipeline stage as an OpenTelemetry span and duration metric."""
    details: dict[str, str] = {}
    started = perf_counter()
    with _TRACER.start_as_current_span(label) as span:
        span.set_attribute("nexus.label", label)
        span.set_attribute("nexus.category", category)
        span.set_attribute("nexus.level", level)
        try:
            yield details
        except Exception as exc:
            span.set_attribute("error.type", type(exc).__name__)
            span.set_status(Status(StatusCode.ERROR))
            raise
        finally:
            if details.get("details"):
                span.set_attribute("nexus.details", details["details"])
            elapsed = perf_counter() - started
            details["elapsed_seconds"] = f"{elapsed:.6f}"
            _SPAN_DURATION.record(
                elapsed,
                {"span": label, "category": category, "level": level},
            )


@contextmanager
def model_request(
    *,
    provider: str,
    model: str,
    operation: str,
    stage: str,
    request_type: str = "generation",
) -> Iterator[Span]:
    """Trace and meter one model API operation without recording prompt content."""
    with _TRACER.start_as_current_span("gen_ai.client.request") as span:
        span.set_attribute("gen_ai.system", provider)
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.operation.name", operation)
        span.set_attribute("nexus.stage", stage)
        span.set_attribute("nexus.request.type", request_type)
        _MODEL_CALLS.add(
            1,
            {
                "gen_ai.system": provider,
                "gen_ai.request.model": model,
                "gen_ai.operation.name": operation,
            },
        )
        try:
            yield span
        except Exception as exc:
            span.set_attribute("error.type", type(exc).__name__)
            span.set_status(Status(StatusCode.ERROR))
            raise


def record_model_usage(
    span: Span,
    *,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    estimated_input: bool = False,
) -> None:
    """Attach token usage to a model span and record OpenTelemetry counters."""
    span_attributes = dict(span.attributes or {})
    provider = str(span_attributes.get("gen_ai.system", "unknown"))
    model = str(span_attributes.get("gen_ai.request.model", "unknown"))
    operation = str(span_attributes.get("gen_ai.operation.name", "unknown"))
    attributes = {
        "gen_ai.system": provider,
        "gen_ai.request.model": model,
        "gen_ai.operation.name": operation,
    }
    if input_tokens is not None:
        value = max(0, int(input_tokens))
        span.set_attribute("gen_ai.usage.input_tokens", value)
        span.set_attribute("gen_ai.usage.input_tokens.estimated", estimated_input)
        _MODEL_TOKENS.add(value, {**attributes, "gen_ai.token.type": "input"})
    if output_tokens is not None:
        value = max(0, int(output_tokens))
        span.set_attribute("gen_ai.usage.output_tokens", value)
        _MODEL_TOKENS.add(value, {**attributes, "gen_ai.token.type": "output"})
    if input_tokens is not None or output_tokens is not None:
        span.set_attribute(
            "gen_ai.usage.total_tokens",
            max(0, int(input_tokens or 0)) + max(0, int(output_tokens or 0)),
        )


def record_model_retry(provider: str, model: str, stage: str) -> None:
    """Increment the provider retry metric."""
    _MODEL_RETRIES.add(
        1,
        {
            "gen_ai.system": provider,
            "gen_ai.request.model": model,
            "nexus.stage": stage,
        },
    )


def span_duration_seconds(span: Span) -> float:
    """Return duration of a completed span in seconds."""
    if span.end_time is None:
        return 0.0
    return max(0, span.end_time - span.start_time) / 1_000_000_000


def current_trace_usage() -> dict[str, Any]:
    """Read model usage from the active trace for final workflow reporting."""
    current = trace.get_current_span().get_span_context()
    if not current.is_valid:
        return aggregate_trace_usage([])
    return aggregate_trace_usage(_trace_records(current.trace_id))


def collect_metrics() -> Any:
    """Collect a snapshot of the OpenTelemetry SDK metrics."""
    return _IN_MEMORY_METRIC_READER.get_metrics_data()


def otlp_export_enabled() -> bool:
    """Return whether an OTLP destination was explicitly configured."""
    return bool(_OTLP_ENDPOINT)
