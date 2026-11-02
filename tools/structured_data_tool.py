"""Deterministic operations over structured document tables."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable


_NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?")
_OPERATION_TERMS = {
    "sum": ("total", "sum", "add", "combined"),
    "average": ("average", "mean", "avg"),
    "maximum": ("maximum", "highest", "largest", "max"),
    "minimum": ("minimum", "lowest", "smallest", "min"),
}


@dataclass(frozen=True)
class CalculationResult:
    """A reproducible result calculated from document table cells."""

    operation: str
    column: str
    values: tuple[Decimal, ...]
    value: Decimal
    source_count: int


def calculate_table_operation(
    question: str,
    chunks: Iterable[dict[str, Any]],
) -> CalculationResult | None:
    """Calculate a supported operation when matching table evidence is present.

    Returns ``None`` rather than guessing when the request, table column, or
    numeric cells cannot be identified deterministically.
    """
    operation = _detect_operation(question)
    if operation is None:
        return None

    for chunk in chunks:
        for table in _tables_from_chunk(chunk):
            result = _calculate_from_table(question, operation, table)
            if result is not None:
                return result
    return None


def format_calculation_answer(
    result: CalculationResult,
    *,
    citation: str | None = None,
) -> str:
    """Format a deterministic result without asking an LLM to do arithmetic."""
    value = _format_decimal(result.value)
    values = ", ".join(_format_decimal(item) for item in result.values)
    answer = (
        f"The {result.operation} of the '{result.column}' values "
        f"({values}) is **{value}**."
    )
    if result.operation == "sum" and result.source_count > 1:
        answer += (
            " These values come from different table rows; this is their sum, "
            "not necessarily an official aggregate benchmark."
        )
    if citation:
        answer = " ".join(
            f"{sentence} {citation}"
            for sentence in re.split(r"(?<=\.)\s+", answer)
        )
    return answer


def _calculate_from_table(
    question: str,
    operation: str,
    table: dict[str, Any],
) -> CalculationResult | None:
    headers = [str(header).strip() for header in table.get("headers") or []]
    rows = table.get("rows") or []
    column_index = _select_column(question, headers, rows)
    if column_index is None:
        return None

    values: list[Decimal] = []
    for row in rows:
        if column_index >= len(row):
            continue
        number = _parse_number(str(row[column_index]))
        if number is not None:
            values.append(number)
    if not values:
        return None

    if operation == "sum":
        value = sum(values, Decimal("0"))
    elif operation == "average":
        value = sum(values, Decimal("0")) / Decimal(len(values))
    elif operation == "maximum":
        value = max(values)
    else:
        value = min(values)

    return CalculationResult(
        operation=operation,
        column=headers[column_index],
        values=tuple(values),
        value=value,
        source_count=len(values),
    )


def _tables_from_chunk(chunk: dict[str, Any]) -> list[dict[str, Any]]:
    """Read native structured tables and markdown tables from retrieved chunks."""
    tables = [
        table for table in (chunk.get("structured_data") or [])
        if table.get("kind") == "table"
    ]
    if tables:
        return tables

    text = str(chunk.get("text") or "")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    parsed: list[dict[str, Any]] = []
    index = 0
    while index + 1 < len(lines):
        if "|" not in lines[index] or not _is_table_separator(lines[index + 1]):
            index += 1
            continue
        headers = _table_cells(lines[index])
        rows: list[list[str]] = []
        index += 2
        while index < len(lines) and "|" in lines[index]:
            rows.append(_table_cells(lines[index]))
            index += 1
        if headers and rows:
            parsed.append({"kind": "table", "headers": headers, "rows": rows})
    return parsed


def _table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip("|").split("|")]


def _is_table_separator(line: str) -> bool:
    cells = _table_cells(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _detect_operation(question: str) -> str | None:
    lowered = question.casefold()
    for operation, terms in _OPERATION_TERMS.items():
        if any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in terms):
            return operation
    return None


def _select_column(
    question: str,
    headers: list[str],
    rows: list[list[Any]],
) -> int | None:
    if not headers:
        return None
    lowered = question.casefold()
    scores: list[tuple[int, int, int]] = []
    for index, header in enumerate(headers):
        words = re.findall(r"[a-z0-9]+", header.casefold())
        lexical_score = sum(1 for word in words if word in lowered)
        scores.append((lexical_score, 0, -index))

    best = max(scores, default=(0, 0, 0))
    if best[0] > 0:
        return -best[2]

    # With no explicit header reference, select the first numeric-looking
    # column only when it is unambiguous. This works for arbitrary schemas
    # without relying on domain-specific names such as "score" or "value".
    numeric_columns = [
        index for index in range(len(headers))
        if _numeric_density(rows, index) >= 0.5
    ]
    if len(numeric_columns) == 1:
        return numeric_columns[0]
    if _contains_position_word(lowered, "first") and numeric_columns:
        return numeric_columns[0]
    if _contains_position_word(lowered, "last") and numeric_columns:
        return numeric_columns[-1]
    return None


def _numeric_density(rows: list[Any], index: int) -> float:
    usable = [row[index] for row in rows if index < len(row)]
    if not usable:
        return 0.0
    return sum(_parse_number(str(value)) is not None for value in usable) / len(usable)


def _contains_position_word(question: str, word: str) -> bool:
    return bool(re.search(rf"\b{re.escape(word)}\b", question))


def _parse_number(value: str) -> Decimal | None:
    match = _NUMBER_RE.search(value.replace("%", ""))
    if not match:
        return None
    try:
        return Decimal(match.group(0).replace(",", ""))
    except InvalidOperation:
        return None


def _format_decimal(value: Decimal) -> str:
    rendered = format(value.normalize(), "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered
