import unittest

from telemetry import aggregate_trace_usage, capture_timings, model_request, record_model_usage


class TelemetryTest(unittest.TestCase):
    def test_query_capture_aggregates_provider_usage(self):
        with capture_timings() as records:
            with model_request(
                provider="groq",
                model="test-model",
                operation="generate",
                stage="writer",
            ) as span:
                record_model_usage(span, input_tokens=25, output_tokens=7)

        model_spans = [record for record in records if record.get("model_call")]
        self.assertEqual(len(model_spans), 1)
        self.assertNotIn("prompt", model_spans[0])
        self.assertNotIn("contents", model_spans[0])
        usage = aggregate_trace_usage(records)
        self.assertEqual(usage["prompt_tokens"], 25)
        self.assertEqual(usage["completion_tokens"], 7)
        self.assertEqual(usage["total_tokens"], 32)
        self.assertEqual(usage["model_calls"], 1)
        self.assertEqual(usage["call_details"][0]["provider"], "groq")

    def test_generation_without_provider_usage_is_not_invented(self):
        with capture_timings() as records:
            with model_request(
                provider="google",
                model="test-model",
                operation="count_tokens",
                stage="summary",
                request_type="count_tokens",
            ):
                pass

        usage = aggregate_trace_usage(records)
        self.assertEqual(usage["model_calls"], 0)
        self.assertEqual(usage["total_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
