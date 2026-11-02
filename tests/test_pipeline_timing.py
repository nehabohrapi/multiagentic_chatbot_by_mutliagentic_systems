import io
import os
import time
import unittest
from contextlib import redirect_stdout
from functools import wraps
from time import perf_counter
from unittest.mock import patch

from dotenv import load_dotenv

from agents import graph as graph_module
from agents.retrieval_agent import USE_MULTI_AGENT_RAG, run_rag
from config import DOCUMENTS_DIR
from tools.document_store import SUPPORTED_EXTENSIONS


load_dotenv()


def _timed(stage, function, measurements):
    @wraps(function)
    def wrapped(*args, **kwargs):
        started = perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            elapsed = perf_counter() - started
            measurements.append((stage, elapsed))
            print(f"[timing] {stage}: {elapsed:.3f}s")

    return wrapped


class PipelineTimingTests(unittest.TestCase):
    def test_timed_stage_returns_result_and_prints_elapsed_time(self):
        measurements = []
        output = io.StringIO()

        with redirect_stdout(output):
            result = _timed("example_stage", lambda value: value + 1, measurements)(4)

        self.assertEqual(result, 5)
        self.assertEqual(measurements[0][0], "example_stage")
        self.assertGreaterEqual(measurements[0][1], 0)
        self.assertIn("[timing] example_stage:", output.getvalue())

    @unittest.skipUnless(
        os.getenv("NEXUS_RUN_TIMING_TEST") == "1",
        "Set NEXUS_RUN_TIMING_TEST=1 to run the live pipeline timing test.",
    )
    def test_live_query_to_answer_timing(self):
        if not os.getenv("GROQ_API_KEY"):
            self.skipTest("GROQ_API_KEY is required for a live timing run.")
        if not USE_MULTI_AGENT_RAG:
            self.skipTest("The live timing test requires USE_MULTI_AGENT_RAG=True.")
        if not any(
            path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            for path in DOCUMENTS_DIR.rglob("*")
        ):
            self.skipTest("Upload at least one supported document before running.")

        question = os.getenv(
            "NEXUS_TIMING_QUERY",
            "What are the main findings in the uploaded documents?",
        ).strip()
        measurements = []
        stages = (
            ("orchestrator_node", "orchestrator"),
            ("executor_node", "executor"),
            ("_direct_reply_node", "direct_reply"),
            ("_clarify_node", "clarify"),
            ("_worker_node", "retriever_worker"),
            ("_writer_node", "writer"),
            ("verifier_node", "verifier"),
            ("_finalize_node", "finalize"),
            ("_summarizer_node", "summarizer"),
        )

        with patch.multiple(
            graph_module,
            **{
                attribute: _timed(
                    label,
                    getattr(graph_module, attribute),
                    measurements,
                )
                for attribute, label in stages
            },
        ):
            instrumented_graph = graph_module._build_graph()
            with patch.object(graph_module, "multi_agent_graph", instrumented_graph):
                print(f"[timing] question: {question}")
                started = time.perf_counter()
                result = run_rag(question, history=[])
                total_elapsed = time.perf_counter() - started

        answer = result.get("answer", "")
        self.assertTrue(answer, "The pipeline returned an empty answer.")
        print("[timing] ----- query-to-answer report -----")
        print(f"[timing] total query_to_answer: {total_elapsed:.3f}s")
        print(f"[timing] llm_calls: {result.get('llm_calls', 'not reported')}")
        print(f"[timing] route: {result.get('route', 'not reported')}")
        print(f"[timing] answer:\n{answer}")


if __name__ == "__main__":
    unittest.main(verbosity=2)