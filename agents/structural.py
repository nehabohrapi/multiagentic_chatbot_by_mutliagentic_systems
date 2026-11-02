"""Structural Agent (Subagent 1) for structured table processing.

Coordinates deterministic operations over document tables (Sum, Avg, Min, Max, cell lookups)
without risk of LLM arithmetic hallucinations.
"""

from __future__ import annotations

from typing import Any, Iterable

from telemetry import time_block
from tools.structured_data_tool import (
    CalculationResult,
    calculate_table_operation,
    format_calculation_answer,
)


class StructuralAgent:
    """Subagent specialized in parsing structured tables and performing deterministic calculations."""

    @staticmethod
    def calculate_table_aggregation(
        question: str,
        chunks: Iterable[dict[str, Any]],
    ) -> CalculationResult | None:
        """Run deterministic aggregation (sum, average, min, max) on matching table cells."""
        normalized_chunks = []
        for chunk in chunks:
            sd = chunk.get("structured_data")
            if isinstance(sd, dict) and "headers" in sd and "rows" in sd:
                item = dict(chunk)
                table = dict(sd)
                if "kind" not in table:
                    table["kind"] = "table"
                item["structured_data"] = [table]
                normalized_chunks.append(item)
            else:
                normalized_chunks.append(chunk)
        return calculate_table_operation(question, normalized_chunks)

    @staticmethod
    def format_answer(
        result: CalculationResult,
        *,
        citation: str | None = None,
    ) -> str:
        """Format the calculated table result into a human-readable grounded answer string."""
        return format_calculation_answer(result, citation=citation)

    @classmethod
    def execute(
        cls,
        question: str,
        chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Execute structural data evaluation on retrieved document chunks."""
        with time_block("Structural Agent: Table Evaluation", category="SUBAGENT", level=2) as tb:
            calc_result = cls.calculate_table_aggregation(question, chunks)
            tb["details"] = f"calculated={calc_result is not None}"

            if calc_result is None:
                return {
                    "has_calculation": False,
                    "calculation": None,
                    "formatted_answer": None,
                    "log_entry": {
                        "subagent": "structural_agent",
                        "operation_found": False,
                    },
                }

            formatted = cls.format_answer(calc_result)
            return {
                "has_calculation": True,
                "calculation": calc_result,
                "formatted_answer": formatted,
                "log_entry": {
                    "subagent": "structural_agent",
                    "operation_found": True,
                    "operation": calc_result.operation,
                    "column": calc_result.column,
                    "value": str(calc_result.value),
                },
            }


run_structural_agent = StructuralAgent.execute
