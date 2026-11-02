import unittest
import json
from unittest.mock import patch

from agents.orchestrator import plan_orchestration


def _model_response(plan):
    content = json.dumps(plan)
    return type(
        "Response",
        (),
        {
            "choices": [
                type(
                    "Choice",
                    (),
                    {"message": type("Message", (), {"content": content})()},
                )()
            ]
        },
    )()


class OrchestratorQueryUnderstandingTests(unittest.TestCase):
    def test_implicit_comparison_discovers_entities_before_comparing(self):
        response = _model_response({
            "action": "retrieve",
            "intent": "comparison",
            "rewritten_query": "Compare the different wars",
            "sub_questions": [
                {"id": "q1", "query": "Identify entities in the requested comparison."},
                {"id": "q2", "query": "Compare the requested entities."},
            ],
            "output_format": "table",
            "reason": "Comparison needs joint planning",
        })
        with patch("agents.orchestrator.safe_chat_create", return_value=response) as chat:
            plan = plan_orchestration("Compare the different wars")

        chat.assert_called_once()
        self.assertEqual(plan.intent, "comparison")
        self.assertEqual(plan.output_format, "table")
        self.assertEqual(len(plan.sub_questions), 2)

    def test_comparison_queries_stay_grounded_in_scaling_law_question(self):
        question = (
            "What is the main difference between the KM scaling law and the "
            "Chinchilla scaling law regarding the allocation of compute to "
            "model size and training data?"
        )
        response = _model_response({
            "action": "retrieve",
            "intent": "comparison",
            "rewritten_query": question,
            "sub_questions": [
                {"id": "q1", "query": question},
                {"id": "q2", "query": question},
            ],
            "output_format": "table",
            "reason": "Comparison needs joint planning",
        })
        with patch("agents.orchestrator.safe_chat_create", return_value=response) as chat:
            plan = plan_orchestration(question)

        chat.assert_called_once()
        self.assertEqual(plan.intent, "comparison")
        self.assertEqual(len(plan.sub_questions), 2)
        prompt = chat.call_args.kwargs["messages"][1]["content"]
        self.assertIn("KM scaling law", prompt)
        self.assertIn("Chinchilla scaling law", prompt)
        self.assertIn("compute", prompt)

    def test_clear_factual_query_does_not_call_llm(self):
        with patch("agents.orchestrator.safe_chat_create") as chat:
            plan = plan_orchestration("When did World War II end?")

        chat.assert_not_called()
        self.assertEqual(plan.sub_questions[0].question_type, "temporal")
        self.assertEqual(plan.sub_questions[0].direction, "extract_dates_and_order_events")
        self.assertFalse(plan.needs_verification)

    def test_definition_query_skips_slow_claim_verifier(self):
        with patch("agents.orchestrator.safe_chat_create") as chat:
            plan = plan_orchestration("What do you mean by LLM?")

        chat.assert_not_called()
        self.assertEqual(plan.sub_questions[0].question_type, "definition")
        self.assertEqual(plan.sub_questions[0].query, "LLM definition")
        self.assertFalse(plan.needs_verification)

    def test_malformed_what_is_mean_by_query_routes_to_definition_search(self):
        with patch("agents.orchestrator.safe_chat_create") as chat:
            plan = plan_orchestration("what is mean by DBMS?")

        chat.assert_not_called()
        self.assertEqual(plan.sub_questions[0].question_type, "definition")
        self.assertEqual(plan.sub_questions[0].query, "DBMS definition")
        self.assertFalse(plan.needs_verification)

    def test_plain_what_is_acronym_routes_to_definition_search(self):
        with patch("agents.orchestrator.safe_chat_create") as chat:
            plan = plan_orchestration("what is llm?")

        chat.assert_not_called()
        self.assertEqual(plan.sub_questions[0].question_type, "definition")
        self.assertEqual(plan.sub_questions[0].query, "llm definition")

    def test_explicit_verification_and_numeric_queries_keep_verification(self):
        with patch("agents.orchestrator.safe_chat_create"):
            verification_plan = plan_orchestration("Is the reported result true?")
            numeric_plan = plan_orchestration("What is the total revenue?")

        self.assertTrue(verification_plan.needs_verification)
        self.assertTrue(numeric_plan.needs_verification)

    def test_ambiguous_follow_up_without_history_clarifies_without_llm(self):
        with patch("agents.orchestrator.safe_chat_create") as chat:
            plan = plan_orchestration("Compare them")

        chat.assert_not_called()
        self.assertEqual(plan.action, "clarify")
        self.assertIn("referring", plan.clarifying_question.lower())

    def test_follow_up_with_history_is_planned_without_llm(self):
        history = [{"role": "user", "content": "What are the crop yields?"}]
        with patch("agents.orchestrator.safe_chat_create") as chat:
            plan = plan_orchestration("What about this?", history=history)

        chat.assert_not_called()
        self.assertEqual(plan.action, "retrieve")
        self.assertIn("crop yields", plan.rewritten_query.lower())

    def test_multi_part_question_uses_llm_planning(self):
        response = _model_response({
            "action": "retrieve",
            "intent": "multi_part",
            "rewritten_query": "What caused the war and what was its outcome",
            "sub_questions": [
                {"id": "q1", "query": "What caused the war?"},
                {"id": "q2", "query": "What was its outcome?"},
            ],
            "output_format": "bullets",
            "reason": "Question has independent parts",
        })
        with patch("agents.orchestrator.safe_chat_create", return_value=response) as chat:
            plan = plan_orchestration(
                "What caused the war and what was its outcome"
            )

        chat.assert_called_once()
        self.assertEqual(plan.intent, "multi_part")
        self.assertEqual(len(plan.sub_questions), 2)

    def test_unresolved_definite_reference_is_clarified_without_llm(self):
        with patch("agents.orchestrator.safe_chat_create") as chat:
            plan = plan_orchestration("When did the war take place?")

        chat.assert_not_called()
        self.assertEqual(plan.action, "clarify")
        self.assertIn("war", plan.clarifying_question.lower())

    def test_underspecified_summary_request_routes_to_summarizer(self):
        with patch("agents.orchestrator.safe_chat_create") as chat:
            plan = plan_orchestration("give me summary of document")

        chat.assert_not_called()
        self.assertEqual(plan.action, "summarize")

    def test_follow_up_after_summary_does_not_inherit_summary_intent(self):
        history = [
            {"role": "user", "content": "Give me a summary of the research paper."},
            {
                "role": "assistant",
                "content": "Here is the summary.",
                "workflow": {
                    "orchestrator_plan": {
                        "sub_questions": [{"doc_ids": ["paper-1"]}],
                    },
                },
            },
        ]

        with patch("agents.orchestrator.safe_chat_create") as chat:
            plan = plan_orchestration("What about this?", history=history)

        chat.assert_not_called()
        self.assertEqual(plan.action, "retrieve")
        self.assertNotEqual(plan.intent, "summary")

    def test_question_about_summary_statistics_is_not_summary_request(self):
        with patch("agents.orchestrator.safe_chat_create") as chat:
            plan = plan_orchestration("What are the summary statistics in the report?")

        chat.assert_not_called()
        self.assertEqual(plan.action, "retrieve")
        self.assertNotEqual(plan.intent, "summary")


if __name__ == "__main__":
    unittest.main()
