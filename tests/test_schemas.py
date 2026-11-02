import unittest

from pydantic import ValidationError

from agents.schemas import ClaimCheck


class ClaimCheckTests(unittest.TestCase):
    def test_normalizes_numeric_string_passage_ids(self):
        claim = ClaimCheck(
            claim="The answer is supported.",
            passage_ids=["5", " 2 "],
            supported="yes",
        )

        self.assertEqual(claim.passage_ids, [5, 2])

    def test_rejects_non_numeric_passage_ids(self):
        with self.assertRaises(ValidationError):
            ClaimCheck(
                claim="The answer is supported.",
                passage_ids=["not-a-passage"],
                supported="yes",
            )


if __name__ == "__main__":
    unittest.main()
