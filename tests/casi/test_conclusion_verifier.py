"""
Unit tests for the Conclusion Verifier.
"""

import unittest

from adaptive_engine.casi.conclusion_verifier import ConclusionVerifier
from adaptive_engine.casi.schema_mapper import Mapping


class TestConclusionVerifier(unittest.TestCase):
    def setUp(self):
        """Set up test cases for the Conclusion Verifier."""
        self.target_kb = {
            "relations": [
                {"type": "attracts", "args": ["moon", "earth"]},
                {"type": "orbits", "args": ["moon", "earth"]},
            ],
            "valid_relations": ["attracts", "orbits", "repels"],
        }
        self.verifier = ConclusionVerifier(target_knowledge_base=self.target_kb)
        self.mapping = Mapping(
            base_case_id="base",
            target_case_id="target",
            score=0.8,
            correspondences={"a": "moon", "b": "earth"},
            candidate_inferences=[
                {
                    "type": "relation",
                    "relation": {"type": "attracts", "args": ["moon", "earth"]},
                },
                {
                    "type": "relation",
                    "relation": {"type": "repels", "args": ["moon", "earth"]},
                },
                {
                    "type": "relation",
                    "relation": {"type": "unrelated-op", "args": ["moon", "earth"]},
                },
            ],
        )

    def test_verify_conclusion_valid(self):
        """Test verification of a valid conclusion (already known)."""
        conclusion = {
            "type": "relation",
            "relation": {"type": "attracts", "args": ["moon", "earth"]},
        }
        self.assertTrue(self.verifier.verify_conclusion(self.mapping, conclusion))

    def test_verify_conclusion_contradiction(self):
        """Test that a conclusion contradicting the KB is invalid."""
        # KB says moon attracts earth, so "repels" should be a contradiction.
        contradictory_conclusion = {
            "type": "relation",
            "relation": {"type": "repels", "args": ["moon", "earth"]},
        }
        self.assertFalse(self.verifier.verify_conclusion(self.mapping, contradictory_conclusion))

    def test_verify_conclusion_implausible(self):
        """Test verification of a conclusion with an implausible relation type."""
        conclusion = {
            "type": "relation",
            "relation": {"type": "unrelated-op", "args": ["moon", "earth"]},
        }
        self.assertFalse(self.verifier.verify_conclusion(self.mapping, conclusion))

    def test_filter_valid_inferences(self):
        """Test the filtering of a list of candidate inferences."""
        # Add a plausible but not-yet-known inference to the list
        self.mapping.candidate_inferences.append(
            {
                "type": "relation",
                "relation": {"type": "repels", "args": ["obj1", "obj2"]},
            }
        )
        self.target_kb["valid_relations"].append("repels")
        self.verifier = ConclusionVerifier(self.target_kb)

        valid_inferences = self.verifier.filter_valid_inferences(self.mapping, self.target_kb)
        self.assertEqual(len(valid_inferences), 2) # a repels inference is now a contradiction
        valid_types = {inf["relation"]["type"] for inf in valid_inferences}
        self.assertIn("attracts", valid_types)
        self.assertIn("repels", valid_types)
        self.assertNotIn("unrelated-op", valid_types)


if __name__ == "__main__":
    unittest.main()
