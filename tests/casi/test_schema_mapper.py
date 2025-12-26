"""
Unit tests for the Structure-Mapping Engine (SME).
"""

import unittest

from adaptive_engine.casi.schema_mapper import SchemaMapper


class TestSchemaMapper(unittest.TestCase):
    def setUp(self):
        """Set up test cases for the SME."""
        self.sme = SchemaMapper()
        self.solar_system = {
            "id": "solar-system",
            "entities": ["sun", "earth", "mars"],
            "relations": [
                {"type": "revolves-around", "args": ["earth", "sun"]},
                {"type": "revolves-around", "args": ["mars", "sun"]},
                {"type": "attracts", "args": ["sun", "earth"]},
                {"type": "attracts", "args": ["sun", "mars"]},
            ],
            "attributes": {
                "sun": {"type": "star", "mass": "large"},
                "earth": {"type": "planet", "mass": "small"},
                "mars": {"type": "planet", "mass": "small"},
            },
        }
        self.rutherford_atom = {
            "id": "rutherford-atom",
            "entities": ["nucleus", "electron"],
            "relations": [
                {"type": "revolves-around", "args": ["electron", "nucleus"]},
                {"type": "attracts", "args": ["nucleus", "electron"]},
            ],
            "attributes": {
                "nucleus": {"type": "core", "charge": "positive"},
                "electron": {"type": "particle", "charge": "negative"},
            },
        }
        self.sme.add_case(self.solar_system)
        self.sme.add_case(self.rutherford_atom)

    def test_add_case(self):
        """Test adding a case to the SME."""
        case_to_add = {"id": "new-case", "entities": [], "relations": []}
        self.sme.add_case(case_to_add)
        self.assertIn("new-case", self.sme._cases)

    def test_map_cases_successful(self):
        """Test a successful mapping between two analogous cases."""
        mapping = self.sme.map_cases("solar-system", "rutherford-atom")
        self.assertIsNotNone(mapping)
        self.assertEqual(mapping.base_case_id, "solar-system")
        self.assertEqual(mapping.target_case_id, "rutherford-atom")
        self.assertIn("sun", mapping.correspondences)
        self.assertEqual(mapping.correspondences["sun"], "nucleus")
        self.assertIn("earth", mapping.correspondences)
        self.assertEqual(mapping.correspondences["earth"], "electron")
        self.assertAlmostEqual(mapping.score, 1.0) # All relations are supported

    def test_map_cases_no_common_structure(self):
        """Test mapping between cases with no common relational structure."""
        unrelated_case = {
            "id": "unrelated",
            "entities": ["a", "b"],
            "relations": [{"type": "is-on-top-of", "args": ["a", "b"]}],
        }
        self.sme.add_case(unrelated_case)
        mapping = self.sme.map_cases("solar-system", "unrelated")
        self.assertIsNone(mapping)

    def test_generate_inferences(self):
        """Test the generation of candidate inferences."""
        # Create a modified solar system case with an extra relation
        # to ensure an inference is generated.
        solar_system_with_extra = self.solar_system.copy()
        solar_system_with_extra["relations"].append(
            {"type": "has-moon", "args": ["earth"]}
        )
        self.sme.add_case(solar_system_with_extra)

        mapping = self.sme.map_cases("solar-system", "rutherford-atom")

        # The simple inference generator might not produce this,
        # but a real one would infer that the electron might have something analogous to a moon.
        # For our simple implementation, we check for inferences that are transferable.
        # Let's check a more direct inference.
        base = {
            "id": "base",
            "entities": ["a", "b", "c"],
            "relations": [
                {"type": "likes", "args": ["a", "b"]},
                {"type": "dislikes", "args": ["b", "c"]},
            ],
        }
        target = {
            "id": "target",
            "entities": ["x", "y"],
            "relations": [{"type": "likes", "args": ["x", "y"]}],
        }
        self.sme.add_case(base)
        self.sme.add_case(target)
        mapping = self.sme.map_cases("base", "target")
        self.assertIsNotNone(mapping)
        # It should infer that Y might dislike something, but since C is not mapped,
        # it cannot complete the inference.
        # Let's modify target to make inference possible
        target["entities"].append("z")
        mapping = self.sme.map_cases("base", "target") # remap
        # a -> x, b -> y, what about c? no correspondence.
        # let's add a common relation for c
        base["relations"].append({"type": "related-to", "args": ["c", "a"]})
        target["relations"].append({"type": "related-to", "args": ["z", "x"]})
        self.sme.add_case(base)
        self.sme.add_case(target)
        mapping = self.sme.map_cases("base", "target")
        self.assertIsNotNone(mapping.candidate_inferences)
        self.assertGreater(len(mapping.candidate_inferences), 0)

        # Check if the expected inference is present, regardless of order
        expected_inference = {"type": "dislikes", "args": ["y", "z"]}
        found = False
        for inference in mapping.candidate_inferences:
            if inference["relation"] == expected_inference:
                found = True
                break
        self.assertTrue(found, "Expected inference not found")


if __name__ == "__main__":
    unittest.main()
