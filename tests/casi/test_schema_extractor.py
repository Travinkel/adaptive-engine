"""
Unit tests for the Schema Extractor.
"""

import unittest

from adaptive_engine.casi.schema_extractor import SchemaExtractor
from adaptive_engine.casi.schema_mapper import Mapping


class TestSchemaExtractor(unittest.TestCase):
    def setUp(self):
        """Set up test cases for the Schema Extractor."""
        self.extractor = SchemaExtractor()
        self.base_cases = {
            "base1": {
                "id": "base1",
                "relations": [
                    {"type": "relationA", "args": ["x", "y"]},
                    {"type": "relationB", "args": ["y", "z"]},
                ],
            },
            "base2": {
                "id": "base2",
                "relations": [
                    {"type": "relationA", "args": ["x", "y"]},
                    {"type": "relationC", "args": ["a", "b"]},
                ],
            },
        }
        self.mapping1 = Mapping(base_case_id="base1", target_case_id="t1", score=0.8)
        self.mapping2 = Mapping(base_case_id="base2", target_case_id="t2", score=0.9)

    def test_extract_schema_successful(self):
        """Test successful extraction of a schema from multiple mappings."""
        schema = self.extractor.extract_schema(
            [self.mapping1, self.mapping2], self.base_cases
        )
        self.assertIn("id", schema)
        self.assertIn("entities", schema)
        self.assertEqual(len(schema["relations"]), 1)
        self.assertEqual(schema["relations"][0]["type"], "relationA")
        self.assertIn("relations", schema)
        self.assertGreater(len(schema["entities"]), 0)
        self.assertTrue(all(e.startswith("?VAR") for e in schema["entities"]))
        self.assertGreater(len(schema["relations"]), 0)
        self.assertEqual(len(schema["relations"][0]["args"]), 2)
        self.assertTrue(all(a.startswith("?VAR") for a in schema["relations"][0]["args"]))
        self.assertEqual(len(schema["source_mappings"]), 2)

    def test_extract_schema_no_mappings(self):
        """Test schema extraction with an empty list of mappings."""
        schema = self.extractor.extract_schema([], self.base_cases)
        self.assertEqual(schema, {})

    def test_placeholder_variable_generation(self):
        """Test that placeholders are generated correctly and consistently."""
        # This test is implicitly covered by test_extract_schema_successful,
        # but an explicit test can be more robust.
        schema = self.extractor.extract_schema([self.mapping1], self.base_cases)
        relation = schema["relations"][0]
        self.assertEqual(len(relation["args"]), 2)
        self.assertTrue(all(a.startswith("?VAR") for a in relation["args"]))


if __name__ == "__main__":
    unittest.main()
