"""
Unit tests for the CASI integration with the Learning Engine.
"""

import sys
import unittest
from unittest.mock import patch, MagicMock

# Mock the astartes_shared module before importing LearningEngine
sys.modules['astartes_shared'] = MagicMock()
sys.modules['astartes_shared.database'] = MagicMock()

from adaptive_engine.learning_engine import LearningEngine


class TestLearningEngineCasi(unittest.TestCase):
    def setUp(self):
        """Set up test cases for CASI integration."""
        self.engine = LearningEngine()
        self.cases = [
            {
                "id": "solar-system",
                "entities": ["sun", "earth"],
                "relations": [{"type": "revolves-around", "args": ["earth", "sun"]}],
                "attributes": {"sun": {"type": "star"}, "earth": {"type": "planet"}},
            },
            {
                "id": "rutherford-atom",
                "entities": ["nucleus", "electron"],
                "relations": [
                    {"type": "revolves-around", "args": ["electron", "nucleus"]}
                ],
                "attributes": {"nucleus": {"type": "core"}, "electron": {"type": "particle"}},
            },
            {
                "id": "geothermal-plant",
                "entities": ["core", "turbine"],
                "relations": [{"type": "powered-by", "args": ["turbine", "core"]}],
                 "attributes": {"core": {"type": "heat-source"}},
            },
        ]

    def test_run_casi_analysis_successful_schema_extraction(self):
        """Test a full CASI run that results in a valid schema."""
        result = self.engine.run_casi_analysis(
            base_case_id="solar-system",
            target_case_ids=["rutherford-atom", "geothermal-plant"],
            cases=self.cases,
        )

        self.assertIn("schema", result)
        self.assertIsNotNone(result["schema"])
        self.assertIn("analysis", result)
        self.assertIn("rutherford-atom", result["analysis"])
        self.assertEqual(
            result["analysis"]["rutherford-atom"], "Successful far-transfer mapping"
        )
        self.assertIn("geothermal-plant", result["analysis"])
        self.assertEqual(result["analysis"]["geothermal-plant"], "Mapping failed")

        schema = result["schema"]
        self.assertIn("id", schema)
        self.assertIn("entities", schema)
        self.assertIn("relations", schema)
        self.assertGreater(len(schema["entities"]), 0)


    def test_run_casi_analysis_no_valid_mappings(self):
        """Test a CASI run where no valid far-transfer mappings are found."""
        # In a real scenario, an ontology would be used to differentiate
        # near and far transfers. Here, we'll test a case where the
        # mapping is rejected as superficial.
        self.cases[0]["attributes"]["sun"] = {"type": "star"}
        self.cases[0]["attributes"]["earth"] = {"type": "planet"}

        # Create superficial attribute matches for both correspondences
        self.cases[1]["attributes"]["nucleus"] = {"type": "star"}
        self.cases[1]["attributes"]["electron"] = {"type": "planet"}

        result = self.engine.run_casi_analysis(
            base_case_id="solar-system",
            target_case_ids=["rutherford-atom"],
            cases=self.cases,
        )

        self.assertIn("schema", result)
        self.assertIsNone(result["schema"])
        self.assertIn("rutherford-atom", result["analysis"])
        self.assertEqual(result["analysis"]["rutherford-atom"], "Mapping is superficial")


if __name__ == "__main__":
    unittest.main()
