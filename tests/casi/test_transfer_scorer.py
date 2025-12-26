"""
Unit tests for the Transfer Scorer.
"""

import unittest

from adaptive_engine.casi.schema_mapper import Mapping
from adaptive_engine.casi.transfer_scorer import TransferScorer


class TestTransferScorer(unittest.TestCase):
    def setUp(self):
        """Set up test cases for the Transfer Scorer."""
        self.domain_ontologies = {
            "solar-system": {"category": "astronomy"},
            "rutherford-atom": {"category": "physics"},
            "car-engine": {"category": "mechanics"},
            "truck-engine": {"category": "mechanics"},
        }
        self.scorer = TransferScorer(domain_ontologies=self.domain_ontologies)

    def test_score_transfer_far(self):
        """Test scoring of a far transfer analogy."""
        mapping = Mapping(
            base_case_id="solar-system",
            target_case_id="rutherford-atom",
            score=0.8,
        )
        transfer_type, distance = self.scorer.score_transfer(mapping)
        self.assertEqual(transfer_type, "far")
        self.assertGreater(distance, 0.5)

    def test_score_transfer_near(self):
        """Test scoring of a near transfer analogy."""
        mapping = Mapping(
            base_case_id="car-engine",
            target_case_id="truck-engine",
            score=0.9,
        )
        transfer_type, distance = self.scorer.score_transfer(mapping)
        self.assertEqual(transfer_type, "near")
        self.assertLess(distance, 0.5)

    def test_score_transfer_no_ontology(self):
        """Test scoring when one of the domains is not in the ontology."""
        mapping = Mapping(
            base_case_id="solar-system",
            target_case_id="unknown-domain",
            score=0.7,
        )
        transfer_type, distance = self.scorer.score_transfer(mapping)
        # Should default to far transfer
        self.assertEqual(transfer_type, "far")
        self.assertEqual(distance, 1.0)


if __name__ == "__main__":
    unittest.main()
