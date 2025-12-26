"""
Unit tests for the Surface Validator.
"""

import unittest

from adaptive_engine.casi.schema_mapper import Mapping
from adaptive_engine.casi.surface_validator import SurfaceValidator


class TestSurfaceValidator(unittest.TestCase):
    def setUp(self):
        """Set up test cases for the Surface Validator."""
        self.cases_with_attributes = {
            "base-deep": {
                "attributes": {
                    "a": {"color": "red"},
                    "b": {"shape": "circle"},
                }
            },
            "target-deep": {
                "attributes": {
                    "x": {"color": "blue"},
                    "y": {"shape": "square"},
                }
            },
            "base-surface": {
                "attributes": {
                    "a": {"color": "yellow", "size": "large"},
                    "b": {"color": "blue"},
                }
            },
            "target-surface": {
                "attributes": {
                    "x": {"color": "yellow", "size": "small"},
                    "y": {"color": "green"},
                }
            },
        }
        self.validator = SurfaceValidator(cases_with_attributes=self.cases_with_attributes)

    def test_is_mapping_deep_true(self):
        """Test a mapping that is deep (structural) and should be valid."""
        deep_mapping = Mapping(
            base_case_id="base-deep",
            target_case_id="target-deep",
            score=0.8,
            correspondences={"a": "x", "b": "y"},
        )
        self.assertTrue(self.validator.is_mapping_deep(deep_mapping))

    def test_is_mapping_deep_false_superficial(self):
        """Test a mapping that is superficial and should be invalid."""
        surface_mapping = Mapping(
            base_case_id="base-surface",
            target_case_id="target-surface",
            score=0.5,
            correspondences={"a": "x", "b": "y"},
        )
        # Here, a -> x has a surface match on "color": "yellow".
        # With default threshold of 0.5, and 1 of 2 correspondences being a surface match,
        # the ratio is 0.5, so is_mapping_deep should be True (<= threshold).
        # Let's test the threshold.
        self.assertTrue(self.validator.is_mapping_deep(surface_mapping, threshold=0.5))
        self.assertFalse(self.validator.is_mapping_deep(surface_mapping, threshold=0.4))

    def test_no_attributes(self):
        """Test a mapping where cases have no attributes."""
        mapping = Mapping(
            base_case_id="base-no-attr",
            target_case_id="target-no-attr",
            score=0.8,
            correspondences={"a": "x"},
        )
        # Should be considered deep by default
        self.assertTrue(self.validator.is_mapping_deep(mapping))

    def test_partial_surface_match(self):
        """Test a mapping with a mix of surface and structural correspondences."""
        mixed_mapping = Mapping(
            base_case_id="base-surface",
            target_case_id="target-deep", # mix and match
            score=0.7,
            correspondences={"a": "x", "b": "y"}, # a->x is surface, b->y is not
        )
        self.cases_with_attributes["base-surface"]["attributes"]["a"] = {"color": "blue"}
        self.validator = SurfaceValidator(cases_with_attributes=self.cases_with_attributes)
        self.assertTrue(self.validator.is_mapping_deep(mixed_mapping, threshold=0.5))


if __name__ == "__main__":
    unittest.main()
