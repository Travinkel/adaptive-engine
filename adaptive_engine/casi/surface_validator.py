"""
Conclusion-verified Analogical Schema Induction (CASI)
Part 5: Surface Validator

Verifies that the analogical mapping is not based on superficial
surface similarities between the base and target entities.
"""

from .schema_mapper import Mapping


class SurfaceValidator:
    """
    Validates that a mapping is based on relational structure, not just
    surface-level attribute matching.

    A key principle of analogy is "systematicity," meaning mappings are
    preferred when they form deeply interconnected structures, rather
    than matching on isolated attributes (e.g., mapping "sun" to "nucleus"
    because both are "yellow" would be a surface match).
    """

    def __init__(self, cases_with_attributes: dict | None = None):
        """
        Initialize with case data that includes attributes.

        Args:
            cases_with_attributes: A dict of cases where each case contains
                                   an 'attributes' dict for its entities.
                                   Example:
                                   {
                                       "solar-system": {
                                           "attributes": {
                                               "sun": {"color": "yellow", "temp": "hot"},
                                               "earth": {"color": "blue", "temp": "warm"},
                                           }
                                       }
                                   }
        """
        self._cases = cases_with_attributes or {}

    def is_mapping_deep(self, mapping: Mapping, threshold: float = 0.5) -> bool:
        """
        Check if a mapping is deep (structural) rather than superficial.

        Args:
            mapping: The Mapping object from the SME.
            threshold: A value from 0 to 1. If the ratio of surface matches
                       to total correspondences exceeds this, the mapping is
                       considered superficial.

        Returns:
            True if the mapping is deep, False if it's superficial.
        """
        base_id = mapping.base_case_id
        target_id = mapping.target_case_id

        base_attrs = self._cases.get(base_id, {}).get("attributes", {})
        target_attrs = self._cases.get(target_id, {}).get("attributes", {})

        if not base_attrs or not target_attrs:
            return True  # No attributes to compare, so can't be a surface match

        surface_match_count = 0
        total_correspondences = len(mapping.correspondences)

        if total_correspondences == 0:
            return True  # No mapping, so not a surface one

        for base_entity, target_entity in mapping.correspondences.items():
            b_attrs = base_attrs.get(base_entity, {})
            t_attrs = target_attrs.get(target_entity, {})

            if self._attributes_match(b_attrs, t_attrs):
                surface_match_count += 1

        # Calculate the ratio of surface matches
        surface_match_ratio = surface_match_count / total_correspondences

        # If the ratio is too high, it's a superficial mapping
        return surface_match_ratio <= threshold

    def _attributes_match(self, attrs1: dict, attrs2: dict) -> bool:
        """
        Check if any attributes (key-value pairs) are identical between two entities.
        """
        if not attrs1 or not attrs2:
            return False

        for key, value in attrs1.items():
            if key in attrs2 and attrs2[key] == value:
                return True
        return False
