"""
Conclusion-verified Analogical Schema Induction (CASI)
Part 3: Transfer Scorer

Classifies the analogical transfer as "near" or "far" based on the
semantic distance between the base and target domains.
"""

from .schema_mapper import Mapping


class TransferScorer:
    """
    Scores the transfer distance of an analogy.

    - Near transfer: Analogy between closely related domains (e.g., car to truck).
    - Far transfer: Analogy between distant domains (e.g., solar system to atom).
    """

    def __init__(self, domain_ontologies: dict | None = None):
        """
        Initialize with domain ontologies or semantic information.

        Args:
            domain_ontologies: A dict mapping domain IDs to their semantic
                               properties, like categories or parent domains.
                               Example:
                               {
                                   "solar-system": {"category": "astronomy"},
                                   "rutherford-atom": {"category": "physics"},
                                   "car-engine": {"category": "mechanics"},
                               }
        """
        self._ontologies = domain_ontologies or {}

    def score_transfer(self, mapping: Mapping) -> tuple[str, float]:
        """
        Score the transfer distance for a given mapping.

        Args:
            mapping: The Mapping object from the SME.

        Returns:
            A tuple of ("near" or "far", and a numeric score 0-1).
        """
        base_id = mapping.base_case_id
        target_id = mapping.target_case_id

        base_domain_info = self._ontologies.get(base_id, {})
        target_domain_info = self._ontologies.get(target_id, {})

        # Simplified scoring: if domains are in the same category, it's near transfer.
        base_category = base_domain_info.get("category")
        target_category = target_domain_info.get("category")

        if base_category and target_category:
            if base_category == target_category:
                # Same category, so high similarity = near transfer
                distance_score = 0.2  # Closer to 0 is nearer
                transfer_type = "near"
            else:
                # Different categories, so low similarity = far transfer
                distance_score = 0.8  # Closer to 1 is farther
                transfer_type = "far"
        else:
            # If no ontology info, assume far transfer as a default
            distance_score = 1.0
            transfer_type = "far"

        return transfer_type, distance_score
