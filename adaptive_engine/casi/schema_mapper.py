"""
Conclusion-verified Analogical Schema Induction (CASI)
Part 1: Structure-Mapping Engine (SME)

Implements the core analogical reasoning by mapping a source (base)
case to a target case.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Mapping:
    """Represents a single analogical mapping between two cases."""

    base_case_id: str
    target_case_id: str
    score: float
    correspondences: dict[str, str] = field(default_factory=dict)
    candidate_inferences: list[Any] = field(default_factory=list)


class SchemaMapper:
    """
    Structure-Mapping Engine (SME) for analogical reasoning.

    Identifies common structures between a base and target case,
    generating a mapping and candidate inferences.
    """



    def __init__(self, cases: list[dict] | None = None):
        """
        Initialize the SME with a set of cases.

        Args:
            cases: A list of cases, where each case is a dict
                   with entities, relations, and attributes.
        """
        self._cases = {case["id"]: case for case in cases} if cases else {}

    def add_case(self, case: dict[str, Any]) -> None:
        """
        Add a case to the SME's knowledge base.

        A case should have at least 'id', 'entities', and 'relations'.
        Example:
        {
            "id": "solar-system",
            "entities": ["sun", "earth", "mars"],
            "relations": [
                {"type": "revolves-around", "args": ["earth", "sun"]},
                {"type": "attracts", "args": ["sun", "earth"]},
            ],
            "attributes": {
                "sun": {"type": "star", "mass": "large"},
                "earth": {"type": "planet", "mass": "small"},
            }
        }
        """
        if "id" not in case:
            raise ValueError("Case must have an 'id' field.")
        self._cases[case["id"]] = case

    def map_cases(self, base_id: str, target_id: str) -> Mapping | None:
        """
        Perform structure-mapping from a base case to a target case.

        Args:
            base_id: ID of the base case (source of analogy)
            target_id: ID of the target case

        Returns:
            A Mapping object with correspondences and inferences, or None
        """
        base = self._cases.get(base_id)
        target = self._cases.get(target_id)

        if not base or not target:
            raise ValueError("Base or target case not found.")

        # This is a simplified placeholder for a real SME implementation.
        # A real implementation would involve complex graph matching algorithms.
        correspondences, score = self._find_correspondences(base, target)

        if not correspondences:
            return None

        inferences = self._generate_inferences(base, target, correspondences)

        return Mapping(
            base_case_id=base_id,
            target_case_id=target_id,
            score=score,
            correspondences=correspondences,
            candidate_inferences=inferences,
        )

    def _find_correspondences(self, base: dict, target: dict) -> tuple[dict, float]:
        """
        Find correspondences using a greedy, iterative algorithm.
        This is a more robust implementation that builds a mapping by
        finding consistent relational pairings.
        """
        correspondences = {}
        base_relations = base.get("relations", [])
        target_relations = target.get("relations", [])

        # Use a set of tuples for efficient lookup of target relations
        target_rel_set = {
            (r["type"], tuple(r["args"])) for r in target.get("relations", [])
        }

        # Iteratively build the mapping
        changed = True
        while changed:
            changed = False
            for b_rel in base_relations:
                for t_rel in target_relations:
                    if b_rel["type"] == t_rel["type"] and len(b_rel["args"]) == len(
                        t_rel["args"]
                    ):
                        b_args = b_rel["args"]
                        t_args = t_rel["args"]
                        can_map = True
                        new_mappings = {}

                        for b_arg, t_arg in zip(b_args, t_args):
                            if b_arg in correspondences:
                                if correspondences[b_arg] != t_arg:
                                    can_map = False
                                    break
                            elif b_arg in new_mappings:
                                if new_mappings[b_arg] != t_arg:
                                    can_map = False
                                    break
                            else:
                                new_mappings[b_arg] = t_arg

                        if can_map and new_mappings:
                            correspondences.update(new_mappings)
                            changed = True

        if not correspondences:
            return {}, 0.0

        # Score the mapping based on the proportion of supported relations
        supported_relations_count = 0
        for b_rel in base_relations:
            mapped_args = [correspondences.get(arg) for arg in b_rel["args"]]
            if all(mapped_args):
                inferred_target_rel = (b_rel["type"], tuple(mapped_args))
                if inferred_target_rel in target_rel_set:
                    supported_relations_count += 1

        score = (
            supported_relations_count / len(base_relations) if base_relations else 0.0
        )

        return correspondences, score

    def _generate_inferences(
        self, base: dict, target: dict, correspondences: dict
    ) -> list:
        """
        Generate candidate inferences based on the mapping.
        An inference is a relation from the base that can be projected
        onto the target using the established correspondences.
        """
        inferences = []
        base_relations = base.get("relations", [])

        for rel in base_relations:
            is_transferable = True
            inferred_args = []

            for arg in rel["args"]:
                if arg in correspondences:
                    inferred_args.append(correspondences[arg])
                else:
                    is_transferable = False
                    break

            if is_transferable:
                inferences.append(
                    {
                        "type": "relation",
                        "relation": {"type": rel["type"], "args": inferred_args},
                    }
                )

        return inferences
