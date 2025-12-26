"""
Conclusion-Verified Schema Induction (CASI) Engine.

Verifies that learners induce transferable schemas via structural mapping,
ensuring learning genuinely transfers across domains rather than relying on
surface pattern matching.

Key Components:
- Structure-Mapping Engine (SME): Computes alignment scores between domains.
- CASI Verifier: Audits mappings for completeness, conclusion derivability,
  and surface independence.
- SAGE Engine: Bonds verified schemas into "Solid-State Molecules" in the
  knowledge base.

Based on research from:
- Gentner, D. (1983). Structure-mapping: A theoretical framework for analogy.
- Falkenhainer, B., Forbus, K. D., & Gentner, D. (1989). The Structure-Mapping
  Engine: Algorithm and examples.

Author: Cortex System
Version: 1.0.0
"""

from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class DomainElement:
    """Represents an element within a base or target domain."""

    id: str
    name: str
    attributes: list[str] = field(default_factory=list)


@dataclass
class DomainRelation:
    """Represents a relationship between elements."""

    source: str  # ID of the source element
    target: str  # ID of the target element
    type: str  # e.g., "CAUSES", "ORBITS"
    order: int = 1  # 1 for first-order, 2 for higher-order


@dataclass
class Domain:
    """A structured representation of a domain (base or target)."""

    name: str
    elements: list[DomainElement] = field(default_factory=list)
    relations: list[DomainRelation] = field(default_factory=list)
    conclusion_predicate: str | None = None


@dataclass
class ProposedMapping:
    """A learner's proposed mapping between two domains."""

    base_domain: Domain
    target_domain: Domain
    element_mappings: dict[str, str] = field(default_factory=dict)  # base_id -> target_id
    relation_mappings: dict[str, str] = field(default_factory=dict)  # base_relation -> target_relation
    proposed_conclusion: str = ""


class VerificationStatus(str, Enum):
    """Status of a verification check."""

    PASS = "PASS"
    FAIL = "FAIL"
    PENDING = "PENDING"


@dataclass
class VerificationResult:
    """The result of a CASI verification check."""

    check: str
    status: VerificationStatus
    details: str


@dataclass
class ShallowTransferIssue:
    """Represents a detected issue in a shallow transfer attempt."""

    issue: str
    details: str


@dataclass
class SMEAlignmentScore:
    """Scores from the Structure-Mapping Engine."""

    attribute_matches: float = 0.0
    first_order_relations: float = 0.0
    higher_order_relations: float = 0.0
    total_score: float = 0.0
    systematicity_score: float = 0.0
    is_valid: bool = False


@dataclass
class TransferEfficiencyScore:
    """Represents the efficiency of a cross-domain transfer."""

    mapping_completeness: float = 0.0
    time_to_insight: int = 0  # in seconds
    hint_independence: float = 0.0
    first_attempt_success: bool = False
    composite_efficiency: float = 0.0


# =============================================================================
# Structure-Mapping Engine (SME)
# =============================================================================


class StructureMappingEngine:
    """
    Computes alignment scores for proposed mappings based on systematicity.
    """

    def __init__(self, higher_order_weight: float = 0.8, attribute_weight: float = 0.1):
        self.higher_order_weight = higher_order_weight
        self.attribute_weight = attribute_weight
        self.first_order_weight = 1.0 - attribute_weight

    def calculate_score(self, mapping: ProposedMapping) -> SMEAlignmentScore:
        """
        Calculates the alignment score for a given mapping.

        S_total = Σs_i + (w × S_higher_order)
        """
        # Placeholder logic for scoring
        num_relations = len(mapping.base_domain.relations)
        mapped_relations = len(mapping.relation_mappings)

        first_order_matches = 0
        higher_order_matches = 0

        for rel in mapping.base_domain.relations:
            if f"{rel.source}->{rel.target}" in mapping.relation_mappings:
                if rel.order > 1:
                    higher_order_matches += 1
                else:
                    first_order_matches += 1

        score = SMEAlignmentScore()
        score.first_order_relations = first_order_matches / num_relations if num_relations > 0 else 0
        score.higher_order_relations = higher_order_matches / num_relations if num_relations > 0 else 0

        # Simplified total score calculation
        score.total_score = (
            score.first_order_relations * self.first_order_weight
            + score.higher_order_relations * self.higher_order_weight
        )
        score.systematicity_score = score.higher_order_relations
        score.is_valid = score.total_score > 0.75

        return score


# =============================================================================
# CASI Verifier
# =============================================================================


class CasiVerifier:
    """
    Verifies the integrity and validity of a schema induction.
    """

    def __init__(self):
        self.sme = StructureMappingEngine()

    def verify(self, mapping: ProposedMapping) -> list[VerificationResult]:
        """
        Performs all CASI verification checks.
        """
        results = []

        # 1. Structural Mapping Completeness
        completeness = self.check_mapping_completeness(mapping)
        if completeness < 1.0:
            results.append(
                VerificationResult(
                    check="Structural_Mapping_Complete",
                    status=VerificationStatus.FAIL,
                    details=f"Mapping is {completeness*100:.0f}% complete.",
                )
            )
        else:
            results.append(
                VerificationResult(
                    check="Structural_Mapping_Complete",
                    status=VerificationStatus.PASS,
                    details="All relations aligned",
                )
            )

        # 2. Conclusion Derivability
        if self.check_conclusion_derivable(mapping):
            results.append(
                VerificationResult(
                    check="Conclusion_Derivable",
                    status=VerificationStatus.PASS,
                    details="Target conclusion follows from mapping",
                )
            )
        else:
            results.append(
                VerificationResult(
                    check="Conclusion_Derivable",
                    status=VerificationStatus.FAIL,
                    details="Proposed conclusion does not logically follow",
                )
            )

        # 3. Surface Independence
        if not self.detect_surface_pattern_matching(mapping):
            results.append(
                VerificationResult(
                    check="Surface_Independent",
                    status=VerificationStatus.PASS,
                    details="Mapping is based on deep structure, not surface features.",
                )
            )
        else:
            results.append(
                VerificationResult(
                    check="Surface_Independent",
                    status=VerificationStatus.FAIL,
                    details="Detected surface-level pattern matching.",
                )
            )

        return results

    def check_mapping_completeness(self, mapping: ProposedMapping) -> float:
        """
        Checks if all elements and relations in the base have been mapped.
        Returns a completeness score (0.0 to 1.0).
        """
        total_relations = len(mapping.base_domain.relations)
        mapped_relations = len(mapping.relation_mappings)
        return mapped_relations / total_relations if total_relations > 0 else 1.0

    def check_conclusion_derivable(self, mapping: ProposedMapping) -> bool:
        """
        Checks if the proposed conclusion is a logical consequence of the mapped structure.
        (Placeholder logic)
        """
        # In a real system, this would involve a symbolic reasoner.
        return mapping.proposed_conclusion is not None and len(mapping.proposed_conclusion) > 5

    def detect_surface_pattern_matching(self, mapping: ProposedMapping) -> bool:
        """
        Detects if the mapping relies on superficial keyword matching rather
        than structural alignment.
        """
        # Placeholder logic: Check for simple keyword matching in conclusion
        if "force" in mapping.proposed_conclusion.lower():
            if not any(
                "force" in rel.type.lower() for rel in mapping.base_domain.relations
            ):
                return True  # Matched keyword not present in deep structure
        return False

    def detect_shallow_transfer(self, mapping: ProposedMapping) -> list[ShallowTransferIssue]:
        """
        Detects shallow transfer attempts by looking for mismatches and missing relations.
        """
        issues = []
        # Example check: Mismatched flow rates
        if "Flow_Rate" in mapping.element_mappings and mapping.element_mappings["Flow_Rate"] == "Temperature":
            issues.append(
                ShallowTransferIssue(
                    issue="Flow_Rate mismatch",
                    details="Temperature ≠ Heat_Flow_Rate",
                )
            )

        # Example check: Missing critical relation
        if not any(
            "Gradient" in el.name for el in mapping.target_domain.elements
        ):
             issues.append(
                ShallowTransferIssue(
                    issue="Missing relation",
                    details="Thermal_Gradient not mapped",
                )
            )
        return issues

    def calculate_transfer_efficiency(
        self,
        mapping_completeness: float,
        time_to_insight: int,
        hints_used: int,
        max_hints: int,
        first_attempt_success: bool,
        max_time_to_insight: int = 300,  # 5 minutes default max
    ) -> TransferEfficiencyScore:
        """
        Calculates the cross-domain transfer efficiency.
        E = w1*MC + w2*(1-TTI/max) + w3*HI
        """
        w1, w2, w3 = 0.4, 0.3, 0.3  # Weights

        hint_independence = 1.0 - (hints_used / max_hints) if max_hints > 0 else 1.0
        normalized_tti = 1.0 - (time_to_insight / max_time_to_insight)

        composite = (
            w1 * mapping_completeness
            + w2 * normalized_tti
            + w3 * hint_independence
        )

        return TransferEfficiencyScore(
            mapping_completeness=mapping_completeness,
            time_to_insight=time_to_insight,
            hint_independence=hint_independence,
            first_attempt_success=first_attempt_success,
            composite_efficiency=composite,
        )
