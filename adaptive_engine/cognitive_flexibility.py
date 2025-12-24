"""
Cognitive Flexibility Theory (CFT) Criss-Crossing Engine.

Implements Spiro et al.'s Cognitive Flexibility Theory for presenting
concepts through multiple expert perspectives and entry points,
helping learners avoid reductive biases and develop multidimensional
mental models.

Work Order: WO-AE-008
Tags: @CFT, @CrissCrossing, @Spiro, @Domain-Cognitive
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
import statistics


class ExpertLens(str, Enum):
    """Expert perspective lenses for viewing concepts."""

    SPIVAK = "spivak"      # Formal mathematical rigor
    RICHTER = "richter"    # Systems/implementation safety
    KNUTH = "knuth"        # Algorithmic efficiency
    GOTZSCHE = "gotzsche"  # Evidence-based skepticism
    KING = "king"          # Narrative/communication craft


class DomainType(str, Enum):
    """Classification of domain structure."""

    WELL_STRUCTURED = "well_structured"   # Clear rules, single solutions
    ILL_STRUCTURED = "ill_structured"     # Complex, multiple valid approaches


class TransferStatus(str, Enum):
    """Transfer readiness status."""

    NOT_READY = "not_ready"
    DEVELOPING = "developing"
    READY = "ready"
    CERTIFIED = "certified"


@dataclass
class LensConfiguration:
    """Configuration for an expert lens."""

    lens_id: ExpertLens
    agent_name: str
    domain_focus: str
    key_questions: List[str] = field(default_factory=list)
    evaluation_criteria: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "lens_id": self.lens_id.value,
            "agent_name": self.agent_name,
            "domain_focus": self.domain_focus,
            "key_questions": self.key_questions,
            "evaluation_criteria": self.evaluation_criteria,
        }


# Default lens configurations
DEFAULT_LENSES: Dict[ExpertLens, LensConfiguration] = {
    ExpertLens.SPIVAK: LensConfiguration(
        lens_id=ExpertLens.SPIVAK,
        agent_name="Spivak_Agent",
        domain_focus="Formal mathematical rigor",
        key_questions=[
            "What is the precise definition?",
            "What are the boundary conditions?",
            "How would you prove this formally?",
        ],
        evaluation_criteria=["Precision", "Completeness", "Logical consistency"],
    ),
    ExpertLens.RICHTER: LensConfiguration(
        lens_id=ExpertLens.RICHTER,
        agent_name="Richter_Agent",
        domain_focus="Systems/implementation safety",
        key_questions=[
            "What could go wrong in production?",
            "What are the failure modes?",
            "How do we handle edge cases safely?",
        ],
        evaluation_criteria=["Safety", "Robustness", "Error handling"],
    ),
    ExpertLens.KNUTH: LensConfiguration(
        lens_id=ExpertLens.KNUTH,
        agent_name="Knuth_Agent",
        domain_focus="Algorithmic efficiency",
        key_questions=[
            "What is the time complexity?",
            "What is the space complexity?",
            "Can we optimize this further?",
        ],
        evaluation_criteria=["Performance", "Scalability", "Resource usage"],
    ),
    ExpertLens.GOTZSCHE: LensConfiguration(
        lens_id=ExpertLens.GOTZSCHE,
        agent_name="Gotzsche_Agent",
        domain_focus="Evidence-based skepticism",
        key_questions=[
            "What is the evidence for this claim?",
            "What biases might affect this?",
            "What are the counterarguments?",
        ],
        evaluation_criteria=["Evidence quality", "Bias awareness", "Critical thinking"],
    ),
    ExpertLens.KING: LensConfiguration(
        lens_id=ExpertLens.KING,
        agent_name="King_Agent",
        domain_focus="Narrative/communication craft",
        key_questions=[
            "How would you explain this simply?",
            "What's the story here?",
            "How does this connect to the reader?",
        ],
        evaluation_criteria=["Clarity", "Engagement", "Accessibility"],
    ),
}


@dataclass
class EntryPoint:
    """An entry point into a concept's landscape."""

    name: str
    domain_frame: str
    first_concept: str
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "domain_frame": self.domain_frame,
            "first_concept": self.first_concept,
            "description": self.description,
        }


@dataclass
class ConceptLandscape:
    """A concept with multiple entry points and perspectives."""

    concept_id: str
    concept_name: str
    entry_points: List[EntryPoint] = field(default_factory=list)
    available_lenses: Set[ExpertLens] = field(default_factory=set)
    domain_type: DomainType = DomainType.WELL_STRUCTURED

    def add_entry_point(self, entry_point: EntryPoint) -> None:
        """Add an entry point to the concept."""
        self.entry_points.append(entry_point)

    def to_dict(self) -> dict:
        return {
            "concept_id": self.concept_id,
            "concept_name": self.concept_name,
            "entry_points": [ep.to_dict() for ep in self.entry_points],
            "available_lenses": [lens.value for lens in self.available_lenses],
            "domain_type": self.domain_type.value,
        }


@dataclass
class CaseStudy:
    """A case study for ill-structured domain learning."""

    case_id: str
    context: str
    relevant_concepts: List[str] = field(default_factory=list)
    domain: str = ""
    complexity: str = "medium"  # low, medium, high

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "context": self.context,
            "relevant_concepts": self.relevant_concepts,
            "domain": self.domain,
            "complexity": self.complexity,
        }


@dataclass
class PerspectiveCoverage:
    """Tracks which perspectives have been applied to concepts."""

    perspective: str
    coverage_percent: float
    gap_areas: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "perspective": self.perspective,
            "coverage_percent": round(self.coverage_percent, 1),
            "gap_areas": self.gap_areas,
        }


@dataclass
class LearnerCFTState:
    """Tracks a learner's CFT progress."""

    learner_id: str
    entry_points_used: Dict[str, List[str]] = field(default_factory=dict)  # concept -> entry points
    lenses_applied: Dict[str, Set[ExpertLens]] = field(default_factory=dict)  # concept -> lenses
    perspective_coverage: Dict[str, float] = field(default_factory=dict)  # perspective -> coverage %
    concepts_with_bias: Set[str] = field(default_factory=set)  # concepts with detected bias
    criss_cross_count: int = 0  # total perspective switches

    def record_entry_point(self, concept_id: str, entry_point: str) -> None:
        """Record that an entry point was used for a concept."""
        if concept_id not in self.entry_points_used:
            self.entry_points_used[concept_id] = []
        if entry_point not in self.entry_points_used[concept_id]:
            self.entry_points_used[concept_id].append(entry_point)

    def record_lens_use(self, concept_id: str, lens: ExpertLens) -> None:
        """Record that a lens was applied to a concept."""
        if concept_id not in self.lenses_applied:
            self.lenses_applied[concept_id] = set()
        if lens not in self.lenses_applied[concept_id]:
            self.lenses_applied[concept_id].add(lens)
            self.criss_cross_count += 1

    def get_entry_point_count(self, concept_id: str) -> int:
        """Get number of entry points used for a concept."""
        return len(self.entry_points_used.get(concept_id, []))

    def get_lens_count(self, concept_id: str) -> int:
        """Get number of lenses applied to a concept."""
        return len(self.lenses_applied.get(concept_id, set()))

    def to_dict(self) -> dict:
        return {
            "learner_id": self.learner_id,
            "entry_points_used": self.entry_points_used,
            "lenses_applied": {k: [l.value for l in v] for k, v in self.lenses_applied.items()},
            "perspective_coverage": self.perspective_coverage,
            "concepts_with_bias": list(self.concepts_with_bias),
            "criss_cross_count": self.criss_cross_count,
        }


@dataclass
class BiasDetection:
    """Result of bias detection analysis."""

    is_biased: bool
    dominant_perspective: Optional[str] = None
    ignored_perspectives: List[str] = field(default_factory=list)
    recommended_action: str = ""
    contrasting_case: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "is_biased": self.is_biased,
            "dominant_perspective": self.dominant_perspective,
            "ignored_perspectives": self.ignored_perspectives,
            "recommended_action": self.recommended_action,
            "contrasting_case": self.contrasting_case,
        }


@dataclass
class TransferReadinessMetrics:
    """Metrics for assessing transfer readiness."""

    entry_points_navigated: float  # avg per concept
    perspectives_integrated: float  # avg per concept
    novel_assembly_success: float  # 0-100%
    reductive_bias_score: float  # 0-1, lower is better
    status: TransferStatus = TransferStatus.NOT_READY

    # Thresholds
    ENTRY_POINT_THRESHOLD = 3.0
    PERSPECTIVE_THRESHOLD = 3.0
    ASSEMBLY_THRESHOLD = 70.0  # percent
    BIAS_THRESHOLD = 0.30

    def evaluate(self) -> TransferStatus:
        """Evaluate transfer readiness based on metrics."""
        passes = 0
        if self.entry_points_navigated >= self.ENTRY_POINT_THRESHOLD:
            passes += 1
        if self.perspectives_integrated >= self.PERSPECTIVE_THRESHOLD:
            passes += 1
        if self.novel_assembly_success >= self.ASSEMBLY_THRESHOLD:
            passes += 1
        if self.reductive_bias_score <= self.BIAS_THRESHOLD:
            passes += 1

        if passes == 4:
            self.status = TransferStatus.CERTIFIED
        elif passes >= 3:
            self.status = TransferStatus.READY
        elif passes >= 2:
            self.status = TransferStatus.DEVELOPING
        else:
            self.status = TransferStatus.NOT_READY

        return self.status

    def to_dict(self) -> dict:
        return {
            "entry_points_navigated": round(self.entry_points_navigated, 2),
            "perspectives_integrated": round(self.perspectives_integrated, 2),
            "novel_assembly_success": round(self.novel_assembly_success, 1),
            "reductive_bias_score": round(self.reductive_bias_score, 2),
            "status": self.status.value,
            "thresholds": {
                "entry_points": self.ENTRY_POINT_THRESHOLD,
                "perspectives": self.PERSPECTIVE_THRESHOLD,
                "assembly": self.ASSEMBLY_THRESHOLD,
                "bias": self.BIAS_THRESHOLD,
            },
        }


@dataclass
class CrissCrossAction:
    """Action to take during criss-crossing."""

    action_type: str
    purpose: str
    details: str = ""

    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type,
            "purpose": self.purpose,
            "details": self.details,
        }


class CaseLibrary:
    """Library of case studies for ill-structured domains."""

    def __init__(self):
        self.cases: Dict[str, CaseStudy] = {}
        self.domain_cases: Dict[str, List[str]] = {}  # domain -> case_ids
        self.concept_cases: Dict[str, List[str]] = {}  # concept -> case_ids

    def add_case(self, case: CaseStudy) -> None:
        """Add a case study to the library."""
        self.cases[case.case_id] = case

        # Index by domain
        if case.domain not in self.domain_cases:
            self.domain_cases[case.domain] = []
        self.domain_cases[case.domain].append(case.case_id)

        # Index by concept
        for concept in case.relevant_concepts:
            if concept not in self.concept_cases:
                self.concept_cases[concept] = []
            self.concept_cases[concept].append(case.case_id)

    def get_cases_for_concept(self, concept: str) -> List[CaseStudy]:
        """Get all cases that involve a concept."""
        case_ids = self.concept_cases.get(concept, [])
        return [self.cases[cid] for cid in case_ids if cid in self.cases]

    def get_cases_for_domain(self, domain: str) -> List[CaseStudy]:
        """Get all cases in a domain."""
        case_ids = self.domain_cases.get(domain, [])
        return [self.cases[cid] for cid in case_ids if cid in self.cases]

    def get_case(self, case_id: str) -> Optional[CaseStudy]:
        """Get a specific case by ID."""
        return self.cases.get(case_id)


class CFTEngine:
    """
    Cognitive Flexibility Theory Engine.

    Implements criss-crossing through multiple perspectives and entry points
    to build flexible, transferable knowledge in ill-structured domains.
    """

    # Minimum entry points required for mastery
    MIN_ENTRY_POINTS_FOR_MASTERY = 3

    # Minimum perspectives for avoiding bias
    MIN_PERSPECTIVES_FOR_BALANCE = 3

    def __init__(self):
        self.concepts: Dict[str, ConceptLandscape] = {}
        self.learner_states: Dict[str, LearnerCFTState] = {}
        self.case_library = CaseLibrary()
        self.lenses = DEFAULT_LENSES.copy()
        self._landscape_explorer_active = False

    def add_concept(
        self,
        concept_id: str,
        name: str,
        domain_type: DomainType = DomainType.WELL_STRUCTURED,
        entry_points: Optional[List[EntryPoint]] = None,
        lenses: Optional[Set[ExpertLens]] = None,
    ) -> ConceptLandscape:
        """Add a concept to the CFT knowledge graph."""
        concept = ConceptLandscape(
            concept_id=concept_id,
            concept_name=name,
            entry_points=entry_points or [],
            available_lenses=lenses or set(ExpertLens),
            domain_type=domain_type,
        )
        self.concepts[concept_id] = concept
        return concept

    def add_entry_point(
        self,
        concept_id: str,
        name: str,
        domain_frame: str,
        first_concept: str,
        description: str = "",
    ) -> None:
        """Add an entry point to a concept."""
        if concept_id not in self.concepts:
            raise ValueError(f"Concept {concept_id} not found")

        entry_point = EntryPoint(
            name=name,
            domain_frame=domain_frame,
            first_concept=first_concept,
            description=description,
        )
        self.concepts[concept_id].add_entry_point(entry_point)

    def add_case_study(self, case: CaseStudy) -> None:
        """Add a case study to the library."""
        self.case_library.add_case(case)

    def get_or_create_learner_state(self, learner_id: str) -> LearnerCFTState:
        """Get or create learner state."""
        if learner_id not in self.learner_states:
            self.learner_states[learner_id] = LearnerCFTState(learner_id=learner_id)
        return self.learner_states[learner_id]

    def activate_landscape_explorer(self) -> None:
        """Activate Landscape Explorer mode."""
        self._landscape_explorer_active = True

    def deactivate_landscape_explorer(self) -> None:
        """Deactivate Landscape Explorer mode."""
        self._landscape_explorer_active = False

    def is_landscape_explorer_active(self) -> bool:
        """Check if Landscape Explorer mode is active."""
        return self._landscape_explorer_active

    def navigate_entry_point(
        self,
        learner_id: str,
        concept_id: str,
        entry_point_name: str,
    ) -> List[CrissCrossAction]:
        """
        Navigate a concept via a specific entry point.

        Returns actions to take based on CFT principles.
        """
        if concept_id not in self.concepts:
            raise ValueError(f"Concept {concept_id} not found")

        state = self.get_or_create_learner_state(learner_id)
        concept = self.concepts[concept_id]

        # Record entry point usage
        state.record_entry_point(concept_id, entry_point_name)

        actions = []
        previous_count = state.get_entry_point_count(concept_id) - 1

        # If this is a revisit via different entry point
        if previous_count > 0:
            actions.append(CrissCrossAction(
                action_type="Track_Entry_Points_Used",
                purpose=f"Recording entry points: {state.entry_points_used[concept_id]}",
                details=f"Now using {state.get_entry_point_count(concept_id)} entry points",
            ))

            actions.append(CrissCrossAction(
                action_type="Highlight_New_Connections",
                purpose="Showing connections between perspectives",
                details=f"Connect {entry_point_name} view to previous views",
            ))

            actions.append(CrissCrossAction(
                action_type="Require_Integration_Task",
                purpose="Map concepts between entry points",
                details=f"Integrate {entry_point_name} with previous understanding",
            ))

        return actions

    def apply_lens(
        self,
        learner_id: str,
        concept_id: str,
        lens: ExpertLens,
    ) -> Tuple[LensConfiguration, Optional[CaseStudy]]:
        """
        Apply an expert lens to a concept.

        Returns the lens configuration and an optional case study.
        """
        if concept_id not in self.concepts:
            raise ValueError(f"Concept {concept_id} not found")

        state = self.get_or_create_learner_state(learner_id)
        state.record_lens_use(concept_id, lens)

        lens_config = self.lenses[lens]

        # Find a relevant case study
        concept = self.concepts[concept_id]
        cases = self.case_library.get_cases_for_concept(concept.concept_name)
        case_study = cases[0] if cases else None

        return lens_config, case_study

    def check_mastery_requirements(
        self,
        learner_id: str,
        concept_id: str,
    ) -> Tuple[bool, str]:
        """
        Check if mastery requirements are met for a concept.

        Requires minimum entry points and perspective balance.
        """
        state = self.get_or_create_learner_state(learner_id)

        entry_count = state.get_entry_point_count(concept_id)
        lens_count = state.get_lens_count(concept_id)

        if entry_count < self.MIN_ENTRY_POINTS_FOR_MASTERY:
            return False, f"Need {self.MIN_ENTRY_POINTS_FOR_MASTERY} entry points, have {entry_count}"

        if lens_count < self.MIN_PERSPECTIVES_FOR_BALANCE:
            return False, f"Need {self.MIN_PERSPECTIVES_FOR_BALANCE} perspectives, have {lens_count}"

        return True, "Mastery requirements met"

    def detect_reductive_bias(
        self,
        learner_id: str,
        concept_id: str,
        available_perspectives: List[str],
    ) -> BiasDetection:
        """
        Detect if learner has reductive bias for a concept.

        Checks if they consistently use only one perspective.
        """
        state = self.get_or_create_learner_state(learner_id)

        lenses_used = state.lenses_applied.get(concept_id, set())
        lenses_count = len(lenses_used)

        # No bias if using multiple perspectives
        if lenses_count >= self.MIN_PERSPECTIVES_FOR_BALANCE:
            return BiasDetection(is_biased=False)

        # Identify dominant and ignored perspectives
        used_names = [l.value for l in lenses_used]
        ignored = [p for p in available_perspectives if p not in used_names]

        if lenses_count == 0:
            dominant = None
        else:
            dominant = used_names[0] if used_names else None

        # Mark as biased
        state.concepts_with_bias.add(concept_id)

        return BiasDetection(
            is_biased=True,
            dominant_perspective=dominant,
            ignored_perspectives=ignored,
            recommended_action="Require_Multi_Criteria comparison on 3+ dimensions",
            contrasting_case=f"Consider alternative perspective: {ignored[0]}" if ignored else None,
        )

    def analyze_perspective_coverage(
        self,
        learner_id: str,
        concepts: List[str],
        perspectives: List[str],
    ) -> List[PerspectiveCoverage]:
        """
        Analyze how well perspectives are covered across concepts.
        """
        state = self.get_or_create_learner_state(learner_id)
        results = []

        for perspective in perspectives:
            # Count concepts where this perspective was used
            covered = 0
            gaps = []

            for concept_id in concepts:
                lenses_used = state.lenses_applied.get(concept_id, set())
                lens_names = [l.value for l in lenses_used]

                if perspective.lower() in [n.lower() for n in lens_names]:
                    covered += 1
                else:
                    concept = self.concepts.get(concept_id)
                    if concept:
                        gaps.append(concept.concept_name)

            coverage = (covered / len(concepts) * 100) if concepts else 0

            results.append(PerspectiveCoverage(
                perspective=perspective,
                coverage_percent=coverage,
                gap_areas=gaps[:5],  # Limit to top 5 gaps
            ))

            # Update learner state
            state.perspective_coverage[perspective] = coverage

        return results

    def assess_transfer_readiness(
        self,
        learner_id: str,
        concepts: List[str],
        novel_problem_success_rate: float = 0.0,
    ) -> TransferReadinessMetrics:
        """
        Assess learner's transfer readiness based on CFT metrics.
        """
        state = self.get_or_create_learner_state(learner_id)

        # Calculate average entry points per concept
        entry_counts = [
            state.get_entry_point_count(c) for c in concepts
            if c in state.entry_points_used
        ]
        avg_entry_points = statistics.mean(entry_counts) if entry_counts else 0.0

        # Calculate average perspectives per concept
        lens_counts = [
            state.get_lens_count(c) for c in concepts
            if c in state.lenses_applied
        ]
        avg_perspectives = statistics.mean(lens_counts) if lens_counts else 0.0

        # Calculate bias score (lower is better)
        total_concepts = len(concepts)
        biased_count = len(state.concepts_with_bias.intersection(set(concepts)))
        bias_score = biased_count / total_concepts if total_concepts > 0 else 0.0

        metrics = TransferReadinessMetrics(
            entry_points_navigated=avg_entry_points,
            perspectives_integrated=avg_perspectives,
            novel_assembly_success=novel_problem_success_rate,
            reductive_bias_score=bias_score,
        )
        metrics.evaluate()

        return metrics

    def get_criss_cross_recommendation(
        self,
        learner_id: str,
        concept_id: str,
    ) -> Optional[str]:
        """
        Get recommendation for next criss-crossing action.
        """
        state = self.get_or_create_learner_state(learner_id)

        entry_count = state.get_entry_point_count(concept_id)
        lens_count = state.get_lens_count(concept_id)

        if entry_count < self.MIN_ENTRY_POINTS_FOR_MASTERY:
            return f"Explore concept from {self.MIN_ENTRY_POINTS_FOR_MASTERY - entry_count} more entry points"

        if lens_count < self.MIN_PERSPECTIVES_FOR_BALANCE:
            unused_lenses = set(ExpertLens) - state.lenses_applied.get(concept_id, set())
            if unused_lenses:
                return f"Apply {list(unused_lenses)[0].value} lens for broader perspective"

        return None

    def build_ill_structured_domain(
        self,
        domain_name: str,
        cases: List[CaseStudy],
    ) -> int:
        """
        Build a case library for an ill-structured domain.

        Returns number of cases added.
        """
        for case in cases:
            case.domain = domain_name
            self.add_case_study(case)
        return len(cases)


def create_default_cft_engine() -> CFTEngine:
    """Create a CFT engine with default configuration."""
    return CFTEngine()
