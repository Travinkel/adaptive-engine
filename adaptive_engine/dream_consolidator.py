"""
Dream Consolidator Engine - Automated Consolidation and Dream Agent Processing.

Implements overnight knowledge consolidation based on Synaptic Consolidation Theory.
Expert agents "dream" (simulate) knowledge states while learner is offline,
optimizing the next session for fragile memory nodes.

Work Order: WO-AE-009
Tags: @Domain-Cognitive, @Consolidation, @Sleep, @DreamAgents, @Synaptic

Research Foundation:
- Synaptic Consolidation Theory (Diekelmann & Born, 2010)
- Spacing Effect (Cepeda et al., 2006, d=0.70)
- Interference Theory (McGeoch, 1932; Underwood, 1957)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
import statistics


class WeaknessType(str, Enum):
    """Types of memory weakness detected by dream agents."""

    RECENT_DECAY = "recent_decay"  # Memory decaying due to time
    HIGH_INTERFERENCE = "high_interference"  # Confusion with similar concepts
    LOW_MASTERY = "low_mastery"  # Never well-learned
    OVERCONFIDENCE = "overconfidence"  # High confidence but poor performance
    FRAGILE_ENCODING = "fragile_encoding"  # Weak initial learning


class AtomType(str, Enum):
    """Types of remediation atoms for consolidation."""

    RETRIEVAL_PRACTICE = "retrieval_practice"
    DISCRIMINATION_TASK = "discrimination_task"
    SOCRATIC_DIALOGUE = "socratic_dialogue"
    PREDICTION_CHALLENGE = "prediction_challenge"
    MULTI_MODAL_REVIEW = "multi_modal_review"
    COMPARATIVE_TASK = "comparative_task"


class ReviewUrgency(str, Enum):
    """Review urgency levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    MAINTENANCE = "maintenance"


class InterferenceRisk(str, Enum):
    """Interference risk levels between concept pairs."""

    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"


class StabilityRange(str, Enum):
    """Stability ranges for categorizing concepts."""

    INTENSIVE = "intensive"  # < 3 days
    STANDARD = "standard"  # 3-7 days
    MAINTENANCE = "maintenance"  # 7-30 days
    OCCASIONAL = "occasional"  # 30+ days


@dataclass
class ConceptState:
    """State of a concept in the learner's knowledge graph."""

    concept_id: str
    name: str
    mastery: float = 0.0  # 0-1
    stability: float = 1.0  # days until 50% retention
    last_review: Optional[datetime] = None
    last_accuracy: float = 0.0
    confidence: float = 0.5  # learner's self-reported confidence
    embedding: Optional[List[float]] = None  # for similarity computation
    retrieval_count: int = 0
    difficulty: float = 0.5  # 0-1

    @property
    def days_since_review(self) -> float:
        """Days since last review."""
        if self.last_review is None:
            return float("inf")
        delta = datetime.now() - self.last_review
        return delta.total_seconds() / 86400

    def current_retention(self) -> float:
        """Calculate current retention using exponential decay."""
        if self.last_review is None:
            return 0.0
        t = self.days_since_review
        # R(t) = e^(-t/S) where S is stability
        return math.exp(-t / max(self.stability, 0.1))

    def project_retention(self, hours_ahead: float) -> float:
        """Project retention forward in time."""
        if self.last_review is None:
            return 0.0
        future_days = self.days_since_review + (hours_ahead / 24)
        return math.exp(-future_days / max(self.stability, 0.1))

    def to_dict(self) -> dict:
        return {
            "concept_id": self.concept_id,
            "name": self.name,
            "mastery": round(self.mastery, 3),
            "stability": round(self.stability, 2),
            "last_review": self.last_review.isoformat() if self.last_review else None,
            "current_retention": round(self.current_retention(), 3),
            "days_since_review": round(self.days_since_review, 1),
        }


@dataclass
class FragilityScore:
    """Fragility analysis result for a concept."""

    concept_id: str
    fragility: float  # 0-1, higher = more fragile
    simulated_retrieval: float  # 0-1, simulated success probability
    weakness_type: WeaknessType
    urgency: ReviewUrgency
    factors: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "concept_id": self.concept_id,
            "fragility": round(self.fragility, 2),
            "simulated_retrieval": round(self.simulated_retrieval, 2),
            "weakness_type": self.weakness_type.value,
            "urgency": self.urgency.value,
            "factors": {k: round(v, 3) for k, v in self.factors.items()},
        }


@dataclass
class InterferencePair:
    """A pair of concepts with interference risk."""

    concept_a: str
    concept_b: str
    similarity: float  # 0-1, based on embedding distance
    risk: InterferenceRisk
    confusion_history: int = 0  # times confused in past

    def to_dict(self) -> dict:
        return {
            "concept_a": self.concept_a,
            "concept_b": self.concept_b,
            "similarity": round(self.similarity, 2),
            "risk": self.risk.value,
            "confusion_history": self.confusion_history,
        }


@dataclass
class ConsolidationAtom:
    """An atom generated by overnight consolidation."""

    priority: int
    concept_id: str
    concept_name: str
    weakness_type: WeaknessType
    atom_type: AtomType
    rationale: str
    estimated_time_minutes: float = 2.0
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "priority": self.priority,
            "concept_id": self.concept_id,
            "concept_name": self.concept_name,
            "weakness_type": self.weakness_type.value,
            "atom_type": self.atom_type.value,
            "rationale": self.rationale,
            "estimated_time_minutes": self.estimated_time_minutes,
            "tags": self.tags,
        }


@dataclass
class DecayProjection:
    """Projection of forgetting curve into the future."""

    concept_id: str
    retention_now: float
    retention_projected: float
    hours_projected: float
    urgency: ReviewUrgency
    review_recommended_within_hours: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "concept_id": self.concept_id,
            "retention_now": round(self.retention_now, 2),
            "retention_projected": round(self.retention_projected, 2),
            "hours_projected": self.hours_projected,
            "urgency": self.urgency.value,
            "review_recommended_within_hours": self.review_recommended_within_hours,
        }


@dataclass
class SchedulingAnalysis:
    """Analysis of items due for review."""

    overdue: List[str] = field(default_factory=list)
    due_today: List[str] = field(default_factory=list)
    due_tomorrow: List[str] = field(default_factory=list)
    preemptive: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "overdue": len(self.overdue),
            "due_today": len(self.due_today),
            "due_tomorrow": len(self.due_tomorrow),
            "preemptive": len(self.preemptive),
            "overdue_items": self.overdue[:10],  # First 10
        }


@dataclass
class StabilityDistribution:
    """Distribution of stability across concepts."""

    intensive: List[str] = field(default_factory=list)  # < 3 days
    standard: List[str] = field(default_factory=list)  # 3-7 days
    maintenance: List[str] = field(default_factory=list)  # 7-30 days
    occasional: List[str] = field(default_factory=list)  # 30+ days

    def to_dict(self) -> dict:
        return {
            "intensive": {"count": len(self.intensive), "recommendation": "Needs intensive review"},
            "standard": {"count": len(self.standard), "recommendation": "Standard review cycle"},
            "maintenance": {"count": len(self.maintenance), "recommendation": "Maintenance mode"},
            "occasional": {"count": len(self.occasional), "recommendation": "Occasional check-ins"},
        }


@dataclass
class MorningBootInfo:
    """Information for morning TUI boot sequence."""

    consolidation_summary: str
    priority_items_count: int
    predicted_session_minutes: float
    motivation_quote: str
    priority_atoms: List[ConsolidationAtom]
    consolidation_ran_at: datetime

    def to_dict(self) -> dict:
        return {
            "consolidation_summary": self.consolidation_summary,
            "priority_items_count": self.priority_items_count,
            "predicted_session_minutes": round(self.predicted_session_minutes, 0),
            "motivation_quote": self.motivation_quote,
            "priority_atoms": [a.to_dict() for a in self.priority_atoms],
            "consolidation_ran_at": self.consolidation_ran_at.isoformat(),
        }


@dataclass
class ConsolidationEffectiveness:
    """Metrics measuring consolidation effectiveness."""

    morning_retrieval_with: float  # success rate with consolidation
    morning_retrieval_without: float  # baseline without
    interference_errors_with: float
    interference_errors_without: float
    overdue_rate_with: float
    overdue_rate_without: float
    days_to_solid_state_with: float
    days_to_solid_state_without: float

    @property
    def retrieval_improvement(self) -> float:
        """Percentage improvement in retrieval success."""
        if self.morning_retrieval_without == 0:
            return 0
        return ((self.morning_retrieval_with - self.morning_retrieval_without)
                / self.morning_retrieval_without * 100)

    @property
    def interference_reduction(self) -> float:
        """Percentage reduction in interference errors."""
        if self.interference_errors_without == 0:
            return 0
        return ((self.interference_errors_without - self.interference_errors_with)
                / self.interference_errors_without * 100)

    def to_dict(self) -> dict:
        return {
            "morning_retrieval_success": {
                "with_consolidation": f"{self.morning_retrieval_with:.0%}",
                "without": f"{self.morning_retrieval_without:.0%}",
                "improvement": f"+{self.retrieval_improvement:.0f}%",
            },
            "interference_errors": {
                "with_consolidation": f"{self.interference_errors_with:.0%}",
                "without": f"{self.interference_errors_without:.0%}",
                "reduction": f"-{self.interference_reduction:.0f}%",
            },
        }


@dataclass
class LearnerPattern:
    """Detected learner usage patterns."""

    avg_session_gap_hours: float = 18.0
    morning_preference_hour: int = 7
    avg_session_length_minutes: float = 25.0
    last_active: Optional[datetime] = None
    days_inactive: float = 0.0

    def to_dict(self) -> dict:
        return {
            "avg_session_gap_hours": self.avg_session_gap_hours,
            "morning_preference_hour": self.morning_preference_hour,
            "avg_session_length_minutes": self.avg_session_length_minutes,
            "days_inactive": round(self.days_inactive, 1),
        }


@dataclass
class SimulationResults:
    """Results from dream simulation."""

    total_retrievals: int
    success_count: int
    failure_count: int
    decay_applied: int
    interference_events: int
    concepts_needing_strengthening: List[str]

    @property
    def success_rate(self) -> float:
        if self.total_retrievals == 0:
            return 0
        return self.success_count / self.total_retrievals

    def to_dict(self) -> dict:
        return {
            "total_retrievals": self.total_retrievals,
            "success_rate": f"{self.success_rate:.0%}",
            "decay_applied": self.decay_applied,
            "interference_events": self.interference_events,
            "concepts_needing_strengthening": len(self.concepts_needing_strengthening),
        }


class WeakNodeScanner:
    """
    Scans for weak/fragile memory nodes via simulated retrieval.

    Based on retrieval strength theory - memories that are harder to
    retrieve are more fragile and need reinforcement.
    """

    # Thresholds for fragility classification
    FRAGILITY_CRITICAL = 0.70
    FRAGILITY_WEAK = 0.50
    FRAGILITY_MODERATE = 0.30

    def __init__(self):
        self.scan_history: List[Dict] = []

    def simulate_retrieval(self, concept: ConceptState) -> float:
        """
        Simulate a retrieval attempt for a concept.

        Returns probability of successful retrieval (0-1).
        """
        # Base retrieval probability from retention
        base_prob = concept.current_retention()

        # Adjust for mastery
        mastery_factor = 0.3 + (0.7 * concept.mastery)

        # Adjust for stability (more stable = easier to retrieve)
        stability_factor = min(1.0, concept.stability / 30)  # Cap at 30 days

        # Adjust for recency (more recent = easier)
        recency_factor = 1.0
        if concept.days_since_review > 7:
            recency_factor = max(0.5, 1 - (concept.days_since_review - 7) / 30)

        # Combine factors
        retrieval_prob = base_prob * mastery_factor * stability_factor * recency_factor

        # Add some noise to simulate real retrieval variability
        noise = random.gauss(0, 0.05)
        retrieval_prob = max(0, min(1, retrieval_prob + noise))

        return retrieval_prob

    def calculate_fragility(self, concept: ConceptState) -> FragilityScore:
        """
        Calculate fragility score for a concept.

        Fragility = 1 - simulated retrieval success
        Higher fragility = more urgent need for review.
        """
        simulated = self.simulate_retrieval(concept)
        fragility = 1 - simulated

        # Determine weakness type
        weakness_type = self._classify_weakness(concept, fragility)

        # Determine urgency
        urgency = self._classify_urgency(fragility)

        # Build factor breakdown
        factors = {
            "retention": concept.current_retention(),
            "mastery": concept.mastery,
            "stability": concept.stability,
            "days_since_review": concept.days_since_review,
            "confidence": concept.confidence,
        }

        return FragilityScore(
            concept_id=concept.concept_id,
            fragility=fragility,
            simulated_retrieval=simulated,
            weakness_type=weakness_type,
            urgency=urgency,
            factors=factors,
        )

    def _classify_weakness(
        self, concept: ConceptState, fragility: float
    ) -> WeaknessType:
        """Classify the type of weakness based on concept state."""
        # Check for overconfidence (high confidence but low performance)
        if concept.confidence > 0.7 and concept.last_accuracy < 0.5:
            return WeaknessType.OVERCONFIDENCE

        # Check for recent decay
        if concept.days_since_review > 3 and concept.current_retention() < 0.7:
            return WeaknessType.RECENT_DECAY

        # Check for low mastery
        if concept.mastery < 0.4:
            return WeaknessType.LOW_MASTERY

        # Check for fragile encoding (reviewed recently but still fragile)
        if concept.days_since_review < 2 and fragility > 0.4:
            return WeaknessType.FRAGILE_ENCODING

        # Default to interference
        return WeaknessType.HIGH_INTERFERENCE

    def _classify_urgency(self, fragility: float) -> ReviewUrgency:
        """Classify review urgency based on fragility."""
        if fragility >= self.FRAGILITY_CRITICAL:
            return ReviewUrgency.CRITICAL
        elif fragility >= self.FRAGILITY_WEAK:
            return ReviewUrgency.HIGH
        elif fragility >= self.FRAGILITY_MODERATE:
            return ReviewUrgency.MODERATE
        else:
            return ReviewUrgency.LOW

    def scan_all(
        self, concepts: List[ConceptState], threshold: float = 0.5
    ) -> List[FragilityScore]:
        """
        Scan all concepts and return those above fragility threshold.

        Args:
            concepts: List of concepts to scan
            threshold: Minimum fragility to include (default 0.5)

        Returns:
            List of FragilityScore for fragile concepts, sorted by fragility
        """
        results = []
        for concept in concepts:
            score = self.calculate_fragility(concept)
            if score.fragility >= threshold:
                results.append(score)

        # Sort by fragility (most fragile first)
        results.sort(key=lambda x: x.fragility, reverse=True)

        # Record scan
        self.scan_history.append({
            "timestamp": datetime.now().isoformat(),
            "concepts_scanned": len(concepts),
            "fragile_found": len(results),
            "threshold": threshold,
        })

        return results


class InterferenceAnalyzer:
    """
    Analyzes interference between similar concepts.

    Based on interference theory - similar memories compete during retrieval,
    causing confusion and errors.
    """

    # Thresholds for interference risk
    HIGH_SIMILARITY_THRESHOLD = 0.85
    MODERATE_SIMILARITY_THRESHOLD = 0.70

    def __init__(self):
        self.pair_history: Dict[Tuple[str, str], int] = {}  # Track confusion counts

    def compute_similarity(
        self,
        concept_a: ConceptState,
        concept_b: ConceptState,
    ) -> float:
        """
        Compute similarity between two concepts.

        Uses embedding cosine similarity if available, otherwise
        falls back to name/category heuristics.
        """
        # If both have embeddings, use cosine similarity
        if concept_a.embedding and concept_b.embedding:
            return self._cosine_similarity(concept_a.embedding, concept_b.embedding)

        # Fallback: simple name-based heuristic
        return self._name_similarity(concept_a.name, concept_b.name)

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def _name_similarity(self, name_a: str, name_b: str) -> float:
        """Simple name-based similarity heuristic."""
        words_a = set(name_a.lower().replace("_", " ").split())
        words_b = set(name_b.lower().replace("_", " ").split())

        if not words_a or not words_b:
            return 0.0

        intersection = len(words_a & words_b)
        union = len(words_a | words_b)

        return intersection / union if union > 0 else 0.0

    def classify_risk(self, similarity: float) -> InterferenceRisk:
        """Classify interference risk based on similarity."""
        if similarity >= self.HIGH_SIMILARITY_THRESHOLD:
            return InterferenceRisk.HIGH
        elif similarity >= self.MODERATE_SIMILARITY_THRESHOLD:
            return InterferenceRisk.MODERATE
        else:
            return InterferenceRisk.LOW

    def find_interference_pairs(
        self,
        concepts: List[ConceptState],
        min_similarity: float = 0.70,
    ) -> List[InterferencePair]:
        """
        Find all pairs of concepts with interference risk.

        Args:
            concepts: List of concepts to analyze
            min_similarity: Minimum similarity to report

        Returns:
            List of InterferencePair, sorted by similarity (highest first)
        """
        pairs = []

        for i, concept_a in enumerate(concepts):
            for concept_b in concepts[i + 1:]:
                similarity = self.compute_similarity(concept_a, concept_b)

                if similarity >= min_similarity:
                    pair_key = tuple(sorted([concept_a.concept_id, concept_b.concept_id]))
                    confusion_count = self.pair_history.get(pair_key, 0)

                    pairs.append(InterferencePair(
                        concept_a=concept_a.concept_id,
                        concept_b=concept_b.concept_id,
                        similarity=similarity,
                        risk=self.classify_risk(similarity),
                        confusion_history=confusion_count,
                    ))

        # Sort by similarity (highest risk first)
        pairs.sort(key=lambda p: p.similarity, reverse=True)

        return pairs

    def record_confusion(self, concept_a: str, concept_b: str) -> None:
        """Record that two concepts were confused."""
        pair_key = tuple(sorted([concept_a, concept_b]))
        self.pair_history[pair_key] = self.pair_history.get(pair_key, 0) + 1


class DecayProjector:
    """
    Projects forgetting curves into the future.

    Uses exponential decay model: R(t) = e^(-t/S)
    where R = retention, t = time, S = stability
    """

    # Retention thresholds for urgency
    CRITICAL_RETENTION = 0.50
    HIGH_RETENTION = 0.60
    MODERATE_RETENTION = 0.70

    def project(
        self,
        concept: ConceptState,
        hours_ahead: float = 48,
    ) -> DecayProjection:
        """
        Project retention forward in time.

        Args:
            concept: The concept to project
            hours_ahead: Hours to project forward

        Returns:
            DecayProjection with current and projected retention
        """
        retention_now = concept.current_retention()
        retention_projected = concept.project_retention(hours_ahead)

        urgency = self._classify_urgency(retention_projected)

        # Calculate when to review to stay above threshold
        review_within = self._calculate_review_deadline(concept, self.MODERATE_RETENTION)

        return DecayProjection(
            concept_id=concept.concept_id,
            retention_now=retention_now,
            retention_projected=retention_projected,
            hours_projected=hours_ahead,
            urgency=urgency,
            review_recommended_within_hours=review_within,
        )

    def _classify_urgency(self, retention: float) -> ReviewUrgency:
        """Classify urgency based on projected retention."""
        if retention < self.CRITICAL_RETENTION:
            return ReviewUrgency.CRITICAL
        elif retention < self.HIGH_RETENTION:
            return ReviewUrgency.HIGH
        elif retention < self.MODERATE_RETENTION:
            return ReviewUrgency.MODERATE
        else:
            return ReviewUrgency.LOW

    def _calculate_review_deadline(
        self,
        concept: ConceptState,
        target_retention: float = 0.70,
    ) -> Optional[float]:
        """
        Calculate hours until retention drops below target.

        Returns None if already below target.
        """
        current = concept.current_retention()
        if current < target_retention:
            return 0  # Review now

        # Solve: target = current * e^(-delta_t / S)
        # delta_t = -S * ln(target / current)
        if current <= 0:
            return None

        ratio = target_retention / current
        if ratio >= 1:
            return None  # Will never drop below target

        delta_days = -concept.stability * math.log(ratio)
        return delta_days * 24  # Convert to hours

    def project_batch(
        self,
        concepts: List[ConceptState],
        hours_ahead: float = 48,
    ) -> List[DecayProjection]:
        """Project retention for multiple concepts."""
        projections = [self.project(c, hours_ahead) for c in concepts]

        # Sort by urgency (most urgent first)
        urgency_order = {
            ReviewUrgency.CRITICAL: 0,
            ReviewUrgency.HIGH: 1,
            ReviewUrgency.MODERATE: 2,
            ReviewUrgency.LOW: 3,
            ReviewUrgency.MAINTENANCE: 4,
        }
        projections.sort(key=lambda p: (urgency_order[p.urgency], -p.retention_projected))

        return projections


class ConsolidationPathOptimizer:
    """
    Generates optimal consolidation learning paths for morning sessions.

    Balances multiple factors: decay risk, interference, prerequisites,
    goals, and variety.
    """

    # Target session length
    DEFAULT_SESSION_MINUTES = 20

    def generate_path(
        self,
        fragility_scores: List[FragilityScore],
        interference_pairs: List[InterferencePair],
        target_minutes: float = DEFAULT_SESSION_MINUTES,
    ) -> List[ConsolidationAtom]:
        """
        Generate an ordered list of consolidation atoms for morning review.

        Args:
            fragility_scores: Fragility analysis results
            interference_pairs: Interference pair analysis
            target_minutes: Target session length

        Returns:
            Ordered list of ConsolidationAtom
        """
        atoms = []
        total_time = 0
        avg_atom_time = 3.0  # minutes per atom

        # Priority 1: High decay risk (prevent forgetting)
        decay_concepts = [
            f for f in fragility_scores
            if f.weakness_type == WeaknessType.RECENT_DECAY
        ]
        for score in decay_concepts[:2]:  # Max 2 decay atoms
            if total_time >= target_minutes:
                break
            atoms.append(self._create_atom(
                priority=len(atoms) + 1,
                score=score,
                atom_type=AtomType.RETRIEVAL_PRACTICE,
                rationale="Prevent forgetting - memory decaying",
            ))
            total_time += avg_atom_time

        # Priority 2: High interference (resolve confusion)
        interference_concepts = set()
        for pair in interference_pairs:
            if pair.risk == InterferenceRisk.HIGH:
                interference_concepts.add(pair.concept_a)
                interference_concepts.add(pair.concept_b)

        interference_scores = [
            f for f in fragility_scores
            if f.concept_id in interference_concepts
        ]
        for score in interference_scores[:2]:
            if total_time >= target_minutes:
                break
            atoms.append(self._create_atom(
                priority=len(atoms) + 1,
                score=score,
                atom_type=AtomType.DISCRIMINATION_TASK,
                rationale="Resolve confusion with similar concepts",
            ))
            total_time += avg_atom_time

        # Priority 3: Weak encoding (fragile but recent)
        weak_encoding = [
            f for f in fragility_scores
            if f.weakness_type == WeaknessType.FRAGILE_ENCODING
        ]
        for score in weak_encoding[:1]:
            if total_time >= target_minutes:
                break
            atoms.append(self._create_atom(
                priority=len(atoms) + 1,
                score=score,
                atom_type=AtomType.MULTI_MODAL_REVIEW,
                rationale="Strengthen fragile encoding",
            ))
            total_time += avg_atom_time

        # Priority 4: Overconfidence (calibration needed)
        overconfident = [
            f for f in fragility_scores
            if f.weakness_type == WeaknessType.OVERCONFIDENCE
        ]
        for score in overconfident[:1]:
            if total_time >= target_minutes:
                break
            atoms.append(self._create_atom(
                priority=len(atoms) + 1,
                score=score,
                atom_type=AtomType.PREDICTION_CHALLENGE,
                rationale="Calibrate overconfidence",
            ))
            total_time += avg_atom_time

        # Fill remaining time with other fragile concepts
        used_concepts = {a.concept_id for a in atoms}
        remaining = [
            f for f in fragility_scores
            if f.concept_id not in used_concepts
        ]
        for score in remaining:
            if total_time >= target_minutes:
                break
            atoms.append(self._create_atom(
                priority=len(atoms) + 1,
                score=score,
                atom_type=self._select_atom_type(score),
                rationale="General consolidation review",
            ))
            total_time += avg_atom_time

        return atoms

    def _create_atom(
        self,
        priority: int,
        score: FragilityScore,
        atom_type: AtomType,
        rationale: str,
    ) -> ConsolidationAtom:
        """Create a consolidation atom from fragility score."""
        return ConsolidationAtom(
            priority=priority,
            concept_id=score.concept_id,
            concept_name=score.concept_id.replace("_", " ").title(),
            weakness_type=score.weakness_type,
            atom_type=atom_type,
            rationale=rationale,
            estimated_time_minutes=3.0,
            tags=["Priority_NCDE_Stabilizers"],
        )

    def _select_atom_type(self, score: FragilityScore) -> AtomType:
        """Select appropriate atom type based on weakness."""
        type_map = {
            WeaknessType.RECENT_DECAY: AtomType.RETRIEVAL_PRACTICE,
            WeaknessType.HIGH_INTERFERENCE: AtomType.DISCRIMINATION_TASK,
            WeaknessType.LOW_MASTERY: AtomType.SOCRATIC_DIALOGUE,
            WeaknessType.OVERCONFIDENCE: AtomType.PREDICTION_CHALLENGE,
            WeaknessType.FRAGILE_ENCODING: AtomType.MULTI_MODAL_REVIEW,
        }
        return type_map.get(score.weakness_type, AtomType.RETRIEVAL_PRACTICE)


class DreamConsolidator:
    """
    Main Dream Consolidation Engine.

    Orchestrates overnight processing to optimize learner's next session
    by identifying fragile memories, interference pairs, and generating
    prioritized review atoms.

    Based on:
    - Synaptic Consolidation Theory (Diekelmann & Born, 2010)
    - Spacing Effect (Cepeda et al., 2006)
    - Interference Theory (McGeoch, 1932)
    """

    # Idle detection threshold
    IDLE_THRESHOLD_HOURS = 6

    def __init__(self):
        self.weak_node_scanner = WeakNodeScanner()
        self.interference_analyzer = InterferenceAnalyzer()
        self.decay_projector = DecayProjector()
        self.path_optimizer = ConsolidationPathOptimizer()

        self.concepts: Dict[str, ConceptState] = {}
        self.learner_pattern = LearnerPattern()
        self.last_consolidation: Optional[datetime] = None
        self.consolidation_history: List[Dict] = []
        self._morning_boot_info: Optional[MorningBootInfo] = None

    def add_concept(self, concept: ConceptState) -> None:
        """Add a concept to the knowledge graph."""
        self.concepts[concept.concept_id] = concept

    def set_concept_mastery(self, concept_id: str, mastery: float) -> None:
        """Update mastery for a concept."""
        if concept_id in self.concepts:
            self.concepts[concept_id].mastery = max(0, min(1, mastery))

    def record_review(
        self,
        concept_id: str,
        accuracy: float,
        confidence: float,
    ) -> None:
        """Record a review event for a concept."""
        if concept_id not in self.concepts:
            return

        concept = self.concepts[concept_id]
        concept.last_review = datetime.now()
        concept.last_accuracy = accuracy
        concept.confidence = confidence
        concept.retrieval_count += 1

        # Update stability based on accuracy
        if accuracy >= 0.8:
            concept.stability = min(90, concept.stability * 1.3)
        elif accuracy < 0.5:
            concept.stability = max(1, concept.stability * 0.7)

    def update_learner_pattern(
        self,
        session_gap_hours: Optional[float] = None,
        morning_hour: Optional[int] = None,
        session_minutes: Optional[float] = None,
    ) -> None:
        """Update learner usage patterns."""
        if session_gap_hours:
            self.learner_pattern.avg_session_gap_hours = session_gap_hours
        if morning_hour:
            self.learner_pattern.morning_preference_hour = morning_hour
        if session_minutes:
            self.learner_pattern.avg_session_length_minutes = session_minutes

    def is_idle(self, hours: float = IDLE_THRESHOLD_HOURS) -> bool:
        """Check if learner has been idle for specified hours."""
        if not self.learner_pattern.last_active:
            return True

        delta = datetime.now() - self.learner_pattern.last_active
        return delta.total_seconds() >= hours * 3600

    def run_consolidation(
        self,
        target_session_minutes: float = 20,
    ) -> MorningBootInfo:
        """
        Run full overnight consolidation process.

        1. Scan for weak nodes
        2. Analyze interference pairs
        3. Project forgetting curves
        4. Generate prioritized atoms
        5. Prepare morning boot info

        Returns:
            MorningBootInfo for morning TUI boot
        """
        concept_list = list(self.concepts.values())

        # 1. Weak node scanning
        fragility_scores = self.weak_node_scanner.scan_all(
            concept_list, threshold=0.30
        )

        # 2. Interference analysis
        interference_pairs = self.interference_analyzer.find_interference_pairs(
            concept_list, min_similarity=0.70
        )

        # 3. Decay projection (48 hours forward)
        decay_projections = self.decay_projector.project_batch(
            concept_list, hours_ahead=48
        )

        # 4. Generate consolidation path
        priority_atoms = self.path_optimizer.generate_path(
            fragility_scores,
            interference_pairs,
            target_minutes=target_session_minutes,
        )

        # 5. Prepare morning boot info
        total_time = sum(a.estimated_time_minutes for a in priority_atoms)

        self._morning_boot_info = MorningBootInfo(
            consolidation_summary=f"Analyzed {len(concept_list)} concepts overnight",
            priority_items_count=len(priority_atoms),
            predicted_session_minutes=total_time,
            motivation_quote=self._get_motivation_quote(),
            priority_atoms=priority_atoms,
            consolidation_ran_at=datetime.now(),
        )

        # Record consolidation
        self.last_consolidation = datetime.now()
        self.consolidation_history.append({
            "timestamp": self.last_consolidation.isoformat(),
            "concepts_analyzed": len(concept_list),
            "fragile_found": len(fragility_scores),
            "interference_pairs": len(interference_pairs),
            "atoms_generated": len(priority_atoms),
        })

        return self._morning_boot_info

    def get_morning_boot_info(self) -> Optional[MorningBootInfo]:
        """Get the morning boot info from last consolidation."""
        return self._morning_boot_info

    def run_dream_simulation(self, num_retrievals: int = 1000) -> SimulationResults:
        """
        Run dream simulation - agents "practice" on learner's behalf.

        Simulates random retrievals to identify which concepts would
        benefit most from real practice.
        """
        concept_list = list(self.concepts.values())
        if not concept_list:
            return SimulationResults(0, 0, 0, 0, 0, [])

        success_count = 0
        failure_count = 0
        interference_events = 0
        decay_applied = 0
        weak_concepts: Set[str] = set()

        for _ in range(num_retrievals):
            # Random concept selection
            concept = random.choice(concept_list)

            # Simulate retrieval
            retrieval_prob = self.weak_node_scanner.simulate_retrieval(concept)
            success = random.random() < retrieval_prob

            if success:
                success_count += 1
            else:
                failure_count += 1
                weak_concepts.add(concept.concept_id)

            # Simulate decay (small probability each iteration)
            if random.random() < 0.01:
                decay_applied += 1

            # Simulate interference (if similar concepts in graph)
            if random.random() < 0.02:
                interference_events += 1

        return SimulationResults(
            total_retrievals=num_retrievals,
            success_count=success_count,
            failure_count=failure_count,
            decay_applied=decay_applied,
            interference_events=interference_events,
            concepts_needing_strengthening=list(weak_concepts),
        )

    def analyze_scheduling(self) -> SchedulingAnalysis:
        """Analyze items due for review."""
        analysis = SchedulingAnalysis()
        now = datetime.now()
        today = now.date()
        tomorrow = today + timedelta(days=1)

        for concept in self.concepts.values():
            if concept.last_review is None:
                analysis.overdue.append(concept.concept_id)
                continue

            # Calculate due date based on stability
            due_date = (concept.last_review + timedelta(days=concept.stability)).date()

            if due_date < today:
                analysis.overdue.append(concept.concept_id)
            elif due_date == today:
                analysis.due_today.append(concept.concept_id)
            elif due_date == tomorrow:
                analysis.due_tomorrow.append(concept.concept_id)
            elif concept.mastery > 0.8 and due_date <= today + timedelta(days=3):
                # High-value, nearly due
                analysis.preemptive.append(concept.concept_id)

        return analysis

    def analyze_stability_distribution(self) -> StabilityDistribution:
        """Analyze stability distribution across concepts."""
        dist = StabilityDistribution()

        for concept in self.concepts.values():
            if concept.stability < 3:
                dist.intensive.append(concept.concept_id)
            elif concept.stability < 7:
                dist.standard.append(concept.concept_id)
            elif concept.stability < 30:
                dist.maintenance.append(concept.concept_id)
            else:
                dist.occasional.append(concept.concept_id)

        return dist

    def handle_extended_absence(self, days_absent: int) -> Dict:
        """
        Handle learner absence gracefully.

        Generates a re-entry plan that doesn't overwhelm.
        """
        if days_absent <= 3:
            return {
                "phase": "standard",
                "action": "Standard consolidation, queue growing",
                "recommendation": "Normal morning review",
            }
        elif days_absent <= 5:
            return {
                "phase": "flag_absence",
                "action": "Flag extended absence, preserve critical",
                "recommendation": "Focus on highest priority items only",
            }
        else:
            # Generate re-entry plan
            critical_concepts = [
                c for c in self.concepts.values()
                if c.current_retention() < 0.5
            ][:10]  # Max 10 critical

            return {
                "phase": "re_entry",
                "action": "Generate re-entry plan for return",
                "recommendation": "Gentle re-onboarding, not overwhelming",
                "critical_concepts": [c.concept_id for c in critical_concepts],
                "suggested_session_minutes": 15,  # Shorter than normal
            }

    def _get_motivation_quote(self) -> str:
        """Get a domain-relevant motivation quote."""
        quotes = [
            "The expert in anything was once a beginner. - Helen Hayes",
            "Learning is not attained by chance, it must be sought for. - Abigail Adams",
            "The capacity to learn is a gift; the ability to learn is a skill. - Brian Herbert",
            "Tell me and I forget, teach me and I remember, involve me and I learn. - Benjamin Franklin",
            "The beautiful thing about learning is that no one can take it away from you. - B.B. King",
        ]
        return random.choice(quotes)


def create_dream_consolidator() -> DreamConsolidator:
    """Factory function to create a DreamConsolidator instance."""
    return DreamConsolidator()
