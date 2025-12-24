"""
Zone of Proximal Development (ZPD) Engine for Adaptive Learning.

Implements Vygotsky's ZPD theory integrated with Csikszentmihalyi's Flow Channel
to maintain optimal challenge-skill balance during learning sessions.

Key Concepts:
- ZPD: The gap between what a learner can do independently and what they can
  achieve with scaffolding/guidance
- Flow Channel: The "Goldilocks zone" where challenge matches skill
- Scaffolding: Temporary support that fades as mastery increases

Research Basis:
- Vygotsky (1978): Zone of Proximal Development
- Csikszentmihalyi (1990): Flow: The Psychology of Optimal Experience
- Wood, Bruner, Ross (1976): The role of tutoring in problem solving (scaffolding)
- Kalyuga (2007): Expertise reversal effect

Author: Cortex System
Version: 1.0.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from loguru import logger


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class ScaffoldType(str, Enum):
    """
    Types of scaffolding support, ordered by opacity (amount of help provided).

    Based on faded worked examples research (Renkl, Atkinson, 2003).
    """

    NO_SCAFFOLD = "no_scaffold"           # 0% - Independent performance
    FIRST_STEP_ONLY = "first_step_only"   # 25% - Just the first step shown
    STRUCTURAL_HINT = "structural_hint"   # 50% - Structure/framework shown
    FADED_PARSONS = "faded_parsons"       # 75% - Most steps provided, some blanks
    FULL_WORKED_EXAMPLE = "full_worked"   # 100% - Complete solution walkthrough

    @property
    def opacity(self) -> float:
        """Return opacity as a float 0.0-1.0."""
        return {
            ScaffoldType.NO_SCAFFOLD: 0.0,
            ScaffoldType.FIRST_STEP_ONLY: 0.25,
            ScaffoldType.STRUCTURAL_HINT: 0.50,
            ScaffoldType.FADED_PARSONS: 0.75,
            ScaffoldType.FULL_WORKED_EXAMPLE: 1.0,
        }[self]


class ZPDPosition(str, Enum):
    """Learner's position relative to their ZPD boundaries."""

    BELOW_ZPD = "below_zpd"           # Too easy - causes boredom
    LOWER_ZPD = "lower_zpd"           # Independent zone - comfortable
    OPTIMAL_ZPD = "optimal_zpd"       # Sweet spot - challenge with achievability
    UPPER_ZPD = "upper_zpd"           # Scaffolded zone - needs support
    ABOVE_ZPD = "above_zpd"           # Frustration zone - too hard even with help


class FlowState(str, Enum):
    """Learner's current flow state (Csikszentmihalyi)."""

    BOREDOM = "boredom"        # Skill >> Challenge
    APATHY = "apathy"          # Low skill, low challenge
    RELAXATION = "relaxation"  # Skill > Challenge (slightly)
    FLOW = "flow"              # Skill ≈ Challenge
    CONTROL = "control"        # Skill < Challenge (slightly)
    AROUSAL = "arousal"        # Challenge > Skill (slightly)
    ANXIETY = "anxiety"        # Challenge >> Skill
    WORRY = "worry"            # Low skill, moderate challenge


class StruggleType(str, Enum):
    """Classification of struggle patterns for scaffold selection."""

    CONCEPTUAL_GAP = "conceptual_gap"         # Missing prerequisite knowledge
    PROCEDURAL_STUCK = "procedural_stuck"     # Knows facts, can't sequence steps
    MINOR_CONFUSION = "minor_confusion"       # Small misunderstanding
    VERIFICATION_NEED = "verification_need"   # Unsure if approach is correct
    COGNITIVE_OVERLOAD = "cognitive_overload" # Too many elements to process


# =============================================================================
# THRESHOLDS AND CONFIGURATION
# =============================================================================

ZPD_THRESHOLDS = {
    # Friction thresholds for scaffold selection (NCDE friction vector)
    "friction_collapse": 3.0,       # Requires full worked example
    "friction_high": 2.5,           # Requires faded parsons
    "friction_moderate": 2.0,       # Requires structural hint
    "friction_mild": 1.5,           # Requires first step only
    "friction_optimal": 1.0,        # No scaffold needed (flow state)

    # Flow channel thresholds (challenge - skill delta)
    "boredom_threshold": -0.5,      # skill > challenge by > 0.5
    "flow_lower": -0.1,             # Lower bound of flow channel
    "flow_upper": 0.3,              # Upper bound of flow channel
    "anxiety_threshold": 0.5,       # challenge > skill by > 0.5

    # ZPD boundary measurement parameters
    "independent_success_rate": 0.80,     # 80% success = independent mastery
    "scaffolded_success_rate": 0.80,      # 80% success with help = scaffolded zone
    "frustration_success_rate": 0.50,     # <50% even with help = frustration
    "min_interactions_for_zpd": 30,       # Minimum data points for ZPD calculation

    # Scaffold fading parameters
    "fade_improvement_threshold": 0.15,   # 15% accuracy improvement to fade
    "fade_min_interactions": 5,           # Minimum interactions before fading
    "fade_latency_improvement": 0.20,     # 20% faster = ready to fade

    # IRT parameters for flow channel
    "irt_theta_offset_low": 0.1,    # Optimal difficulty = θ + 0.1
    "irt_theta_offset_high": 0.3,   # Upper bound = θ + 0.3
}


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class ZPDBoundary:
    """
    Represents a specific boundary in the ZPD.

    Difficulty is measured on the IRT scale (typically -3 to +3).
    """

    name: str                    # e.g., "lower_bound", "current_independent", etc.
    difficulty: float            # IRT difficulty (b parameter)
    success_rate: float          # Historical success rate at this level
    requires_scaffold: bool      # Whether scaffold is needed at this level
    interactions_count: int = 0  # Number of interactions informing this boundary


@dataclass
class ZPDState:
    """
    Complete ZPD state for a learner on a specific concept/skill.

    The ZPD has five key boundaries:
    1. Lower Bound: Too easy (>95% success without help)
    2. Current Independent: What they can do alone (80% success)
    3. Upper Independent: Edge of independent ability (50% success alone)
    4. Upper Scaffolded: What they can do with help (80% success WITH help)
    5. Frustration Ceiling: Too hard even with help (<50% success)
    """

    concept_id: str
    learner_id: str

    # IRT-based skill estimate
    theta: float = 0.0           # Learner's ability estimate
    theta_se: float = 1.0        # Standard error of theta estimate

    # ZPD boundaries (difficulty values)
    lower_bound: float = 0.0          # Too easy
    current_independent: float = 0.3  # 80% success without help
    upper_independent: float = 0.55   # 50% success without help
    upper_scaffolded: float = 0.70    # 80% success WITH help
    frustration_ceiling: float = 0.85 # <50% even with help

    # Derived metrics
    @property
    def zpd_width(self) -> float:
        """Width of ZPD = what can be learned with scaffolding."""
        return self.upper_scaffolded - self.current_independent

    @property
    def optimal_difficulty(self) -> float:
        """Optimal difficulty for learning (middle of scaffolded zone)."""
        return (self.current_independent + self.upper_scaffolded) / 2

    # Tracking
    last_updated: datetime = field(default_factory=datetime.now)
    interactions_count: int = 0
    scaffold_threshold: float = 2.0   # NCDE friction threshold for scaffolds

    def get_position(self, item_difficulty: float) -> ZPDPosition:
        """Determine where an item falls relative to ZPD boundaries."""
        if item_difficulty < self.lower_bound:
            return ZPDPosition.BELOW_ZPD
        elif item_difficulty < self.current_independent:
            return ZPDPosition.LOWER_ZPD
        elif item_difficulty < self.upper_independent:
            return ZPDPosition.OPTIMAL_ZPD
        elif item_difficulty <= self.upper_scaffolded:
            return ZPDPosition.UPPER_ZPD
        else:
            return ZPDPosition.ABOVE_ZPD

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "concept_id": self.concept_id,
            "learner_id": self.learner_id,
            "theta": round(self.theta, 3),
            "theta_se": round(self.theta_se, 3),
            "lower_bound": round(self.lower_bound, 3),
            "current_independent": round(self.current_independent, 3),
            "upper_independent": round(self.upper_independent, 3),
            "upper_scaffolded": round(self.upper_scaffolded, 3),
            "frustration_ceiling": round(self.frustration_ceiling, 3),
            "zpd_width": round(self.zpd_width, 3),
            "optimal_difficulty": round(self.optimal_difficulty, 3),
            "interactions_count": self.interactions_count,
            "scaffold_threshold": self.scaffold_threshold,
        }


@dataclass
class FlowChannelState:
    """
    Current flow channel state for session management.

    Based on Csikszentmihalyi's Flow Model where optimal experience
    occurs when challenge ≈ skill.
    """

    skill_level: float           # IRT theta
    current_challenge: float     # Current item difficulty
    flow_state: FlowState        # Classified flow state

    # Optimal range
    optimal_challenge_low: float = 0.0   # Lower bound of flow zone
    optimal_challenge_high: float = 0.0  # Upper bound of flow zone

    # Indicators
    boredom_indicators: list[str] = field(default_factory=list)
    anxiety_indicators: list[str] = field(default_factory=list)

    # Recommended intervention
    intervention: str | None = None

    def in_flow(self) -> bool:
        """Check if learner is in flow state."""
        return self.flow_state == FlowState.FLOW


@dataclass
class ScaffoldDecision:
    """
    Decision about what scaffold to apply (if any).
    """

    scaffold_type: ScaffoldType
    opacity: float               # 0.0-1.0
    reason: str                  # Why this scaffold was chosen
    struggle_type: StruggleType | None = None
    friction_level: float = 0.0  # NCDE friction that triggered this
    auto_fade_seconds: int = 0   # Time before scaffold fades (0 = manual)

    # Content details
    scaffold_content: dict[str, Any] = field(default_factory=dict)


@dataclass
class ZPDGrowthRecord:
    """
    Record of ZPD changes over time for tracking learning progression.
    """

    concept_id: str
    learner_id: str
    timestamp: datetime

    # Boundary snapshots
    lower_bound: float
    upper_bound: float
    zpd_width: float

    # Change from previous
    lower_shift: float = 0.0     # How much lower bound moved
    upper_shift: float = 0.0     # How much upper bound moved
    width_change: float = 0.0    # Change in ZPD width


@dataclass
class NCDEFrictionVector:
    """
    NCDE (Normalized Cognitive Demand Estimate) friction vector.

    Measures real-time cognitive friction during learning:
    - Low friction (< 1.5): Learner is comfortable
    - Moderate friction (1.5-2.5): Optimal challenge zone
    - High friction (> 2.5): Approaching overload
    - Critical friction (> 3.0): Cognitive collapse imminent
    """

    # Component frictions
    retrieval_friction: float = 0.0    # Difficulty recalling information
    integration_friction: float = 0.0  # Difficulty connecting concepts
    execution_friction: float = 0.0    # Difficulty applying knowledge
    metacognitive_friction: float = 0.0  # Uncertainty about own understanding

    @property
    def total_friction(self) -> float:
        """Compute total friction as weighted average."""
        # Weights based on cognitive load theory
        weights = {
            "retrieval": 0.25,
            "integration": 0.30,
            "execution": 0.25,
            "metacognitive": 0.20,
        }
        return (
            weights["retrieval"] * self.retrieval_friction +
            weights["integration"] * self.integration_friction +
            weights["execution"] * self.execution_friction +
            weights["metacognitive"] * self.metacognitive_friction
        )

    @property
    def is_critical(self) -> bool:
        """Check if friction indicates cognitive collapse."""
        return self.total_friction > ZPD_THRESHOLDS["friction_collapse"]

    @property
    def is_optimal(self) -> bool:
        """Check if friction is in optimal learning zone."""
        f = self.total_friction
        return ZPD_THRESHOLDS["friction_optimal"] <= f <= ZPD_THRESHOLDS["friction_moderate"]


# =============================================================================
# ZPD CALCULATOR
# =============================================================================


class ZPDCalculator:
    """
    Calculates and updates ZPD boundaries based on learner performance.

    Uses a Bayesian approach to estimate boundaries:
    1. Prior from population norms
    2. Update with each interaction
    3. Maintain uncertainty estimates
    """

    def __init__(
        self,
        min_interactions: int = ZPD_THRESHOLDS["min_interactions_for_zpd"],
        independent_threshold: float = ZPD_THRESHOLDS["independent_success_rate"],
        scaffolded_threshold: float = ZPD_THRESHOLDS["scaffolded_success_rate"],
        frustration_threshold: float = ZPD_THRESHOLDS["frustration_success_rate"],
    ):
        self.min_interactions = min_interactions
        self.independent_threshold = independent_threshold
        self.scaffolded_threshold = scaffolded_threshold
        self.frustration_threshold = frustration_threshold

        logger.info(
            f"ZPDCalculator initialized: min_interactions={min_interactions}, "
            f"independent={independent_threshold}, scaffolded={scaffolded_threshold}"
        )

    def compute_boundaries(
        self,
        interactions: list[dict[str, Any]],
        concept_id: str,
        learner_id: str,
    ) -> ZPDState:
        """
        Compute ZPD boundaries from interaction history.

        Args:
            interactions: List of interactions with fields:
                - difficulty: IRT difficulty of item
                - is_correct: Whether answer was correct
                - had_scaffold: Whether scaffold was provided
            concept_id: Concept/skill being measured
            learner_id: Learner identifier

        Returns:
            ZPDState with computed boundaries
        """
        if len(interactions) < self.min_interactions:
            logger.warning(
                f"Insufficient interactions ({len(interactions)} < {self.min_interactions}) "
                f"for accurate ZPD calculation. Using defaults."
            )
            return ZPDState(
                concept_id=concept_id,
                learner_id=learner_id,
                interactions_count=len(interactions),
            )

        # Separate scaffolded and unscaffolded interactions
        independent = [i for i in interactions if not i.get("had_scaffold", False)]
        scaffolded = [i for i in interactions if i.get("had_scaffold", False)]

        # Group by difficulty bins
        difficulty_bins = self._bin_by_difficulty(interactions)
        independent_bins = self._bin_by_difficulty(independent) if independent else {}
        scaffolded_bins = self._bin_by_difficulty(scaffolded) if scaffolded else {}

        # Estimate theta (ability) using MLE
        theta = self._estimate_theta(interactions)

        # Find boundaries
        lower_bound = self._find_threshold_difficulty(
            independent_bins, 0.95, default=theta - 0.5
        )
        current_independent = self._find_threshold_difficulty(
            independent_bins, self.independent_threshold, default=theta
        )
        upper_independent = self._find_threshold_difficulty(
            independent_bins, 0.50, default=theta + 0.25
        )
        upper_scaffolded = self._find_threshold_difficulty(
            scaffolded_bins, self.scaffolded_threshold, default=theta + 0.45
        )
        frustration_ceiling = self._find_threshold_difficulty(
            difficulty_bins, self.frustration_threshold, default=theta + 0.60, find_below=True
        )

        # Ensure monotonicity
        if current_independent < lower_bound:
            current_independent = lower_bound + 0.1
        if upper_independent < current_independent:
            upper_independent = current_independent + 0.15
        if upper_scaffolded < upper_independent:
            upper_scaffolded = upper_independent + 0.15
        if frustration_ceiling < upper_scaffolded:
            frustration_ceiling = upper_scaffolded + 0.15

        state = ZPDState(
            concept_id=concept_id,
            learner_id=learner_id,
            theta=theta,
            theta_se=self._estimate_theta_se(interactions, theta),
            lower_bound=lower_bound,
            current_independent=current_independent,
            upper_independent=upper_independent,
            upper_scaffolded=upper_scaffolded,
            frustration_ceiling=frustration_ceiling,
            interactions_count=len(interactions),
        )

        logger.info(
            f"ZPD computed for {learner_id}/{concept_id}: "
            f"θ={theta:.2f}, width={state.zpd_width:.2f}, "
            f"boundaries=[{lower_bound:.2f}, {current_independent:.2f}, "
            f"{upper_independent:.2f}, {upper_scaffolded:.2f}, {frustration_ceiling:.2f}]"
        )

        return state

    def _bin_by_difficulty(
        self,
        interactions: list[dict[str, Any]],
        bin_width: float = 0.2,
    ) -> dict[float, dict[str, int]]:
        """Group interactions into difficulty bins."""
        bins: dict[float, dict[str, int]] = {}

        for interaction in interactions:
            difficulty = interaction.get("difficulty", 0.5)
            bin_center = round(difficulty / bin_width) * bin_width

            if bin_center not in bins:
                bins[bin_center] = {"correct": 0, "total": 0}

            bins[bin_center]["total"] += 1
            if interaction.get("is_correct", False):
                bins[bin_center]["correct"] += 1

        return bins

    def _find_threshold_difficulty(
        self,
        bins: dict[float, dict[str, int]],
        success_threshold: float,
        default: float,
        find_below: bool = False,
    ) -> float:
        """
        Find difficulty level at which success rate crosses threshold.

        Args:
            bins: Difficulty bins with correct/total counts
            success_threshold: Target success rate
            default: Default value if not enough data
            find_below: If True, find where success drops BELOW threshold

        Returns:
            Difficulty level at threshold
        """
        if not bins:
            return default

        # Sort bins by difficulty
        sorted_bins = sorted(bins.items())

        for difficulty, counts in sorted_bins:
            if counts["total"] < 3:
                continue

            success_rate = counts["correct"] / counts["total"]

            if find_below:
                if success_rate < success_threshold:
                    return difficulty
            else:
                if success_rate <= success_threshold:
                    return difficulty

        return default

    def _estimate_theta(self, interactions: list[dict[str, Any]]) -> float:
        """
        Estimate ability (theta) using simple MLE.

        For a more sophisticated implementation, use proper IRT estimation
        with py-irt or similar library.
        """
        if not interactions:
            return 0.0

        # Weighted average: correct answers at high difficulty = high ability
        total_weight = 0.0
        weighted_sum = 0.0

        for interaction in interactions:
            difficulty = interaction.get("difficulty", 0.5)
            is_correct = interaction.get("is_correct", False)

            # Weight by recency (more recent = higher weight)
            weight = 1.0

            if is_correct:
                weighted_sum += (difficulty + 0.5) * weight  # Correct at hard = high ability
            else:
                weighted_sum += (difficulty - 0.5) * weight  # Wrong at easy = low ability

            total_weight += weight

        if total_weight == 0:
            return 0.0

        # Scale to typical IRT range
        raw_theta = weighted_sum / total_weight
        return max(-3.0, min(3.0, raw_theta))

    def _estimate_theta_se(
        self,
        interactions: list[dict[str, Any]],
        theta: float,
    ) -> float:
        """Estimate standard error of theta based on information."""
        # Fisher information approximation
        # I(θ) = Σ P(1-P) where P = probability of correct
        total_info = 0.0

        for interaction in interactions:
            difficulty = interaction.get("difficulty", 0.5)
            # 2PL probability
            p = 1.0 / (1.0 + math.exp(-(theta - difficulty)))
            info = p * (1 - p)
            total_info += info

        if total_info == 0:
            return 1.0  # High uncertainty

        se = 1.0 / math.sqrt(total_info)
        return min(1.0, se)  # Cap at 1.0

    def update_with_interaction(
        self,
        current_state: ZPDState,
        interaction: dict[str, Any],
    ) -> ZPDState:
        """
        Update ZPD state with a new interaction (online update).

        Uses exponential moving average for quick adaptation.
        """
        difficulty = interaction.get("difficulty", 0.5)
        is_correct = interaction.get("is_correct", False)
        had_scaffold = interaction.get("had_scaffold", False)

        # Learning rate (higher for newer learners)
        alpha = 0.1 if current_state.interactions_count > 50 else 0.2

        # Update theta
        if is_correct:
            if difficulty > current_state.theta:
                # Correct at hard item = increase ability estimate
                current_state.theta += alpha * (difficulty - current_state.theta)
        else:
            if difficulty < current_state.theta:
                # Wrong at easy item = decrease ability estimate
                current_state.theta -= alpha * (current_state.theta - difficulty)

        # Update boundaries based on performance
        position = current_state.get_position(difficulty)

        if position == ZPDPosition.OPTIMAL_ZPD and is_correct and not had_scaffold:
            # Success in ZPD without scaffold = expand ZPD upward
            current_state.upper_independent += alpha * 0.1
            current_state.upper_scaffolded += alpha * 0.1

        if position == ZPDPosition.UPPER_ZPD and is_correct and had_scaffold:
            # Success with scaffold = can potentially do without
            current_state.upper_scaffolded += alpha * 0.05

        if position == ZPDPosition.ABOVE_ZPD and not is_correct:
            # Failure above ZPD = frustration ceiling confirmed
            current_state.frustration_ceiling = min(
                current_state.frustration_ceiling,
                difficulty - 0.1
            )

        current_state.interactions_count += 1
        current_state.last_updated = datetime.now()

        return current_state


# =============================================================================
# FLOW CHANNEL MANAGER
# =============================================================================


class FlowChannelManager:
    """
    Manages the Flow Channel to maintain optimal challenge-skill balance.

    Based on Csikszentmihalyi's Flow Model:
    - Boredom: skill >> challenge
    - Flow: skill ≈ challenge
    - Anxiety: challenge >> skill
    """

    def __init__(
        self,
        flow_lower: float = ZPD_THRESHOLDS["flow_lower"],
        flow_upper: float = ZPD_THRESHOLDS["flow_upper"],
        boredom_threshold: float = ZPD_THRESHOLDS["boredom_threshold"],
        anxiety_threshold: float = ZPD_THRESHOLDS["anxiety_threshold"],
    ):
        self.flow_lower = flow_lower
        self.flow_upper = flow_upper
        self.boredom_threshold = boredom_threshold
        self.anxiety_threshold = anxiety_threshold

    def compute_flow_state(
        self,
        skill_theta: float,
        item_difficulty: float,
        session_metrics: dict[str, Any] | None = None,
    ) -> FlowChannelState:
        """
        Compute current flow state and recommendations.

        Args:
            skill_theta: Learner's IRT ability
            item_difficulty: Current/proposed item difficulty
            session_metrics: Optional session metrics for detailed analysis

        Returns:
            FlowChannelState with classification and recommendations
        """
        delta = item_difficulty - skill_theta  # Challenge - Skill

        # Classify flow state
        if delta < self.boredom_threshold:
            flow_state = FlowState.BOREDOM
        elif delta < self.flow_lower:
            flow_state = FlowState.RELAXATION
        elif delta <= self.flow_upper:
            flow_state = FlowState.FLOW
        elif delta <= self.anxiety_threshold:
            flow_state = FlowState.AROUSAL
        else:
            flow_state = FlowState.ANXIETY

        # Compute optimal range
        optimal_low = skill_theta + self.flow_lower
        optimal_high = skill_theta + self.flow_upper

        state = FlowChannelState(
            skill_level=skill_theta,
            current_challenge=item_difficulty,
            flow_state=flow_state,
            optimal_challenge_low=optimal_low,
            optimal_challenge_high=optimal_high,
        )

        # Add indicators and intervention if applicable
        self._add_indicators_and_intervention(state, session_metrics)

        return state

    def _add_indicators_and_intervention(
        self,
        state: FlowChannelState,
        session_metrics: dict[str, Any] | None,
    ) -> None:
        """Add boredom/anxiety indicators and intervention recommendations."""
        if state.flow_state == FlowState.BOREDOM:
            state.boredom_indicators = [
                "Challenge significantly below skill level",
            ]
            if session_metrics:
                if session_metrics.get("response_speed", 0) > 0.8:  # Very fast
                    state.boredom_indicators.append("Very fast response times")
                if session_metrics.get("accuracy", 0) > 0.95:
                    state.boredom_indicators.append("Near-perfect accuracy (98%+)")
                if session_metrics.get("attention_drift", 0) > 0.5:
                    state.boredom_indicators.append("High attention drift detected")

            state.intervention = "increase_difficulty"

        elif state.flow_state == FlowState.ANXIETY:
            state.anxiety_indicators = [
                "Challenge significantly exceeds skill level",
            ]
            if session_metrics:
                if session_metrics.get("response_speed", 0) < 0.3:  # Very slow
                    state.anxiety_indicators.append("Very slow response times")
                if session_metrics.get("error_rate", 0) > 0.5:
                    state.anxiety_indicators.append("High error rate (50%+)")
                if session_metrics.get("friction", 0) > 2.5:
                    state.anxiety_indicators.append("High NCDE friction")

            state.intervention = "inject_scaffold"

        elif state.flow_state == FlowState.RELAXATION:
            state.intervention = "slight_increase"

        elif state.flow_state == FlowState.AROUSAL:
            state.intervention = "offer_hint"

    def get_optimal_difficulty(
        self,
        skill_theta: float,
        context: str = "practice",
    ) -> tuple[float, float]:
        """
        Get optimal difficulty range for item selection.

        Args:
            skill_theta: Learner's IRT ability
            context: "practice" for learning, "assessment" for testing

        Returns:
            (min_difficulty, max_difficulty) tuple
        """
        if context == "assessment":
            # For assessment, target theta exactly
            return (skill_theta - 0.2, skill_theta + 0.2)

        # For practice, target slightly above skill (desirable difficulty)
        return (
            skill_theta + ZPD_THRESHOLDS["irt_theta_offset_low"],
            skill_theta + ZPD_THRESHOLDS["irt_theta_offset_high"],
        )

    def detect_boredom(
        self,
        session_metrics: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        """
        Detect boredom based on session metrics.

        Returns:
            (is_bored, list of indicators)
        """
        indicators = []
        score = 0.0

        # Fast responses = low engagement
        if session_metrics.get("avg_response_ms", 5000) < 2000:
            indicators.append("Very fast responses (<2s avg)")
            score += 0.3

        # High accuracy = too easy
        if session_metrics.get("accuracy", 0) > 0.95:
            indicators.append("Near-perfect accuracy (>95%)")
            score += 0.3

        # Low friction = no challenge
        if session_metrics.get("friction", 1.5) < 0.5:
            indicators.append("Very low NCDE friction (<0.5)")
            score += 0.2

        # Attention drift (if tracked)
        if session_metrics.get("attention_drift", 0) > 0.5:
            indicators.append("High attention drift detected")
            score += 0.2

        return score >= 0.5, indicators

    def detect_anxiety(
        self,
        session_metrics: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        """
        Detect anxiety/frustration based on session metrics.

        Returns:
            (is_anxious, list of indicators)
        """
        indicators = []
        score = 0.0

        # Slow responses = struggling
        if session_metrics.get("avg_response_ms", 5000) > 12000:
            indicators.append("Very slow responses (>12s avg)")
            score += 0.25

        # High error rate = too hard
        if session_metrics.get("error_rate", 0) > 0.5:
            indicators.append("High error rate (>50%)")
            score += 0.3

        # High friction = cognitive overload
        if session_metrics.get("friction", 1.5) > 2.5:
            indicators.append("High NCDE friction (>2.5)")
            score += 0.25

        # Frequent pauses (if tracked)
        if session_metrics.get("pause_frequency", 0) > 0.5:
            indicators.append("Frequent pauses detected")
            score += 0.2

        return score >= 0.5, indicators


# =============================================================================
# SCAFFOLD SELECTOR
# =============================================================================


class ScaffoldSelector:
    """
    Selects appropriate scaffolding based on learner state and struggle type.

    Implements dynamic scaffolding with:
    - Just-in-time delivery (at point of need)
    - Gradual release of responsibility (fading)
    - Expertise reversal awareness (too much scaffold = harmful)
    """

    def __init__(self):
        self.friction_thresholds = {
            ScaffoldType.FULL_WORKED_EXAMPLE: ZPD_THRESHOLDS["friction_collapse"],
            ScaffoldType.FADED_PARSONS: ZPD_THRESHOLDS["friction_high"],
            ScaffoldType.STRUCTURAL_HINT: ZPD_THRESHOLDS["friction_moderate"],
            ScaffoldType.FIRST_STEP_ONLY: ZPD_THRESHOLDS["friction_mild"],
            ScaffoldType.NO_SCAFFOLD: 0.0,
        }

    def select_scaffold(
        self,
        friction: NCDEFrictionVector,
        struggle_type: StruggleType | None = None,
        zpd_state: ZPDState | None = None,
    ) -> ScaffoldDecision:
        """
        Select appropriate scaffold based on current state.

        Args:
            friction: Current NCDE friction vector
            struggle_type: Classified type of struggle (if known)
            zpd_state: Learner's ZPD state (for calibration)

        Returns:
            ScaffoldDecision with type, opacity, and reasoning
        """
        total_friction = friction.total_friction

        # Select scaffold type by friction level
        scaffold_type = ScaffoldType.NO_SCAFFOLD

        if total_friction >= self.friction_thresholds[ScaffoldType.FULL_WORKED_EXAMPLE]:
            scaffold_type = ScaffoldType.FULL_WORKED_EXAMPLE
        elif total_friction >= self.friction_thresholds[ScaffoldType.FADED_PARSONS]:
            scaffold_type = ScaffoldType.FADED_PARSONS
        elif total_friction >= self.friction_thresholds[ScaffoldType.STRUCTURAL_HINT]:
            scaffold_type = ScaffoldType.STRUCTURAL_HINT
        elif total_friction >= self.friction_thresholds[ScaffoldType.FIRST_STEP_ONLY]:
            scaffold_type = ScaffoldType.FIRST_STEP_ONLY

        # Override based on struggle type if provided
        if struggle_type:
            scaffold_type = self._select_by_struggle_type(struggle_type, scaffold_type)

        # Generate reason
        reason = self._generate_reason(scaffold_type, total_friction, struggle_type)

        # Compute auto-fade time (higher opacity = longer before fade)
        auto_fade = int(scaffold_type.opacity * 120)  # 0-120 seconds

        return ScaffoldDecision(
            scaffold_type=scaffold_type,
            opacity=scaffold_type.opacity,
            reason=reason,
            struggle_type=struggle_type,
            friction_level=total_friction,
            auto_fade_seconds=auto_fade,
        )

    def _select_by_struggle_type(
        self,
        struggle_type: StruggleType,
        current_selection: ScaffoldType,
    ) -> ScaffoldType:
        """Override scaffold selection based on struggle type."""
        mapping = {
            StruggleType.CONCEPTUAL_GAP: ScaffoldType.FULL_WORKED_EXAMPLE,
            StruggleType.PROCEDURAL_STUCK: ScaffoldType.FADED_PARSONS,
            StruggleType.MINOR_CONFUSION: ScaffoldType.STRUCTURAL_HINT,
            StruggleType.VERIFICATION_NEED: ScaffoldType.FIRST_STEP_ONLY,
            StruggleType.COGNITIVE_OVERLOAD: ScaffoldType.FULL_WORKED_EXAMPLE,
        }

        suggested = mapping.get(struggle_type, current_selection)

        # Take the higher opacity (more support) if friction suggests it
        if suggested.opacity < current_selection.opacity:
            return current_selection

        return suggested

    def _generate_reason(
        self,
        scaffold_type: ScaffoldType,
        friction: float,
        struggle_type: StruggleType | None,
    ) -> str:
        """Generate human-readable reason for scaffold selection."""
        reasons = {
            ScaffoldType.NO_SCAFFOLD: f"Friction {friction:.2f} is within optimal range. Independent practice.",
            ScaffoldType.FIRST_STEP_ONLY: f"Mild friction ({friction:.2f}). Just need a starting point.",
            ScaffoldType.STRUCTURAL_HINT: f"Moderate friction ({friction:.2f}). Provide structural guidance.",
            ScaffoldType.FADED_PARSONS: f"High friction ({friction:.2f}). Provide partial solution with blanks.",
            ScaffoldType.FULL_WORKED_EXAMPLE: f"Critical friction ({friction:.2f}). Show complete worked example.",
        }

        base_reason = reasons[scaffold_type]

        if struggle_type:
            base_reason += f" Struggle type: {struggle_type.value}."

        return base_reason

    def should_fade_scaffold(
        self,
        current_opacity: float,
        recent_performance: list[dict[str, Any]],
    ) -> tuple[bool, float]:
        """
        Determine if scaffold should fade based on recent performance.

        Args:
            current_opacity: Current scaffold opacity (0.0-1.0)
            recent_performance: Recent interactions with accuracy and latency

        Returns:
            (should_fade, new_opacity)
        """
        if len(recent_performance) < ZPD_THRESHOLDS["fade_min_interactions"]:
            return False, current_opacity

        if current_opacity <= 0.0:
            return False, 0.0

        # Calculate improvement
        correct_count = sum(1 for p in recent_performance if p.get("is_correct", False))
        accuracy = correct_count / len(recent_performance)

        avg_latency = sum(
            p.get("response_time_ms", 5000) for p in recent_performance
        ) / len(recent_performance)

        # Check if performance warrants fading
        if accuracy >= 0.85 and avg_latency < 6000:
            # Good performance - reduce opacity by 25%
            new_opacity = max(0.0, current_opacity - 0.25)
            return True, new_opacity

        return False, current_opacity

    def compute_fade_sequence(
        self,
        starting_opacity: float,
        num_interactions: int = 5,
    ) -> list[float]:
        """
        Compute a gradual fade sequence for scaffolding.

        Used for faded worked examples progression.

        Args:
            starting_opacity: Initial opacity
            num_interactions: Number of interactions to fade over

        Returns:
            List of opacity values, ending at 0.0
        """
        if num_interactions <= 1:
            return [starting_opacity, 0.0]

        step = starting_opacity / num_interactions
        return [max(0.0, starting_opacity - (i * step)) for i in range(num_interactions + 1)]


# =============================================================================
# ZPD ENGINE (Main Orchestrator)
# =============================================================================


class ZPDEngine:
    """
    Main orchestrator for ZPD-based adaptive learning.

    Integrates:
    - ZPD boundary calculation
    - Flow channel management
    - Scaffold selection
    - Real-time adaptation
    """

    def __init__(self):
        self.zpd_calculator = ZPDCalculator()
        self.flow_manager = FlowChannelManager()
        self.scaffold_selector = ScaffoldSelector()

        # Cache for ZPD states
        self._zpd_cache: dict[str, ZPDState] = {}

        logger.info("ZPDEngine initialized")

    def get_zpd_state(
        self,
        concept_id: str,
        learner_id: str,
        interactions: list[dict[str, Any]] | None = None,
    ) -> ZPDState:
        """
        Get or compute ZPD state for a concept/learner pair.

        Args:
            concept_id: Concept identifier
            learner_id: Learner identifier
            interactions: Optional interaction history for computation

        Returns:
            ZPDState
        """
        cache_key = f"{learner_id}:{concept_id}"

        if cache_key in self._zpd_cache and interactions is None:
            return self._zpd_cache[cache_key]

        if interactions:
            state = self.zpd_calculator.compute_boundaries(
                interactions, concept_id, learner_id
            )
            self._zpd_cache[cache_key] = state
            return state

        # Return default state if not cached and no data
        return ZPDState(concept_id=concept_id, learner_id=learner_id)

    def update_zpd(
        self,
        concept_id: str,
        learner_id: str,
        interaction: dict[str, Any],
    ) -> ZPDState:
        """
        Update ZPD state with a new interaction.

        Args:
            concept_id: Concept identifier
            learner_id: Learner identifier
            interaction: New interaction data

        Returns:
            Updated ZPDState
        """
        cache_key = f"{learner_id}:{concept_id}"
        current = self._zpd_cache.get(
            cache_key,
            ZPDState(concept_id=concept_id, learner_id=learner_id)
        )

        updated = self.zpd_calculator.update_with_interaction(current, interaction)
        self._zpd_cache[cache_key] = updated

        return updated

    def get_flow_state(
        self,
        skill_theta: float,
        item_difficulty: float,
        session_metrics: dict[str, Any] | None = None,
    ) -> FlowChannelState:
        """
        Get current flow channel state.

        Args:
            skill_theta: Learner's IRT ability
            item_difficulty: Item difficulty
            session_metrics: Session metrics for detailed analysis

        Returns:
            FlowChannelState
        """
        return self.flow_manager.compute_flow_state(
            skill_theta, item_difficulty, session_metrics
        )

    def select_scaffold(
        self,
        friction: NCDEFrictionVector,
        struggle_type: StruggleType | None = None,
        zpd_state: ZPDState | None = None,
    ) -> ScaffoldDecision:
        """
        Select appropriate scaffold for current state.

        Args:
            friction: NCDE friction vector
            struggle_type: Type of struggle (if known)
            zpd_state: Learner's ZPD state

        Returns:
            ScaffoldDecision
        """
        return self.scaffold_selector.select_scaffold(friction, struggle_type, zpd_state)

    def recommend_item_difficulty(
        self,
        zpd_state: ZPDState,
        context: str = "practice",
    ) -> tuple[float, float, str]:
        """
        Recommend difficulty range for next item.

        Args:
            zpd_state: Learner's ZPD state
            context: "practice" or "assessment"

        Returns:
            (min_difficulty, max_difficulty, reason)
        """
        if context == "assessment":
            # For assessment, target theta
            return (
                zpd_state.theta - 0.2,
                zpd_state.theta + 0.2,
                "Assessment: targeting ability level for accurate measurement"
            )

        # For practice, target optimal ZPD zone
        return (
            zpd_state.current_independent,
            zpd_state.upper_scaffolded,
            f"Practice: targeting ZPD (optimal difficulty: {zpd_state.optimal_difficulty:.2f})"
        )

    def check_zpd_expansion(
        self,
        old_state: ZPDState,
        new_state: ZPDState,
    ) -> ZPDGrowthRecord | None:
        """
        Check if ZPD has expanded and record growth.

        Args:
            old_state: Previous ZPD state
            new_state: Current ZPD state

        Returns:
            ZPDGrowthRecord if expansion detected, None otherwise
        """
        lower_shift = new_state.lower_bound - old_state.lower_bound
        upper_shift = new_state.upper_scaffolded - old_state.upper_scaffolded
        width_change = new_state.zpd_width - old_state.zpd_width

        # Only record if meaningful change
        if abs(upper_shift) > 0.05 or abs(lower_shift) > 0.05:
            return ZPDGrowthRecord(
                concept_id=new_state.concept_id,
                learner_id=new_state.learner_id,
                timestamp=datetime.now(),
                lower_bound=new_state.lower_bound,
                upper_bound=new_state.upper_scaffolded,
                zpd_width=new_state.zpd_width,
                lower_shift=lower_shift,
                upper_shift=upper_shift,
                width_change=width_change,
            )

        return None

    def handle_success_streak(
        self,
        zpd_state: ZPDState,
        streak_length: int,
        avg_latency_ms: int,
    ) -> dict[str, Any]:
        """
        Handle success streak for accelerated learning.

        Implements ZPD expansion scenario: bypass N+1, present N+2 challenge.

        Args:
            zpd_state: Current ZPD state
            streak_length: Number of consecutive successes
            avg_latency_ms: Average response latency

        Returns:
            Recommendation dict with action and difficulty
        """
        # High efficiency = low latency + high accuracy
        high_efficiency = avg_latency_ms < 4000 and streak_length >= 3

        if high_efficiency:
            # Bypass N+1, present N+2 challenge
            skip_difficulty = zpd_state.optimal_difficulty + 0.3

            return {
                "action": "ZPD_EXPANSION",
                "skip_levels": 1,
                "new_target_difficulty": skip_difficulty,
                "reason": f"Success streak ({streak_length}) with high efficiency. Skipping to challenge level.",
                "record_event": "ZPD_EXPANSION",
            }

        # Normal progression
        return {
            "action": "CONTINUE",
            "new_target_difficulty": zpd_state.optimal_difficulty,
            "reason": "Continue at optimal ZPD difficulty",
        }

    def handle_struggle_signal(
        self,
        friction: NCDEFrictionVector,
        zpd_state: ZPDState,
    ) -> dict[str, Any]:
        """
        Handle struggle signal with scaffold injection.

        Implements intelligent scaffolding re-entry on ZPD exit.

        Args:
            friction: Current friction vector
            zpd_state: Current ZPD state

        Returns:
            Intervention recommendation
        """
        if not friction.is_critical:
            return {"action": "CONTINUE", "scaffold": None}

        scaffold = self.select_scaffold(friction, zpd_state=zpd_state)

        return {
            "action": "INJECT_SCAFFOLD",
            "scaffold": scaffold,
            "mark_zpd_limit": True,
            "new_zpd_limit": zpd_state.upper_scaffolded,
            "temporary_scaffolding": True,
            "reason": f"Cognitive overload detected (friction: {friction.total_friction:.2f}). "
                     f"Injecting {scaffold.scaffold_type.value} scaffold.",
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def compute_friction_from_metrics(
    response_time_ms: int,
    accuracy_rate: float,
    error_streak: int = 0,
    session_duration_minutes: float = 0,
) -> NCDEFrictionVector:
    """
    Compute NCDE friction vector from observable metrics.

    This is a simplified computation for when detailed cognitive
    tracking isn't available.
    """
    # Retrieval friction from response time
    retrieval_friction = min(3.0, (response_time_ms - 2000) / 4000)
    retrieval_friction = max(0.0, retrieval_friction)

    # Integration friction from accuracy (inverse)
    integration_friction = min(3.0, (1.0 - accuracy_rate) * 4)

    # Execution friction from error streak
    execution_friction = min(3.0, error_streak * 0.6)

    # Metacognitive friction from session duration
    if session_duration_minutes > 45:
        metacognitive_friction = min(3.0, (session_duration_minutes - 45) / 15)
    else:
        metacognitive_friction = 0.0

    return NCDEFrictionVector(
        retrieval_friction=retrieval_friction,
        integration_friction=integration_friction,
        execution_friction=execution_friction,
        metacognitive_friction=metacognitive_friction,
    )
