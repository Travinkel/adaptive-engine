"""
Bayesian Topic Readiness & Knowledge Tracing.

Implements Bayesian Networks for calculating topic readiness based on
prerequisite mastery, enabling intelligent prerequisite gating and
interleaved practice triggers.

Work Order: WO-AE-006
Tags: @Algorithm-BayesianNetwork, @Science-InformationTheory, @DARPA-DigitalTutor
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set
from uuid import UUID, uuid4
import math


class TopicState(str, Enum):
    """State of a topic in the knowledge graph."""

    LOCKED = "locked"       # Prerequisites not met
    UNLOCKED = "unlocked"   # Ready for learning
    MASTERED = "mastered"   # Fully mastered (P >= 0.85)


class InterventionType(str, Enum):
    """Types of learning interventions."""

    NONE = "none"
    INTERLEAVED_PRACTICE = "interleaved_practice"
    REMEDIATION = "remediation"
    REVIEW = "review"


@dataclass
class ReadinessScore:
    """
    Bayesian readiness score for a topic.

    P(Ready_B) = probability that the learner is ready to learn Topic B,
    computed from the mastery of its prerequisites.
    """

    topic_id: str
    probability: float  # P(Ready_B) - 0.0 to 1.0
    state: TopicState
    contributing_prerequisites: Dict[str, float] = field(default_factory=dict)
    threshold: float = 0.8  # Unlock threshold

    @property
    def is_unlocked(self) -> bool:
        """Topic is unlocked if P(Ready) > threshold."""
        return self.probability > self.threshold

    def to_dict(self) -> dict:
        return {
            "topic_id": self.topic_id,
            "probability": round(self.probability, 4),
            "state": self.state.value,
            "is_unlocked": self.is_unlocked,
            "threshold": self.threshold,
            "contributing_prerequisites": {
                k: round(v, 4) for k, v in self.contributing_prerequisites.items()
            },
        }


@dataclass
class InterventionDecision:
    """Decision about what intervention is needed."""

    intervention_type: InterventionType
    target_topic_id: str
    reason: str
    priority: float = 0.5  # 0-1, higher = more urgent
    prerequisite_topics: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "intervention_type": self.intervention_type.value,
            "target_topic_id": self.target_topic_id,
            "reason": self.reason,
            "priority": round(self.priority, 4),
            "prerequisite_topics": self.prerequisite_topics,
        }


@dataclass
class TopicNode:
    """A topic node in the knowledge graph."""

    topic_id: str
    name: str
    prerequisites: Set[str] = field(default_factory=set)
    mastery_probability: float = 0.0  # P(Mastered)
    state: TopicState = TopicState.LOCKED

    def to_dict(self) -> dict:
        return {
            "topic_id": self.topic_id,
            "name": self.name,
            "prerequisites": list(self.prerequisites),
            "mastery_probability": round(self.mastery_probability, 4),
            "state": self.state.value,
        }


class BayesianReadinessCalculator:
    """
    Calculates topic readiness using Bayesian Networks.

    The core insight: P(Ready_B | Mastered_A) represents how ready
    a learner is for Topic B given their mastery of prerequisite Topic A.

    For multiple prerequisites:
    P(Ready_B) = product of P(Ready_B | Mastered_Ai) for all prereqs

    This implements a simple Noisy-AND gate in Bayesian Network terms.
    """

    # Thresholds
    UNLOCK_THRESHOLD = 0.8       # P(Ready) > 0.8 to unlock
    MASTERY_THRESHOLD = 0.85    # P(Mastered) >= 0.85 means mastered
    RELOCK_THRESHOLD = 0.5      # P(Ready) < 0.5 triggers intervention

    # Conditional probability parameters
    # P(Ready_B | Mastered_A = True) - high if prerequisite mastered
    CPT_READY_GIVEN_MASTERED = 0.95
    # P(Ready_B | Mastered_A = False) - low if prerequisite not mastered
    CPT_READY_GIVEN_NOT_MASTERED = 0.1

    def __init__(self):
        """Initialize the calculator."""
        self.topics: Dict[str, TopicNode] = {}
        self._intervention_history: List[InterventionDecision] = []

    def add_topic(
        self,
        topic_id: str,
        name: str,
        prerequisites: Optional[Set[str]] = None,
        initial_mastery: float = 0.0,
    ) -> TopicNode:
        """
        Add a topic to the knowledge graph.

        Args:
            topic_id: Unique identifier for the topic
            name: Human-readable name
            prerequisites: Set of prerequisite topic IDs
            initial_mastery: Initial mastery probability (0-1)

        Returns:
            The created TopicNode
        """
        node = TopicNode(
            topic_id=topic_id,
            name=name,
            prerequisites=prerequisites or set(),
            mastery_probability=initial_mastery,
            state=TopicState.MASTERED if initial_mastery >= self.MASTERY_THRESHOLD
                  else TopicState.LOCKED,
        )
        self.topics[topic_id] = node
        return node

    def set_mastery(self, topic_id: str, mastery: float) -> None:
        """
        Update mastery probability for a topic.

        Args:
            topic_id: Topic to update
            mastery: New mastery probability (0-1)
        """
        if topic_id not in self.topics:
            raise ValueError(f"Topic {topic_id} not found")

        self.topics[topic_id].mastery_probability = max(0.0, min(1.0, mastery))

        # Update mastered state
        if mastery >= self.MASTERY_THRESHOLD:
            self.topics[topic_id].state = TopicState.MASTERED

    def calculate_readiness(self, topic_id: str) -> ReadinessScore:
        """
        Calculate P(Ready) for a topic using Bayesian inference.

        The conditional probability table:
        - P(Ready_B | Mastered_A = 1) = 0.95
        - P(Ready_B | Mastered_A = 0) = 0.10

        For multiple prerequisites, we use a Noisy-AND:
        P(Ready_B) = product of individual contributions

        Args:
            topic_id: Topic to calculate readiness for

        Returns:
            ReadinessScore with computed probability
        """
        if topic_id not in self.topics:
            raise ValueError(f"Topic {topic_id} not found")

        topic = self.topics[topic_id]

        # No prerequisites = always ready
        if not topic.prerequisites:
            return ReadinessScore(
                topic_id=topic_id,
                probability=1.0,
                state=TopicState.UNLOCKED if topic.state != TopicState.MASTERED
                      else TopicState.MASTERED,
                threshold=self.UNLOCK_THRESHOLD,
            )

        # Calculate P(Ready_B) using conditional probability
        # P(Ready_B) = sum over A of P(Ready_B | A) * P(A)
        #            = P(Ready|Mastered) * P(Mastered) + P(Ready|NotMastered) * P(NotMastered)
        contributing = {}
        combined_probability = 1.0

        for prereq_id in topic.prerequisites:
            if prereq_id not in self.topics:
                # Unknown prerequisite = assume not mastered
                contrib = self.CPT_READY_GIVEN_NOT_MASTERED
            else:
                prereq = self.topics[prereq_id]
                p_mastered = prereq.mastery_probability

                # P(Ready | this prereq) = weighted sum of CPT
                contrib = (
                    self.CPT_READY_GIVEN_MASTERED * p_mastered +
                    self.CPT_READY_GIVEN_NOT_MASTERED * (1 - p_mastered)
                )

            contributing[prereq_id] = contrib
            combined_probability *= contrib

        # Normalize to prevent extreme values
        # Using geometric mean for multiple prerequisites
        if len(contributing) > 1:
            combined_probability = combined_probability ** (1 / len(contributing))

        # Determine state
        if topic.mastery_probability >= self.MASTERY_THRESHOLD:
            state = TopicState.MASTERED
        elif combined_probability > self.UNLOCK_THRESHOLD:
            state = TopicState.UNLOCKED
        else:
            state = TopicState.LOCKED

        return ReadinessScore(
            topic_id=topic_id,
            probability=combined_probability,
            state=state,
            contributing_prerequisites=contributing,
            threshold=self.UNLOCK_THRESHOLD,
        )

    def run_inference(self) -> Dict[str, ReadinessScore]:
        """
        Run Bayesian inference on all topics.

        Updates topic states based on computed readiness scores.

        Returns:
            Dictionary of topic_id -> ReadinessScore
        """
        results = {}

        for topic_id in self.topics:
            score = self.calculate_readiness(topic_id)
            results[topic_id] = score

            # Update topic state (but don't demote from MASTERED)
            if self.topics[topic_id].state != TopicState.MASTERED:
                self.topics[topic_id].state = score.state

        return results

    def check_intervention_needed(
        self,
        topic_id: str,
        previous_score: Optional[ReadinessScore] = None,
    ) -> Optional[InterventionDecision]:
        """
        Check if intervention is needed for a topic.

        Triggers interleaved practice if:
        1. Topic was unlocked but readiness dropped below threshold
        2. Prerequisite mastery has declined

        Args:
            topic_id: Topic to check
            previous_score: Previous readiness score (if available)

        Returns:
            InterventionDecision if intervention needed, None otherwise
        """
        current_score = self.calculate_readiness(topic_id)
        topic = self.topics.get(topic_id)

        if not topic:
            return None

        # Check for readiness drop requiring intervention
        needs_intervention = False
        reason = ""
        priority = 0.5
        prereqs_to_practice = []

        # Case 1: Was unlocked, now locked
        if (previous_score and
            previous_score.is_unlocked and
            not current_score.is_unlocked):
            needs_intervention = True
            reason = f"Readiness dropped from {previous_score.probability:.2f} to {current_score.probability:.2f}"
            priority = 0.8

        # Case 2: Readiness below critical threshold
        if current_score.probability < self.RELOCK_THRESHOLD:
            needs_intervention = True
            reason = f"Readiness critically low: {current_score.probability:.2f}"
            priority = 0.9

        # Identify which prerequisites need practice
        if needs_intervention:
            for prereq_id, contrib in current_score.contributing_prerequisites.items():
                if contrib < 0.7:  # Weak contribution
                    prereqs_to_practice.append(prereq_id)

            decision = InterventionDecision(
                intervention_type=InterventionType.INTERLEAVED_PRACTICE,
                target_topic_id=topic_id,
                reason=reason,
                priority=priority,
                prerequisite_topics=prereqs_to_practice,
            )
            self._intervention_history.append(decision)
            return decision

        return None

    def propagate_mastery_update(
        self,
        topic_id: str,
        new_mastery: float,
    ) -> Dict[str, ReadinessScore]:
        """
        Update mastery for a topic and propagate effects.

        When a prerequisite's mastery changes, all dependent topics
        need their readiness recalculated.

        Args:
            topic_id: Topic whose mastery changed
            new_mastery: New mastery probability

        Returns:
            Dictionary of affected topic_id -> new ReadinessScore
        """
        # Store previous scores for comparison
        previous_scores = {}
        for tid in self.topics:
            previous_scores[tid] = self.calculate_readiness(tid)

        # Update mastery
        self.set_mastery(topic_id, new_mastery)

        # Find all topics that depend on this one (directly or transitively)
        affected = self._find_dependents(topic_id)
        affected.add(topic_id)

        # Recalculate readiness for affected topics
        updated_scores = {}
        for tid in affected:
            new_score = self.calculate_readiness(tid)
            updated_scores[tid] = new_score

            # Check if intervention needed
            self.check_intervention_needed(tid, previous_scores.get(tid))

            # Update state
            if self.topics[tid].state != TopicState.MASTERED:
                self.topics[tid].state = new_score.state

        return updated_scores

    def _find_dependents(self, topic_id: str) -> Set[str]:
        """Find all topics that depend on the given topic."""
        dependents = set()

        for tid, topic in self.topics.items():
            if topic_id in topic.prerequisites:
                dependents.add(tid)
                # Recursive - find topics that depend on the dependent
                dependents.update(self._find_dependents(tid))

        return dependents

    def get_unlock_path(self, topic_id: str) -> List[str]:
        """
        Get ordered list of topics to master to unlock target.

        Uses topological sort on prerequisite graph.

        Args:
            topic_id: Target topic to unlock

        Returns:
            Ordered list of topic IDs to master
        """
        if topic_id not in self.topics:
            return []

        # Collect all prerequisites recursively
        to_visit = [topic_id]
        visited = set()
        path = []

        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue
            visited.add(current)

            topic = self.topics.get(current)
            if topic:
                # Add unmastered prerequisites first
                for prereq in topic.prerequisites:
                    if prereq not in visited:
                        to_visit.append(prereq)

                # Only include if not yet mastered
                if topic.mastery_probability < self.MASTERY_THRESHOLD:
                    path.append(current)

        # Reverse to get proper order (prerequisites first)
        path.reverse()
        return path

    def get_intervention_history(self) -> List[InterventionDecision]:
        """Get history of intervention decisions."""
        return self._intervention_history.copy()

    def clear_intervention_history(self) -> None:
        """Clear intervention history."""
        self._intervention_history.clear()


# Convenience function for simple use cases
def calculate_topic_readiness(
    topic_mastery: float,
    prerequisite_masteries: List[float],
    unlock_threshold: float = 0.8,
) -> float:
    """
    Calculate readiness for a topic given prerequisite masteries.

    Simple function for use without full graph structure.

    Args:
        topic_mastery: Current mastery of the topic
        prerequisite_masteries: List of prerequisite mastery values
        unlock_threshold: Threshold for unlocking

    Returns:
        Readiness probability (0-1)
    """
    if not prerequisite_masteries:
        return 1.0

    # Use same CPT values as the calculator
    CPT_READY_GIVEN_MASTERED = 0.95
    CPT_READY_GIVEN_NOT_MASTERED = 0.1

    contributions = []
    for p_mastered in prerequisite_masteries:
        contrib = (
            CPT_READY_GIVEN_MASTERED * p_mastered +
            CPT_READY_GIVEN_NOT_MASTERED * (1 - p_mastered)
        )
        contributions.append(contrib)

    # Geometric mean for multiple prerequisites
    if len(contributions) == 1:
        return contributions[0]

    product = 1.0
    for c in contributions:
        product *= c

    return product ** (1 / len(contributions))
