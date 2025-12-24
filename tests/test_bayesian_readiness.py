"""
Tests for Bayesian Topic Readiness & Knowledge Tracing.

This file runs the BDD scenarios AND provides additional unit test coverage.

Work Order: WO-AE-006
"""

import pytest
from pytest_bdd import scenarios

# Import step definitions - required for pytest-bdd to find them
from step_defs.bayesian_topic_readiness_steps import *  # noqa: F401, F403

# Load all scenarios from the feature file
scenarios("../features/bayesian_readiness.feature")


# ============================================================================
# Additional Unit Tests for Coverage
# ============================================================================

from adaptive_engine.bayesian_readiness import (
    BayesianReadinessCalculator,
    TopicState,
    InterventionType,
    ReadinessScore,
    TopicNode,
    calculate_topic_readiness,
)


class TestBayesianReadinessCalculator:
    """Unit tests for BayesianReadinessCalculator."""

    def test_add_topic(self):
        """Test adding topics to the calculator."""
        calc = BayesianReadinessCalculator()

        topic = calc.add_topic("t1", "Test Topic", set(), 0.5)

        assert topic.topic_id == "t1"
        assert topic.name == "Test Topic"
        assert topic.mastery_probability == 0.5
        assert "t1" in calc.topics

    def test_add_topic_with_prerequisites(self):
        """Test adding topic with prerequisites."""
        calc = BayesianReadinessCalculator()

        calc.add_topic("prereq", "Prerequisite", set(), 0.8)
        calc.add_topic("main", "Main Topic", {"prereq"}, 0.0)

        assert "prereq" in calc.topics["main"].prerequisites

    def test_set_mastery(self):
        """Test updating mastery values."""
        calc = BayesianReadinessCalculator()
        calc.add_topic("t1", "Test", set(), 0.0)

        calc.set_mastery("t1", 0.75)

        assert calc.topics["t1"].mastery_probability == 0.75

    def test_set_mastery_clamps_values(self):
        """Test that mastery is clamped to [0, 1]."""
        calc = BayesianReadinessCalculator()
        calc.add_topic("t1", "Test", set(), 0.5)

        calc.set_mastery("t1", 1.5)  # Over 1
        assert calc.topics["t1"].mastery_probability == 1.0

        calc.set_mastery("t1", -0.5)  # Under 0
        assert calc.topics["t1"].mastery_probability == 0.0

    def test_set_mastery_unknown_topic(self):
        """Test error on unknown topic."""
        calc = BayesianReadinessCalculator()

        with pytest.raises(ValueError, match="not found"):
            calc.set_mastery("unknown", 0.5)

    def test_calculate_readiness_no_prereqs(self):
        """Topic with no prerequisites should always be ready."""
        calc = BayesianReadinessCalculator()
        calc.add_topic("t1", "Test", set(), 0.0)

        score = calc.calculate_readiness("t1")

        assert score.probability == 1.0
        assert score.is_unlocked

    def test_calculate_readiness_mastered_prereq(self):
        """High readiness when prerequisite is mastered."""
        calc = BayesianReadinessCalculator()
        calc.add_topic("prereq", "Prereq", set(), 0.95)
        calc.add_topic("main", "Main", {"prereq"}, 0.0)

        score = calc.calculate_readiness("main")

        # P(Ready) = 0.95 * 0.95 + 0.10 * 0.05 ≈ 0.9075
        assert score.probability > 0.85
        assert score.is_unlocked

    def test_calculate_readiness_unmastered_prereq(self):
        """Low readiness when prerequisite is not mastered."""
        calc = BayesianReadinessCalculator()
        calc.add_topic("prereq", "Prereq", set(), 0.1)
        calc.add_topic("main", "Main", {"prereq"}, 0.0)

        score = calc.calculate_readiness("main")

        # P(Ready) = 0.95 * 0.1 + 0.10 * 0.9 = 0.095 + 0.09 = 0.185
        assert score.probability < 0.5
        assert not score.is_unlocked

    def test_calculate_readiness_multiple_prereqs(self):
        """Readiness with multiple prerequisites."""
        calc = BayesianReadinessCalculator()
        calc.add_topic("p1", "Prereq 1", set(), 0.90)
        calc.add_topic("p2", "Prereq 2", set(), 0.80)
        calc.add_topic("main", "Main", {"p1", "p2"}, 0.0)

        score = calc.calculate_readiness("main")

        # Both prerequisites are well-mastered, should be unlocked
        assert score.probability > 0.7
        assert len(score.contributing_prerequisites) == 2

    def test_calculate_readiness_mixed_prereqs(self):
        """Readiness with one weak prerequisite."""
        calc = BayesianReadinessCalculator()
        calc.add_topic("strong", "Strong Prereq", set(), 0.95)
        calc.add_topic("weak", "Weak Prereq", set(), 0.20)
        calc.add_topic("main", "Main", {"strong", "weak"}, 0.0)

        score = calc.calculate_readiness("main")

        # Weak prerequisite should drag down readiness
        assert score.probability < 0.8

    def test_run_inference(self):
        """Test running inference on entire graph."""
        calc = BayesianReadinessCalculator()
        calc.add_topic("a", "A", set(), 0.90)
        calc.add_topic("b", "B", {"a"}, 0.0)
        calc.add_topic("c", "C", {"b"}, 0.0)

        results = calc.run_inference()

        assert len(results) == 3
        assert results["a"].probability == 1.0  # No prereqs
        assert results["b"].probability > 0.8  # A is mastered
        # C depends on B which isn't mastered yet

    def test_check_intervention_needed(self):
        """Test intervention detection."""
        calc = BayesianReadinessCalculator()
        calc.add_topic("prereq", "Prereq", set(), 0.95)
        calc.add_topic("main", "Main", {"prereq"}, 0.3)

        # Get initial score (should be unlocked)
        initial = calc.calculate_readiness("main")
        assert initial.is_unlocked

        # Drop prerequisite mastery
        calc.set_mastery("prereq", 0.30)

        # Check for intervention
        intervention = calc.check_intervention_needed("main", initial)

        assert intervention is not None
        assert intervention.intervention_type == InterventionType.INTERLEAVED_PRACTICE
        assert "prereq" in intervention.prerequisite_topics

    def test_propagate_mastery_update(self):
        """Test mastery propagation through graph."""
        calc = BayesianReadinessCalculator()
        calc.add_topic("a", "A", set(), 0.50)
        calc.add_topic("b", "B", {"a"}, 0.0)
        calc.add_topic("c", "C", {"a"}, 0.0)

        # Initial state - B and C should have low readiness
        initial_b = calc.calculate_readiness("b")
        assert not initial_b.is_unlocked

        # Update A's mastery
        updated = calc.propagate_mastery_update("a", 0.95)

        # B and C should now have higher readiness
        assert "b" in updated
        assert "c" in updated
        assert updated["b"].is_unlocked
        assert updated["c"].is_unlocked

    def test_find_dependents(self):
        """Test finding dependent topics."""
        calc = BayesianReadinessCalculator()
        calc.add_topic("a", "A", set(), 0.0)
        calc.add_topic("b", "B", {"a"}, 0.0)
        calc.add_topic("c", "C", {"b"}, 0.0)  # Transitive dependency

        dependents = calc._find_dependents("a")

        assert "b" in dependents
        assert "c" in dependents  # Transitive

    def test_get_unlock_path(self):
        """Test getting path to unlock a topic."""
        calc = BayesianReadinessCalculator()
        calc.add_topic("a", "A", set(), 0.0)
        calc.add_topic("b", "B", {"a"}, 0.0)
        calc.add_topic("c", "C", {"b"}, 0.0)

        path = calc.get_unlock_path("c")

        # Should return ordered path: a -> b -> c
        assert path[0] == "a"  # Must learn A first
        assert "c" in path  # Target is in path

    def test_intervention_history(self):
        """Test intervention history tracking."""
        calc = BayesianReadinessCalculator()
        calc.add_topic("a", "A", set(), 0.95)
        calc.add_topic("b", "B", {"a"}, 0.0)

        initial = calc.calculate_readiness("b")
        calc.set_mastery("a", 0.20)
        calc.check_intervention_needed("b", initial)

        history = calc.get_intervention_history()
        assert len(history) >= 1

        calc.clear_intervention_history()
        assert len(calc.get_intervention_history()) == 0


class TestReadinessScore:
    """Tests for ReadinessScore dataclass."""

    def test_is_unlocked_above_threshold(self):
        """Score above threshold is unlocked."""
        score = ReadinessScore(
            topic_id="t1",
            probability=0.85,
            state=TopicState.UNLOCKED,
            threshold=0.8,
        )
        assert score.is_unlocked

    def test_is_unlocked_below_threshold(self):
        """Score below threshold is locked."""
        score = ReadinessScore(
            topic_id="t1",
            probability=0.75,
            state=TopicState.LOCKED,
            threshold=0.8,
        )
        assert not score.is_unlocked

    def test_to_dict(self):
        """Test serialization to dict."""
        score = ReadinessScore(
            topic_id="t1",
            probability=0.85,
            state=TopicState.UNLOCKED,
            contributing_prerequisites={"p1": 0.9},
        )
        d = score.to_dict()

        assert d["topic_id"] == "t1"
        assert d["probability"] == 0.85
        assert d["state"] == "unlocked"
        assert d["is_unlocked"] is True


class TestTopicNode:
    """Tests for TopicNode dataclass."""

    def test_to_dict(self):
        """Test serialization."""
        node = TopicNode(
            topic_id="t1",
            name="Test",
            prerequisites={"p1", "p2"},
            mastery_probability=0.75,
            state=TopicState.UNLOCKED,
        )
        d = node.to_dict()

        assert d["topic_id"] == "t1"
        assert len(d["prerequisites"]) == 2


class TestConvenienceFunction:
    """Tests for calculate_topic_readiness helper."""

    def test_no_prerequisites(self):
        """No prerequisites means always ready."""
        readiness = calculate_topic_readiness(0.5, [])
        assert readiness == 1.0

    def test_single_mastered_prerequisite(self):
        """Single mastered prerequisite gives high readiness."""
        readiness = calculate_topic_readiness(0.0, [0.95])
        assert readiness > 0.85

    def test_single_unmastered_prerequisite(self):
        """Single unmastered prerequisite gives low readiness."""
        readiness = calculate_topic_readiness(0.0, [0.1])
        assert readiness < 0.3

    def test_multiple_prerequisites(self):
        """Multiple prerequisites are combined."""
        readiness = calculate_topic_readiness(0.0, [0.9, 0.8, 0.7])
        assert 0.5 < readiness < 1.0
