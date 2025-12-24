"""
Step definitions for Bayesian Topic Readiness & Knowledge Tracing.

Feature: libs/adaptive_engine/features/bayesian_readiness.feature
Work Order: WO-AE-006
"""

import pytest
from pytest_bdd import scenarios, given, when, then, parsers

# Import the module under test
from adaptive_engine.bayesian_readiness import (
    BayesianReadinessCalculator,
    TopicState,
    InterventionType,
    ReadinessScore,
    TopicNode,
)

# Load scenarios from feature file
scenarios("../../features/bayesian_readiness.feature")


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def calculator():
    """Create a fresh BayesianReadinessCalculator."""
    return BayesianReadinessCalculator()


@pytest.fixture
def knowledge_graph(calculator):
    """
    Context for the knowledge graph.

    Stores the calculator and state for scenario steps.
    """
    return {
        "calculator": calculator,
        "inference_results": {},
        "topic_a_id": None,
        "topic_b_id": None,
        "previous_scores": {},
        "interventions": [],
    }


# ============================================================================
# Background Steps
# ============================================================================


@given("the Knowledge Graph `G` contains Nodes `N` and Edges `E`")
def given_knowledge_graph(knowledge_graph):
    """Initialize the knowledge graph structure."""
    # Graph is initialized via the calculator fixture
    assert knowledge_graph["calculator"] is not None


@given("`P(Ready_B | Mastered_A)` represents the conditional probability")
def given_conditional_probability(knowledge_graph):
    """
    Confirm the conditional probability table is defined.

    P(Ready_B | Mastered_A = True) = 0.95 (high readiness if prereq mastered)
    P(Ready_B | Mastered_A = False) = 0.10 (low readiness if prereq not mastered)
    """
    calc = knowledge_graph["calculator"]
    assert calc.CPT_READY_GIVEN_MASTERED == 0.95
    assert calc.CPT_READY_GIVEN_NOT_MASTERED == 0.10


# ============================================================================
# Scenario: Propagating Readiness Scores
# ============================================================================


@given("Topic A is a prerequisite for Topic B")
def given_topic_a_prerequisite_for_b(knowledge_graph):
    """Set up the prerequisite relationship A -> B."""
    calc = knowledge_graph["calculator"]

    # Create Topic A (the prerequisite)
    topic_a = calc.add_topic(
        topic_id="topic_a",
        name="Topic A",
        prerequisites=set(),  # No prerequisites for A
        initial_mastery=0.0,
    )
    knowledge_graph["topic_a_id"] = "topic_a"

    # Create Topic B (depends on A)
    topic_b = calc.add_topic(
        topic_id="topic_b",
        name="Topic B",
        prerequisites={"topic_a"},  # A is prerequisite
        initial_mastery=0.0,
    )
    knowledge_graph["topic_b_id"] = "topic_b"

    # Verify relationship
    assert "topic_a" in calc.topics["topic_b"].prerequisites


@given(parsers.parse("the user has mastered Topic A (P(Mastered_A) = {mastery:f})"))
def given_user_mastered_topic_a(knowledge_graph, mastery: float):
    """Set Topic A's mastery probability."""
    calc = knowledge_graph["calculator"]
    topic_a_id = knowledge_graph["topic_a_id"]

    # Store previous score for comparison
    knowledge_graph["previous_scores"]["topic_b"] = calc.calculate_readiness("topic_b")

    # Update mastery
    calc.set_mastery(topic_a_id, mastery)

    # Verify mastery was set
    assert calc.topics[topic_a_id].mastery_probability == mastery


@when("the Bayesian inference runs")
def when_bayesian_inference_runs(knowledge_graph):
    """Execute Bayesian inference on the knowledge graph."""
    calc = knowledge_graph["calculator"]
    knowledge_graph["inference_results"] = calc.run_inference()


@then("P(Ready_B) should increase significantly")
def then_readiness_increases(knowledge_graph):
    """
    Verify P(Ready_B) increased after mastering prerequisite.

    With P(Mastered_A) = 0.95:
    P(Ready_B) = 0.95 * 0.95 + 0.10 * 0.05 = 0.9025 + 0.005 = 0.9075

    This is significantly higher than if A were not mastered:
    P(Ready_B | not mastered) = 0.95 * 0.0 + 0.10 * 1.0 = 0.10
    """
    results = knowledge_graph["inference_results"]
    topic_b_id = knowledge_graph["topic_b_id"]

    assert topic_b_id in results
    readiness = results[topic_b_id]

    # Should be significantly above the low-mastery baseline
    assert readiness.probability > 0.5, (
        f"Expected P(Ready_B) > 0.5, got {readiness.probability}"
    )

    # With 0.95 mastery, should be close to CPT_READY_GIVEN_MASTERED
    assert readiness.probability > 0.85, (
        f"Expected P(Ready_B) > 0.85 with high prerequisite mastery, "
        f"got {readiness.probability}"
    )


@then('if P(Ready_B) > 0.8, Topic B becomes "Unlocked"')
def then_topic_b_unlocked(knowledge_graph):
    """Verify Topic B is unlocked when readiness exceeds threshold."""
    calc = knowledge_graph["calculator"]
    results = knowledge_graph["inference_results"]
    topic_b_id = knowledge_graph["topic_b_id"]

    readiness = results[topic_b_id]

    # If readiness > 0.8, should be unlocked
    if readiness.probability > 0.8:
        assert readiness.is_unlocked, (
            f"Topic B should be unlocked with P(Ready) = {readiness.probability}"
        )
        assert readiness.state in (TopicState.UNLOCKED, TopicState.MASTERED), (
            f"Expected UNLOCKED or MASTERED, got {readiness.state}"
        )


# ============================================================================
# Scenario: Interleaved Practice Trigger
# ============================================================================


@given("Topic B is unlocked")
def given_topic_b_unlocked(knowledge_graph):
    """Set up Topic B as unlocked (prerequisite A is mastered)."""
    calc = knowledge_graph["calculator"]

    # Create Topic A (mastered)
    calc.add_topic(
        topic_id="topic_a",
        name="Topic A",
        prerequisites=set(),
        initial_mastery=0.95,  # Mastered
    )
    knowledge_graph["topic_a_id"] = "topic_a"

    # Create Topic B (unlocked because A is mastered)
    calc.add_topic(
        topic_id="topic_b",
        name="Topic B",
        prerequisites={"topic_a"},
        initial_mastery=0.3,  # Attempting but not mastered
    )
    knowledge_graph["topic_b_id"] = "topic_b"

    # Run initial inference to unlock B
    results = calc.run_inference()
    knowledge_graph["previous_scores"]["topic_b"] = results["topic_b"]

    # Verify B is unlocked
    assert results["topic_b"].is_unlocked, (
        "Topic B should start as unlocked for this scenario"
    )


@when("the user fails Topic B repeatedly (P(Mastered_B) drops)")
def when_user_fails_topic_b(knowledge_graph):
    """Simulate repeated failures on Topic B."""
    calc = knowledge_graph["calculator"]

    # Simulate mastery drop on Topic A (the prerequisite)
    # This models "failing Topic B reveals gaps in Topic A understanding"
    calc.set_mastery("topic_a", 0.40)  # Drop from 0.95 to 0.40

    # Also drop Topic B mastery
    calc.set_mastery("topic_b", 0.15)  # Drop from 0.30 to 0.15


@then("the system should re-evaluate P(Ready_B)")
def then_reevaluate_readiness(knowledge_graph):
    """Re-run inference after mastery changes."""
    calc = knowledge_graph["calculator"]
    knowledge_graph["inference_results"] = calc.run_inference()

    # Verify readiness was recalculated
    assert "topic_b" in knowledge_graph["inference_results"]

    # Readiness should have dropped
    new_readiness = knowledge_graph["inference_results"]["topic_b"]
    previous = knowledge_graph["previous_scores"]["topic_b"]

    assert new_readiness.probability < previous.probability, (
        f"Readiness should have dropped: {previous.probability} -> {new_readiness.probability}"
    )


@then('if it drops below threshold, trigger "Interleaved Practice" for Topic A')
def then_trigger_interleaved_practice(knowledge_graph):
    """Verify interleaved practice is triggered for the prerequisite."""
    calc = knowledge_graph["calculator"]
    results = knowledge_graph["inference_results"]
    previous = knowledge_graph["previous_scores"]["topic_b"]

    # Check for intervention
    intervention = calc.check_intervention_needed(
        "topic_b",
        previous_score=previous,
    )

    # If readiness dropped below threshold, intervention should be triggered
    if results["topic_b"].probability < calc.UNLOCK_THRESHOLD:
        assert intervention is not None, (
            "Intervention should be triggered when readiness drops below threshold"
        )
        assert intervention.intervention_type == InterventionType.INTERLEAVED_PRACTICE, (
            f"Expected INTERLEAVED_PRACTICE, got {intervention.intervention_type}"
        )
        assert "topic_a" in intervention.prerequisite_topics, (
            "Topic A should be in the list of prerequisites needing practice"
        )

        # Store intervention for verification
        knowledge_graph["interventions"].append(intervention)


# ============================================================================
# Additional Test Helpers
# ============================================================================


@pytest.fixture
def sample_graph():
    """Create a sample multi-topic graph for testing."""
    calc = BayesianReadinessCalculator()

    # Create a diamond dependency:
    #       A
    #      / \
    #     B   C
    #      \ /
    #       D

    calc.add_topic("A", "Fundamentals", set(), initial_mastery=0.90)
    calc.add_topic("B", "Algebra", {"A"}, initial_mastery=0.70)
    calc.add_topic("C", "Geometry", {"A"}, initial_mastery=0.60)
    calc.add_topic("D", "Calculus", {"B", "C"}, initial_mastery=0.0)

    return calc
