"""
Unit tests for ZPD (Zone of Proximal Development) module.

Tests cover:
- ZPD boundary calculation
- Flow channel management
- Scaffold selection
- Boredom/anxiety detection
- ZPD expansion tracking
"""
import pytest
from datetime import datetime, timedelta

import sys
from pathlib import Path

# Add the adaptive_engine package to the path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


# =============================================================================
# Test ScaffoldType Enum
# =============================================================================


class TestScaffoldType:
    """Tests for ScaffoldType enum."""

    def test_scaffold_opacity_values(self):
        from adaptive_engine.zpd import ScaffoldType

        assert ScaffoldType.NO_SCAFFOLD.opacity == 0.0
        assert ScaffoldType.FIRST_STEP_ONLY.opacity == 0.25
        assert ScaffoldType.STRUCTURAL_HINT.opacity == 0.50
        assert ScaffoldType.FADED_PARSONS.opacity == 0.75
        assert ScaffoldType.FULL_WORKED_EXAMPLE.opacity == 1.0

    def test_scaffold_values(self):
        from adaptive_engine.zpd import ScaffoldType

        assert ScaffoldType.NO_SCAFFOLD.value == "no_scaffold"
        assert ScaffoldType.FULL_WORKED_EXAMPLE.value == "full_worked"


# =============================================================================
# Test ZPDPosition and FlowState Enums
# =============================================================================


class TestEnums:
    """Tests for ZPD-related enums."""

    def test_zpd_position_values(self):
        from adaptive_engine.zpd import ZPDPosition

        assert ZPDPosition.BELOW_ZPD.value == "below_zpd"
        assert ZPDPosition.OPTIMAL_ZPD.value == "optimal_zpd"
        assert ZPDPosition.ABOVE_ZPD.value == "above_zpd"

    def test_flow_state_values(self):
        from adaptive_engine.zpd import FlowState

        assert FlowState.BOREDOM.value == "boredom"
        assert FlowState.FLOW.value == "flow"
        assert FlowState.ANXIETY.value == "anxiety"

    def test_struggle_type_values(self):
        from adaptive_engine.zpd import StruggleType

        assert StruggleType.CONCEPTUAL_GAP.value == "conceptual_gap"
        assert StruggleType.PROCEDURAL_STUCK.value == "procedural_stuck"
        assert StruggleType.COGNITIVE_OVERLOAD.value == "cognitive_overload"


# =============================================================================
# Test NCDEFrictionVector
# =============================================================================


class TestNCDEFrictionVector:
    """Tests for NCDE friction vector."""

    def test_total_friction_calculation(self):
        from adaptive_engine.zpd import NCDEFrictionVector

        friction = NCDEFrictionVector(
            retrieval_friction=2.0,
            integration_friction=2.0,
            execution_friction=2.0,
            metacognitive_friction=2.0,
        )

        # Weighted average: 0.25*2 + 0.30*2 + 0.25*2 + 0.20*2 = 2.0
        assert friction.total_friction == 2.0

    def test_is_critical_true(self):
        from adaptive_engine.zpd import NCDEFrictionVector

        friction = NCDEFrictionVector(
            retrieval_friction=3.5,
            integration_friction=3.5,
            execution_friction=3.5,
            metacognitive_friction=3.5,
        )
        assert friction.is_critical is True

    def test_is_critical_false(self):
        from adaptive_engine.zpd import NCDEFrictionVector

        friction = NCDEFrictionVector(
            retrieval_friction=1.0,
            integration_friction=1.0,
            execution_friction=1.0,
            metacognitive_friction=1.0,
        )
        assert friction.is_critical is False

    def test_is_optimal_true(self):
        from adaptive_engine.zpd import NCDEFrictionVector

        friction = NCDEFrictionVector(
            retrieval_friction=1.5,
            integration_friction=1.5,
            execution_friction=1.5,
            metacognitive_friction=1.5,
        )
        assert friction.is_optimal is True

    def test_is_optimal_false_too_low(self):
        from adaptive_engine.zpd import NCDEFrictionVector

        friction = NCDEFrictionVector(
            retrieval_friction=0.2,
            integration_friction=0.2,
            execution_friction=0.2,
            metacognitive_friction=0.2,
        )
        assert friction.is_optimal is False


# =============================================================================
# Test ZPDState
# =============================================================================


class TestZPDState:
    """Tests for ZPDState dataclass."""

    def test_zpd_width_calculation(self):
        from adaptive_engine.zpd import ZPDState

        state = ZPDState(
            concept_id="test",
            learner_id="user1",
            current_independent=0.5,
            upper_scaffolded=0.8,
        )

        assert abs(state.zpd_width - 0.3) < 0.001

    def test_optimal_difficulty(self):
        from adaptive_engine.zpd import ZPDState

        state = ZPDState(
            concept_id="test",
            learner_id="user1",
            current_independent=0.4,
            upper_scaffolded=0.8,
        )

        assert abs(state.optimal_difficulty - 0.6) < 0.001  # (0.4 + 0.8) / 2

    def test_get_position_below_zpd(self):
        from adaptive_engine.zpd import ZPDState, ZPDPosition

        state = ZPDState(
            concept_id="test",
            learner_id="user1",
            lower_bound=0.3,
            current_independent=0.5,
        )

        position = state.get_position(0.1)
        assert position == ZPDPosition.BELOW_ZPD

    def test_get_position_optimal_zpd(self):
        from adaptive_engine.zpd import ZPDState, ZPDPosition

        state = ZPDState(
            concept_id="test",
            learner_id="user1",
            current_independent=0.4,
            upper_independent=0.6,
        )

        position = state.get_position(0.5)
        assert position == ZPDPosition.OPTIMAL_ZPD

    def test_get_position_above_zpd(self):
        from adaptive_engine.zpd import ZPDState, ZPDPosition

        state = ZPDState(
            concept_id="test",
            learner_id="user1",
            upper_scaffolded=0.7,
        )

        position = state.get_position(0.9)
        assert position == ZPDPosition.ABOVE_ZPD

    def test_to_dict(self):
        from adaptive_engine.zpd import ZPDState

        state = ZPDState(
            concept_id="test",
            learner_id="user1",
            theta=0.5,
        )

        d = state.to_dict()
        assert d["concept_id"] == "test"
        assert d["learner_id"] == "user1"
        assert d["theta"] == 0.5
        assert "zpd_width" in d


# =============================================================================
# Test ZPDCalculator
# =============================================================================


class TestZPDCalculator:
    """Tests for ZPDCalculator."""

    @pytest.fixture
    def sample_interactions(self):
        """Generate sample interactions."""
        interactions = []
        for i in range(50):
            difficulty = 0.3 + (i * 0.02)
            success_prob = max(0.1, 1.0 - (difficulty - 0.3) * 1.5)
            is_correct = (i % 10) < int(success_prob * 10)

            interactions.append({
                "difficulty": difficulty,
                "is_correct": is_correct,
                "had_scaffold": difficulty > 0.7 and is_correct,
            })
        return interactions

    def test_compute_boundaries_insufficient_data(self):
        from adaptive_engine.zpd import ZPDCalculator

        calculator = ZPDCalculator(min_interactions=30)

        # Only 10 interactions
        interactions = [{"difficulty": 0.5, "is_correct": True}] * 10

        state = calculator.compute_boundaries(
            interactions=interactions,
            concept_id="test",
            learner_id="user1",
        )

        # Should return default state
        assert state.interactions_count == 10

    def test_compute_boundaries_with_sufficient_data(self, sample_interactions):
        from adaptive_engine.zpd import ZPDCalculator

        calculator = ZPDCalculator()

        state = calculator.compute_boundaries(
            interactions=sample_interactions,
            concept_id="test",
            learner_id="user1",
        )

        # Boundaries should be monotonic
        assert state.lower_bound <= state.current_independent
        assert state.current_independent <= state.upper_independent
        assert state.upper_independent <= state.upper_scaffolded
        assert state.upper_scaffolded <= state.frustration_ceiling

    def test_estimate_theta_empty(self):
        from adaptive_engine.zpd import ZPDCalculator

        calculator = ZPDCalculator()
        theta = calculator._estimate_theta([])

        assert theta == 0.0

    def test_estimate_theta_with_data(self):
        from adaptive_engine.zpd import ZPDCalculator

        calculator = ZPDCalculator()

        # High ability: correct on hard items
        interactions = [
            {"difficulty": 0.8, "is_correct": True},
            {"difficulty": 0.9, "is_correct": True},
        ]

        theta = calculator._estimate_theta(interactions)
        assert theta > 0.5

    def test_update_with_interaction(self):
        from adaptive_engine.zpd import ZPDCalculator, ZPDState

        calculator = ZPDCalculator()

        initial_state = ZPDState(
            concept_id="test",
            learner_id="user1",
            theta=0.5,
            upper_independent=0.6,
            interactions_count=10,
        )

        # Success at harder difficulty
        interaction = {
            "difficulty": 0.7,
            "is_correct": True,
            "had_scaffold": False,
        }

        updated = calculator.update_with_interaction(initial_state, interaction)

        # Theta should increase
        assert updated.theta >= initial_state.theta
        assert updated.interactions_count == 11

    def test_bin_by_difficulty(self):
        from adaptive_engine.zpd import ZPDCalculator

        calculator = ZPDCalculator()

        interactions = [
            {"difficulty": 0.5, "is_correct": True},
            {"difficulty": 0.52, "is_correct": False},
            {"difficulty": 0.55, "is_correct": True},
        ]

        bins = calculator._bin_by_difficulty(interactions, bin_width=0.2)

        # Verify bins were created (floating point may cause 1-2 bins)
        assert len(bins) >= 1
        total_items = sum(b["total"] for b in bins.values())
        correct_items = sum(b["correct"] for b in bins.values())
        assert total_items == 3
        assert correct_items == 2


# =============================================================================
# Test FlowChannelManager
# =============================================================================


class TestFlowChannelManager:
    """Tests for FlowChannelManager."""

    def test_compute_flow_state_boredom(self):
        from adaptive_engine.zpd import FlowChannelManager, FlowState

        manager = FlowChannelManager()

        state = manager.compute_flow_state(
            skill_theta=1.0,
            item_difficulty=0.3,  # Much easier than skill
        )

        assert state.flow_state == FlowState.BOREDOM

    def test_compute_flow_state_flow(self):
        from adaptive_engine.zpd import FlowChannelManager, FlowState

        manager = FlowChannelManager()

        state = manager.compute_flow_state(
            skill_theta=1.0,
            item_difficulty=1.1,  # Slightly challenging
        )

        assert state.flow_state == FlowState.FLOW

    def test_compute_flow_state_anxiety(self):
        from adaptive_engine.zpd import FlowChannelManager, FlowState

        manager = FlowChannelManager()

        state = manager.compute_flow_state(
            skill_theta=1.0,
            item_difficulty=1.8,  # Much harder than skill
        )

        assert state.flow_state == FlowState.ANXIETY

    def test_in_flow_method(self):
        from adaptive_engine.zpd import FlowChannelManager

        manager = FlowChannelManager()

        state = manager.compute_flow_state(
            skill_theta=1.0,
            item_difficulty=1.1,
        )

        assert state.in_flow() is True

    def test_get_optimal_difficulty_practice(self):
        from adaptive_engine.zpd import FlowChannelManager

        manager = FlowChannelManager()

        low, high = manager.get_optimal_difficulty(skill_theta=1.0, context="practice")

        # Should be slightly above skill
        assert low > 1.0
        assert high > low

    def test_get_optimal_difficulty_assessment(self):
        from adaptive_engine.zpd import FlowChannelManager

        manager = FlowChannelManager()

        low, high = manager.get_optimal_difficulty(skill_theta=1.0, context="assessment")

        # Should be centered on skill
        assert low < 1.0
        assert high > 1.0

    def test_detect_boredom_positive(self):
        from adaptive_engine.zpd import FlowChannelManager

        manager = FlowChannelManager()

        metrics = {
            "avg_response_ms": 1500,  # Very fast
            "accuracy": 0.98,  # Near perfect
            "friction": 0.3,  # Very low
        }

        is_bored, indicators = manager.detect_boredom(metrics)
        assert is_bored is True
        assert len(indicators) > 0

    def test_detect_boredom_negative(self):
        from adaptive_engine.zpd import FlowChannelManager

        manager = FlowChannelManager()

        metrics = {
            "avg_response_ms": 5000,  # Normal
            "accuracy": 0.75,  # Normal
            "friction": 1.5,  # Optimal
        }

        is_bored, indicators = manager.detect_boredom(metrics)
        assert is_bored is False

    def test_detect_anxiety_positive(self):
        from adaptive_engine.zpd import FlowChannelManager

        manager = FlowChannelManager()

        metrics = {
            "avg_response_ms": 15000,  # Very slow
            "error_rate": 0.6,  # High
            "friction": 3.0,  # High
        }

        is_anxious, indicators = manager.detect_anxiety(metrics)
        assert is_anxious is True
        assert len(indicators) > 0

    def test_detect_anxiety_negative(self):
        from adaptive_engine.zpd import FlowChannelManager

        manager = FlowChannelManager()

        metrics = {
            "avg_response_ms": 5000,  # Normal
            "error_rate": 0.2,  # Low
            "friction": 1.5,  # Optimal
        }

        is_anxious, indicators = manager.detect_anxiety(metrics)
        assert is_anxious is False


# =============================================================================
# Test ScaffoldSelector
# =============================================================================


class TestScaffoldSelector:
    """Tests for ScaffoldSelector."""

    def test_select_scaffold_no_friction(self):
        from adaptive_engine.zpd import ScaffoldSelector, NCDEFrictionVector, ScaffoldType

        selector = ScaffoldSelector()

        friction = NCDEFrictionVector(
            retrieval_friction=0.5,
            integration_friction=0.5,
            execution_friction=0.5,
            metacognitive_friction=0.5,
        )

        decision = selector.select_scaffold(friction)
        assert decision.scaffold_type == ScaffoldType.NO_SCAFFOLD

    def test_select_scaffold_high_friction(self):
        from adaptive_engine.zpd import ScaffoldSelector, NCDEFrictionVector, ScaffoldType

        selector = ScaffoldSelector()

        friction = NCDEFrictionVector(
            retrieval_friction=2.8,
            integration_friction=2.8,
            execution_friction=2.8,
            metacognitive_friction=2.8,
        )

        decision = selector.select_scaffold(friction)
        assert decision.scaffold_type in [
            ScaffoldType.FADED_PARSONS,
            ScaffoldType.FULL_WORKED_EXAMPLE,
        ]

    def test_select_scaffold_with_struggle_type(self):
        from adaptive_engine.zpd import (
            ScaffoldSelector, NCDEFrictionVector, ScaffoldType, StruggleType
        )

        selector = ScaffoldSelector()

        friction = NCDEFrictionVector(
            retrieval_friction=2.5,
            integration_friction=2.5,
            execution_friction=2.5,
            metacognitive_friction=2.5,
        )

        decision = selector.select_scaffold(
            friction=friction,
            struggle_type=StruggleType.CONCEPTUAL_GAP,
        )

        # Conceptual gap should trigger full worked example
        assert decision.scaffold_type == ScaffoldType.FULL_WORKED_EXAMPLE

    def test_scaffold_auto_fade_time(self):
        from adaptive_engine.zpd import ScaffoldSelector, NCDEFrictionVector

        selector = ScaffoldSelector()

        friction = NCDEFrictionVector(
            retrieval_friction=2.8,
            integration_friction=2.8,
            execution_friction=2.8,
            metacognitive_friction=2.8,
        )

        decision = selector.select_scaffold(friction)

        # Higher opacity = longer fade time
        assert decision.auto_fade_seconds > 0

    def test_should_fade_scaffold_insufficient_data(self):
        from adaptive_engine.zpd import ScaffoldSelector

        selector = ScaffoldSelector()

        # Only 2 interactions
        performance = [
            {"is_correct": True, "response_time_ms": 3000},
            {"is_correct": True, "response_time_ms": 3000},
        ]

        should_fade, new_opacity = selector.should_fade_scaffold(0.75, performance)
        assert should_fade is False
        assert new_opacity == 0.75

    def test_should_fade_scaffold_positive(self):
        from adaptive_engine.zpd import ScaffoldSelector

        selector = ScaffoldSelector()

        # Good performance
        performance = [
            {"is_correct": True, "response_time_ms": 3000},
            {"is_correct": True, "response_time_ms": 2500},
            {"is_correct": True, "response_time_ms": 2000},
            {"is_correct": True, "response_time_ms": 2500},
            {"is_correct": True, "response_time_ms": 3000},
        ]

        should_fade, new_opacity = selector.should_fade_scaffold(0.75, performance)
        assert should_fade is True
        assert new_opacity < 0.75

    def test_compute_fade_sequence(self):
        from adaptive_engine.zpd import ScaffoldSelector

        selector = ScaffoldSelector()

        sequence = selector.compute_fade_sequence(starting_opacity=1.0, num_interactions=4)

        assert len(sequence) == 5  # 4 + 1 (initial)
        assert sequence[0] == 1.0
        assert sequence[-1] == 0.0

        # Should be monotonically decreasing
        for i in range(1, len(sequence)):
            assert sequence[i] <= sequence[i - 1]


# =============================================================================
# Test ZPDEngine
# =============================================================================


class TestZPDEngine:
    """Tests for ZPDEngine main orchestrator."""

    def test_get_zpd_state_default(self):
        from adaptive_engine.zpd import ZPDEngine

        engine = ZPDEngine()

        state = engine.get_zpd_state(
            concept_id="test",
            learner_id="user1",
        )

        assert state.concept_id == "test"
        assert state.learner_id == "user1"

    def test_update_zpd(self):
        from adaptive_engine.zpd import ZPDEngine

        engine = ZPDEngine()

        # First, get a state
        initial = engine.get_zpd_state("test", "user1")

        # Update with success
        interaction = {
            "difficulty": 0.7,
            "is_correct": True,
            "had_scaffold": False,
        }

        updated = engine.update_zpd("test", "user1", interaction)

        assert updated.interactions_count > initial.interactions_count

    def test_get_flow_state(self):
        from adaptive_engine.zpd import ZPDEngine, FlowState

        engine = ZPDEngine()

        state = engine.get_flow_state(
            skill_theta=1.0,
            item_difficulty=1.1,
        )

        assert state.flow_state == FlowState.FLOW

    def test_select_scaffold(self):
        from adaptive_engine.zpd import ZPDEngine, NCDEFrictionVector

        engine = ZPDEngine()

        friction = NCDEFrictionVector(
            retrieval_friction=2.5,
            integration_friction=2.5,
            execution_friction=2.5,
            metacognitive_friction=2.5,
        )

        decision = engine.select_scaffold(friction)

        assert decision.opacity > 0

    def test_recommend_item_difficulty_practice(self):
        from adaptive_engine.zpd import ZPDEngine, ZPDState

        engine = ZPDEngine()

        zpd_state = ZPDState(
            concept_id="test",
            learner_id="user1",
            current_independent=0.5,
            upper_scaffolded=0.8,
        )

        low, high, reason = engine.recommend_item_difficulty(zpd_state, context="practice")

        assert low >= zpd_state.current_independent
        assert high <= zpd_state.upper_scaffolded
        assert "Practice" in reason

    def test_recommend_item_difficulty_assessment(self):
        from adaptive_engine.zpd import ZPDEngine, ZPDState

        engine = ZPDEngine()

        zpd_state = ZPDState(
            concept_id="test",
            learner_id="user1",
            theta=0.7,
        )

        low, high, reason = engine.recommend_item_difficulty(zpd_state, context="assessment")

        assert "Assessment" in reason

    def test_check_zpd_expansion_detected(self):
        from adaptive_engine.zpd import ZPDEngine, ZPDState

        engine = ZPDEngine()

        old_state = ZPDState(
            concept_id="test",
            learner_id="user1",
            lower_bound=0.3,
            upper_scaffolded=0.7,
        )

        new_state = ZPDState(
            concept_id="test",
            learner_id="user1",
            lower_bound=0.4,  # Shifted up
            upper_scaffolded=0.8,  # Shifted up
        )

        record = engine.check_zpd_expansion(old_state, new_state)

        assert record is not None
        assert abs(record.upper_shift - 0.1) < 0.001
        assert abs(record.lower_shift - 0.1) < 0.001

    def test_check_zpd_expansion_not_detected(self):
        from adaptive_engine.zpd import ZPDEngine, ZPDState

        engine = ZPDEngine()

        old_state = ZPDState(
            concept_id="test",
            learner_id="user1",
            lower_bound=0.3,
            upper_scaffolded=0.7,
        )

        new_state = ZPDState(
            concept_id="test",
            learner_id="user1",
            lower_bound=0.31,  # Negligible change
            upper_scaffolded=0.71,
        )

        record = engine.check_zpd_expansion(old_state, new_state)

        assert record is None

    def test_handle_success_streak_expansion(self):
        from adaptive_engine.zpd import ZPDEngine, ZPDState

        engine = ZPDEngine()

        zpd_state = ZPDState(
            concept_id="test",
            learner_id="user1",
            theta=0.5,
            current_independent=0.5,
            upper_scaffolded=0.7,  # This gives optimal_difficulty = 0.6
        )

        result = engine.handle_success_streak(
            zpd_state=zpd_state,
            streak_length=4,
            avg_latency_ms=2000,  # Fast
        )

        assert result["action"] == "ZPD_EXPANSION"
        assert result["skip_levels"] >= 1

    def test_handle_success_streak_continue(self):
        from adaptive_engine.zpd import ZPDEngine, ZPDState

        engine = ZPDEngine()

        zpd_state = ZPDState(
            concept_id="test",
            learner_id="user1",
        )

        result = engine.handle_success_streak(
            zpd_state=zpd_state,
            streak_length=2,  # Short streak
            avg_latency_ms=6000,  # Slow
        )

        assert result["action"] == "CONTINUE"

    def test_handle_struggle_signal_inject(self):
        from adaptive_engine.zpd import ZPDEngine, ZPDState, NCDEFrictionVector

        engine = ZPDEngine()

        zpd_state = ZPDState(
            concept_id="test",
            learner_id="user1",
        )

        friction = NCDEFrictionVector(
            retrieval_friction=3.5,
            integration_friction=3.5,
            execution_friction=3.5,
            metacognitive_friction=3.5,
        )

        result = engine.handle_struggle_signal(friction, zpd_state)

        assert result["action"] == "INJECT_SCAFFOLD"
        assert result["scaffold"] is not None
        assert result["temporary_scaffolding"] is True

    def test_handle_struggle_signal_continue(self):
        from adaptive_engine.zpd import ZPDEngine, ZPDState, NCDEFrictionVector

        engine = ZPDEngine()

        zpd_state = ZPDState(
            concept_id="test",
            learner_id="user1",
        )

        friction = NCDEFrictionVector(
            retrieval_friction=1.0,
            integration_friction=1.0,
            execution_friction=1.0,
            metacognitive_friction=1.0,
        )

        result = engine.handle_struggle_signal(friction, zpd_state)

        assert result["action"] == "CONTINUE"


# =============================================================================
# Test Utility Functions
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_compute_friction_from_metrics_low(self):
        from adaptive_engine.zpd import compute_friction_from_metrics

        friction = compute_friction_from_metrics(
            response_time_ms=2000,
            accuracy_rate=0.9,
            error_streak=0,
            session_duration_minutes=10,
        )

        assert friction.total_friction < 1.5

    def test_compute_friction_from_metrics_high(self):
        from adaptive_engine.zpd import compute_friction_from_metrics

        friction = compute_friction_from_metrics(
            response_time_ms=15000,  # Very slow
            accuracy_rate=0.3,  # Low accuracy
            error_streak=5,  # Many errors
            session_duration_minutes=60,  # Long session
        )

        assert friction.total_friction > 2.0

    def test_compute_friction_from_metrics_bounds(self):
        from adaptive_engine.zpd import compute_friction_from_metrics

        # Test with extreme values
        friction = compute_friction_from_metrics(
            response_time_ms=100000,
            accuracy_rate=0.0,
            error_streak=100,
            session_duration_minutes=200,
        )

        # Should be capped at 3.0 per component
        assert friction.total_friction <= 3.0


# =============================================================================
# Test ZPDGrowthRecord
# =============================================================================


class TestZPDGrowthRecord:
    """Tests for ZPD growth record."""

    def test_growth_record_creation(self):
        from adaptive_engine.zpd import ZPDGrowthRecord

        record = ZPDGrowthRecord(
            concept_id="test",
            learner_id="user1",
            timestamp=datetime.now(),
            lower_bound=0.4,
            upper_bound=0.8,
            zpd_width=0.4,
            lower_shift=0.1,
            upper_shift=0.1,
        )

        assert record.concept_id == "test"
        assert record.zpd_width == 0.4
        assert record.lower_shift == 0.1


# =============================================================================
# Test FlowChannelState
# =============================================================================


class TestFlowChannelState:
    """Tests for FlowChannelState."""

    def test_in_flow_true(self):
        from adaptive_engine.zpd import FlowChannelState, FlowState

        state = FlowChannelState(
            skill_level=1.0,
            current_challenge=1.1,
            flow_state=FlowState.FLOW,
        )

        assert state.in_flow() is True

    def test_in_flow_false(self):
        from adaptive_engine.zpd import FlowChannelState, FlowState

        state = FlowChannelState(
            skill_level=1.0,
            current_challenge=0.3,
            flow_state=FlowState.BOREDOM,
        )

        assert state.in_flow() is False


# =============================================================================
# Test ScaffoldDecision
# =============================================================================


class TestScaffoldDecision:
    """Tests for ScaffoldDecision."""

    def test_scaffold_decision_creation(self):
        from adaptive_engine.zpd import ScaffoldDecision, ScaffoldType, StruggleType

        decision = ScaffoldDecision(
            scaffold_type=ScaffoldType.FADED_PARSONS,
            opacity=0.75,
            reason="High friction detected",
            struggle_type=StruggleType.PROCEDURAL_STUCK,
            friction_level=2.8,
            auto_fade_seconds=90,
        )

        assert decision.scaffold_type == ScaffoldType.FADED_PARSONS
        assert decision.opacity == 0.75
        assert decision.auto_fade_seconds == 90
