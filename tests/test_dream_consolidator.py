"""
Tests for Automated Consolidation and Dream Agent Processing.

This file runs the BDD scenarios AND provides additional unit test coverage.

Work Order: WO-AE-009
"""

import pytest
from pytest_bdd import scenarios
from datetime import datetime, timedelta
import math

# Import step definitions - required for pytest-bdd to find them
from step_defs.consolidation_dream_agents_steps import *  # noqa: F401, F403

# Load all scenarios from the feature file
scenarios("../features/consolidation_dream_agents.feature")


# ============================================================================
# Additional Unit Tests for Coverage
# ============================================================================

from adaptive_engine.dream_consolidator import (
    DreamConsolidator,
    ConceptState,
    WeakNodeScanner,
    InterferenceAnalyzer,
    DecayProjector,
    ConsolidationPathOptimizer,
    WeaknessType,
    AtomType,
    ReviewUrgency,
    InterferenceRisk,
    FragilityScore,
    InterferencePair,
    ConsolidationAtom,
    DecayProjection,
    MorningBootInfo,
    SimulationResults,
    SchedulingAnalysis,
    StabilityDistribution,
    ConsolidationEffectiveness,
    LearnerPattern,
    create_dream_consolidator,
)


class TestConceptState:
    """Unit tests for ConceptState."""

    def test_days_since_review_with_review(self):
        """Test days since review calculation."""
        concept = ConceptState(
            "c1", "Concept 1",
            last_review=datetime.now() - timedelta(days=5),
        )
        assert 4.9 < concept.days_since_review < 5.1

    def test_days_since_review_no_review(self):
        """Test days since review when never reviewed."""
        concept = ConceptState("c1", "Concept 1")
        assert concept.days_since_review == float("inf")

    def test_current_retention(self):
        """Test retention calculation."""
        concept = ConceptState(
            "c1", "Concept 1",
            stability=7,
            last_review=datetime.now() - timedelta(days=7),
        )
        # R(t) = e^(-t/S) = e^(-7/7) = e^(-1) ≈ 0.368
        assert 0.35 < concept.current_retention() < 0.40

    def test_current_retention_no_review(self):
        """Test retention with no review."""
        concept = ConceptState("c1", "Concept 1")
        assert concept.current_retention() == 0.0

    def test_project_retention(self):
        """Test forward projection of retention."""
        concept = ConceptState(
            "c1", "Concept 1",
            stability=10,
            last_review=datetime.now(),
        )
        # Project 48 hours (2 days) forward
        retention = concept.project_retention(48)
        # R(2) = e^(-2/10) = e^(-0.2) ≈ 0.818
        assert 0.80 < retention < 0.85

    def test_to_dict(self):
        """Test serialization."""
        concept = ConceptState(
            "c1", "Concept 1",
            mastery=0.75,
            stability=10,
            last_review=datetime.now() - timedelta(days=2),
        )
        d = concept.to_dict()

        assert d["concept_id"] == "c1"
        assert d["mastery"] == 0.75
        assert "current_retention" in d


class TestWeakNodeScanner:
    """Unit tests for WeakNodeScanner."""

    def test_simulate_retrieval(self):
        """Test retrieval simulation."""
        scanner = WeakNodeScanner()
        concept = ConceptState(
            "c1", "Concept 1",
            mastery=0.8,
            stability=10,
            last_review=datetime.now() - timedelta(days=1),
        )

        prob = scanner.simulate_retrieval(concept)
        assert 0 <= prob <= 1

    def test_simulate_retrieval_high_mastery(self):
        """Test retrieval with high mastery."""
        scanner = WeakNodeScanner()
        concept = ConceptState(
            "c1", "Concept 1",
            mastery=0.95,
            stability=30,
            last_review=datetime.now() - timedelta(hours=1),
        )

        prob = scanner.simulate_retrieval(concept)
        assert prob > 0.5  # Should be relatively high

    def test_simulate_retrieval_low_mastery(self):
        """Test retrieval with low mastery."""
        scanner = WeakNodeScanner()
        concept = ConceptState(
            "c1", "Concept 1",
            mastery=0.1,
            stability=1,
            last_review=datetime.now() - timedelta(days=10),
        )

        prob = scanner.simulate_retrieval(concept)
        assert prob < 0.5  # Should be relatively low

    def test_calculate_fragility(self):
        """Test fragility calculation."""
        scanner = WeakNodeScanner()
        concept = ConceptState(
            "c1", "Concept 1",
            mastery=0.3,
            stability=2,
            last_review=datetime.now() - timedelta(days=5),
        )

        score = scanner.calculate_fragility(concept)

        assert 0 <= score.fragility <= 1
        assert score.weakness_type is not None
        assert score.urgency is not None

    def test_classify_weakness_overconfidence(self):
        """Test overconfidence detection."""
        scanner = WeakNodeScanner()
        concept = ConceptState(
            "c1", "Concept 1",
            mastery=0.5,
            stability=5,
            confidence=0.9,
            last_accuracy=0.3,
            last_review=datetime.now() - timedelta(days=1),
        )

        score = scanner.calculate_fragility(concept)
        assert score.weakness_type == WeaknessType.OVERCONFIDENCE

    def test_scan_all(self):
        """Test scanning all concepts."""
        scanner = WeakNodeScanner()
        concepts = [
            ConceptState("c1", "C1", mastery=0.9, stability=30,
                        last_review=datetime.now() - timedelta(days=1)),
            ConceptState("c2", "C2", mastery=0.2, stability=1,
                        last_review=datetime.now() - timedelta(days=10)),
            ConceptState("c3", "C3", mastery=0.5, stability=5,
                        last_review=datetime.now() - timedelta(days=3)),
        ]

        results = scanner.scan_all(concepts, threshold=0.3)

        # Should find some fragile concepts
        assert len(results) >= 1
        # Should be sorted by fragility
        if len(results) >= 2:
            assert results[0].fragility >= results[1].fragility


class TestInterferenceAnalyzer:
    """Unit tests for InterferenceAnalyzer."""

    def test_compute_similarity_with_embeddings(self):
        """Test similarity with embeddings."""
        analyzer = InterferenceAnalyzer()

        a = ConceptState("a", "A", embedding=[1, 0, 0])
        b = ConceptState("b", "B", embedding=[0.9, 0.1, 0])

        similarity = analyzer.compute_similarity(a, b)
        assert 0.9 < similarity < 1.0

    def test_compute_similarity_orthogonal(self):
        """Test similarity of orthogonal vectors."""
        analyzer = InterferenceAnalyzer()

        a = ConceptState("a", "A", embedding=[1, 0, 0])
        b = ConceptState("b", "B", embedding=[0, 1, 0])

        similarity = analyzer.compute_similarity(a, b)
        assert similarity < 0.1

    def test_compute_similarity_no_embeddings(self):
        """Test similarity without embeddings (name-based)."""
        analyzer = InterferenceAnalyzer()

        a = ConceptState("binary_search", "Binary Search")
        b = ConceptState("binary_tree", "Binary Tree")

        similarity = analyzer.compute_similarity(a, b)
        assert 0 < similarity < 1  # Some overlap in words

    def test_classify_risk(self):
        """Test risk classification."""
        analyzer = InterferenceAnalyzer()

        assert analyzer.classify_risk(0.95) == InterferenceRisk.HIGH
        assert analyzer.classify_risk(0.80) == InterferenceRisk.MODERATE
        assert analyzer.classify_risk(0.50) == InterferenceRisk.LOW

    def test_find_interference_pairs(self):
        """Test finding interference pairs."""
        analyzer = InterferenceAnalyzer()

        concepts = [
            ConceptState("a", "A", embedding=[1, 0]),
            ConceptState("b", "B", embedding=[0.95, 0.1]),  # Similar to A
            ConceptState("c", "C", embedding=[0, 1]),  # Different
        ]

        pairs = analyzer.find_interference_pairs(concepts, min_similarity=0.7)

        # Should find A-B pair
        assert len(pairs) >= 1
        assert pairs[0].similarity > 0.7

    def test_record_confusion(self):
        """Test recording confusion events."""
        analyzer = InterferenceAnalyzer()

        analyzer.record_confusion("a", "b")
        analyzer.record_confusion("a", "b")

        pair_key = ("a", "b")
        assert analyzer.pair_history.get(pair_key, 0) == 2


class TestDecayProjector:
    """Unit tests for DecayProjector."""

    def test_project(self):
        """Test decay projection."""
        projector = DecayProjector()
        concept = ConceptState(
            "c1", "C1",
            stability=10,
            last_review=datetime.now() - timedelta(days=2),
        )

        projection = projector.project(concept, hours_ahead=48)

        assert projection.retention_now > projection.retention_projected
        assert projection.urgency is not None

    def test_project_high_urgency(self):
        """Test projection with high urgency."""
        projector = DecayProjector()
        concept = ConceptState(
            "c1", "C1",
            stability=2,
            last_review=datetime.now() - timedelta(days=5),
        )

        projection = projector.project(concept, hours_ahead=48)

        assert projection.urgency in [ReviewUrgency.CRITICAL, ReviewUrgency.HIGH]

    def test_calculate_review_deadline(self):
        """Test review deadline calculation."""
        projector = DecayProjector()
        concept = ConceptState(
            "c1", "C1",
            stability=10,
            last_review=datetime.now(),  # Just reviewed
        )

        projection = projector.project(concept, hours_ahead=24)

        # Should have time before review needed
        if projection.review_recommended_within_hours:
            assert projection.review_recommended_within_hours > 0

    def test_project_batch(self):
        """Test batch projection."""
        projector = DecayProjector()
        concepts = [
            ConceptState("c1", "C1", stability=2,
                        last_review=datetime.now() - timedelta(days=5)),
            ConceptState("c2", "C2", stability=20,
                        last_review=datetime.now() - timedelta(days=1)),
        ]

        projections = projector.project_batch(concepts, hours_ahead=48)

        assert len(projections) == 2
        # Should be sorted by urgency
        urgency_order = {
            ReviewUrgency.CRITICAL: 0, ReviewUrgency.HIGH: 1,
            ReviewUrgency.MODERATE: 2, ReviewUrgency.LOW: 3,
            ReviewUrgency.MAINTENANCE: 4,
        }
        if len(projections) >= 2:
            assert urgency_order[projections[0].urgency] <= urgency_order[projections[1].urgency]


class TestConsolidationPathOptimizer:
    """Unit tests for ConsolidationPathOptimizer."""

    def test_generate_path(self):
        """Test path generation."""
        optimizer = ConsolidationPathOptimizer()

        scores = [
            FragilityScore("c1", 0.8, 0.2, WeaknessType.RECENT_DECAY, ReviewUrgency.HIGH),
            FragilityScore("c2", 0.7, 0.3, WeaknessType.HIGH_INTERFERENCE, ReviewUrgency.MODERATE),
            FragilityScore("c3", 0.6, 0.4, WeaknessType.OVERCONFIDENCE, ReviewUrgency.MODERATE),
        ]
        pairs = [
            InterferencePair("c2", "cx", 0.9, InterferenceRisk.HIGH),
        ]

        atoms = optimizer.generate_path(scores, pairs, target_minutes=15)

        assert len(atoms) >= 1
        # Should be ordered by priority
        for i, atom in enumerate(atoms):
            assert atom.priority == i + 1

    def test_generate_path_respects_time_limit(self):
        """Test that path respects time limit."""
        optimizer = ConsolidationPathOptimizer()

        scores = [
            FragilityScore(f"c{i}", 0.8, 0.2, WeaknessType.RECENT_DECAY, ReviewUrgency.HIGH)
            for i in range(20)
        ]

        atoms = optimizer.generate_path(scores, [], target_minutes=10)

        total_time = sum(a.estimated_time_minutes for a in atoms)
        assert total_time <= 15  # Allow some buffer

    def test_select_atom_type(self):
        """Test atom type selection."""
        optimizer = ConsolidationPathOptimizer()

        score = FragilityScore("c1", 0.8, 0.2, WeaknessType.RECENT_DECAY, ReviewUrgency.HIGH)
        atom_type = optimizer._select_atom_type(score)
        assert atom_type == AtomType.RETRIEVAL_PRACTICE

        score2 = FragilityScore("c2", 0.7, 0.3, WeaknessType.OVERCONFIDENCE, ReviewUrgency.MODERATE)
        atom_type2 = optimizer._select_atom_type(score2)
        assert atom_type2 == AtomType.PREDICTION_CHALLENGE


class TestDreamConsolidator:
    """Unit tests for DreamConsolidator."""

    def test_create_consolidator(self):
        """Test consolidator creation."""
        consolidator = DreamConsolidator()

        assert consolidator.weak_node_scanner is not None
        assert consolidator.interference_analyzer is not None
        assert consolidator.decay_projector is not None
        assert consolidator.path_optimizer is not None

    def test_add_concept(self):
        """Test adding concepts."""
        consolidator = DreamConsolidator()
        concept = ConceptState("c1", "Concept 1")

        consolidator.add_concept(concept)

        assert "c1" in consolidator.concepts

    def test_set_concept_mastery(self):
        """Test updating mastery."""
        consolidator = DreamConsolidator()
        consolidator.add_concept(ConceptState("c1", "C1", mastery=0.5))

        consolidator.set_concept_mastery("c1", 0.8)

        assert consolidator.concepts["c1"].mastery == 0.8

    def test_set_concept_mastery_clamps(self):
        """Test mastery clamping."""
        consolidator = DreamConsolidator()
        consolidator.add_concept(ConceptState("c1", "C1"))

        consolidator.set_concept_mastery("c1", 1.5)
        assert consolidator.concepts["c1"].mastery == 1.0

        consolidator.set_concept_mastery("c1", -0.5)
        assert consolidator.concepts["c1"].mastery == 0.0

    def test_record_review(self):
        """Test recording a review."""
        consolidator = DreamConsolidator()
        consolidator.add_concept(ConceptState("c1", "C1", stability=5))

        consolidator.record_review("c1", accuracy=0.9, confidence=0.8)

        concept = consolidator.concepts["c1"]
        assert concept.last_review is not None
        assert concept.last_accuracy == 0.9
        assert concept.confidence == 0.8
        assert concept.stability > 5  # Increased due to good accuracy

    def test_record_review_failure(self):
        """Test recording a failed review."""
        consolidator = DreamConsolidator()
        consolidator.add_concept(ConceptState("c1", "C1", stability=10))

        consolidator.record_review("c1", accuracy=0.3, confidence=0.7)

        concept = consolidator.concepts["c1"]
        assert concept.stability < 10  # Decreased due to poor accuracy

    def test_is_idle(self):
        """Test idle detection."""
        consolidator = DreamConsolidator()

        # No activity recorded
        assert consolidator.is_idle(6)

        # Recent activity
        consolidator.learner_pattern.last_active = datetime.now() - timedelta(hours=2)
        assert not consolidator.is_idle(6)

        # Old activity
        consolidator.learner_pattern.last_active = datetime.now() - timedelta(hours=10)
        assert consolidator.is_idle(6)

    def test_run_consolidation(self):
        """Test full consolidation run."""
        consolidator = DreamConsolidator()

        # Add concepts
        for i in range(5):
            consolidator.add_concept(ConceptState(
                f"c{i}", f"Concept {i}",
                mastery=0.3 + i * 0.1,
                stability=2 + i,
                last_review=datetime.now() - timedelta(days=1 + i % 3),
            ))

        boot_info = consolidator.run_consolidation(target_session_minutes=15)

        assert boot_info is not None
        assert boot_info.consolidation_summary
        assert boot_info.priority_items_count >= 0
        assert len(boot_info.priority_atoms) >= 0

    def test_run_dream_simulation(self):
        """Test dream simulation."""
        consolidator = DreamConsolidator()

        for i in range(10):
            consolidator.add_concept(ConceptState(
                f"c{i}", f"C{i}",
                mastery=0.5, stability=5,
                last_review=datetime.now() - timedelta(days=2),
            ))

        results = consolidator.run_dream_simulation(100)

        assert results.total_retrievals == 100
        assert results.success_count + results.failure_count == 100
        assert 0 <= results.success_rate <= 1

    def test_analyze_scheduling(self):
        """Test scheduling analysis."""
        consolidator = DreamConsolidator()

        # Add overdue concept
        consolidator.add_concept(ConceptState(
            "overdue", "Overdue",
            stability=1,
            last_review=datetime.now() - timedelta(days=10),
        ))

        # Add due today
        consolidator.add_concept(ConceptState(
            "today", "Today",
            stability=2,
            last_review=datetime.now() - timedelta(days=2),
        ))

        analysis = consolidator.analyze_scheduling()

        assert len(analysis.overdue) >= 1

    def test_analyze_stability_distribution(self):
        """Test stability distribution analysis."""
        consolidator = DreamConsolidator()

        # Add concepts with different stabilities
        consolidator.add_concept(ConceptState("a", "A", stability=1))
        consolidator.add_concept(ConceptState("b", "B", stability=5))
        consolidator.add_concept(ConceptState("c", "C", stability=15))
        consolidator.add_concept(ConceptState("d", "D", stability=45))

        dist = consolidator.analyze_stability_distribution()

        assert len(dist.intensive) >= 1
        assert len(dist.standard) >= 1
        assert len(dist.maintenance) >= 1
        assert len(dist.occasional) >= 1

    def test_handle_extended_absence_short(self):
        """Test handling short absence."""
        consolidator = DreamConsolidator()

        result = consolidator.handle_extended_absence(2)

        assert result["phase"] == "standard"

    def test_handle_extended_absence_medium(self):
        """Test handling medium absence."""
        consolidator = DreamConsolidator()

        result = consolidator.handle_extended_absence(5)

        assert result["phase"] == "flag_absence"

    def test_handle_extended_absence_long(self):
        """Test handling long absence."""
        consolidator = DreamConsolidator()

        # Add some concepts
        for i in range(20):
            consolidator.add_concept(ConceptState(
                f"c{i}", f"C{i}",
                mastery=0.3, stability=2,
                last_review=datetime.now() - timedelta(days=10),
            ))

        result = consolidator.handle_extended_absence(7)

        assert result["phase"] == "re_entry"
        assert "critical_concepts" in result
        assert result["suggested_session_minutes"] <= 20


class TestDataclasses:
    """Tests for supporting dataclasses."""

    def test_fragility_score_to_dict(self):
        """Test FragilityScore serialization."""
        score = FragilityScore(
            concept_id="c1",
            fragility=0.75,
            simulated_retrieval=0.25,
            weakness_type=WeaknessType.RECENT_DECAY,
            urgency=ReviewUrgency.HIGH,
            factors={"retention": 0.5, "mastery": 0.6},
        )
        d = score.to_dict()

        assert d["concept_id"] == "c1"
        assert d["fragility"] == 0.75
        assert d["weakness_type"] == "recent_decay"

    def test_interference_pair_to_dict(self):
        """Test InterferencePair serialization."""
        pair = InterferencePair(
            concept_a="a",
            concept_b="b",
            similarity=0.92,
            risk=InterferenceRisk.HIGH,
            confusion_history=3,
        )
        d = pair.to_dict()

        assert d["concept_a"] == "a"
        assert d["similarity"] == 0.92
        assert d["risk"] == "high"

    def test_consolidation_atom_to_dict(self):
        """Test ConsolidationAtom serialization."""
        atom = ConsolidationAtom(
            priority=1,
            concept_id="c1",
            concept_name="Concept 1",
            weakness_type=WeaknessType.RECENT_DECAY,
            atom_type=AtomType.RETRIEVAL_PRACTICE,
            rationale="Prevent forgetting",
            tags=["Priority_NCDE_Stabilizers"],
        )
        d = atom.to_dict()

        assert d["priority"] == 1
        assert d["atom_type"] == "retrieval_practice"
        assert "Priority_NCDE_Stabilizers" in d["tags"]

    def test_decay_projection_to_dict(self):
        """Test DecayProjection serialization."""
        projection = DecayProjection(
            concept_id="c1",
            retention_now=0.75,
            retention_projected=0.55,
            hours_projected=48,
            urgency=ReviewUrgency.HIGH,
            review_recommended_within_hours=24,
        )
        d = projection.to_dict()

        assert d["retention_now"] == 0.75
        assert d["hours_projected"] == 48

    def test_morning_boot_info_to_dict(self):
        """Test MorningBootInfo serialization."""
        boot = MorningBootInfo(
            consolidation_summary="Analyzed 100 concepts",
            priority_items_count=5,
            predicted_session_minutes=20,
            motivation_quote="Learning is a journey.",
            priority_atoms=[],
            consolidation_ran_at=datetime.now(),
        )
        d = boot.to_dict()

        assert d["priority_items_count"] == 5
        assert d["predicted_session_minutes"] == 20

    def test_simulation_results_success_rate(self):
        """Test SimulationResults success rate."""
        results = SimulationResults(
            total_retrievals=100,
            success_count=75,
            failure_count=25,
            decay_applied=5,
            interference_events=3,
            concepts_needing_strengthening=["a", "b"],
        )

        assert results.success_rate == 0.75

    def test_consolidation_effectiveness_improvement(self):
        """Test ConsolidationEffectiveness calculations."""
        metrics = ConsolidationEffectiveness(
            morning_retrieval_with=0.85,
            morning_retrieval_without=0.68,
            interference_errors_with=0.08,
            interference_errors_without=0.22,
            overdue_rate_with=0.05,
            overdue_rate_without=0.18,
            days_to_solid_state_with=45,
            days_to_solid_state_without=60,
        )

        assert metrics.retrieval_improvement > 20  # ~25% improvement
        assert metrics.interference_reduction > 60  # ~64% reduction


class TestFactoryFunction:
    """Tests for factory functions."""

    def test_create_dream_consolidator(self):
        """Test factory function."""
        consolidator = create_dream_consolidator()

        assert isinstance(consolidator, DreamConsolidator)
        assert consolidator.weak_node_scanner is not None
