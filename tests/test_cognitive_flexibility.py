"""
Tests for Cognitive Flexibility Theory Criss-Crossing.

This file runs the BDD scenarios AND provides additional unit test coverage.

Work Order: WO-AE-008
"""

import pytest
from pytest_bdd import scenarios

# Import step definitions - required for pytest-bdd to find them
from step_defs.cognitive_flexibility_steps import *  # noqa: F401, F403

# Load all scenarios from the feature file
scenarios("../features/cognitive_flexibility_criss_crossing.feature")


# ============================================================================
# Additional Unit Tests for Coverage
# ============================================================================

from adaptive_engine.cognitive_flexibility import (
    CFTEngine,
    ExpertLens,
    DomainType,
    TransferStatus,
    EntryPoint,
    CaseStudy,
    ConceptLandscape,
    LensConfiguration,
    LearnerCFTState,
    BiasDetection,
    TransferReadinessMetrics,
    CrissCrossAction,
    CaseLibrary,
    PerspectiveCoverage,
    DEFAULT_LENSES,
    create_default_cft_engine,
)


class TestCFTEngine:
    """Unit tests for CFTEngine."""

    def test_create_engine(self):
        """Test engine creation."""
        engine = CFTEngine()
        assert engine is not None
        assert len(engine.lenses) == 5

    def test_add_concept(self):
        """Test adding concepts to the engine."""
        engine = CFTEngine()

        concept = engine.add_concept(
            "test_concept",
            "Test Concept",
            DomainType.WELL_STRUCTURED,
        )

        assert concept.concept_id == "test_concept"
        assert concept.concept_name == "Test Concept"
        assert "test_concept" in engine.concepts

    def test_add_concept_with_entry_points(self):
        """Test adding concept with initial entry points."""
        engine = CFTEngine()

        entry_points = [
            EntryPoint("Math", "Proof", "Axiom"),
            EntryPoint("Code", "Function", "Recursion"),
        ]

        concept = engine.add_concept(
            "recursion",
            "Recursion",
            DomainType.ILL_STRUCTURED,
            entry_points=entry_points,
        )

        assert len(concept.entry_points) == 2

    def test_add_entry_point(self):
        """Test adding entry points to existing concept."""
        engine = CFTEngine()
        engine.add_concept("c1", "Concept 1")

        engine.add_entry_point(
            "c1",
            "Mathematical",
            "Proof_Theory",
            "Induction",
            "Start with base case",
        )

        assert len(engine.concepts["c1"].entry_points) == 1
        assert engine.concepts["c1"].entry_points[0].name == "Mathematical"

    def test_add_entry_point_unknown_concept(self):
        """Test error on unknown concept."""
        engine = CFTEngine()

        with pytest.raises(ValueError, match="not found"):
            engine.add_entry_point("unknown", "Math", "Proof", "Axiom")

    def test_apply_lens(self):
        """Test applying a lens to a concept."""
        engine = CFTEngine()
        engine.add_concept("c1", "Concept 1")

        lens_config, case_study = engine.apply_lens(
            "learner1",
            "c1",
            ExpertLens.SPIVAK,
        )

        assert lens_config.lens_id == ExpertLens.SPIVAK
        assert lens_config.agent_name == "Spivak_Agent"

    def test_apply_lens_records_criss_cross(self):
        """Test that applying lenses increments criss-cross count."""
        engine = CFTEngine()
        engine.add_concept("c1", "Concept 1")

        engine.apply_lens("learner1", "c1", ExpertLens.SPIVAK)
        engine.apply_lens("learner1", "c1", ExpertLens.KNUTH)

        state = engine.get_or_create_learner_state("learner1")
        assert state.criss_cross_count == 2

    def test_apply_lens_same_lens_no_increment(self):
        """Test that applying same lens twice doesn't double-count."""
        engine = CFTEngine()
        engine.add_concept("c1", "Concept 1")

        engine.apply_lens("learner1", "c1", ExpertLens.SPIVAK)
        engine.apply_lens("learner1", "c1", ExpertLens.SPIVAK)  # Same lens

        state = engine.get_or_create_learner_state("learner1")
        assert state.criss_cross_count == 1  # Only counted once

    def test_navigate_entry_point(self):
        """Test navigating via entry point."""
        engine = CFTEngine()
        engine.add_concept("c1", "Concept 1")
        engine.add_entry_point("c1", "Math", "Proof", "Axiom")

        actions = engine.navigate_entry_point("learner1", "c1", "Math")

        # First navigation returns no special actions
        assert isinstance(actions, list)

    def test_navigate_multiple_entry_points(self):
        """Test navigating via multiple entry points generates actions."""
        engine = CFTEngine()
        engine.add_concept("c1", "Concept 1")
        engine.add_entry_point("c1", "Math", "Proof", "Axiom")
        engine.add_entry_point("c1", "Code", "Function", "Stack")

        engine.navigate_entry_point("learner1", "c1", "Math")
        actions = engine.navigate_entry_point("learner1", "c1", "Code")

        # Second navigation generates actions
        assert len(actions) >= 1
        action_types = [a.action_type for a in actions]
        assert "Track_Entry_Points_Used" in action_types

    def test_check_mastery_requirements_not_met(self):
        """Test mastery requirements not met with insufficient coverage."""
        engine = CFTEngine()
        engine.add_concept("c1", "Concept 1")

        engine.navigate_entry_point("learner1", "c1", "Math")
        engine.apply_lens("learner1", "c1", ExpertLens.SPIVAK)

        met, reason = engine.check_mastery_requirements("learner1", "c1")

        assert not met
        assert "entry points" in reason.lower() or "perspectives" in reason.lower()

    def test_check_mastery_requirements_met(self):
        """Test mastery requirements met with good coverage."""
        engine = CFTEngine()
        engine.add_concept("c1", "Concept 1")

        # Add 3+ entry points
        for ep in ["Math", "Code", "Visual", "Linguistic"]:
            engine.navigate_entry_point("learner1", "c1", ep)

        # Add 3+ lenses
        for lens in [ExpertLens.SPIVAK, ExpertLens.KNUTH, ExpertLens.RICHTER]:
            engine.apply_lens("learner1", "c1", lens)

        met, reason = engine.check_mastery_requirements("learner1", "c1")

        assert met
        assert "met" in reason.lower()

    def test_landscape_explorer_mode(self):
        """Test Landscape Explorer mode activation."""
        engine = CFTEngine()

        assert not engine.is_landscape_explorer_active()

        engine.activate_landscape_explorer()
        assert engine.is_landscape_explorer_active()

        engine.deactivate_landscape_explorer()
        assert not engine.is_landscape_explorer_active()

    def test_detect_reductive_bias_no_bias(self):
        """Test bias detection with balanced perspectives."""
        engine = CFTEngine()
        engine.add_concept("c1", "Concept 1")

        # Apply multiple lenses
        for lens in [ExpertLens.SPIVAK, ExpertLens.KNUTH, ExpertLens.RICHTER]:
            engine.apply_lens("learner1", "c1", lens)

        result = engine.detect_reductive_bias(
            "learner1", "c1", ["spivak", "knuth", "richter"]
        )

        assert not result.is_biased

    def test_detect_reductive_bias_with_bias(self):
        """Test bias detection with single perspective."""
        engine = CFTEngine()
        engine.add_concept("c1", "Concept 1")

        engine.apply_lens("learner1", "c1", ExpertLens.KNUTH)

        result = engine.detect_reductive_bias(
            "learner1", "c1", ["spivak", "knuth", "richter", "gotzsche"]
        )

        assert result.is_biased
        assert result.dominant_perspective == "knuth"
        assert len(result.ignored_perspectives) >= 2

    def test_analyze_perspective_coverage(self):
        """Test perspective coverage analysis."""
        engine = CFTEngine()

        concepts = []
        for i in range(5):
            cid = f"concept_{i}"
            engine.add_concept(cid, f"Concept {i}")
            concepts.append(cid)

            # Apply Knuth to all, Spivak to some
            engine.apply_lens("learner1", cid, ExpertLens.KNUTH)
            if i < 3:
                engine.apply_lens("learner1", cid, ExpertLens.SPIVAK)

        results = engine.analyze_perspective_coverage(
            "learner1", concepts, ["knuth", "spivak"]
        )

        assert len(results) == 2
        knuth_cov = next(r for r in results if r.perspective == "knuth")
        spivak_cov = next(r for r in results if r.perspective == "spivak")

        assert knuth_cov.coverage_percent == 100.0
        assert spivak_cov.coverage_percent == 60.0

    def test_assess_transfer_readiness_not_ready(self):
        """Test transfer readiness assessment - not ready."""
        engine = CFTEngine()
        engine.add_concept("c1", "Concept 1")

        metrics = engine.assess_transfer_readiness(
            "learner1", ["c1"], novel_problem_success_rate=30.0
        )

        assert metrics.status == TransferStatus.NOT_READY

    def test_assess_transfer_readiness_ready(self):
        """Test transfer readiness assessment - ready."""
        engine = CFTEngine()

        concepts = []
        for i in range(5):
            cid = f"concept_{i}"
            engine.add_concept(cid, f"Concept {i}")
            concepts.append(cid)

            # Good coverage
            for ep in ["A", "B", "C", "D"]:
                engine.navigate_entry_point("learner1", cid, ep)
            for lens in [ExpertLens.SPIVAK, ExpertLens.KNUTH, ExpertLens.RICHTER, ExpertLens.KING]:
                engine.apply_lens("learner1", cid, lens)

        metrics = engine.assess_transfer_readiness(
            "learner1", concepts, novel_problem_success_rate=85.0
        )

        assert metrics.status in [TransferStatus.READY, TransferStatus.CERTIFIED]

    def test_get_criss_cross_recommendation_need_entry_points(self):
        """Test recommendation when more entry points needed."""
        engine = CFTEngine()
        engine.add_concept("c1", "Concept 1")

        engine.navigate_entry_point("learner1", "c1", "Math")

        rec = engine.get_criss_cross_recommendation("learner1", "c1")

        assert rec is not None
        assert "entry point" in rec.lower()

    def test_get_criss_cross_recommendation_need_lenses(self):
        """Test recommendation when more lenses needed."""
        engine = CFTEngine()
        engine.add_concept("c1", "Concept 1")

        # Enough entry points
        for ep in ["A", "B", "C"]:
            engine.navigate_entry_point("learner1", "c1", ep)

        # Not enough lenses
        engine.apply_lens("learner1", "c1", ExpertLens.KNUTH)

        rec = engine.get_criss_cross_recommendation("learner1", "c1")

        assert rec is not None
        assert "lens" in rec.lower()

    def test_build_ill_structured_domain(self):
        """Test building case library for ill-structured domain."""
        engine = CFTEngine()

        cases = [
            CaseStudy("C1", "Context 1", ["A", "B"]),
            CaseStudy("C2", "Context 2", ["B", "C"]),
            CaseStudy("C3", "Context 3", ["A", "C"]),
        ]

        count = engine.build_ill_structured_domain("Test Domain", cases)

        assert count == 3
        assert engine.case_library.get_case("C1") is not None


class TestCaseLibrary:
    """Tests for CaseLibrary."""

    def test_add_case(self):
        """Test adding cases to library."""
        library = CaseLibrary()
        case = CaseStudy("C1", "Context", ["Concept1", "Concept2"], "Domain1")

        library.add_case(case)

        assert library.get_case("C1") == case

    def test_get_cases_for_concept(self):
        """Test retrieving cases by concept."""
        library = CaseLibrary()
        library.add_case(CaseStudy("C1", "Ctx1", ["A", "B"]))
        library.add_case(CaseStudy("C2", "Ctx2", ["B", "C"]))
        library.add_case(CaseStudy("C3", "Ctx3", ["A", "C"]))

        a_cases = library.get_cases_for_concept("A")
        b_cases = library.get_cases_for_concept("B")

        assert len(a_cases) == 2
        assert len(b_cases) == 2

    def test_get_cases_for_domain(self):
        """Test retrieving cases by domain."""
        library = CaseLibrary()
        library.add_case(CaseStudy("C1", "Ctx1", [], "D1"))
        library.add_case(CaseStudy("C2", "Ctx2", [], "D1"))
        library.add_case(CaseStudy("C3", "Ctx3", [], "D2"))

        d1_cases = library.get_cases_for_domain("D1")
        d2_cases = library.get_cases_for_domain("D2")

        assert len(d1_cases) == 2
        assert len(d2_cases) == 1


class TestLearnerCFTState:
    """Tests for LearnerCFTState."""

    def test_record_entry_point(self):
        """Test recording entry points."""
        state = LearnerCFTState("learner1")

        state.record_entry_point("c1", "Math")
        state.record_entry_point("c1", "Code")

        assert state.get_entry_point_count("c1") == 2

    def test_record_entry_point_duplicate(self):
        """Test that duplicate entry points aren't counted twice."""
        state = LearnerCFTState("learner1")

        state.record_entry_point("c1", "Math")
        state.record_entry_point("c1", "Math")  # Duplicate

        assert state.get_entry_point_count("c1") == 1

    def test_record_lens_use(self):
        """Test recording lens usage."""
        state = LearnerCFTState("learner1")

        state.record_lens_use("c1", ExpertLens.SPIVAK)
        state.record_lens_use("c1", ExpertLens.KNUTH)

        assert state.get_lens_count("c1") == 2
        assert state.criss_cross_count == 2

    def test_to_dict(self):
        """Test serialization."""
        state = LearnerCFTState("learner1")
        state.record_entry_point("c1", "Math")
        state.record_lens_use("c1", ExpertLens.SPIVAK)

        d = state.to_dict()

        assert d["learner_id"] == "learner1"
        assert "c1" in d["entry_points_used"]


class TestTransferReadinessMetrics:
    """Tests for TransferReadinessMetrics."""

    def test_evaluate_not_ready(self):
        """Test evaluation - not ready."""
        metrics = TransferReadinessMetrics(
            entry_points_navigated=1.0,
            perspectives_integrated=1.0,
            novel_assembly_success=30.0,
            reductive_bias_score=0.8,
        )

        status = metrics.evaluate()

        assert status == TransferStatus.NOT_READY

    def test_evaluate_developing(self):
        """Test evaluation - developing."""
        metrics = TransferReadinessMetrics(
            entry_points_navigated=3.5,
            perspectives_integrated=3.5,
            novel_assembly_success=50.0,
            reductive_bias_score=0.5,
        )

        status = metrics.evaluate()

        assert status == TransferStatus.DEVELOPING

    def test_evaluate_ready(self):
        """Test evaluation - ready (meets 3 of 4 thresholds)."""
        metrics = TransferReadinessMetrics(
            entry_points_navigated=3.5,  # PASS (>= 3.0)
            perspectives_integrated=3.5,  # PASS (>= 3.0)
            novel_assembly_success=75.0,  # PASS (>= 70.0)
            reductive_bias_score=0.35,    # FAIL (> 0.30 threshold)
        )

        status = metrics.evaluate()

        # With 3 passes, should be READY (not CERTIFIED which requires all 4)
        assert status == TransferStatus.READY

    def test_evaluate_certified(self):
        """Test evaluation - certified."""
        metrics = TransferReadinessMetrics(
            entry_points_navigated=4.5,
            perspectives_integrated=4.0,
            novel_assembly_success=85.0,
            reductive_bias_score=0.1,
        )

        status = metrics.evaluate()

        assert status == TransferStatus.CERTIFIED

    def test_to_dict(self):
        """Test serialization."""
        metrics = TransferReadinessMetrics(
            entry_points_navigated=3.5,
            perspectives_integrated=3.0,
            novel_assembly_success=75.0,
            reductive_bias_score=0.2,
        )
        metrics.evaluate()

        d = metrics.to_dict()

        assert "entry_points_navigated" in d
        assert "thresholds" in d


class TestDataclasses:
    """Tests for supporting dataclasses."""

    def test_lens_configuration_to_dict(self):
        """Test LensConfiguration serialization."""
        config = LensConfiguration(
            lens_id=ExpertLens.SPIVAK,
            agent_name="Spivak_Agent",
            domain_focus="Math rigor",
            key_questions=["Q1", "Q2"],
        )

        d = config.to_dict()

        assert d["lens_id"] == "spivak"
        assert d["agent_name"] == "Spivak_Agent"

    def test_entry_point_to_dict(self):
        """Test EntryPoint serialization."""
        ep = EntryPoint("Math", "Proof", "Axiom", "Description")

        d = ep.to_dict()

        assert d["name"] == "Math"
        assert d["domain_frame"] == "Proof"

    def test_concept_landscape_to_dict(self):
        """Test ConceptLandscape serialization."""
        landscape = ConceptLandscape(
            concept_id="c1",
            concept_name="Concept 1",
            entry_points=[EntryPoint("Math", "Proof", "Axiom")],
            available_lenses={ExpertLens.SPIVAK, ExpertLens.KNUTH},
        )

        d = landscape.to_dict()

        assert d["concept_id"] == "c1"
        assert len(d["entry_points"]) == 1
        assert len(d["available_lenses"]) == 2

    def test_case_study_to_dict(self):
        """Test CaseStudy serialization."""
        case = CaseStudy(
            case_id="C1",
            context="Context",
            relevant_concepts=["A", "B"],
            domain="Domain",
            complexity="high",
        )

        d = case.to_dict()

        assert d["case_id"] == "C1"
        assert d["complexity"] == "high"

    def test_bias_detection_to_dict(self):
        """Test BiasDetection serialization."""
        bias = BiasDetection(
            is_biased=True,
            dominant_perspective="efficiency",
            ignored_perspectives=["safety", "memory"],
            recommended_action="Broaden perspective",
            contrasting_case="Case 1",
        )

        d = bias.to_dict()

        assert d["is_biased"] is True
        assert len(d["ignored_perspectives"]) == 2

    def test_perspective_coverage_to_dict(self):
        """Test PerspectiveCoverage serialization."""
        coverage = PerspectiveCoverage(
            perspective="Efficiency",
            coverage_percent=85.5,
            gap_areas=["Area1", "Area2"],
        )

        d = coverage.to_dict()

        assert d["coverage_percent"] == 85.5

    def test_criss_cross_action_to_dict(self):
        """Test CrissCrossAction serialization."""
        action = CrissCrossAction(
            action_type="Track",
            purpose="Record",
            details="Details here",
        )

        d = action.to_dict()

        assert d["action_type"] == "Track"


class TestDefaultLenses:
    """Tests for default lens configurations."""

    def test_all_lenses_configured(self):
        """Test all 5 lenses are configured."""
        assert len(DEFAULT_LENSES) == 5
        assert ExpertLens.SPIVAK in DEFAULT_LENSES
        assert ExpertLens.RICHTER in DEFAULT_LENSES
        assert ExpertLens.KNUTH in DEFAULT_LENSES
        assert ExpertLens.GOTZSCHE in DEFAULT_LENSES
        assert ExpertLens.KING in DEFAULT_LENSES

    def test_lens_has_questions(self):
        """Test each lens has key questions."""
        for lens, config in DEFAULT_LENSES.items():
            assert len(config.key_questions) >= 1, f"{lens} missing questions"

    def test_lens_has_criteria(self):
        """Test each lens has evaluation criteria."""
        for lens, config in DEFAULT_LENSES.items():
            assert len(config.evaluation_criteria) >= 1, f"{lens} missing criteria"


class TestCreateDefaultEngine:
    """Tests for factory function."""

    def test_create_default_engine(self):
        """Test creating default engine."""
        engine = create_default_cft_engine()

        assert isinstance(engine, CFTEngine)
        assert len(engine.lenses) == 5
