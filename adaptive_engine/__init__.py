"""
Adaptive Learning Engine package.

Note: To keep package import lightweight for tests that only need stateless
utilities (e.g., PathSequencer helpers), submodule imports are guarded.
This avoids importing database-dependent modules during package import.
"""

# Guarded imports to avoid raising on optional dependencies during test collection.
# Do NOT import DB-dependent modules here to keep package import light.
LearningEngine = None  # type: ignore
MasteryCalculator = None  # type: ignore

try:  # Data models (pure)
    from .models import (  # type: ignore
        COMPREHENSION_LEVELS,
        AnswerResult,
        AtomPresentation,
        BlockingPrerequisite,
        ChapterReadingProgress,
        ConceptMastery,
        ContentFeatures,
        GatingType,
        KnowledgeBreakdown,
        KnowledgeGap,
        LearningPath,
        MasteryLevel,
        RemediationPlan,
        ReReadRecommendation,
        SessionMode,
        SessionState,
        SessionStatus,
        SuitabilityScore,
        TriggerType,
        UnlockStatus,
    )
except Exception:  # pragma: no cover
    # Provide minimal fallbacks so attribute access doesn't crash in light tests
    COMPREHENSION_LEVELS = None  # type: ignore
    AnswerResult = None  # type: ignore
    AtomPresentation = None  # type: ignore
    BlockingPrerequisite = None  # type: ignore
    ChapterReadingProgress = None  # type: ignore
    ConceptMastery = None  # type: ignore
    ContentFeatures = None  # type: ignore
    GatingType = None  # type: ignore
    KnowledgeBreakdown = None  # type: ignore
    KnowledgeGap = None  # type: ignore
    LearningPath = None  # type: ignore
    MasteryLevel = None  # type: ignore
    RemediationPlan = None  # type: ignore
    ReReadRecommendation = None  # type: ignore
    SessionMode = None  # type: ignore
    SessionState = None  # type: ignore
    SessionStatus = None  # type: ignore
    SuitabilityScore = None  # type: ignore
    TriggerType = None  # type: ignore
    UnlockStatus = None  # type: ignore

try:  # Sequencer and auxiliary scorers (should be pure or lazily import DB)
    from .path_sequencer import PathSequencer  # type: ignore
except Exception:  # pragma: no cover
    PathSequencer = None  # type: ignore

try:
    from .remediation_router import RemediationRouter  # type: ignore
except Exception:  # pragma: no cover
    RemediationRouter = None  # type: ignore

try:
    from .suitability_scorer import SuitabilityScorer  # type: ignore
except Exception:  # pragma: no cover
    SuitabilityScorer = None  # type: ignore

try:
    from .jit_generator import (  # type: ignore
        ContentType,
        GenerationRequest,
        GenerationResult,
        GenerationTrigger,
        JITGenerationService,
    )
except Exception:  # pragma: no cover
    JITGenerationService = None  # type: ignore
    GenerationRequest = None  # type: ignore
    GenerationResult = None  # type: ignore
    GenerationTrigger = None  # type: ignore
    ContentType = None  # type: ignore

try:  # NCDE Struggle Weight Update Functions
    from .ncde_pipeline import (  # type: ignore
        StruggleUpdateData,
        prepare_struggle_update,
        update_struggle_weight_async,
        update_struggle_weight_sync,
    )
except Exception:  # pragma: no cover
    StruggleUpdateData = None  # type: ignore
    prepare_struggle_update = None  # type: ignore
    update_struggle_weight_async = None  # type: ignore
    update_struggle_weight_sync = None  # type: ignore

try:  # ZPD (Zone of Proximal Development) Engine
    from .zpd import (  # type: ignore
        # Main engine
        ZPDEngine,
        # Calculators
        ZPDCalculator,
        FlowChannelManager,
        ScaffoldSelector,
        # State classes
        ZPDState,
        ZPDBoundary,
        ZPDGrowthRecord,
        FlowChannelState,
        ScaffoldDecision,
        NCDEFrictionVector,
        # Enums
        ScaffoldType,
        ZPDPosition,
        FlowState,
        StruggleType,
        # Utilities
        compute_friction_from_metrics,
        ZPD_THRESHOLDS,
    )
except Exception:  # pragma: no cover
    ZPDEngine = None  # type: ignore
    ZPDCalculator = None  # type: ignore
    FlowChannelManager = None  # type: ignore
    ScaffoldSelector = None  # type: ignore
    ZPDState = None  # type: ignore
    ZPDBoundary = None  # type: ignore
    ZPDGrowthRecord = None  # type: ignore
    FlowChannelState = None  # type: ignore
    ScaffoldDecision = None  # type: ignore
    NCDEFrictionVector = None  # type: ignore
    ScaffoldType = None  # type: ignore
    ZPDPosition = None  # type: ignore
    FlowState = None  # type: ignore
    StruggleType = None  # type: ignore
    compute_friction_from_metrics = None  # type: ignore
    ZPD_THRESHOLDS = None  # type: ignore

try:  # Bayesian Topic Readiness (WO-AE-006)
    from .bayesian_readiness import (  # type: ignore
        BayesianReadinessCalculator,
        TopicState,
        InterventionType,
        ReadinessScore,
        TopicNode,
        InterventionDecision,
        calculate_topic_readiness,
    )
except Exception:  # pragma: no cover
    BayesianReadinessCalculator = None  # type: ignore
    TopicState = None  # type: ignore
    InterventionType = None  # type: ignore
    ReadinessScore = None  # type: ignore
    TopicNode = None  # type: ignore
    InterventionDecision = None  # type: ignore
    calculate_topic_readiness = None  # type: ignore

try:  # Cognitive Flexibility Theory (WO-AE-008)
    from .cognitive_flexibility import (  # type: ignore
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
except Exception:  # pragma: no cover
    CFTEngine = None  # type: ignore
    ExpertLens = None  # type: ignore
    DomainType = None  # type: ignore
    TransferStatus = None  # type: ignore
    EntryPoint = None  # type: ignore
    CaseStudy = None  # type: ignore
    ConceptLandscape = None  # type: ignore
    LensConfiguration = None  # type: ignore
    LearnerCFTState = None  # type: ignore
    BiasDetection = None  # type: ignore
    TransferReadinessMetrics = None  # type: ignore
    CrissCrossAction = None  # type: ignore
    CaseLibrary = None  # type: ignore
    PerspectiveCoverage = None  # type: ignore
    DEFAULT_LENSES = None  # type: ignore
    create_default_cft_engine = None  # type: ignore

try:  # Dream Consolidation Engine (WO-AE-009)
    from .dream_consolidator import (  # type: ignore
        # Main engine
        DreamConsolidator,
        # Component classes
        WeakNodeScanner,
        InterferenceAnalyzer,
        DecayProjector,
        ConsolidationPathOptimizer,
        # State/data classes
        ConceptState,
        FragilityScore,
        InterferencePair,
        ConsolidationAtom,
        DecayProjection,
        SchedulingAnalysis,
        StabilityDistribution,
        MorningBootInfo,
        ConsolidationEffectiveness,
        LearnerPattern,
        SimulationResults,
        # Enums
        WeaknessType,
        AtomType,
        ReviewUrgency,
        InterferenceRisk,
        StabilityRange,
        # Factory
        create_dream_consolidator,
    )
except Exception:  # pragma: no cover
    DreamConsolidator = None  # type: ignore
    WeakNodeScanner = None  # type: ignore
    InterferenceAnalyzer = None  # type: ignore
    DecayProjector = None  # type: ignore
    ConsolidationPathOptimizer = None  # type: ignore
    ConceptState = None  # type: ignore
    FragilityScore = None  # type: ignore
    InterferencePair = None  # type: ignore
    ConsolidationAtom = None  # type: ignore
    DecayProjection = None  # type: ignore
    SchedulingAnalysis = None  # type: ignore
    StabilityDistribution = None  # type: ignore
    MorningBootInfo = None  # type: ignore
    ConsolidationEffectiveness = None  # type: ignore
    LearnerPattern = None  # type: ignore
    SimulationResults = None  # type: ignore
    WeaknessType = None  # type: ignore
    AtomType = None  # type: ignore
    ReviewUrgency = None  # type: ignore
    InterferenceRisk = None  # type: ignore
    StabilityRange = None  # type: ignore
    create_dream_consolidator = None  # type: ignore

__all__ = [
    # Main engine
    "LearningEngine",
    # Component classes
    "MasteryCalculator",
    "PathSequencer",
    "RemediationRouter",
    "SuitabilityScorer",
    # Data models
    "ConceptMastery",
    "KnowledgeBreakdown",
    "LearningPath",
    "RemediationPlan",
    "SuitabilityScore",
    "ContentFeatures",
    "SessionState",
    "AtomPresentation",
    "AnswerResult",
    "KnowledgeGap",
    "UnlockStatus",
    "BlockingPrerequisite",
    # Reading progress
    "ChapterReadingProgress",
    "ReReadRecommendation",
    "COMPREHENSION_LEVELS",
    # Enums
    "MasteryLevel",
    "GatingType",
    "TriggerType",
    "SessionMode",
    "SessionStatus",
    # JIT Generation
    "JITGenerationService",
    "GenerationRequest",
    "GenerationResult",
    "GenerationTrigger",
    "ContentType",
    # NCDE Struggle Weight Update
    "StruggleUpdateData",
    "prepare_struggle_update",
    "update_struggle_weight_async",
    "update_struggle_weight_sync",
    # ZPD (Zone of Proximal Development)
    "ZPDEngine",
    "ZPDCalculator",
    "FlowChannelManager",
    "ScaffoldSelector",
    "ZPDState",
    "ZPDBoundary",
    "ZPDGrowthRecord",
    "FlowChannelState",
    "ScaffoldDecision",
    "NCDEFrictionVector",
    "ScaffoldType",
    "ZPDPosition",
    "FlowState",
    "StruggleType",
    "compute_friction_from_metrics",
    "ZPD_THRESHOLDS",
    # Bayesian Topic Readiness (WO-AE-006)
    "BayesianReadinessCalculator",
    "TopicState",
    "InterventionType",
    "ReadinessScore",
    "TopicNode",
    "InterventionDecision",
    "calculate_topic_readiness",
    # Cognitive Flexibility Theory (WO-AE-008)
    "CFTEngine",
    "ExpertLens",
    "DomainType",
    "TransferStatus",
    "EntryPoint",
    "CaseStudy",
    "ConceptLandscape",
    "LensConfiguration",
    "LearnerCFTState",
    "BiasDetection",
    "TransferReadinessMetrics",
    "CrissCrossAction",
    "CaseLibrary",
    "PerspectiveCoverage",
    "DEFAULT_LENSES",
    "create_default_cft_engine",
    # Dream Consolidation Engine (WO-AE-009)
    "DreamConsolidator",
    "WeakNodeScanner",
    "InterferenceAnalyzer",
    "DecayProjector",
    "ConsolidationPathOptimizer",
    "ConceptState",
    "FragilityScore",
    "InterferencePair",
    "ConsolidationAtom",
    "DecayProjection",
    "SchedulingAnalysis",
    "StabilityDistribution",
    "MorningBootInfo",
    "ConsolidationEffectiveness",
    "LearnerPattern",
    "SimulationResults",
    "WeaknessType",
    "AtomType",
    "ReviewUrgency",
    "InterferenceRisk",
    "StabilityRange",
    "create_dream_consolidator",
]
