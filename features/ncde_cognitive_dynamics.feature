@Adaptive @NCDE @Priority-Critical @Regression
@Node-AdaptiveEngine @DARPA-DigitalTutor @Algorithm-NeuralODE
Feature: Neural Controlled Differential Equations for Cognitive Dynamics
  """
  NCDE (Neural Controlled Differential Equations) models the continuous
  evolution of learner cognitive state over time. Unlike discrete FSRS,
  NCDE captures the dynamics of knowledge acquisition and decay as a
  continuous-time dynamical system.

  Scientific Foundation:
  - Kidger et al. (2020). Neural Controlled Differential Equations
  - Anderson & Schooler (1991). Memory decay follows power law
  - Pavlik & Anderson (2008). ACT-R declarative memory model
  - Lindsey et al. (2014). Optimal scheduling with forgetting curves

  Mathematical Model:
  dh(t)/dt = f_θ(h(t), X(t))

  Where:
  - h(t) = hidden cognitive state at time t
  - X(t) = control path (learning events, context)
  - f_θ = neural network parameterizing dynamics

  Key Metrics Tracked:
  - Retrieval Latency: Time to access knowledge (ms)
  - Error Complexity: C_error = Δn × n_s (misconception depth × spread)
  - Lattice Stability: How well knowledge graph holds under perturbation
  - Decay Velocity: Rate of forgetting (power law exponent)
  """

  Background:
    Given the NCDE model is initialized with pretrained weights
    And the learner has a cognitive state vector h(t)
    And learning events are logged as control path X(t)

  # ============================================================================
  # COGNITIVE STATE EVOLUTION
  # ============================================================================

  @Smoke @CognitiveState @ODE-Solver
  Scenario: Update cognitive state after learning event
    """
    When a learner completes an activity, the NCDE updates their
    cognitive state by solving the ODE forward in time.
    """
    Given learner "L1" has cognitive state h(t₀) at time t₀
    And learner completes activity with:
      | field          | value                |
      | atom_id        | atom-binary-subnet-1 |
      | response_time  | 2300ms               |
      | correct        | true                 |
      | confidence     | 0.8                  |
    When the NCDE solver advances to time t₁
    Then cognitive state h(t₁) reflects:
      | dimension          | change_direction | magnitude |
      | subnet_mastery     | increase         | +0.12     |
      | retrieval_strength | increase         | +0.08     |
      | confidence_calibration | stable       | ~0        |

  @Regression @CognitiveState @DecayModeling
  Scenario: Model knowledge decay between sessions
    """
    Anderson & Schooler (1991) showed memory follows power-law decay:
    m(t) = a × t^(-d) where d ≈ 0.5 for well-learned material.

    NCDE learns this decay function from data rather than assuming it.
    """
    Given learner "L1" last reviewed concept "C1" at time t₀
    And current time is t₁ = t₀ + 7 days
    And no learning events occurred for "C1" in this period
    When the NCDE extrapolates cognitive state to t₁
    Then the decay model predicts:
      | metric              | value            |
      | retrieval_probability | 0.62           |
      | decay_exponent      | -0.48            |
      | optimal_review_time | t₀ + 5 days      |
    And the prediction aligns with power-law decay (R² > 0.95)

  @Regression @CognitiveState @Spacing
  Scenario: Capture spacing effect in cognitive dynamics
    """
    The spacing effect (Cepeda et al., 2006) shows distributed practice
    leads to better retention than massed practice. NCDE should capture
    this by learning higher stability for spaced reviews.
    """
    Given two learners with identical initial states:
      | learner | practice_schedule |
      | L1      | massed (all day 1) |
      | L2      | spaced (days 1,3,7) |
    And both complete the same total practice
    When 14 days have elapsed
    Then NCDE predicts retention:
      | learner | retrieval_probability | stability |
      | L1      | 0.41                  | 3.2 days  |
      | L2      | 0.73                  | 8.7 days  |
    And the spacing benefit is d = 0.32 (Cohen's d)

  # ============================================================================
  # RETRIEVAL LATENCY TRACKING
  # ============================================================================

  @Smoke @RetrievalLatency @RealTime
  Scenario: Track retrieval latency as mastery indicator
    """
    Retrieval latency (response time) is a key indicator of knowledge
    accessibility. Faster retrieval indicates more automatic knowledge.

    Reference: Rickard (1997) - Instance-based random-sample retrieval
    """
    Given learner responds to flashcard with:
      | field         | value |
      | response_time | 850ms |
      | correct       | true  |
    When the retrieval latency is analyzed
    Then latency metrics are computed:
      | metric               | value      |
      | raw_latency          | 850ms      |
      | adjusted_latency     | 720ms      |
      | latency_percentile   | 65th       |
      | automaticity_score   | 0.72       |
    And the cognitive state is updated with automaticity signal

  @Regression @RetrievalLatency @TrendAnalysis
  Scenario: Detect fluency building through latency trends
    """
    As knowledge becomes proceduralized, retrieval becomes faster.
    The power law of practice (Newell & Rosenbloom, 1981):
    T(n) = T(1) × n^(-α) where α ≈ 0.4
    """
    Given learner has response time history for concept "C1":
      | trial | response_time |
      | 1     | 3200ms        |
      | 2     | 2100ms        |
      | 3     | 1600ms        |
      | 4     | 1350ms        |
      | 5     | 1200ms        |
    When the power law fit is computed
    Then the learning rate is:
      | metric          | value |
      | alpha           | 0.42  |
      | predicted_T(10) | 890ms |
      | fluency_status  | building |

  # ============================================================================
  # ERROR COMPLEXITY ANALYSIS
  # ============================================================================

  @Smoke @ErrorComplexity @MisconceptionDetection
  Scenario: Calculate error complexity for misconception depth
    """
    Error complexity measures how deep and widespread a misconception is:
    C_error = Δn × n_s

    Where:
    - Δn = number of nodes between correct and incorrect concepts in knowledge graph
    - n_s = number of related concepts affected (spread)

    High C_error triggers MAARTA agent recruitment for remediation.
    """
    Given learner makes an error on "subnet mask calculation"
    And the error reveals misconception:
      | field              | value                    |
      | expected_answer    | 255.255.255.0            |
      | actual_answer      | 255.255.0.0              |
      | misconception_root | bit boundary confusion   |
    When error complexity is calculated
    Then complexity metrics are:
      | metric              | value |
      | node_distance (Δn)  | 3     |
      | concept_spread (n_s)| 5     |
      | error_complexity    | 15    |
      | severity            | high  |
    And MAARTA recruitment is triggered for remediation

  @Regression @ErrorComplexity @PatternRecognition
  Scenario: Detect systematic error patterns across concepts
    """
    Some errors are isolated; others reveal systematic gaps.
    Pattern detection identifies root causes.
    """
    Given learner has error history:
      | concept              | error_type           | timestamp |
      | binary_to_decimal    | off_by_one           | t1        |
      | subnet_calculation   | boundary_confusion   | t2        |
      | CIDR_notation        | prefix_length_error  | t3        |
    When error patterns are analyzed
    Then a systematic gap is detected:
      | field           | value                              |
      | pattern_type    | positional_value_confusion         |
      | affected_concepts | ["binary", "subnet", "CIDR"]     |
      | root_cause      | weak binary number system mental model |
      | remediation     | return_to_binary_fundamentals      |

  # ============================================================================
  # LATTICE STABILITY TRACKING
  # ============================================================================

  @Smoke @LatticeStability @KnowledgeGraph
  Scenario: Measure knowledge graph stability under perturbation
    """
    A stable knowledge lattice means the learner can access related
    concepts even when starting from different entry points.

    We measure this by probing adjacent concepts after a successful retrieval.
    """
    Given learner successfully retrieves concept "C1"
    When adjacent concepts are probed:
      | concept | relationship     |
      | C2      | prerequisite_of  |
      | C3      | analogy_to       |
      | C4      | contrasts_with   |
    Then lattice stability is measured:
      | metric                    | value |
      | adjacent_retrieval_rate   | 0.85  |
      | spreading_activation_score| 0.72  |
      | lattice_stability_index   | 0.78  |

  @Regression @LatticeStability @Fragility
  Scenario: Detect fragile knowledge structures
    """
    Fragile knowledge is correct in isolation but fails under
    contextual variation or when integrated with other concepts.
    """
    Given learner shows:
      | context          | performance |
      | isolated_drill   | 95%         |
      | mixed_practice   | 62%         |
      | transfer_task    | 41%         |
    When fragility is assessed
    Then fragility indicators are:
      | metric              | value      |
      | context_dependence  | high       |
      | transfer_gap        | 0.54       |
      | fragility_score     | 0.68       |
      | recommendation      | interleave_practice |

  # ============================================================================
  # SCHEDULING INTEGRATION
  # ============================================================================

  @Smoke @Scheduling @ZScorePriority
  Scenario: NCDE feeds into Z-Score priority scheduler
    """
    NCDE cognitive state feeds into the scheduling priority function:

    Priority = 0.35×Decay + 0.25×Centrality + 0.25×ProjectRelevance + 0.15×Novelty

    Where Decay is derived from NCDE retrieval probability predictions.
    """
    Given NCDE predicts for concept "C1":
      | metric                  | value |
      | retrieval_probability   | 0.45  |
      | decay_urgency           | 0.78  |
      | error_complexity        | 8     |
    And concept "C1" has:
      | metric              | value |
      | centrality_score    | 0.65  |
      | project_relevance   | 0.90  |
      | novelty_score       | 0.20  |
    When priority Z-score is calculated
    Then the weighted priority is:
      | component          | weight | value | contribution |
      | decay_urgency      | 0.35   | 0.78  | 0.273        |
      | centrality         | 0.25   | 0.65  | 0.163        |
      | project_relevance  | 0.25   | 0.90  | 0.225        |
      | novelty            | 0.15   | 0.20  | 0.030        |
      | total_priority     | -      | -     | 0.691        |

  @Regression @Scheduling @OptimalTiming
  Scenario: NCDE determines optimal review timing
    """
    Lindsey et al. (2014) showed optimal scheduling requires predicting
    future memory strength. NCDE provides this continuous-time prediction.
    """
    Given learner has stability 5.2 days for concept "C1"
    And target retention is 0.85
    When NCDE computes optimal review time
    Then the recommendation is:
      | metric              | value        |
      | optimal_delay       | 4.1 days     |
      | predicted_retention | 0.86         |
      | stability_gain      | +1.8 days    |
    And the review is scheduled within optimal window

  # ============================================================================
  # REAL-TIME ADAPTATION
  # ============================================================================

  @Smoke @RealTime @MomentumCapitalization
  Scenario: Capitalize on learning momentum
    """
    When a learner is "in flow", NCDE detects positive momentum
    and recommends continuing with related challenging material.

    Reference: Csikszentmihalyi (1990) - Flow theory
    """
    Given learner has completed 5 activities in last 15 minutes:
      | activity | correct | response_time | difficulty |
      | A1       | true    | 1200ms        | 0.6        |
      | A2       | true    | 1100ms        | 0.65       |
      | A3       | true    | 950ms         | 0.7        |
      | A4       | true    | 900ms         | 0.72       |
      | A5       | true    | 850ms         | 0.75       |
    When momentum is assessed
    Then flow state is detected:
      | metric           | value    |
      | momentum_score   | 0.85     |
      | flow_probability | 0.78     |
      | recommended_action | increase_challenge |
    And next activity difficulty is scaled up by 0.1

  @Regression @RealTime @FrustrationDetection
  Scenario: Detect and respond to frustration signals
    """
    Frustration (too hard) and boredom (too easy) both indicate
    suboptimal difficulty. NCDE detects via response patterns.
    """
    Given learner shows frustration signals:
      | signal             | value    |
      | error_rate         | 0.65     |
      | response_time_trend| increasing |
      | skip_rate          | 0.30     |
      | retry_attempts     | 3.2 avg  |
    When affective state is inferred
    Then the assessment is:
      | metric              | value       |
      | frustration_score   | 0.72        |
      | boredom_score       | 0.15        |
      | optimal_difficulty_delta | -0.15  |
      | recommended_action  | scaffold_and_simplify |
