@Adaptive @PredictiveCoding @Priority-Critical @Regression
@Node-AdaptiveEngine @DARPA-DigitalTutor @Phase6-Sovereign
@Friston @ActiveInference @FreeEnergy @SynapticPlasticity
Feature: Generative Predictive Coding - Active Inference for Deep Learning
  """
  Predictive Coding (Friston, 2010) posits that the brain is fundamentally
  a "prediction machine." Learning occurs when there is Prediction Error -
  the difference between what we expected and what we observed.

  The Free Energy Principle:
  - The brain minimizes "surprise" (prediction error)
  - Learning = updating internal models to predict better
  - Maximum learning occurs at optimal surprise levels

  Implementation for Learning:
  1. Present context/header only (hide conclusion)
  2. Force learner to PREDICT the outcome
  3. Reveal truth and compute Prediction Error Density
  4. High-error predictions create maximum synaptic update

  Scientific Foundation:
  - Friston (2010). The Free Energy Principle
  - Rao & Ballard (1999). Predictive Coding in Visual Cortex
  - Clark (2013). Whatever Next? Predictive Brains
  - Keller & Mrsic-Flogel (2018). Predictive Processing

  Key Insight: "Surprise is the currency of learning."
  Maximum surprise (within tolerance) = Maximum plasticity

  Effect Size Enhancement: +0.3σ to +0.7σ through prediction-first learning
  """

  Background:
    Given the predictive coding engine ("Friston-Mode") is available
    And learner's prediction history is tracked
    And prediction error density is computed per concept

  # ============================================================================
  # PREDICTION-BASED INGESTION
  # ============================================================================

  @Smoke @PredictionFirst @ActiveInference
  Scenario: Prediction-Based Ingestion of new content
    """
    Force prediction BEFORE revealing content to maximize
    the surprise signal when truth is revealed.
    """
    Given a new "Silver" node is about to be revealed in the Right Pane
    When "Friston-Mode" is active
    Then the Right Pane displays only:
      | visible_element    | content                        |
      | context_header     | "Chapter 5: Memory Management" |
      | setup_premise      | "Given a heap with fragmentation..." |
      | hidden_conclusion  | [CONCEALED]                    |
    And the Left Pane prompts:
      """
      Predict the core relational conclusion of this axiom.
      What do you expect the solution to address?
      """
    And only after the user's "Predictive Input" is saved
    Then the Right Pane reveals the actual content
    And prediction error is computed

  @Regression @PredictionFirst @ErrorDensity
  Scenario: Calculate Prediction Error Density
    """
    Prediction Error Density measures the "surprise" level.
    Optimal surprise is neither too low (boring) nor too high (overwhelming).
    """
    Given learner predicted:
      """
      The solution involves compacting memory to reduce fragmentation.
      """
    And actual content reveals:
      """
      The solution uses a buddy allocator to prevent fragmentation.
      """
    When prediction error density is calculated
    Then the metrics are:
      | metric                | value | interpretation           |
      | semantic_distance     | 0.45  | moderate divergence      |
      | structural_alignment  | 0.62  | partially correct        |
      | surprise_level        | 0.58  | optimal learning zone    |
      | concept_gap           | ["buddy_allocator"] | novel element |
    And the high-error concept gets priority encoding

  @Regression @PredictionFirst @DifficultyAdjustment
  Scenario: Adjust difficulty based on prediction accuracy
    """
    If predictions are consistently accurate, difficulty should increase.
    If predictions are consistently wrong, difficulty should decrease.
    """
    Given prediction history shows:
      | window_size | avg_error_density | trend      |
      | last_5      | 0.25              | decreasing |
      | last_10     | 0.35              | decreasing |
    When difficulty adjustment is calculated
    Then:
      | condition              | adjustment           |
      | error_too_low (<0.3)   | increase_difficulty  |
      | error_optimal (0.3-0.6)| maintain_level       |
      | error_too_high (>0.6)  | decrease_difficulty  |
    And current case triggers: increase_difficulty

  # ============================================================================
  # HIERARCHICAL PREDICTION
  # ============================================================================

  @Smoke @Hierarchical @MultiLevel
  Scenario: Multi-level hierarchical predictions
    """
    Predictive coding operates at multiple levels:
    - Low level: Next word/symbol
    - Mid level: Next step in procedure
    - High level: Overall conclusion/pattern
    """
    Given a multi-step proof is being presented
    When hierarchical prediction mode is active
    Then predictions are requested at each level:
      | level  | prediction_prompt                        |
      | high   | "What will this proof ultimately show?"  |
      | mid    | "What technique will step 3 use?"        |
      | low    | "Complete this equation: ∫... = ___"     |
    And error is computed at each level independently
    And high-level errors indicate schema gaps
    And low-level errors indicate procedural gaps

  @Regression @Hierarchical @TopDown
  Scenario: Top-down prediction shapes perception
    """
    High-level predictions influence how lower-level content
    is interpreted (perception is prediction).
    """
    Given learner's high-level prediction: "This will use dynamic programming"
    When actual content uses "greedy algorithm"
    Then:
      | observation               | implication                    |
      | high_level_mismatch       | schema needs updating          |
      | re-parsing_required       | re-read with new lens          |
      | prediction_model_update   | when to use greedy vs DP       |
    And the prediction model is refined

  # ============================================================================
  # SURPRISE OPTIMIZATION
  # ============================================================================

  @Smoke @SurpriseOptimization @OptimalZone
  Scenario: Maintain optimal surprise level
    """
    Too little surprise = boring, no learning
    Too much surprise = overwhelming, confusion
    Optimal surprise = maximum plasticity, engaged learning
    """
    Given current surprise metrics:
      | metric           | value |
      | surprise_level   | 0.75  |
      | frustration_risk | 0.35  |
      | engagement       | 0.82  |
    When surprise optimization runs
    Then adjustments are made:
      | adjustment_type      | direction |
      | content_complexity   | decrease  |
      | scaffold_availability| increase  |
      | prediction_granularity| coarsen  |
    And surprise level returns to optimal band (0.4-0.6)

  @Regression @SurpriseOptimization @PersonalCalibration
  Scenario: Personalize surprise tolerance
    """
    Different learners have different optimal surprise levels.
    Some thrive on high surprise; others need stability.
    """
    Given learner profile indicates:
      | trait                    | value    |
      | novelty_seeking          | high     |
      | frustration_tolerance    | high     |
      | prior_expertise          | moderate |
    When personal surprise target is calibrated
    Then the target is:
      | metric                   | value |
      | optimal_surprise_target  | 0.55  |
      | tolerance_band           | ±0.15 |
    And learner can handle more surprise than default

  # ============================================================================
  # PREDICTIVE LEARNING SEQUENCES
  # ============================================================================

  @Smoke @Sequence @IncrementalReveal
  Scenario: Incremental reveal maximizes prediction opportunities
    """
    Instead of showing full content, reveal incrementally with
    prediction prompts at each stage.
    """
    Given a worked example is being presented
    When incremental reveal mode is active
    Then the sequence is:
      | step | revealed_content              | prediction_prompt           |
      | 1    | problem_statement             | "What approach would you take?"|
      | 2    | first_step + prediction       | "What comes next?"          |
      | 3    | second_step + prediction      | "What comes next?"          |
      | 4    | solution_reveal               | "Was your path correct?"    |
    And each step generates prediction error signal
    And errors guide which steps need more practice

  @Regression @Sequence @ExplicitPrediction
  Scenario: Explicit prediction verbalization
    """
    Verbalized predictions create stronger memory traces than
    implicit predictions (generation effect + prediction).
    """
    Given prediction opportunity arises
    When explicit prediction mode is active
    Then learner must TYPE their prediction:
      | element              | requirement              |
      | prediction_text      | minimum 20 characters    |
      | confidence_rating    | 1-5 scale                |
      | reasoning            | optional but encouraged  |
    And the typed prediction becomes part of learning record
    And prediction accuracy is tracked over time

  # ============================================================================
  # ERROR-DRIVEN LEARNING
  # ============================================================================

  @Smoke @ErrorDriven @HighErrorPriority
  Scenario: High prediction error concepts get priority
    """
    Concepts where prediction error is consistently high represent
    the biggest gaps in the learner's mental model.
    """
    Given prediction error history:
      | concept          | avg_error_density | review_priority |
      | heap_allocation  | 0.72              | 1st             |
      | stack_frames     | 0.45              | 3rd             |
      | memory_layout    | 0.65              | 2nd             |
    When scheduling prioritizes high-error concepts
    Then "heap_allocation" appears first in next session
    And the system targets maximum model-improvement

  @Regression @ErrorDriven @ModelUpdate
  Scenario: Prediction errors update internal model
    """
    Each prediction error is an opportunity to update the
    learner's mental model (Bayesian belief update).
    """
    Given prediction error on "garbage collection timing"
    When model update occurs
    Then:
      | update_type          | description                      |
      | concept_link_added   | GC timing ↔ reference counting   |
      | misconception_flagged| "GC runs continuously" → wrong   |
      | new_understanding    | "GC runs on allocation failure"  |
    And the updated model improves future predictions

  # ============================================================================
  # INTEGRATION WITH MASTERY ENGINE
  # ============================================================================

  @Integration @MasteryGating @PredictionAccuracy
  Scenario: Prediction accuracy gates mastery claims
    """
    You can't claim mastery if you can't predict outcomes.
    Prediction accuracy is a mastery requirement.
    """
    Given concept "C1" has:
      | metric              | value |
      | IRT_theta           | 1.8   |
      | CASI_verified       | true  |
      | prediction_accuracy | 0.45  |
    When mastery status is evaluated
    Then:
      | criterion           | status | required |
      | IRT_mastery         | PASS   | > 1.5    |
      | CASI_transfer       | PASS   | verified |
      | prediction_accuracy | FAIL   | > 0.70   |
      | final_status        | BLOCKED | need prediction improvement |
    And prediction practice is scheduled

  @Integration @NCDE @SurpriseSignal
  Scenario: Surprise signal feeds into NCDE state vector
    """
    Prediction error / surprise is a component of the cognitive
    state vector, influencing trajectory modeling.
    """
    Given prediction error event occurs
    When NCDE state is updated
    Then state vector reflects:
      | component            | change          |
      | surprise_accumulator | +0.12           |
      | learning_rate_local  | increased       |
      | attention_allocation | toward_gap_area |
