@priority-p1 @domain-adaptive @logic-metrics @technical-ml @linking-wo-ae-005
Feature: VOA Proficiency Metrics System
  As the adaptive engine
  I need to implement Virtual Objective Assessment metrics
  So that I can identify exactly which behaviors distinguish experts from novices

  Background:
    Given the VOA metrics system is initialized
    And behavioral telemetry collection is active

  # ═══════════════════════════════════════════════════════════════════════════
  # METRIC COLLECTION
  # ═══════════════════════════════════════════════════════════════════════════

  @collection @telemetry
  Scenario: Collect raw behavioral metrics from learning session
    """
    Adapted from NeuroVR surgical metrics to learning context.
    """
    Given an active learning session
    When I collect behavioral telemetry
    Then I should capture the following metric categories:
      | category        | metrics_count | example_metric           |
      | accuracy        | 15            | error_rate               |
      | timing          | 25            | response_latency         |
      | effort          | 20            | cognitive_friction       |
      | strategy        | 30            | path_efficiency          |
      | automaticity    | 15            | latency_variance         |
      | meta_cognition  | 10            | confidence_calibration   |
    And raw values should be normalized for comparison

  @collection @surgical-adaptation
  Scenario: Map surgical VOA metrics to learning equivalents
    Given the original NeuroVR/McGill surgical metrics
    When I adapt them for learning assessment
    Then the mapping should be:
      | surgical_metric   | learning_equivalent    | purpose            |
      | bleeding_rate     | error_rate            | accuracy           |
      | instrument_force  | cognitive_friction    | effort measurement |
      | path_efficiency   | response_optimality   | strategy quality   |
      | tremor_index      | latency_variance      | automaticity       |
      | tip_separation    | tool_switching_time   | coordination       |
      | tissue_damage     | misconception_depth   | error severity     |

  @collection @real-time
  Scenario: Update metrics in real-time during session
    Given an ongoing learning session
    And 10 learning events have occurred
    When a new event occurs with telemetry:
      | metric           | value |
      | response_time    | 2.3s  |
      | was_correct      | false |
      | confidence       | 0.8   |
      | friction_signal  | 0.65  |
    Then metrics should be updated immediately
    And running statistics should be maintained
    And anomalies should be flagged in real-time

  # ═══════════════════════════════════════════════════════════════════════════
  # METRIC SELECTION (FORWARD/BACKWARD)
  # ═══════════════════════════════════════════════════════════════════════════

  @selection @forward
  Scenario: Forward selection to identify discriminative metrics
    """
    From 270+ candidate metrics, select the minimal set that
    maximizes discrimination between skill levels.
    """
    Given 270 candidate behavioral metrics
    And labeled data with expert/intermediate/novice classifications
    When I perform forward selection
    Then I should iteratively:
      | step | action                           | criterion              |
      | 1    | Start with empty metric set      | -                      |
      | 2    | Add metric that most improves    | Δ classification_acc   |
      | 3    | Repeat until diminishing returns | Δ < 0.01               |
    And I should identify ~15-20 most discriminative metrics

  @selection @backward
  Scenario: Backward elimination to validate selection
    Given the forward-selected metric set
    When I perform backward elimination
    Then I should iteratively:
      | step | action                        | criterion            |
      | 1    | Start with full selected set  | -                    |
      | 2    | Remove least important metric | Δ accuracy < 0.005   |
      | 3    | Continue until accuracy drops | Δ accuracy >= 0.01   |
    And the final set should be stable and minimal

  @selection @cross-validation
  Scenario: Cross-validate metric selection
    Given the selected metric set
    When I perform 5-fold cross-validation
    Then each fold should achieve classification accuracy > 85%
    And the metric importance ranking should be consistent across folds

  # ═══════════════════════════════════════════════════════════════════════════
  # SVM CLASSIFICATION
  # ═══════════════════════════════════════════════════════════════════════════

  @svm @linear
  Scenario: Train linear SVM for skill classification
    """
    Linear SVM allows interpretation of θ weights to understand
    which specific behaviors contribute to skill level.
    """
    Given the selected discriminative metrics
    And labeled training data:
      | skill_level  | sample_count |
      | novice       | 500          |
      | intermediate | 300          |
      | expert       | 200          |
    When I train a linear SVM classifier
    Then the decision function should be: f(x) = Σ θᵢxᵢ + b
    And classification accuracy should be > 85%
    And θ weights should be extractable for interpretation

  @svm @multiclass
  Scenario: Handle multi-class skill classification
    Given skill levels: novice, intermediate, proficient, expert
    When I train the SVM with one-vs-rest strategy
    Then I should produce 4 binary classifiers
    And each classifier should have interpretable θ weights
    And combined predictions should be consistent

  @svm @confidence
  Scenario: Provide confidence scores for classifications
    Given a trained SVM classifier
    And a learner's metric vector
    When I classify the learner
    Then I should return:
      | output              | type    | example |
      | predicted_level     | string  | "proficient" |
      | confidence_score    | float   | 0.87 |
      | margin_distance     | float   | 1.23 |
      | probability_dist    | dict    | {novice: 0.05, ...} |

  # ═══════════════════════════════════════════════════════════════════════════
  # WEIGHT INTERPRETATION
  # ═══════════════════════════════════════════════════════════════════════════

  @interpretation @theta-weights
  Scenario: Interpret θ weights for behavioral feedback
    Given a trained linear SVM with θ weights:
      | metric              | weight  | interpretation                    |
      | error_rate          | -2.5    | High errors strongly indicate novice |
      | response_optimality | +1.8    | Efficient strategy indicates expert |
      | latency_variance    | -1.2    | Inconsistent timing indicates novice |
      | cognitive_friction  | -0.9    | High friction indicates struggle |
      | confidence_calibration | +1.5 | Well-calibrated indicates expert |
    When I analyze the weights
    Then I should identify the top behaviors distinguishing experts
    And I should generate targeted improvement recommendations

  @interpretation @contribution
  Scenario: Calculate metric contribution to skill score
    Given a learner's metric values and θ weights
    When I calculate each metric's contribution
    Then I should produce:
      | metric              | raw_value | θ_weight | contribution | rank |
      | error_rate          | 0.15      | -2.5     | -0.375       | 1    |
      | response_optimality | 0.72      | +1.8     | +1.296       | 2    |
      | latency_variance    | 0.35      | -1.2     | -0.420       | 3    |
    And the learner should see which behaviors need improvement

  @interpretation @feedback
  Scenario: Generate actionable feedback from weights
    Given a novice learner's classification
    And the top 3 behaviors dragging down their score:
      | behavior            | current | expert_range | gap   |
      | error_rate          | 0.25    | 0.05-0.10    | large |
      | response_optimality | 0.45    | 0.75-0.90    | large |
      | latency_variance    | 0.50    | 0.15-0.25    | medium|
    When I generate feedback
    Then I should produce specific recommendations:
      """
      1. Focus on accuracy: Your error rate (25%) is higher than experts (5-10%).
         Practice deliberate problem-solving before answering.
      2. Improve strategy: Your path efficiency (45%) suggests trial-and-error.
         Try planning your approach before starting.
      3. Build automaticity: Your timing varies significantly.
         Consistent practice will reduce this variation.
      """

  # ═══════════════════════════════════════════════════════════════════════════
  # INTEGRATION WITH ADAPTIVE ENGINE
  # ═══════════════════════════════════════════════════════════════════════════

  @integration @ncde
  Scenario: Feed proficiency signals to NCDE cognitive state
    Given a learner's VOA proficiency classification
    When I update the NCDE cognitive state
    Then the state vector should incorporate:
      | component           | source                  |
      | skill_level_estimate | SVM classification     |
      | behavior_strengths   | Positive θ contributions |
      | behavior_gaps        | Negative θ contributions |
      | confidence_in_estimate | SVM margin distance |

  @integration @agents
  Scenario: Inform remediation agents with VOA insights
    Given a learner classified as "struggling"
    And specific behavioral gaps identified
    When a MAARTA agent is recruited
    Then the agent should receive:
      | context                 | purpose                        |
      | specific_weak_behaviors | Target remediation accurately  |
      | θ_weight_magnitudes     | Prioritize most impactful gaps |
      | expert_benchmarks       | Set concrete improvement goals |

  @integration @scheduling
  Scenario: Adjust scheduling based on proficiency trajectory
    Given a learner's proficiency classification over time:
      | session | classification | confidence |
      | 1       | novice        | 0.92       |
      | 2       | novice        | 0.85       |
      | 3       | intermediate  | 0.68       |
      | 4       | intermediate  | 0.81       |
    When I analyze the trajectory
    Then I should detect skill level transition
    And I should adjust difficulty accordingly
    And I should update FSRS parameters for new level
