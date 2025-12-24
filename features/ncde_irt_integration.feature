@Domain-Cognitive @DARPA-DigitalTutor @NCDE @IRT @Priority-Regression @Mastery
Feature: NCDE-IRT Integrated Mastery Estimation
  As the Adaptive Engine
  I want to combine NCDE continuous dynamics with IRT discrete estimation
  So that learner ability is tracked with both granularity and stability for 2.0σ+ effect sizes.

  Background:
    Given the 3-parameter IRT model is active with parameters (a, b, c)
    And the NCDE model tracks continuous state evolution via dθ_t = f(θ_t, X_t)dX_t
    And the FSRS scheduler provides review intervals
    And the PostgreSQL ncde_ledger is initialized
    And the retrieval latency threshold is 2500ms for "Solid-State" detection

  # ─────────────────────────────────────────────────────────────────────────
  # IRT Ability Estimation (Discrete Updates)
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Smoke @IRT @AbilityEstimation
  Scenario: Bayesian ability update after correct response
    Given a learner with current ability estimate:
      | Metric          | Value |
      | Theta           | 0.5   |
      | Standard_Error  | 0.3   |
    And they answer an item with IRT parameters:
      | Parameter | Value | Description          |
      | a         | 1.2   | Discrimination       |
      | b         | 1.0   | Difficulty           |
      | c         | 0.25  | Guessing (4 options) |
    When they respond correctly
    Then the Bayesian update should yield:
      | Metric          | Before | After  | Change   |
      | Theta           | 0.5    | 0.62   | +0.12    |
      | Standard_Error  | 0.3    | 0.25   | -0.05    |
    And confidence should increase by approximately 17%
    And the update should be logged to ncde_ledger

  @Positive @IRT @IncorrectResponse
  Scenario: Bayesian ability update after incorrect response
    Given a learner with current ability estimate Theta = 0.7
    And they answer an easy item (b = -0.5) incorrectly
    When the Bayesian update executes
    Then Theta should decrease significantly (unexpected error on easy item)
    And the system should flag potential:
      | Flag_Type        | Reason                              |
      | Careless_Error   | High ability + easy item + wrong    |
      | Knowledge_Gap    | Possible misconception              |

  @Positive @IRT @ItemCalibration
  Scenario: Automatic item difficulty calibration
    Given a Gold atom served 100+ times with response data
    When the IRT calibration engine analyzes responses
    Then it should update item parameters:
      | Parameter | Initial | Calibrated | Method                |
      | a         | 1.0     | 1.35       | Marginal MLE          |
      | b         | 0.5     | 0.72       | Empirical difficulty  |
      | c         | 0.25    | 0.22       | Low-ability asymptote |
    And items with poor fit (RMSEA > 0.08) should be flagged for review

  # ─────────────────────────────────────────────────────────────────────────
  # NCDE Continuous State Tracking
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @NCDE @ContinuousTracking
  Scenario: Continuous state evolution during study session
    Given a learner in a 30-minute study session
    And the NCDE model receives behavioral stream:
      | Timestamp | Event           | Latency_ms | Precision |
      | 0:02      | MCQ_Correct     | 2500       | 1.0       |
      | 0:05      | Parsons_Partial | 8000       | 0.7       |
      | 0:09      | Socratic_Fail   | 12000      | 0.2       |
      | 0:15      | MCQ_Correct     | 3500       | 1.0       |
      | 0:20      | MCQ_Correct     | 2200       | 1.0       |
    When the NCDE computes trajectory using controlled differential equation
    Then it should detect state variables:
      | State_Variable | Trend      | Value_at_0:20 | Alert              |
      | Theta_dot      | Recovering | +0.05/min     | -                  |
      | Precision_avg  | Improving  | 0.78          | -                  |
      | Fatigue_index  | Increasing | 0.45          | Monitor            |
    And interventions should be triggered if trends are negative

  @Positive @NCDE @IrregularSampling
  Scenario: Handling irregular time intervals between interactions
    Given behavioral data with irregular timestamps:
      | Event    | Time_Since_Last |
      | Response | 30 seconds      |
      | Response | 5 minutes       |
      | Response | 45 seconds      |
      | Response | 10 minutes      |
    When the NCDE processes the irregular stream
    Then it should use controlled differential equation:
      "dθ_t = f(θ_t, NCDE_Metrics_t) dX_t"
    And the model should handle gaps gracefully
    And longer gaps should decay toward prior estimate

  @Positive @NCDE @Gradients
  Scenario: Computing mastery gradients for adaptive decisions
    Given accumulated NCDE trajectory for concept "Recursion"
    When the gradient analyzer computes dθ/dt
    Then it should return:
      | Metric              | Value  | Interpretation                |
      | dTheta/dt           | +0.02  | Positive learning rate        |
      | d²Theta/dt²         | -0.001 | Decelerating (approaching max)|
      | Time_to_Mastery_Est | 45 min | Estimated at current rate     |

  # ─────────────────────────────────────────────────────────────────────────
  # Solid-State Knowledge Detection
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @SolidState @LatticeHardening
  Scenario: Detecting Solid-State mastery for scaffold fading
    Given a learner has completed 50+ interactions on concept "Binary Search"
    And retrieval latency history shows:
      | Interaction_Range | Avg_Latency_ms | Accuracy |
      | 1-10              | 8500           | 0.60     |
      | 11-25             | 5200           | 0.75     |
      | 26-40             | 3100           | 0.88     |
      | 41-50             | 2300           | 0.95     |
    When the NCDE stability detector analyzes the trajectory
    Then it should detect:
      | State         | Criteria_Met                              |
      | Solid-State   | dθ/dt → 0 AND Latency ≤ 2500ms           |
    And trigger Adaptive_Fading event:
      | Action                | Value          |
      | Scaffold_Opacity      | 0% (fade out)  |
      | Hint_Availability     | Disabled       |
      | Difficulty_Multiplier | 1.5x           |

  @Positive @SolidState @Phases
  Scenario Outline: Knowledge state phase transitions
    Given a learner's knowledge state metrics:
      | Metric         | Value         |
      | Theta          | <Theta>       |
      | Retrieval_ms   | <Latency>     |
      | dTheta/dt      | <Gradient>    |
    When the phase classifier evaluates
    Then it should classify as "<Phase>" with scaffold opacity <Opacity>

    Examples:
      | Theta | Latency | Gradient | Phase      | Opacity |
      | 0.2   | 10000   | +0.05    | Gaseous    | 100%    |
      | 0.4   | 6000    | +0.03    | Liquid     | 75%     |
      | 0.6   | 4000    | +0.01    | Crystalline| 50%     |
      | 0.8   | 2500    | +0.005   | Solid      | 25%     |
      | 0.95  | 1800    | 0.0      | Solid-State| 0%      |

  # ─────────────────────────────────────────────────────────────────────────
  # Integrated NCDE-IRT Inference
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Integration @HybridModel
  Scenario: Hybrid NCDE-IRT inference for robust estimation
    Given:
      - IRT provides discrete ability estimate: θ_IRT = 0.65 (SE = 0.15)
      - NCDE provides continuous trajectory: θ_NCDE = 0.72 (based on recent session)
    When the hybrid model combines estimates
    Then it should use weighted combination:
      | Source     | Weight | Contribution |
      | IRT        | 0.6    | Historical stability |
      | NCDE       | 0.4    | Recent dynamics      |
    And final estimate: θ_hybrid ≈ 0.68
    And confidence interval should be narrower than either alone

  @Positive @Integration @ConflictResolution
  Scenario: Resolving IRT-NCDE conflicts
    Given:
      - IRT estimate: θ = 0.75 (based on 200 responses)
      - NCDE trajectory: θ declining sharply in current session
    When conflict detection identifies divergence > 0.2
    Then the system should:
      | Action                    | Rationale                       |
      | Flag_Session_Anomaly      | Possible fatigue or distraction |
      | Increase_NCDE_Weight      | Recent data more relevant       |
      | Recommend_Session_Break   | If fatigue indicators high      |

  # ─────────────────────────────────────────────────────────────────────────
  # Forgetting Curve Integration
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Forgetting @Decay
  Scenario: Modeling memory decay with exponential forgetting
    Given a concept was last reviewed 7 days ago
    And at last review: θ = 0.85, stability = 14 days
    When the forgetting model computes current retention
    Then it should apply: R = e^(-t/s) where t=7, s=14
    And current retention estimate: R ≈ 0.61
    And adjusted ability: θ_adjusted = θ × R ≈ 0.52
    And the concept should be scheduled for review

  @Positive @Forgetting @StabilityGrowth
  Scenario: Stability growth after successful review
    Given a concept with:
      | Metric     | Value  |
      | Stability  | 10 days|
      | Difficulty | 0.3    |
    When the learner successfully recalls after 10 days
    Then stability should grow:
      | New_Stability | Formula                      |
      | 15-20 days    | S_new = S × (1 + growth_rate)|
    And growth rate depends on difficulty (harder = more growth)

  # ─────────────────────────────────────────────────────────────────────────
  # Adaptive Scheduling Outputs
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Scheduling @NextItem
  Scenario: Selecting next item based on integrated model
    Given a pool of 50 available atoms for concept cluster "Algorithms"
    And learner's current state from NCDE-IRT model
    When the scheduler selects next item
    Then selection should optimize:
      | Criterion              | Weight | Rationale                    |
      | Information_Gain       | 0.4    | Reduce uncertainty in θ      |
      | Mastery_Progress       | 0.3    | Move toward learning goal    |
      | Cognitive_Load_Fit     | 0.2    | Match current fatigue level  |
      | Variety/Interleaving   | 0.1    | Desirable difficulty benefit |
    And selected item should have appropriate IRT parameters for current θ
