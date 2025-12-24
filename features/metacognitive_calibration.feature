@Domain-Cognitive @Metacognition @SRL @Zimmerman @Calibration @Priority-Regression @6Sigma-Sovereign
Feature: Metacognitive Calibration and Self-Regulated Learning
  As an AI Thought Partner
  I want learners to predict their performance before attempting atoms
  So that the metacognitive gap can be measured and narrowed for Ramanujan-tier self-awareness.

  Background:
    Given Self-Regulated Learning theory (Zimmerman) is active
    And the metacognition tracker is initialized
    And the Gotzsche Agent is available for metacognitive audits
    And calibration metrics are stored in PostgreSQL:
      | Metric                | Description                          |
      | Predicted_Confidence  | Learner's self-estimated success %   |
      | Actual_Performance    | Binary or graded outcome             |
      | Calibration_Error     | |Predicted - Actual|                 |
      | Overconfidence_Bias   | Predicted > Actual pattern           |
      | Underconfidence_Bias  | Predicted < Actual pattern           |

  # ─────────────────────────────────────────────────────────────────────────
  # Pre-Interaction Calibration
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Smoke @Calibration @PreInteraction
  Scenario: Pre-interaction confidence prediction
    Given a "Gold" atom is selected by the Dispatcher
    When the TUI displays the "Calibration Prompt"
    Then the learner must:
      | Step | Prompt                                    | Input_Type      |
      | 1    | "How confident are you? (0-100%)"         | Slider or number|
      | 2    | "What might be challenging?"              | Optional text   |
    And after the interaction, the system should calculate:
      | Metric              | Formula                           |
      | Calibration_Error   | |Confidence - Actual_Score|       |
      | Direction           | Over or Under confident           |
    And if error > 30%, the Gotzsche Agent should provide metacognitive audit

  @Positive @Calibration @Granular
  Scenario: Granular calibration by question type
    Given learner has 50+ calibration data points
    When calibration analysis runs by question type
    Then it should report:
      | Question_Type   | Avg_Prediction | Avg_Actual | Bias_Direction |
      | MCQ             | 78%            | 82%        | Underconfident |
      | Parsons         | 85%            | 65%        | Overconfident  |
      | Free_Response   | 60%            | 55%        | Well_Calibrated|
      | Socratic        | 70%            | 45%        | Overconfident  |
    And Parsons and Socratic should receive calibration intervention
    And feedback: "You tend to overestimate on construction tasks"

  @Positive @Calibration @DomainSpecific
  Scenario: Domain-specific calibration tracking
    Given learner studies multiple domains
    When domain-wise calibration analysis runs
    Then it should identify:
      | Domain              | Calibration_Score | Interpretation          |
      | Data_Structures     | 0.85              | Well calibrated         |
      | Algorithms          | 0.62              | Needs improvement       |
      | System_Design       | 0.45              | Significantly miscalibrated|
    And recommend: "More reflection needed in System_Design"

  # ─────────────────────────────────────────────────────────────────────────
  # Metacognitive Interventions
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Intervention @GotzscheAudit
  Scenario: Gotzsche Agent metacognitive audit
    Given learner has high calibration error (> 30%) on recent items
    When Gotzsche Agent audit triggers
    Then it should present:
      | Audit_Component       | Content                              |
      | Evidence_Review       | "You predicted 90%, scored 55%"      |
      | Pattern_Analysis      | "This is 4th time in Algorithms"     |
      | Cognitive_Bias_Check  | "Possible Dunning-Kruger effect"     |
      | Reflection_Prompt     | "What made you so confident?"        |
      | Remediation_Task      | Accuracy prediction drill            |
    And learner must complete reflection to continue
    And future predictions should improve

  @Positive @Intervention @RetrospectiveCalibration
  Scenario: Retrospective calibration training
    Given calibration error is consistently high
    When retrospective training activates
    Then it should present:
      | Phase | Activity                              | Purpose                |
      | 1     | Review 5 recent errors                | Build awareness        |
      | 2     | Identify common error patterns        | Pattern recognition    |
      | 3     | Predict on similar items              | Immediate practice     |
      | 4     | Compare prediction vs outcome         | Feedback loop          |
    And training should continue until calibration improves by 20%

  @Positive @Intervention @ConfidenceAnchoring
  Scenario: Providing calibration anchors for better prediction
    Given learner struggles with absolute confidence estimation
    When anchoring assistance is enabled
    Then the prompt should include:
      | Anchor_Level | Description                          | Example_Reference |
      | 90-100%      | "Would bet money, rarely wrong"      | 2+2=4 confident   |
      | 70-89%       | "Pretty sure, might miss details"    | Capital of France |
      | 50-69%       | "Coin flip, genuinely uncertain"     | Complex proof step|
      | Below 50%    | "Guessing, more likely wrong"        | Unfamiliar topic  |
    And anchors improve calibration accuracy by ~15%

  # ─────────────────────────────────────────────────────────────────────────
  # Self-Regulated Learning Cycle
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @SRL @ForethoughtPhase
  Scenario: Forethought phase - goal setting and planning
    Given learner begins a study session
    When SRL forethought phase activates
    Then it should guide:
      | Step | Prompt                                    | Input                  |
      | 1    | "What's your learning goal today?"        | Topic or skill         |
      | 2    | "How much time do you have?"              | Duration               |
      | 3    | "Rate your current mastery (1-10)"        | Self-assessment        |
      | 4    | "What strategies will you use?"           | [Retrieval, Elaboration]|
    And session should be planned around stated goals
    And goals should be revisited in reflection phase

  @Positive @SRL @PerformancePhase
  Scenario: Performance phase - self-monitoring during study
    Given learner is in active study session
    When SRL performance monitoring is active
    Then system should track:
      | Monitor                 | Frequency | Action_If_Triggered        |
      | Attention_Drift         | 30 sec    | Gentle refocus prompt      |
      | Difficulty_Mismatch     | Per item  | Adjust difficulty          |
      | Strategy_Adherence      | 5 min     | "Still using retrieval?"   |
      | Time_Management         | 10 min    | Progress toward goal check |
    And learner can request self-monitoring reports

  @Positive @SRL @ReflectionPhase
  Scenario: Reflection phase - post-session self-evaluation
    Given study session is ending
    When SRL reflection phase activates
    Then it should guide:
      | Reflection_Question           | Purpose                        |
      | "Did you meet your goal?"     | Goal attainment assessment     |
      | "What was challenging?"       | Difficulty awareness           |
      | "What strategy worked best?"  | Strategy evaluation            |
      | "What would you do differently?"| Adaptive adjustment         |
    And learner responses should inform next session planning
    And reflection quality should improve over time

  # ─────────────────────────────────────────────────────────────────────────
  # Predictive Coding Integration (Friston Active Inference)
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Friston @PredictionError
  Scenario: Active inference prediction before content reveal
    Given a new "Silver" concept is about to be revealed
    When "Friston Mode" (Active Inference) is active
    Then the Right Pane should display only the "Context/Header"
    And the Left Pane should prompt:
      | Prompt                                    | Input_Type |
      | "Predict the core conclusion of this axiom" | Free text |
    And only AFTER prediction is saved, the Right Pane reveals truth
    And system calculates "Prediction Error Density":
      | Metric                  | Formula                          |
      | Prediction_Error        | Semantic distance to truth       |
      | Surprise_Signal         | -log(P(prediction|context))      |
      | Learning_Potential      | High surprise = high learning    |

  @Positive @Friston @ErrorDrivenScheduling
  Scenario: Scheduling based on prediction error history
    Given learner has prediction error history for concepts
    When scheduling algorithm considers prediction errors
    Then it should prioritize:
      | Concept              | Avg_Prediction_Error | Priority_Boost |
      | Dynamic_Programming  | 0.75                 | +30%           |
      | Binary_Search        | 0.20                 | +5%            |
      | Sorting_Basics       | 0.10                 | +0%            |
    And high prediction error concepts get more practice
    And surprising concepts create strongest memories

  @Positive @Friston @PredictiveProcessing
  Scenario: Building predictive mental models
    Given learner has studied "Algorithm Complexity"
    When encountering new algorithm
    Then system should:
      | Phase                 | Action                              |
      | Prime_Prediction      | "Based on structure, what's Big-O?" |
      | Reveal_Answer         | Show actual complexity              |
      | Calculate_Error       | Measure prediction accuracy         |
      | Update_Model          | Strengthen correct predictions      |
    And prediction accuracy should improve with practice

  # ─────────────────────────────────────────────────────────────────────────
  # Metacognitive Progress Tracking
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Progress @CalibrationImprovement
  Scenario: Tracking metacognitive improvement over time
    Given learner has 6 months of calibration data
    When progress analysis runs
    Then it should show:
      | Period          | Calibration_Score | Trend       |
      | Month 1         | 0.45              | Baseline    |
      | Month 2         | 0.52              | +7%         |
      | Month 3         | 0.58              | +6%         |
      | Month 4         | 0.65              | +7%         |
      | Month 5         | 0.72              | +7%         |
      | Month 6         | 0.78              | +6%         |
    And celebrate: "Metacognitive calibration improved 73%!"
    And well-calibrated learners achieve better outcomes

  @Positive @Progress @ExpertCalibration
  Scenario: Reaching expert-level metacognitive calibration
    Given learner achieves calibration score > 0.85 consistently
    When expert calibration is detected
    Then system should:
      | Action                    | Details                          |
      | Reduce_Calibration_Prompts| Every 5th item vs every item     |
      | Award_Achievement         | "Metacognitive Master" badge     |
      | Enable_Self_Scheduling    | Trust learner's judgment more    |
    And expert calibration indicates "knowing what you know"

  @Positive @Progress @DunningKrugerDetection
  Scenario: Detecting Dunning-Kruger effect patterns
    Given learner is new to domain "Machine Learning"
    When calibration patterns show:
      | Phase               | Confidence | Actual_Performance | Gap    |
      | Early_Exposure      | 85%        | 35%                | +50%   |
      | Partial_Learning    | 70%        | 55%                | +15%   |
      | Deeper_Learning     | 60%        | 65%                | -5%    |
      | Expertise           | 75%        | 80%                | -5%    |
    Then system should recognize Dunning-Kruger curve
    And provide feedback: "Initial overconfidence is normal - you're now more accurately calibrated"

