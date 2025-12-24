@Adaptive @Metacognition @Priority-High @Regression
@Node-AdaptiveEngine @DARPA-DigitalTutor @Phase6-Sovereign
@SRL @Zimmerman @CalibrationAccuracy @DunningKruger
Feature: Metacognitive Calibration - Self-Regulated Learning System
  """
  Metacognition is "thinking about thinking." A critical skill for expert
  performance is knowing what you know and what you don't know (calibration).

  The Problem:
  - Dunning-Kruger Effect: Low performers overestimate ability
  - Imposter Syndrome: High performers underestimate ability
  - Poor metacognition leads to inefficient study strategies

  Self-Regulated Learning (Zimmerman):
  1. Forethought: Planning, goal-setting, self-motivation
  2. Performance: Self-monitoring, self-control
  3. Self-Reflection: Self-evaluation, adaptation

  Calibration Accuracy:
  CA = 1 - |Confidence - Actual_Performance|

  Perfect calibration: CA = 1.0 (confidence matches performance)
  Over-confident: Confidence > Performance
  Under-confident: Confidence < Performance

  Scientific Foundation:
  - Zimmerman (2000). Self-Regulated Learning
  - Dunning & Kruger (1999). Unskilled and Unaware
  - Azevedo & Cromley (2004). Metacognition and Hypermedia
  - Nelson & Narens (1990). Metamemory Framework

  Effect Size Enhancement: +0.2σ to +0.4σ through improved calibration
  """

  Background:
    Given the metacognitive calibration engine is active
    And prediction/judgment prompts are enabled
    And calibration history is tracked

  # ============================================================================
  # PRE-INTERACTION CALIBRATION
  # ============================================================================

  @Smoke @PreCalibration @ConfidenceJudgment
  Scenario: Pre-Interaction Calibration prompt
    """
    Before attempting an atom, the learner must predict their
    performance, exposing their metacognitive model.
    """
    Given a "Gold" atom is selected by the Dispatcher
    When the TUI displays the "Calibration Prompt"
    Then the learner must estimate:
      | prompt                                    | input_type |
      | "How confident are you? (0-100%)"         | slider     |
      | "Will you get this right on first try?"   | yes/no     |
      | "How long will this take? (seconds)"      | number     |
    And after the interaction, the system calculates:
      | metric              | formula                           |
      | calibration_error   | |confidence - actual_performance| |
      | time_estimation_error | |predicted_time - actual_time|  |

  @Regression @PreCalibration @OverConfidence
  Scenario: Detect and address overconfidence
    """
    When calibration error is consistently positive (overconfident),
    the Gøtzsche Agent provides a "Metacognitive Audit."
    """
    Given calibration history shows:
      | trial | confidence | actual | error | direction      |
      | 1     | 0.90       | 0.60   | 0.30  | overconfident  |
      | 2     | 0.85       | 0.55   | 0.30  | overconfident  |
      | 3     | 0.80       | 0.50   | 0.30  | overconfident  |
    When overconfidence pattern is detected
    Then Gøtzsche Agent intervenes:
      """
      Metacognitive Audit Alert 🔍

      Your confidence (85%) significantly exceeds your performance (55%).
      This pattern suggests overconfidence bias.

      Questions for reflection:
      - What made you so confident?
      - What signals did you miss that indicated difficulty?
      - How will you calibrate your confidence on the next attempt?
      """

  @Regression @PreCalibration @UnderConfidence
  Scenario: Detect and address underconfidence
    """
    Underconfidence (imposter syndrome) also needs correction.
    """
    Given calibration history shows:
      | trial | confidence | actual | error | direction       |
      | 1     | 0.30       | 0.85   | 0.55  | underconfident  |
      | 2     | 0.35       | 0.90   | 0.55  | underconfident  |
      | 3     | 0.40       | 0.88   | 0.48  | underconfident  |
    When underconfidence pattern is detected
    Then encouragement intervention occurs:
      """
      You're better than you think! 🌟

      Your performance (88%) far exceeds your confidence (40%).
      This is sometimes called "imposter syndrome."

      Evidence of your competence:
      - You've correctly solved 12 of last 15 problems
      - Your error rate is in the top 20% of learners
      - Trust your preparation!
      """

  # ============================================================================
  # JUDGMENT OF LEARNING (JOL)
  # ============================================================================

  @Smoke @JOL @PostLearning
  Scenario: Judgment of Learning after studying
    """
    After studying new material, learners predict future recall
    (Judgment of Learning). This is compared against actual retention.
    """
    Given learner has studied concept "VLSM"
    When study phase completes
    Then JOL prompt appears:
      | prompt                                          | input    |
      | "How well will you remember this tomorrow?"     | 0-100%   |
      | "How well will you remember this in a week?"    | 0-100%   |
    And these predictions are stored for verification
    And actual retention is measured at those intervals
    And calibration accuracy is computed

  @Regression @JOL @DelayedVerification
  Scenario: Verify JOL predictions over time
    """
    JOL predictions are verified when the interval arrives,
    providing concrete feedback on metacognitive accuracy.
    """
    Given learner predicted 1-week retention of 80%
    When 1 week has passed
    And retention test is administered
    Then actual retention is measured:
      | metric              | value |
      | actual_retention    | 0.65  |
      | predicted_retention | 0.80  |
      | JOL_error           | 0.15  |
    And feedback is provided:
      """
      Your prediction: 80% retention after 1 week
      Your actual: 65% retention
      Calibration gap: 15%

      This material decays faster than you expected.
      Consider more frequent review for similar content.
      """

  # ============================================================================
  # FEELING OF KNOWING (FOK)
  # ============================================================================

  @Smoke @FOK @RetrievalPrediction
  Scenario: Feeling of Knowing during retrieval failure
    """
    When learner can't recall an answer, they judge whether
    they would recognize it if shown (Feeling of Knowing).
    """
    Given learner fails to recall answer
    When FOK prompt appears:
      | prompt                                          | input  |
      | "Would you recognize the answer if shown?"      | yes/no |
      | "How close are you to remembering? (0-100%)"    | slider |
    Then the correct answer is shown
    And FOK accuracy is computed:
      | metric         | value | interpretation              |
      | fok_prediction | 0.80  | "I would recognize it"      |
      | recognized     | true  | actually recognized         |
      | fok_accurate   | true  | prediction matched reality  |

  @Regression @FOK @TipOfTongue
  Scenario: "Tip of tongue" states and resolution
    """
    High FOK with retrieval failure = "tip of tongue" state.
    These are especially valuable learning moments.
    """
    Given learner reports high FOK (90%) but fails retrieval
    When "tip of tongue" is detected
    Then special scaffolding is provided:
      | scaffold_type      | content                        |
      | first_letter_hint  | "The answer starts with 'E'..."  |
      | category_hint      | "It's a type of scheduling..." |
      | phonetic_hint      | "It rhymes with 'bursting'..."  |
    And when learner finally retrieves (or recognizes)
    Then the resolution strengthens memory significantly

  # ============================================================================
  # SELF-REGULATION PHASES
  # ============================================================================

  @Smoke @SRL @Forethought
  Scenario: Forethought phase - planning and goal-setting
    """
    Before each study session, prompt for planning and goals.
    """
    Given learner starts a new session
    When forethought phase is initiated
    Then learner is prompted:
      | prompt                                    | purpose           |
      | "What's your main learning goal today?"   | goal_setting      |
      | "Which concepts will you focus on?"       | planning          |
      | "How long will you study?"                | time_management   |
      | "What strategies will you use?"           | strategy_selection|
    And goals are recorded for later reflection

  @Regression @SRL @Performance
  Scenario: Performance phase - real-time self-monitoring
    """
    During learning, prompt for self-monitoring at intervals.
    """
    Given learner is mid-session (15 minutes in)
    When monitoring prompt appears
    Then learner is asked:
      | prompt                                    | purpose           |
      | "Are you making progress toward your goal?" | progress_check  |
      | "Is your strategy working?"               | strategy_check    |
      | "Do you need to adjust anything?"         | adaptation_prompt |
    And responses inform real-time adaptation

  @Regression @SRL @SelfReflection
  Scenario: Self-reflection phase - post-session evaluation
    """
    After session, prompt for self-evaluation and attribution.
    """
    Given session has ended
    When self-reflection phase is initiated
    Then learner is prompted:
      | prompt                                    | purpose           |
      | "Did you achieve your goal?"              | outcome_evaluation|
      | "What worked well?"                       | success_analysis  |
      | "What would you do differently?"          | improvement_plan  |
      | "Why do you think you performed as you did?" | attribution    |
    And reflections are stored for longitudinal analysis

  # ============================================================================
  # CALIBRATION TRAINING
  # ============================================================================

  @Smoke @CalibrationTraining @Feedback
  Scenario: Train calibration accuracy through feedback
    """
    Consistent feedback on prediction accuracy improves
    metacognitive calibration over time.
    """
    Given learner has 20+ calibration data points
    When calibration training is triggered
    Then personalized feedback is provided:
      | feedback_element          | content                          |
      | overall_calibration       | "Your CA score: 0.72 (improving!)"|
      | bias_direction            | "You tend to be 15% overconfident"|
      | domain_variation          | "Better calibrated in math than coding"|
      | improvement_trend         | "Calibration improved 12% this month"|

  @Regression @CalibrationTraining @CalibrationGraph
  Scenario: Visualize calibration over time
    """
    Show the learner their calibration curve compared to
    perfect calibration (the diagonal).
    """
    Given calibration history is available
    When calibration graph is generated
    Then the visualization shows:
      | element                  | description                      |
      | x_axis                   | confidence level (0-100%)        |
      | y_axis                   | actual accuracy (0-100%)         |
      | perfect_calibration      | 45° diagonal line                |
      | learner_curve            | actual calibration curve         |
      | overconfidence_region    | above the diagonal               |
      | underconfidence_region   | below the diagonal               |
    And gaps between learner curve and diagonal are highlighted

  # ============================================================================
  # INTEGRATION WITH MASTERY ENGINE
  # ============================================================================

  @Integration @CalibrationAccuracy @MasteryWeight
  Scenario: Calibration accuracy weights mastery claims
    """
    A learner with poor calibration has unreliable self-assessment.
    This affects how we weight their mastery claims.
    """
    Given learner has:
      | metric              | value |
      | IRT_theta           | 1.8   |
      | calibration_accuracy| 0.45  |
    When mastery confidence is computed
    Then calibration affects confidence:
      | calculation                        | result     |
      | raw_mastery_claim                  | 1.8 θ      |
      | calibration_discount               | × 0.80     |
      | adjusted_mastery_confidence        | less certain |
    And additional verification is recommended

  @Integration @CalibrationAccuracy @StudyRecommendations
  Scenario: Calibration informs study recommendations
    """
    Overconfident learners should do more retrieval practice.
    Underconfident learners need more positive feedback.
    """
    Given learner's calibration profile:
      | profile_type     | recommended_intervention          |
      | overconfident    | more retrieval practice, less re-reading |
      | underconfident   | more positive feedback, evidence review |
      | well_calibrated  | continue current approach          |
    When study recommendations are generated
    Then recommendations are personalized to calibration profile
