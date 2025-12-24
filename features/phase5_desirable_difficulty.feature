@Adaptive @DesirableDifficulty @Priority-Critical @Regression
@Node-AdaptiveEngine @DARPA-DigitalTutor @Phase5-Advanced
@Bjork @Interleaving @Spacing @RetrievalPractice
Feature: Bjork's Desirable Difficulty - Interleaving and Strategic Struggle
  """
  Desirable Difficulties (Bjork, 1994) are learning conditions that appear
  to slow acquisition but enhance long-term retention and transfer.

  The Paradox:
  - Easy learning leads to fast forgetting
  - Difficult learning leads to durable retention
  - "Storage Strength" vs "Retrieval Strength" distinction

  Key Desirable Difficulties:
  1. Spacing: Distribute practice over time (vs massed)
  2. Interleaving: Mix different topics (vs blocked)
  3. Testing Effect: Retrieve rather than re-study
  4. Generation: Produce answers rather than recognize
  5. Variation: Practice in varied conditions

  Scientific Foundation:
  - Bjork & Bjork (1992). New Theory of Disuse
  - Rohrer & Taylor (2007). Interleaving benefits
  - Karpicke & Roediger (2008). Testing effect
  - Kornell & Bjork (2008). Learning vs Performance

  The "Bjork Timer":
  When performance is TOO good (high accuracy streak), the system
  deliberately disrupts with a "Context Jolt" to prevent fluency illusion.

  Effect Size Enhancement: +0.5σ to +1.0σ through strategic difficulty
  """

  Background:
    Given the Bjork difficulty engine is active
    And the learner has a session history for spacing calculations
    And interleaving entropy is being tracked

  # ============================================================================
  # INTERLEAVING AND CONTEXT JOLTS
  # ============================================================================

  @Smoke @Interleaving @ContextJolt
  Scenario: Interleaved Switch - The Context Jolt
    """
    When a learner is in a high-accuracy streak (too easy), disrupt
    with a completely different domain to prevent false fluency.
    """
    Given I am in a high-accuracy streak for "Calculus" (8 correct in row)
    When the "Bjork-Timer" expires (90 seconds of easy success)
    Then the CLI should perform a "Context Jolt"
    And the Right Pane should immediately switch to ".NET Garbage Collection"
    And the Left Pane should present a "Cold Retrieval" atom with zero hints
    And the learner experiences productive confusion (desirable)

  @Regression @Interleaving @EntropyThreshold
  Scenario: Calculate interleaving entropy for jolt timing
    """
    Interleaving entropy measures how "blocked" recent practice has been.
    Low entropy (same topic repeatedly) triggers intervention.
    """
    Given recent practice history:
      | trial | domain        |
      | 1     | calculus      |
      | 2     | calculus      |
      | 3     | calculus      |
      | 4     | calculus      |
      | 5     | calculus      |
    When interleaving entropy is calculated
    Then entropy is LOW (blocked practice detected):
      | metric              | value   | threshold |
      | interleaving_entropy| 0.15    | < 0.40    |
      | blocked_run_length  | 5       | > 3       |
    And context jolt is scheduled within 2 trials

  @Regression @Interleaving @OptimalMix
  Scenario: Maintain optimal interleaving ratio
    """
    Research suggests ~3:1 ratio of related to unrelated items
    maximizes both learning and engagement.
    """
    Given target interleaving ratio is 3:1
    When session planning occurs
    Then the schedule interleaves:
      | position | topic_relatedness |
      | 1        | primary           |
      | 2        | primary           |
      | 3        | primary           |
      | 4        | UNRELATED_JOLT    |
      | 5        | primary           |
      | ...      | ...               |
    And jolts are unpredictable within the 3:1 ratio band

  # ============================================================================
  # SCAFFOLDING REMOVAL
  # ============================================================================

  @Smoke @ScaffoldRemoval @TotalRemoval
  Scenario: Deliberate Scaffolding Removal for hardened nodes
    """
    Once a node is Silver-Hardened, remove ALL scaffolding to force
    pure retrieval from memory (testing effect).
    """
    Given a node has achieved a "Silver-Hardened" state
    When the session starts and this node is selected
    Then the Right Pane should be "Blank" (Total Scaffolding Removal)
    And the Left Pane displays only: "Reproduce the axiomatic core"
    And the user must type the core axiom to unlock any scaffold
    And only after successful retrieval does content appear

  @Regression @ScaffoldRemoval @GradualFading
  Scenario: Gradual scaffold fading based on mastery
    """
    Scaffolding is faded progressively, not removed all at once,
    following the I+1 principle (just beyond current ability).
    """
    Given concept "recursion" at mastery level 0.65
    When scaffold level is determined
    Then scaffolding density matches mastery:
      | mastery_level | scaffold_components_hidden |
      | 0.0 - 0.3     | 0% (full scaffold)         |
      | 0.3 - 0.5     | 25%                        |
      | 0.5 - 0.7     | 50%                        |
      | 0.7 - 0.85    | 75%                        |
      | 0.85 - 1.0    | 100% (no scaffold)         |

  @Regression @ScaffoldRemoval @RetrievalFirst
  Scenario: Attempt retrieval before showing answer (testing effect)
    """
    The testing effect shows retrieval attempt > restudying.
    Force retrieval attempt before any hint is available.
    """
    Given a flashcard is presented
    When "Retrieval First" mode is active
    Then the answer is hidden for minimum 10 seconds
    And hint button is disabled during this window
    And retrieval attempt must be typed before reveal
    And even WRONG attempts benefit memory (retrieval effort)

  # ============================================================================
  # SPACING OPTIMIZATION
  # ============================================================================

  @Smoke @Spacing @OptimalInterval
  Scenario: Calculate optimal spacing interval
    """
    Spacing interval depends on:
    - Target retention rate
    - Current stability (FSRS S)
    - Time since last review

    Optimal interval maximizes long-term retention per review.
    """
    Given concept with:
      | metric         | value    |
      | stability_days | 8.5      |
      | target_retention | 0.85   |
    When optimal interval is calculated
    Then the recommendation is:
      | metric              | value       |
      | optimal_delay       | 6.2 days    |
      | expected_retention  | 0.86        |
      | stability_gain      | +2.1 days   |
    And reviewing earlier wastes effort (massed practice)
    And reviewing later risks forgetting (too spaced)

  @Regression @Spacing @ExpandingIntervals
  Scenario: Expanding retrieval intervals over time
    """
    Each successful retrieval at the right difficulty increases
    the next interval (expanding retrieval practice).
    """
    Given successful retrieval history:
      | review_number | interval_days | correct |
      | 1             | 1             | yes     |
      | 2             | 3             | yes     |
      | 3             | 7             | yes     |
      | 4             | 14            | yes     |
    When next interval is calculated
    Then interval expansion continues:
      | metric           | value    |
      | next_interval    | 28 days  |
      | expansion_ratio  | 2.0x     |
    And the pattern follows power law of practice

  # ============================================================================
  # GENERATION EFFECT
  # ============================================================================

  @Smoke @Generation @ActiveProduction
  Scenario: Generate answers rather than recognize them
    """
    The generation effect: producing information from memory
    strengthens traces more than recognition/selection.
    """
    Given a concept review is due
    When activity type is selected
    Then generative activities are preferred:
      | activity_type    | generation_level | preference |
      | free_recall      | high             | 1st choice |
      | cued_recall      | medium           | 2nd choice |
      | short_answer     | medium           | 3rd choice |
      | multiple_choice  | low              | last resort|
    And MCQ is only used for initial exposure or assessment

  @Regression @Generation @ErrorCorrection
  Scenario: Errors during generation enhance learning
    """
    Counterintuitively, generating wrong answers before seeing
    the correct answer enhances eventual learning (hypercorrection).
    """
    Given learner generates incorrect answer with high confidence
    When the correct answer is revealed
    Then the "hypercorrection" effect is triggered:
      | metric               | value  |
      | confidence_before    | 0.85   |
      | answer_correct       | false  |
      | surprise_signal      | high   |
      | memory_boost         | +40%   |
    And high-confidence errors are best learning opportunities

  # ============================================================================
  # VARIATION
  # ============================================================================

  @Smoke @Variation @ContextVariety
  Scenario: Practice in varied conditions
    """
    Varying practice conditions (context, format, framing) prevents
    context-dependent learning and enables flexible retrieval.
    """
    Given concept "subnet calculation" to practice
    When variation is applied
    Then practice conditions vary:
      | trial | context_variation                    |
      | 1     | classroom-style word problem         |
      | 2     | network diagram with subnets         |
      | 3     | troubleshooting scenario             |
      | 4     | design specification format          |
      | 5     | exam-style multiple choice           |
    And same underlying skill, different surface features

  @Regression @Variation @TransferPreparation
  Scenario: Variation prepares for transfer
    """
    Varied practice creates more generalizable representations
    that transfer to novel situations (like CASI requires).
    """
    Given learner practiced "recursion" with variation:
      | variation_count | transfer_test_score |
      | 1 (blocked)     | 45%                 |
      | 3               | 62%                 |
      | 5               | 78%                 |
    When transfer to new domain is tested
    Then high variation predicts better transfer
    And this prepares learner for CASI verification

  # ============================================================================
  # BJORK TIMER IMPLEMENTATION
  # ============================================================================

  @Smoke @BjorkTimer @Monitoring
  Scenario: Bjork Timer monitors for easy streaks
    """
    The Bjork Timer watches for signs that learning is "too easy":
    - High accuracy streak
    - Fast response times
    - Low cognitive friction

    When triggered, it forces desirable difficulty.
    """
    Given real-time monitoring is active
    When the following pattern is detected:
      | metric              | value  | threshold |
      | accuracy_streak     | 8      | > 5       |
      | avg_response_time   | 1.2s   | < 2.0s    |
      | cognitive_friction  | 0.15   | < 0.30    |
    Then Bjork Timer triggers "Difficulty Injection":
      | intervention          | probability |
      | context_jolt          | 0.5         |
      | scaffold_removal      | 0.3         |
      | generation_switch     | 0.2         |

  @Regression @BjorkTimer @MetaLearning
  Scenario: Learner understands difficulty is productive
    """
    Metacognitive insight: Learners must understand that struggle
    is productive, not a sign of failure.
    """
    Given a context jolt causes temporary performance drop
    When metacognitive prompt appears
    Then the system explains:
      """
      Your accuracy dropped because we introduced a challenge.
      This temporary difficulty STRENGTHENS long-term memory.
      Easy practice feels good but leads to fast forgetting.
      """
    And learner's metacognitive calibration improves
