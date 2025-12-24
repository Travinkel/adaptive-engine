@Adaptive @ZPD @Priority-Critical @Regression
@Node-AdaptiveEngine @DARPA-DigitalTutor @Phase6-Sovereign
@Vygotsky @Scaffolding @FlowChannel @RealTimeAdaptation
Feature: Dynamic ZPD Flow-Channel Maintenance
  """
  Vygotsky's Zone of Proximal Development (ZPD) is the narrow band between
  what a learner can do independently and what they cannot do even with help.

  The Challenge:
  - Too Easy: Below ZPD → boredom, no growth
  - Too Hard: Above ZPD → frustration, learned helplessness
  - Just Right: Within ZPD → maximum learning, flow state

  Dynamic ZPD Scaling:
  Using real-time NCDE cognitive friction signals, we adjust scaffold
  density continuously to keep the learner in the "Flow Channel"
  (Csikszentmihalyi, 1990).

  Scientific Foundation:
  - Vygotsky (1978). Mind in Society
  - Wood, Bruner & Ross (1976). Scaffolding and the ZPD
  - Csikszentmihalyi (1990). Flow: The Psychology of Optimal Experience
  - Puntambekar & Hubscher (2005). Scaffolding in complex learning

  The Vygotsky Coefficient (V_c):
  V_c = (task_difficulty - current_ability) / scaffold_available

  Optimal: V_c ∈ [0.3, 0.7] (challenging but achievable with support)

  Effect Size Enhancement: +0.4σ to +0.8σ through optimal challenge
  """

  Background:
    Given the ZPD scaling engine is active
    And NCDE friction signals are streaming
    And scaffold injection/removal is real-time capable

  # ============================================================================
  # REAL-TIME SCAFFOLD INJECTION
  # ============================================================================

  @Smoke @ScaffoldInjection @RealTime
  Scenario: Real-Time Scaffold Injection on cognitive deadlock
    """
    When the learner's NCDE friction vector exceeds threshold,
    immediately inject a scaffold BEFORE frustration cascades.
    """
    Given the learner's "NCDE Friction" vector exceeds 2.5 threshold
    When the system detects a "Logic Deadlock" in the Left Pane
    Then the CLI should "Inject" a temporary scaffold into the Right Pane
    And the scaffold should be a "Faded-Parsons" version of the prerequisite node
    And once the friction vector drops below 1.5
    Then the scaffold should be "Vaporized" immediately
    And the learner continues with reduced support

  @Regression @ScaffoldInjection @FrictionSignals
  Scenario: Monitor friction signals for scaffold timing
    """
    Multiple signals indicate cognitive overload requiring intervention:
    - Response time increasing
    - Keystroke velocity decreasing
    - Pause frequency increasing
    - Error rate spiking
    """
    Given real-time monitoring shows:
      | signal              | baseline | current | trend     |
      | response_time       | 2.1s     | 4.8s    | increasing|
      | keystroke_velocity  | 120/min  | 45/min  | decreasing|
      | pause_ratio         | 0.15     | 0.52    | increasing|
      | local_error_rate    | 0.10     | 0.45    | spiking   |
    When friction is calculated
    Then:
      | metric            | value | threshold | action_needed |
      | friction_vector   | 3.2   | 2.5       | YES           |
      | frustration_prob  | 0.78  | 0.60      | YES           |
    And scaffold injection is triggered

  @Regression @ScaffoldInjection @ScaffoldTypes
  Scenario: Select appropriate scaffold type for the deadlock
    """
    Different deadlock types require different scaffold interventions.
    """
    Given a logic deadlock is detected
    When scaffold selection runs
    Then the appropriate type is chosen:
      | deadlock_type        | scaffold_type           | content                    |
      | missing_prerequisite | prerequisite_summary    | Faded version of prereq    |
      | procedural_stuck     | next_step_hint          | First step only            |
      | conceptual_confusion | analogy_scaffold        | Familiar domain mapping    |
      | syntax_error         | syntax_template         | Correct syntax structure   |
      | integration_failure  | worked_example_partial  | Similar problem solved     |

  # ============================================================================
  # SCAFFOLD FADING
  # ============================================================================

  @Smoke @ScaffoldFading @GradualRemoval
  Scenario: Gradual scaffold fading as mastery increases
    """
    Scaffolds should fade as learner becomes more capable,
    following Wood's contingent scaffolding principle.
    """
    Given learner is working on "recursion" with scaffold
    And local performance improves:
      | trial | correct | friction |
      | 1     | false   | 2.8      |
      | 2     | true    | 2.1      |
      | 3     | true    | 1.5      |
      | 4     | true    | 1.0      |
    When scaffold fading is evaluated
    Then scaffolding density decreases:
      | trial | scaffold_level | fade_percentage |
      | 1     | full           | 0%              |
      | 2     | 75%            | 25%             |
      | 3     | 50%            | 50%             |
      | 4     | 25%            | 75%             |

  @Regression @ScaffoldFading @Vaporization
  Scenario: Immediate scaffold vaporization when no longer needed
    """
    Once the learner demonstrates capability, the scaffold must
    disappear immediately to avoid dependency.
    """
    Given scaffold is currently displayed
    And learner successfully completes:
      | condition                    | met    |
      | correct_answer               | yes    |
      | without_using_scaffold_hint  | yes    |
      | friction < 1.5               | yes    |
    When vaporization check runs
    Then scaffold is immediately removed:
      | action           | timing      |
      | fade_out_visual  | 500ms       |
      | remove_content   | after fade  |
      | log_event        | "scaffold_vaporized" |

  @Regression @ScaffoldFading @PreventDependency
  Scenario: Prevent scaffold dependency formation
    """
    If a learner consistently uses scaffolds without attempting
    first, intervention prevents learned helplessness.
    """
    Given scaffold usage pattern:
      | trial | attempted_first | used_scaffold |
      | 1     | no              | yes           |
      | 2     | no              | yes           |
      | 3     | no              | yes           |
    When dependency risk is assessed
    Then intervention is triggered:
      | intervention         | description                          |
      | delay_scaffold       | 10 second delay before available     |
      | require_attempt      | Must type something before scaffold  |
      | metacognitive_prompt | "Try first - struggle is learning"   |

  # ============================================================================
  # VYGOTSKY COEFFICIENT CALCULATION
  # ============================================================================

  @Smoke @VygotskyCoefficent @Calculation
  Scenario: Calculate Vygotsky Coefficient for task calibration
    """
    V_c = (task_difficulty - current_ability) / scaffold_available

    V_c < 0.3: Too easy, increase difficulty or remove scaffold
    V_c ∈ [0.3, 0.7]: Optimal ZPD
    V_c > 0.7: Too hard, add scaffold or reduce difficulty
    """
    Given:
      | parameter           | value |
      | task_difficulty     | 0.75  |
      | current_ability     | 0.50  |
      | scaffold_available  | 0.40  |
    When V_c is calculated
    Then:
      | metric   | value | interpretation     |
      | V_c      | 0.625 | within optimal ZPD |
    And no adjustment is needed

  @Regression @VygotskyCoefficent @Adjustment
  Scenario: Adjust parameters when V_c is out of range
    """
    When V_c drifts outside optimal range, adjust dynamically.
    """
    Given V_c = 0.85 (too hard)
    When adjustment is triggered
    Then options are evaluated:
      | adjustment_option    | new_V_c | selected |
      | increase_scaffold    | 0.62    | yes      |
      | decrease_difficulty  | 0.55    | possible |
      | reduce_task_scope    | 0.48    | possible |
    And the least disruptive option is chosen (increase_scaffold)

  # ============================================================================
  # FLOW CHANNEL MAINTENANCE
  # ============================================================================

  @Smoke @FlowChannel @OptimalChallenge
  Scenario: Maintain learner in flow channel
    """
    Flow occurs when challenge matches skill level.
    The system continuously adjusts to keep learner in flow.
    """
    Given flow channel defined as:
      | axis      | low_bound | high_bound |
      | challenge | skill-0.2 | skill+0.3  |
      | skill     | measured  | measured   |
    When current state is:
      | skill  | challenge | in_flow |
      | 0.65   | 0.80      | yes     |
    Then no adjustment needed
    And flow state is maintained

  @Regression @FlowChannel @Boredom
  Scenario: Detect and correct boredom (under-challenge)
    """
    Signs of boredom: fast responses, high accuracy, low engagement.
    Intervention: increase challenge or remove scaffolds.
    """
    Given boredom signals:
      | signal           | value  | threshold |
      | response_time    | 0.8s   | < 1.5s    |
      | accuracy_streak  | 10     | > 7       |
      | engagement_proxy | 0.45   | < 0.50    |
    When boredom is detected
    Then intervention increases challenge:
      | action                | magnitude |
      | remove_scaffold       | 50%       |
      | select_harder_atom    | +0.2 b    |
      | context_jolt          | possible  |

  @Regression @FlowChannel @Anxiety
  Scenario: Detect and correct anxiety (over-challenge)
    """
    Signs of anxiety: slow responses, high error rate, friction.
    Intervention: add scaffolds, reduce difficulty.
    """
    Given anxiety signals:
      | signal           | value  | threshold |
      | response_time    | 8.2s   | > 5.0s    |
      | error_rate       | 0.55   | > 0.40    |
      | friction_vector  | 2.8    | > 2.5     |
    When anxiety is detected
    Then intervention reduces challenge:
      | action                | magnitude |
      | inject_scaffold       | level_2   |
      | select_easier_atom    | -0.15 b   |
      | encouragement_prompt  | yes       |

  # ============================================================================
  # PERSONALIZED ZPD
  # ============================================================================

  @Smoke @PersonalizedZPD @LearnerProfile
  Scenario: Personalize ZPD width based on learner profile
    """
    Some learners have wider ZPDs (tolerate more challenge/support variance).
    Others need tighter calibration.
    """
    Given learner profile:
      | trait                 | value    |
      | frustration_tolerance | high     |
      | growth_mindset        | strong   |
      | prior_domain_exposure | moderate |
    When personalized ZPD is calculated
    Then ZPD width is expanded:
      | parameter          | default | personalized |
      | zpd_width          | 0.4     | 0.6          |
      | scaffold_threshold | 2.5     | 3.2          |
    And learner can handle more challenge before intervention

  @Regression @PersonalizedZPD @ContextSpecific
  Scenario: ZPD varies by domain/concept
    """
    The same learner may have different ZPD widths in different domains
    based on their background knowledge and confidence.
    """
    Given learner's domain-specific ZPDs:
      | domain        | zpd_width | reason                    |
      | algorithms    | 0.7       | strong background         |
      | networking    | 0.3       | novice, needs support     |
      | mathematics   | 0.5       | moderate familiarity      |
    When activity in "networking" is selected
    Then scaffolding is more aggressive:
      | parameter           | networking | algorithms |
      | scaffold_threshold  | 1.8        | 3.5        |
      | scaffold_density    | high       | low        |
