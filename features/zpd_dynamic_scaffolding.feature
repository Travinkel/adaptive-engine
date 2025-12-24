@Domain-Cognitive @ZPD @Vygotsky @Scaffolding @Flow @Priority-Regression @6Sigma-Sovereign
Feature: Dynamic Zone of Proximal Development Scaling
  As an Adaptive Orchestrator
  I want to adjust scaffold density based on real-time NCDE friction
  So that learners remain in the "Flow Channel" without cognitive collapse or boredom.

  Background:
    Given Zone of Proximal Development theory (Vygotsky) is active
    And Flow Channel theory (Csikszentmihalyi) informs difficulty calibration
    And NCDE friction vector is computed in real-time
    And scaffold types are available:
      | Scaffold_Type       | Opacity_Range | Trigger_Condition            |
      | Full_Worked_Example | 100%          | Friction > 3.0 (collapse)    |
      | Faded_Parsons       | 75%           | Friction 2.5-3.0             |
      | Structural_Hint     | 50%           | Friction 2.0-2.5             |
      | First_Step_Only     | 25%           | Friction 1.5-2.0             |
      | No_Scaffold         | 0%            | Friction < 1.5               |

  # ─────────────────────────────────────────────────────────────────────────
  # Real-Time Scaffold Injection
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Smoke @ZPD @ScaffoldInjection
  Scenario: Real-time scaffold injection on cognitive friction
    Given the learner's NCDE Friction vector exceeds 2.5 threshold
    When the system detects a "Logic Deadlock" in the Left Pane
    Then the CLI should inject a temporary scaffold:
      | Action                    | Details                          |
      | Inject_Faded_Parsons      | Prerequisite node structure      |
      | Display_In_Right_Pane     | Non-intrusive overlay            |
      | Set_Decay_Timer           | Auto-fade after use              |
    And once the friction vector drops below 2.0, scaffold should "vaporize"
    And injection should feel seamless, not jarring

  @Positive @ZPD @GradualRelease
  Scenario: Gradual release of responsibility
    Given learner is working on "Recursion" concept
    And initial scaffolding is at 75%
    When performance improves over 5 interactions:
      | Interaction | Accuracy | Latency_ms | Friction |
      | 1           | 60%      | 8000       | 2.8      |
      | 2           | 70%      | 6500       | 2.4      |
      | 3           | 75%      | 5000       | 2.1      |
      | 4           | 85%      | 4000       | 1.8      |
      | 5           | 90%      | 3500       | 1.5      |
    Then scaffolding should progressively fade:
      | Interaction | Scaffold_Opacity |
      | 1           | 75%              |
      | 2           | 60%              |
      | 3           | 45%              |
      | 4           | 25%              |
      | 5           | 0%               |
    And learner should not notice explicit transitions

  @Positive @ZPD @AdaptiveExpansion
  Scenario: Expanding ZPD through successful challenge
    Given learner's current ZPD upper bound is "Medium Difficulty"
    When they successfully complete challenging items:
      | Challenge_Level | Success_Rate | Confidence |
      | Medium-High     | 85%          | Increasing |
      | High            | 75%          | Stable     |
      | Very_High       | 65%          | Stable     |
    Then ZPD should expand:
      | Metric              | Before    | After     |
      | Upper_Bound         | Medium    | High      |
      | Optimal_Difficulty  | 0.55      | 0.70      |
      | Scaffold_Threshold  | 2.0       | 2.5       |
    And learner can now handle harder material without scaffolds

  # ─────────────────────────────────────────────────────────────────────────
  # Flow Channel Maintenance
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Flow @OptimalChallenge
  Scenario: Maintaining optimal challenge-skill balance
    Given learner's skill level is tracked via IRT theta
    When item selection considers flow channel
    Then it should maintain:
      | Skill_Level (θ) | Optimal_Difficulty (b) | Flow_Zone        |
      | 0.5             | 0.6-0.8                | Challenge + 0.1-0.3|
      | 1.0             | 1.1-1.3                | Challenge + 0.1-0.3|
      | 1.5             | 1.6-1.8                | Challenge + 0.1-0.3|
    And items too easy (b < θ - 0.5) cause boredom
    And items too hard (b > θ + 0.5) cause anxiety

  @Positive @Flow @BoredomIntervention
  Scenario: Detecting and intervening on boredom
    Given learner shows boredom indicators:
      | Indicator             | Value          | Normal_Range     |
      | Response_Speed        | Very fast      | Moderate         |
      | Attention_Drift       | High           | Low              |
      | Accuracy              | 98%            | 70-85%           |
      | NCDE_Friction         | 0.5            | 1.5-2.5          |
    When boredom detection triggers
    Then system should:
      | Intervention          | Purpose                          |
      | Increase_Difficulty   | Move toward challenge zone       |
      | Inject_Novel_Topic    | Break monotony                   |
      | Remove_Scaffolding    | Force deeper engagement          |
      | Context_Jolt          | Surprise with interleaving       |
    And learner should return to flow state

  @Positive @Flow @AnxietyIntervention
  Scenario: Detecting and intervening on anxiety/frustration
    Given learner shows anxiety indicators:
      | Indicator             | Value          | Normal_Range     |
      | Response_Speed        | Very slow      | Moderate         |
      | Error_Rate            | 60%            | 15-30%           |
      | NCDE_Friction         | 3.5            | 1.5-2.5          |
      | Pause_Frequency       | High           | Moderate         |
    When anxiety detection triggers
    Then system should:
      | Intervention          | Purpose                          |
      | Inject_Scaffold       | Provide immediate support        |
      | Reduce_Difficulty     | Move toward comfort zone         |
      | Offer_Break           | Allow recovery                   |
      | Success_Guarantee     | Easy item for confidence         |
    And learner should return to flow state

  # ─────────────────────────────────────────────────────────────────────────
  # Scaffold Types and Selection
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Scaffold @TypeSelection
  Scenario: Selecting appropriate scaffold type
    Given learner is struggling with "Dynamic Programming"
    When scaffold selector evaluates the situation
    Then it should choose based on:
      | Struggle_Type         | Scaffold_Type           | Rationale           |
      | Conceptual_Gap        | Worked_Example          | Show full solution  |
      | Procedural_Stuck      | Faded_Parsons           | Partial structure   |
      | Minor_Confusion       | Structural_Hint         | Point in direction  |
      | Verification_Need     | Feedback_Only           | Confirm approach    |
    And scaffold type should match the specific difficulty

  @Positive @Scaffold @FadedWorkedExamples
  Scenario: Faded worked examples progression
    Given concept "Merge Sort" has 4 worked examples
    When learner progresses through fading sequence
    Then examples should fade:
      | Example | Completeness | Learner_Contribution      |
      | 1       | 100%         | Observe only              |
      | 2       | 75%          | Complete final quarter    |
      | 3       | 50%          | Complete second half      |
      | 4       | 25%          | Only first step provided  |
      | Test    | 0%           | Full independent solution |
    And each transition should feel achievable

  @Positive @Scaffold @JustInTime
  Scenario: Just-in-time scaffolding delivery
    Given learner is mid-problem and showing specific struggle
    When JIT scaffold system detects bottleneck
    Then it should provide targeted help:
      | Detected_Issue        | JIT_Scaffold                      |
      | Wrong_approach        | "Consider: what if you tried X?"  |
      | Missing_prerequisite  | Mini-review of prereq concept     |
      | Syntax_error          | Syntax reference popup            |
      | Logic_error           | Step-by-step trace hint           |
    And scaffold should appear at point of need, not before

  # ─────────────────────────────────────────────────────────────────────────
  # Social Scaffolding (Ghost Agents)
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Social @GhostAgents
  Scenario: Multi-perspective author debate for scaffolding
    Given learner has submitted solution for "Knuth Algorithm"
    When the "Ghost Panel" is summoned
    Then multiple expert agents should critique:
      | Agent           | Focus                    | Critique_Type        |
      | Knuth_Agent     | Algorithmic efficiency   | Performance critique |
      | Richter_Agent   | .NET implementation safety| Memory/safety critique|
      | Spivak_Agent    | Mathematical correctness | Formal verification  |
    And learner must provide "Synthesis Response" satisfying all agents
    And only then is the node marked "Elite-Hardened"
    And this represents highest-level social scaffolding

  @Positive @Social @PeerSimulation
  Scenario: Simulated peer collaboration scaffold
    Given learner is stuck on difficult problem
    When peer simulation activates
    Then simulated peer should:
      | Peer_Action           | Purpose                          |
      | Ask_Clarifying_Q      | Force learner to articulate      |
      | Suggest_Wrong_Path    | Trigger correction/teaching      |
      | Partial_Solution      | Model thinking process           |
      | Celebrate_Progress    | Motivational support             |
    And teaching others is powerful scaffold for own understanding

  # ─────────────────────────────────────────────────────────────────────────
  # ZPD Measurement and Tracking
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Measurement @ZPDBoundaries
  Scenario: Measuring ZPD boundaries dynamically
    Given learner has 30+ interactions on concept cluster
    When ZPD analyzer computes boundaries
    Then it should determine:
      | Boundary              | Difficulty | Evidence                    |
      | Lower_Bound           | 0.3        | 95% success without help    |
      | Current_Independent   | 0.55       | 80% success without help    |
      | Upper_Independent     | 0.70       | 50% success without help    |
      | Upper_Scaffolded      | 0.85       | 80% success WITH help       |
      | Frustration_Ceiling   | 1.0        | <50% success even with help |
    And ZPD width = Upper_Scaffolded - Current_Independent = 0.30

  @Positive @Measurement @ZPDGrowth
  Scenario: Tracking ZPD expansion over time
    Given learner has 3 months of learning data
    When ZPD growth analysis runs
    Then it should show:
      | Month | ZPD_Lower | ZPD_Upper | Width | Growth    |
      | 1     | 0.3       | 0.7       | 0.4   | Baseline  |
      | 2     | 0.45      | 0.85      | 0.4   | +0.15     |
      | 3     | 0.6       | 1.0       | 0.4   | +0.15     |
    And ZPD should shift upward while maintaining width
    And this indicates genuine learning progression

