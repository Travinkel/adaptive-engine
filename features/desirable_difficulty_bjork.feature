@Domain-Cognitive @DesirableDifficulty @Bjork @Interleaving @Spacing @Priority-Regression @6Sigma-Sovereign
Feature: Desirable Difficulty - Interleaving and Spacing
  As a Rigorous Learning Engineer
  I want to intentionally make learning harder in the short term
  So that encoding is forced into deep, long-term storage via Bjork's desirable difficulties.

  Background:
    Given Desirable Difficulty theory (Bjork) is active
    And the Bjork Timer monitors performance streaks
    And interleaving parameters are configured:
      | Parameter              | Value | Description                    |
      | Context_Jolt_Threshold | 85%   | Accuracy before forcing switch |
      | Min_Streak_Length      | 5     | Items before jolt eligible     |
      | Interleave_Ratio       | 0.30  | 30% of items from other domains|
      | Spacing_Multiplier     | 2.5   | Interval growth factor         |
    And scaffolding removal triggers are defined

  # ─────────────────────────────────────────────────────────────────────────
  # Context Jolts (Interleaving)
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Smoke @Interleaving @ContextJolt
  Scenario: Interleaved context jolt during high-accuracy streak
    Given learner is in a "Calculus" session with 90% accuracy over 7 items
    When the Bjork Timer expires (context jolt threshold reached)
    Then the CLI should perform a "Context Jolt":
      | Action                    | Details                          |
      | Switch_Domain             | Calculus → .NET Garbage Collection|
      | Remove_Scaffolding        | Right pane goes blank            |
      | Present_Cold_Retrieval    | Zero hints available             |
    And after the jolt, return to Calculus with refreshed encoding
    And log: "Context Jolt applied - interleaving for retention"

  @Positive @Interleaving @HyperInterleave
  Scenario: Hyper-interleaving across fundamentally different domains
    Given learner is in "Spivak Calculus" session with high accuracy
    When "Interleaving Entropy" threshold is reached
    Then the CLI should "Hot-Swap" to a distant domain:
      | Source_Domain    | Target_Domain         | Structural_Challenge      |
      | Spivak_Calculus  | Stephen_King_Writing  | Map rate of change to tension|
      | Knuth_Algorithms | Medical_Diagnosis     | Map divide-conquer to DDx |
      | .NET_Memory      | Music_Theory          | Map GC cycles to rhythm   |
    And the Left Pane should present a "Rhythmic Isomorphism" atom
    And learner must identify the cross-domain structural pattern

  @Positive @Interleaving @MixedPractice
  Scenario: Mixed practice within same domain
    Given learner studying "Sorting Algorithms"
    When mixed practice mode is active
    Then items should interleave:
      | Item | Algorithm    | Type     | Prediction |
      | 1    | QuickSort    | MCQ      | -          |
      | 2    | MergeSort    | Parsons  | -          |
      | 3    | HeapSort     | MCQ      | -          |
      | 4    | QuickSort    | Trace    | -          |
      | 5    | InsertionSort| MCQ      | -          |
      | 6    | MergeSort    | MCQ      | -          |
    And NO blocked practice (all QuickSort then all MergeSort)
    And discrimination learning should be enhanced

  # ─────────────────────────────────────────────────────────────────────────
  # Scaffolding Removal
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Scaffolding @DeliberateRemoval
  Scenario: Deliberate scaffolding removal for hardened nodes
    Given a concept has achieved "Silver-Hardened" state
    When the study session starts
    Then scaffolding should be removed:
      | Removal_Level   | Right_Pane_State        | Hint_Availability |
      | Full            | Completely blank        | None              |
    And the learner must type the "Axiomatic Core" to unlock scaffold
    And this tests true retrieval without recognition cues

  @Positive @Scaffolding @GradualFading
  Scenario: Gradual scaffolding fading based on performance
    Given learner has 70% accuracy on concept "Binary Search"
    When fading algorithm calculates scaffold level
    Then it should apply:
      | Accuracy_Range | Scaffold_Level | Hint_Types_Available      |
      | 0-50%          | Full (100%)    | All hints, worked examples|
      | 50-70%         | Moderate (50%) | Structural hints only     |
      | 70-85%         | Minimal (25%)  | First-step hints only     |
      | 85-95%         | Near-None (10%)| Emergency hints only      |
      | 95%+           | None (0%)      | No scaffolding            |
    And fading should be per-concept, not global

  @Positive @Scaffolding @RetrievalPractice
  Scenario: Forced retrieval practice before scaffold access
    Given learner requests hint for "Recursion" concept
    When retrieval gate is active
    Then system should require:
      | Step | Challenge                         | Purpose             |
      | 1    | "State the recursive definition"  | Active recall       |
      | 2    | "What is the base case?"          | Core understanding  |
    And only after retrieval attempt, show scaffold
    And partial retrieval should still strengthen memory

  # ─────────────────────────────────────────────────────────────────────────
  # Spacing Effects
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Spacing @OptimalIntervals
  Scenario: Calculating optimal spacing intervals
    Given concept "Hash Tables" was successfully reviewed
    And current stability = 10 days
    When FSRS scheduler calculates next interval
    Then it should apply spacing multiplier:
      | Factor              | Value | Impact                     |
      | Base_Stability      | 10d   | Foundation                 |
      | Success_Bonus       | 1.3x  | Successful retrieval       |
      | Difficulty_Adjust   | 0.9x  | Moderate difficulty        |
      | Spacing_Multiplier  | 2.5x  | Bjork desirable difficulty |
    And next review should be in: 10 × 1.3 × 0.9 × 2.5 ≈ 29 days
    And longer spacing increases retention strength

  @Positive @Spacing @ExpandingRetrieval
  Scenario: Expanding retrieval practice schedule
    Given a new concept "Graph Traversal" is learned
    When expanding retrieval schedule is generated
    Then review intervals should follow:
      | Review | Interval | Cumulative_Days |
      | 1      | 1 day    | 1               |
      | 2      | 3 days   | 4               |
      | 3      | 7 days   | 11              |
      | 4      | 14 days  | 25              |
      | 5      | 30 days  | 55              |
      | 6      | 60 days  | 115             |
    And each successful review extends next interval
    And lapses reset to shorter interval

  # ─────────────────────────────────────────────────────────────────────────
  # Generation Effect
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Generation @ActiveProduction
  Scenario: Requiring generation over recognition
    Given multiple question types are available:
      | Type            | Generation_Required |
      | MCQ             | Low                 |
      | Fill_In_Blank   | Medium              |
      | Free_Response   | High                |
      | Teach_Back      | Very High           |
    When desirable difficulty scheduler selects
    Then it should prefer higher generation items:
      | Learner_State      | Preferred_Type     | Reason              |
      | Novice             | MCQ + Fill_In      | Build foundation    |
      | Intermediate       | Fill_In + Free     | Increase difficulty |
      | Advanced           | Free + Teach_Back  | Maximum generation  |
    And generation effect enhances long-term retention

  @Positive @Generation @ErrorfulLearning
  Scenario: Allowing productive errors during generation
    Given learner is attempting difficult generation task
    When they produce an incorrect response
    Then system should:
      | Action                    | Details                          |
      | NOT_Immediately_Correct   | Allow error to register          |
      | Record_Error_Type         | Track misconception pattern      |
      | Provide_Delayed_Feedback  | After retrieval attempt complete |
      | Explain_Correction        | Detailed remediation             |
    And productive failure leads to better final learning
    And error → correction sequence strengthens memory

  # ─────────────────────────────────────────────────────────────────────────
  # Bjork Timer Management
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @BjorkTimer @Configuration
  Scenario: Configuring Bjork Timer based on learner profile
    Given learner profile indicates:
      | Trait               | Value    |
      | Tolerance_For_Difficulty | Medium |
      | Domain_Breadth      | Narrow   |
      | Learning_Goal       | Deep Mastery |
    When Bjork Timer is configured
    Then parameters should be:
      | Parameter              | Default | Adjusted | Reason                 |
      | Context_Jolt_Threshold | 85%     | 80%      | Earlier interleaving   |
      | Min_Streak_Length      | 5       | 4        | More frequent jolts    |
      | Interleave_Ratio       | 0.30    | 0.25     | Moderate cross-domain  |
    And timer should adapt as learner tolerance builds

  @Positive @BjorkTimer @FatigueAwareness
  Scenario: Suspending desirable difficulty during fatigue
    Given NCDE detects high cognitive fatigue (friction > 3.0)
    When Bjork Timer would normally trigger context jolt
    Then it should:
      | Decision              | Reason                           |
      | Delay_Context_Jolt    | Fatigue makes difficulty harmful |
      | Reduce_Interleaving   | Temporarily ease cognitive load  |
      | Maintain_Spacing      | Spacing still beneficial         |
    And resume desirable difficulty after fatigue recovery
    And log: "Bjork difficulty suspended due to fatigue"

