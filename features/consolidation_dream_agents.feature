@Domain-Cognitive @Consolidation @Sleep @DreamAgents @Synaptic @Priority-Regression @6Sigma-Sovereign
Feature: Automated Consolidation and Dream Agent Processing
  As a Sovereign Intelligence System
  I want expert agents to "dream" (simulate) knowledge states while learner is offline
  So that the next session is optimized for the most fragile nodes in memory.

  Background:
    Given Synaptic Consolidation Theory is active
    And the Consolidator Agent has access to the PostgreSQL Master Ledger
    And idle detection triggers after > 6 hours of inactivity
    And dream simulation uses lightweight inference:
      | Simulation_Type   | Purpose                              |
      | Weak_Node_Scan    | Identify fragile memories            |
      | Interference_Test | Find confusion-prone pairs           |
      | Decay_Projection  | Predict what will be forgotten       |
      | Priority_Ranking  | Order morning atoms by urgency       |

  # ─────────────────────────────────────────────────────────────────────────
  # Background Synaptic Consolidation
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Smoke @Consolidation @OfflineProcessing
  Scenario: Morning hardening generation from overnight consolidation
    Given the system has been idle for > 8 hours (learner sleeping)
    When the Consolidator Agent scans the PostgreSQL Ledger
    Then it should generate 5 "High-Entropy" atoms:
      | Priority | Target_Concept      | Weakness_Type         | Recommended_Atom    |
      | 1        | Recursion           | Recent decay (3 days) | Retrieval_Practice  |
      | 2        | Hash_Tables         | High interference     | Discrimination_Task |
      | 3        | Async_Patterns      | Low mastery + priority| Socratic_Dialogue   |
      | 4        | Binary_Search       | Overconfidence signal | Prediction_Challenge|
      | 5        | Graph_Traversal     | Fragile encoding      | Multi_Modal_Review  |
    And these atoms should be tagged as "Priority_NCDE_Stabilizers"
    And they should appear first in morning TUI boot

  @Positive @Consolidation @WeakNodeIdentification
  Scenario: Identifying weak nodes via simulated retrieval
    Given 100 concepts have mastery data
    When overnight weak node scan runs
    Then it should simulate retrieval for each:
      | Concept             | Simulated_Retrieval | Fragility_Score |
      | Sorting_Algorithms  | Success (0.95)      | 0.05 (Strong)   |
      | Dynamic_Programming | Partial (0.60)      | 0.40 (Moderate) |
      | Red_Black_Trees     | Failure (0.30)      | 0.70 (Weak)     |
      | Monads              | Failure (0.20)      | 0.80 (Critical) |
    And concepts with fragility > 0.50 flagged for review
    And fragility based on: decay, stability, last accuracy

  @Positive @Consolidation @InterferencePrediction
  Scenario: Predicting interference between similar concepts
    Given concepts have embedding similarity data
    When interference analyzer runs overnight
    Then it should identify confusion-prone pairs:
      | Concept_A         | Concept_B           | Similarity | Interference_Risk |
      | BFS               | DFS                 | 0.92       | HIGH              |
      | Stack             | Queue               | 0.88       | MODERATE          |
      | Merge_Sort        | Quick_Sort          | 0.85       | MODERATE          |
      | Binary_Tree       | BST                 | 0.90       | HIGH              |
    And high-interference pairs should receive discrimination atoms
    And morning session should include comparative tasks

  # ─────────────────────────────────────────────────────────────────────────
  # Dream Agent Simulation
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Dream @SimulatedPractice
  Scenario: Agents "practicing" on learner's behalf
    Given learner's knowledge graph has 50 active concepts
    When dream simulation runs 1000 simulated retrievals
    Then it should:
      | Simulation_Action         | Output                           |
      | Random_Retrieval_Attempts | Success/failure per concept      |
      | Decay_Application         | Time-based forgetting applied    |
      | Interference_Events       | Similar concepts confused        |
      | Strengthen_Estimation     | Which retrievals would strengthen|
    And simulation results guide morning priorities
    And no actual learner effort required

  @Positive @Dream @ForgettingProjection
  Scenario: Projecting forgetting curves into the future
    Given concept "Graph Algorithms" has:
      | Metric        | Value     |
      | Last_Review   | 5 days ago|
      | Stability     | 7 days    |
      | Current_Retention| 0.71   |
    When dream agent projects 48 hours forward
    Then it should calculate:
      | Projection        | Value     | Formula               |
      | Retention_Now     | 0.71      | e^(-5/7)              |
      | Retention_48h     | 0.54      | e^(-7/7)              |
      | Review_Urgency    | HIGH      | Below 0.60 threshold  |
    And schedule morning review before further decay

  @Positive @Dream @ConsolidationPath
  Scenario: Generating optimal consolidation learning path
    Given 10 concepts need overnight consolidation
    When path optimizer runs
    Then it should order by:
      | Order | Concept             | Priority_Factor                  |
      | 1     | High_Decay_Risk     | Prevent forgetting first         |
      | 2     | High_Interference   | Resolve confusion early          |
      | 3     | Prerequisite_Gaps   | Foundation before advanced       |
      | 4     | Goal_Aligned        | User's stated learning goal      |
      | 5     | Variety_Balance     | Interleave domains               |
    And morning path should take ~20 minutes

  # ─────────────────────────────────────────────────────────────────────────
  # Spaced Repetition Integration
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Spaced @OptimalReviewTime
  Scenario: Calculating optimal review timing overnight
    Given 200 cards have FSRS scheduling data
    When overnight scheduler recalculates
    Then it should identify:
      | Review_Urgency    | Card_Count | Criteria                   |
      | Overdue           | 15         | Due date < today           |
      | Due_Today         | 25         | Due date = today           |
      | Due_Tomorrow      | 18         | Due date = tomorrow        |
      | Preemptive        | 12         | High-value, nearly due     |
    And morning session should start with overdue items
    And preemptive reviews prevent future overdue buildup

  @Positive @Spaced @StabilityAnalysis
  Scenario: Analyzing stability distribution overnight
    Given all concepts have stability values
    When stability analysis runs
    Then it should identify:
      | Stability_Range   | Count | Recommendation                 |
      | < 3 days          | 20    | Needs intensive review         |
      | 3-7 days          | 35    | Standard review cycle          |
      | 7-30 days         | 80    | Maintenance mode               |
      | 30+ days          | 65    | Occasional check-ins           |
    And low-stability concepts get priority in morning queue

  # ─────────────────────────────────────────────────────────────────────────
  # Morning Boot Sequence
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Morning @BootSequence
  Scenario: Morning TUI boot with consolidation results
    Given learner opens cortex-cli after overnight processing
    When morning boot sequence runs
    Then it should display:
      | Boot_Phase        | Content                              |
      | Consolidation_Summary | "Analyzed 100 concepts overnight"  |
      | Priority_Items    | "5 critical reviews identified"      |
      | Predicted_Session | "~25 min to clear priority queue"    |
      | Motivation_Quote  | Domain-relevant inspiration          |
    And learner can accept or customize morning plan
    And "Start Priority Review" should be prominent

  @Positive @Morning @PriorityReview
  Scenario: Executing priority review from consolidation
    Given 5 priority atoms were generated overnight
    When learner accepts priority review
    Then review session should:
      | Phase             | Content                              |
      | 1. High_Decay     | Retrieval for decaying concepts      |
      | 2. Interference   | Discrimination for confusable pairs  |
      | 3. Weak_Encoding  | Multi-modal review for fragile nodes |
      | 4. Prediction_Test| Calibration for overconfident areas  |
      | 5. Celebration    | Summary of consolidation success     |
    And session should feel like "tuning up" memory

  # ─────────────────────────────────────────────────────────────────────────
  # Adaptive Consolidation Triggers
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Adaptive @DynamicTriggers
  Scenario: Adapting consolidation based on learner patterns
    Given learner usage data shows:
      | Pattern               | Value                    |
      | Avg_Session_Gap       | 18 hours                 |
      | Morning_Preference    | 7-8 AM                   |
      | Session_Length        | 25 minutes               |
    When consolidation scheduler adapts
    Then it should:
      | Adaptation            | Details                          |
      | Run_Consolidation_At  | 5 AM (2 hours before session)    |
      | Generate_Items_For    | 25 min session                   |
      | Optimize_For          | Morning cognitive state          |
    And consolidation should be ready before learner wakes

  @Positive @Adaptive @NoActivityHandling
  Scenario: Handling extended learner absence
    Given learner has been inactive for 7 days
    When consolidation runs daily during absence
    Then it should:
      | Day | Action                                    |
      | 1-3 | Standard consolidation, queue growing     |
      | 4-5 | Flag extended absence, preserve critical  |
      | 6-7 | Generate "Re-entry Plan" for return       |
    And on return, provide gentle re-onboarding
    And not overwhelm with 7 days of accumulated reviews

  # ─────────────────────────────────────────────────────────────────────────
  # Consolidation Effectiveness Tracking
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Effectiveness @MeasuringImpact
  Scenario: Measuring consolidation effectiveness
    Given learner has used consolidation for 30 days
    When effectiveness analysis runs
    Then it should compute:
      | Metric                      | With_Consolidation | Without | Impact |
      | Morning_Retrieval_Success   | 85%                | 68%     | +25%   |
      | Interference_Errors         | 8%                 | 22%     | -64%   |
      | Overdue_Card_Rate           | 5%                 | 18%     | -72%   |
      | Time_To_Solid_State         | 45 days            | 60 days | -25%   |
    And consolidation clearly improves retention
    And feedback: "Overnight processing boosted retention 25%"

