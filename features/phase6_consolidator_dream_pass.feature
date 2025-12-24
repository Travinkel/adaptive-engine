@Infrastructure @Consolidator @Priority-High @Regression
@Node-AdaptiveEngine @DARPA-DigitalTutor @Phase6-Sovereign
@SynapticConsolidation @OfflineProcessing @BackgroundAgent @DreamPass
Feature: Automated Consolidator - Background Synaptic Simulation
  """
  The Consolidator Agent runs while the learner is offline (sleeping,
  working, living), simulating the brain's synaptic consolidation process.
  It identifies weak points in the knowledge graph and prepares optimized
  "Hardening Passes" for the next session.

  Biological Basis:
  During sleep, the hippocampus "replays" experiences to the neocortex,
  strengthening important memories and pruning weak ones. The Consolidator
  simulates this process computationally.

  Consolidator Functions:
  1. Simulate forgetting (run NCDE forward with no input)
  2. Identify fragile nodes (approaching retrieval threshold)
  3. Detect schema inconsistencies
  4. Generate optimized morning "Hardening Pass"
  5. Pre-compute optimal difficulty atoms

  Scientific Foundation:
  - Diekelmann & Born (2010). Sleep and Memory Consolidation
  - Walker & Stickgold (2006). Sleep-Dependent Learning
  - McClelland et al. (1995). Complementary Learning Systems
  - Rasch & Born (2013). Memory Replay During Sleep

  Effect Size Enhancement: +0.2σ to +0.4σ through optimized next-session prep
  """

  Background:
    Given the Consolidator Agent is running as background service
    And it has access to the PostgreSQL knowledge ledger
    And NCDE model is loaded for state simulation

  # ============================================================================
  # BACKGROUND CONSOLIDATION
  # ============================================================================

  @Smoke @BackgroundProcess @IdleDetection
  Scenario: Trigger consolidation after extended idle
    """
    When the system detects the learner has been idle for
    a significant period (sleep), consolidation begins.
    """
    Given the system has been idle for > 8 hours
    When the "Consolidator Agent" awakens
    Then it scans the PostgreSQL Ledger for:
      | analysis_type           | purpose                          |
      | fragile_nodes           | nodes approaching forgetting     |
      | recent_errors           | misconceptions needing repair    |
      | schema_gaps             | incomplete knowledge structures  |
      | high_entropy_zones      | unstable knowledge clusters      |
    And analysis results are stored for morning session

  @Regression @BackgroundProcess @SimulatedDecay
  Scenario: Simulate knowledge decay during offline period
    """
    Run NCDE forward in time with no learning events to
    predict current cognitive state when learner returns.
    """
    Given learner's last known state h(t₀) at logout
    And current time is t₁ = t₀ + 9 hours
    When NCDE decay simulation runs
    Then predicted state h(t₁) is computed:
      | component              | change          |
      | retrieval_probability  | decreased       |
      | stability              | unchanged       |
      | momentum               | reset to 0      |
    And nodes below retrieval threshold are flagged

  @Regression @BackgroundProcess @ReplaySimulation
  Scenario: Simulate memory replay for consolidation
    """
    Like hippocampal replay, re-process recent learning
    to strengthen important connections.
    """
    Given recent learning session contained:
      | concept_learned    | importance_score | connections_formed |
      | VLSM_subnetting    | 0.85             | 5                  |
      | route_aggregation  | 0.72             | 3                  |
      | NAT_concepts       | 0.45             | 2                  |
    When replay simulation runs
    Then consolidation priority is computed:
      | concept            | replay_priority | reason                    |
      | VLSM_subnetting    | 1st             | high importance + connections |
      | route_aggregation  | 2nd             | builds on VLSM            |
      | NAT_concepts       | 3rd             | lower importance          |

  # ============================================================================
  # MORNING HARDENING GENERATION
  # ============================================================================

  @Smoke @MorningPass @AtomGeneration
  Scenario: Morning Hardening Generation
    """
    Generate 5 "High-Entropy" atoms targeting the weakest points
    identified during consolidation simulation.
    """
    Given the Consolidator has identified weak points:
      | weak_point           | fragility_score | recommended_action        |
      | subnet_boundaries    | 0.82            | retrieval_practice        |
      | OSPF_adjacency       | 0.75            | worked_example            |
      | TCP_windowing        | 0.68            | interleaved_review        |
    When morning atoms are generated
    Then 5 "Priority_NCDE_Stabilizers" are created:
      | atom_id | target_weakness    | atom_type       | difficulty |
      | MA-001  | subnet_boundaries  | cold_retrieval  | matched    |
      | MA-002  | OSPF_adjacency     | worked_example  | b = θ+0.2  |
      | MA-003  | TCP_windowing      | interleaved     | mixed      |
      | MA-004  | schema_integration | transfer_task   | high       |
      | MA-005  | metacognitive      | confidence_check| meta       |
    And these are queued as first activities in morning session

  @Regression @MorningPass @PersonalizedWakeUp
  Scenario: Personalized "wake-up" routine based on consolidation
    """
    The morning session starts with activities optimized by
    overnight consolidation analysis.
    """
    Given learner opens cortex-cli in the morning
    When session starts
    Then the "Wake-Up Pass" is presented:
      """
      Good morning! While you rested, your Shadow PC analyzed your knowledge.

      🧠 Consolidation Report:
      - 3 concepts approaching forgetting threshold
      - 1 schema integration opportunity detected
      - Recommended: 15-minute Hardening Pass before new content

      Start Hardening Pass? [Y/n]
      """
    And pressing Y begins the optimized hardening sequence

  @Regression @MorningPass @AdaptiveSequencing
  Scenario: Adapt morning sequence based on real-time performance
    """
    If the learner performs better/worse than predicted,
    adjust the remaining morning atoms dynamically.
    """
    Given morning pass has 5 atoms planned
    And learner completes atoms 1-2 with performance:
      | atom | predicted_score | actual_score |
      | 1    | 0.65            | 0.90         |
      | 2    | 0.70            | 0.85         |
    When real-time adaptation runs
    Then remaining atoms are adjusted:
      | adjustment_type     | reason                          |
      | skip_easy_atoms     | stronger than expected          |
      | increase_difficulty | can handle more challenge       |
      | add_transfer_task   | ready for schema work           |

  # ============================================================================
  # WEAK POINT IDENTIFICATION
  # ============================================================================

  @Smoke @WeakPointAnalysis @FragilityDetection
  Scenario: Identify fragile knowledge nodes
    """
    Fragile nodes are those where small perturbation (time, context)
    could push retrieval below threshold.
    """
    Given knowledge graph with mastery states
    When fragility analysis runs
    Then nodes are classified:
      | node_id | stability | retrieval_prob | fragility | classification |
      | N1      | 12 days   | 0.88           | 0.15      | stable         |
      | N2      | 3 days    | 0.72           | 0.45      | fragile        |
      | N3      | 1 day     | 0.58           | 0.78      | critical       |
    And critical/fragile nodes are prioritized for morning pass

  @Regression @WeakPointAnalysis @SchemaGaps
  Scenario: Detect incomplete schemas
    """
    A schema is incomplete if key relationships are missing
    or weakly established.
    """
    Given schema "networking_layers" has components:
      | component          | present | strength |
      | physical_layer     | yes     | 0.85     |
      | data_link_layer    | yes     | 0.72     |
      | network_layer      | yes     | 0.90     |
      | transport_layer    | partial | 0.45     |
      | layer_interactions | weak    | 0.35     |
    When schema completeness is analyzed
    Then gaps are identified:
      | gap_type             | location           | severity |
      | weak_component       | transport_layer    | medium   |
      | missing_relation     | layer_interactions | high     |
    And targeted atoms are generated to fill gaps

  @Regression @WeakPointAnalysis @EntropyZones
  Scenario: Identify high-entropy knowledge zones
    """
    High entropy = unstable, contradictory, or confused knowledge.
    These areas need consolidation before they corrupt adjacent nodes.
    """
    Given recent error patterns in zone "routing_protocols":
      | error_type              | frequency |
      | OSPF_EIGRP_confusion    | 5         |
      | metric_type_mixup       | 3         |
      | administrative_distance | 2         |
    When entropy analysis runs
    Then the zone is flagged:
      | zone              | entropy_score | recommendation          |
      | routing_protocols | 0.72          | contrastive_remediation |
    And the Consolidator generates contrastive comparison atoms

  # ============================================================================
  # PREDICTIVE SCHEDULING
  # ============================================================================

  @Smoke @PredictiveScheduling @OptimalTiming
  Scenario: Pre-compute optimal review times for next 24 hours
    """
    Based on decay simulation, pre-compute when each concept
    should ideally be reviewed in the next day.
    """
    Given consolidation analysis complete
    When optimal timing is computed
    Then a 24-hour schedule is generated:
      | concept            | optimal_review_time | reason                |
      | VLSM_subnetting    | +2 hours (morning)  | critical fragility    |
      | OSPF_basics        | +6 hours (afternoon)| moderate decay        |
      | TCP_three_way      | +12 hours (evening) | stable but due        |
    And the schedule is ready when learner opens app

  @Regression @PredictiveScheduling @ContextAwareness
  Scenario: Factor in learner's typical schedule
    """
    The Consolidator learns the learner's typical active hours
    and schedules optimally within those windows.
    """
    Given learner's typical schedule:
      | day_type   | active_windows          |
      | weekday    | 7-8am, 12-1pm, 8-10pm   |
      | weekend    | 9am-12pm, 4-8pm         |
    When scheduling for Monday
    Then reviews are placed in active windows:
      | review_slot | window_used | atom_count |
      | 1           | 7-8am       | 5          |
      | 2           | 12-1pm      | 3          |
      | 3           | 8-10pm      | 7          |

  # ============================================================================
  # INTEGRATION
  # ============================================================================

  @Integration @StateVector @Overnight
  Scenario: Update state vector with consolidation effects
    """
    Consolidation simulation should update the state vector
    to reflect expected state at wake-up.
    """
    Given overnight consolidation ran
    When learner's state vector is queried in morning
    Then it reflects:
      | component              | effect                    |
      | retrieval_decay        | simulated from last state |
      | consolidation_boost    | +5% for practiced items   |
      | predicted_readiness    | per-concept estimates     |

  @Integration @Notification @MorningBrief
  Scenario: Optional morning brief notification
    """
    If enabled, send a brief summary of consolidation findings
    to prompt the learner to study.
    """
    Given notifications are enabled
    When morning brief is generated
    Then notification includes:
      """
      🌅 Morning Knowledge Brief

      Overnight analysis found:
      • 2 concepts need urgent review (< 60% retention)
      • 1 schema gap detected in networking
      • Estimated optimal study time: 18 minutes

      Ready when you are!
      """
