@Domain-Cognitive @CFT @CrissCrossing @Spiro @Priority-Regression @6Sigma-Sovereign
Feature: Cognitive Flexibility Theory Criss-Crossing
  As a Senior Learning Scientist
  I want to present concepts through multiple expert perspectives and entry points
  So that learners avoid reductive biases and develop multidimensional mental models.

  Background:
    Given Cognitive Flexibility Theory (CFT) is active (Spiro et al.)
    And the knowledge graph supports multiple "Expert Lenses":
      | Lens_ID       | Expert_Agent  | Domain_Focus                    |
      | Spivak_Lens   | Spivak_Agent  | Formal mathematical rigor       |
      | Richter_Lens  | Richter_Agent | Systems/implementation safety   |
      | Knuth_Lens    | Knuth_Agent   | Algorithmic efficiency          |
      | Gotzsche_Lens | Gotzsche_Agent| Evidence-based skepticism       |
      | King_Lens     | King_Agent    | Narrative/communication craft   |
    And the Landscape Explorer mode is available
    And case studies are tagged by domain for cross-application

  # ─────────────────────────────────────────────────────────────────────────
  # Multi-Perspective Criss-Crossing
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Smoke @CFT @MultiPerspective
  Scenario Outline: Multi-perspective criss-crossing through expert lenses
    Given learner has mastered a concept's base logic via "<InitialAgent>"
    When the "Landscape-Explorer" mode is activated
    Then the CLI should re-render the concept using the "<SecondaryAgent>" lens
    And the Right Pane should display a case study from "<CaseStudyDomain>"
    And the Left Pane should require a "Re-Representation" of the logic
    And mastery is only confirmed when both perspectives are reconciled

    Examples:
      | InitialAgent  | SecondaryAgent | CaseStudyDomain            |
      | Spivak_Agent  | Richter_Agent  | Memory Allocation Bounds   |
      | Knuth_Agent   | Gotzsche_Agent | Algorithmic Bias in Medicine|
      | Richter_Agent | Spivak_Agent   | Functional Lambda Calculus |
      | King_Agent    | Knuth_Agent    | Technical Writing Efficiency|
      | Gotzsche_Agent| King_Agent     | Communicating Uncertainty  |

  @Positive @CFT @LandscapeNavigation
  Scenario: Navigating conceptual landscape from multiple entry points
    Given concept "Recursion" exists in knowledge graph
    And it has multiple entry point paths:
      | Entry_Point         | Domain_Frame          | First_Concept_Encountered |
      | Mathematical        | Proof_By_Induction    | Base_Case_Axiom           |
      | Programming         | Function_Calls        | Stack_Frame_Mechanics     |
      | Linguistic          | Self_Reference        | Chomsky_Grammars          |
      | Visual              | Fractal_Patterns      | Self_Similarity           |
    When learner selects "Programming" entry point
    And later revisits via "Mathematical" entry point
    Then CFT engine should:
      | Action                      | Purpose                          |
      | Track_Entry_Points_Used     | [Programming, Mathematical]      |
      | Highlight_New_Connections   | Induction ↔ Stack unwinding      |
      | Require_Integration_Task    | "Map proof base case to return"  |
    And mastery requires traversing ≥ 3 entry points

  @Positive @CFT @CaseLibrary
  Scenario: Building a case library for ill-structured domains
    Given domain "Software Architecture" is marked as ill-structured
    When CFT case library is constructed
    Then it should include diverse cases:
      | Case_ID | Context                  | Relevant_Concepts           |
      | ARCH_01 | Microservices_Migration  | Coupling, Cohesion, Latency |
      | ARCH_02 | Monolith_Scaling         | Database_Sharding, Caching  |
      | ARCH_03 | Event_Sourcing_Adoption  | CQRS, Eventual_Consistency  |
      | ARCH_04 | Legacy_Integration       | Anti_Corruption_Layer       |
    And each case should be linkable to multiple concepts
    And learners should encounter same concept across different cases

  # ─────────────────────────────────────────────────────────────────────────
  # Reductive Bias Prevention
  # ─────────────────────────────────────────────────────────────────────────

  @Negative @CFT @ReductiveBias
  Scenario: Detecting and preventing reductive understanding
    Given learner consistently approaches "Sorting" only via efficiency lens
    And they ignore:
      | Ignored_Perspective   | Relevant_Consideration           |
      | Memory_Usage          | In-place vs auxiliary space      |
      | Stability             | Equal element ordering           |
      | Parallelizability     | Merge sort vs quicksort          |
    When the bias detector analyzes learning patterns
    Then it should:
      | Action                    | Details                          |
      | Flag_Narrow_Perspective   | Only efficiency considered       |
      | Inject_Contrasting_Case   | "Sort 10M records with 1GB RAM"  |
      | Require_Multi_Criteria    | Compare on 3+ dimensions         |
    And the learner must demonstrate broader understanding

  @Positive @CFT @PerspectiveTracking
  Scenario: Tracking perspective coverage across concepts
    Given learner has studied 20 concepts in "Data Structures"
    When perspective coverage analysis runs
    Then it should report:
      | Perspective       | Coverage | Gap_Areas                  |
      | Efficiency        | 95%      | -                          |
      | Memory            | 60%      | Trees, Graphs              |
      | Real_World_App    | 45%      | Most concepts              |
      | Historical_Context| 20%      | Nearly all concepts        |
    And recommend: "Explore real-world applications and historical development"

  # ─────────────────────────────────────────────────────────────────────────
  # Flexible Knowledge Assembly
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @CFT @KnowledgeAssembly
  Scenario: Assembling knowledge for novel problem
    Given learner encounters unprecedented problem:
      | Problem              | "Design a rate limiter for distributed API" |
      | Required_Concepts    | [Counting, Time_Windows, Distributed_State] |
      | Cross_Domain_Links   | Algorithms, Distributed_Systems, Networking |
    When CFT assembly assistant activates
    Then it should guide:
      | Step | Action                          | Purpose                    |
      | 1    | Identify relevant concepts      | Build concept inventory    |
      | 2    | Map cross-domain connections    | Find structural bridges    |
      | 3    | Present multiple solution frames| Leaky bucket, sliding window|
      | 4    | Require trade-off analysis      | Choose with justification  |
    And novel problem-solving demonstrates genuine flexibility

  @Positive @CFT @TransferReadiness
  Scenario: Assessing transfer readiness via CFT metrics
    Given learner completes CFT-enhanced curriculum
    When transfer readiness is assessed
    Then metrics should include:
      | Metric                    | Value | Threshold | Status    |
      | Entry_Points_Navigated    | 4.2   | 3.0       | PASS      |
      | Perspectives_Integrated   | 3.8   | 3.0       | PASS      |
      | Novel_Assembly_Success    | 78%   | 70%       | PASS      |
      | Reductive_Bias_Score      | 0.15  | 0.30      | PASS      |
    And learner is certified as "Flexible Thinker" for domain

