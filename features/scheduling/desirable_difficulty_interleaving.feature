@Adaptive @DARPA-DigitalTutor @NCDE-Metrics @Logic-Positive @Priority-Smoke @Domain-Scheduling
Feature: Desirable Difficulty through Spacing and Interleaving
  As a Cognitive Engineer
  I want the CLI to schedule and interleave learning atoms based on the "Desirable Difficulty" framework
  So that long-term retention is maximized even if short-term performance is lower.

  Background:
    Given the PostgreSQL ledger "astartes_kernel" contains historical performance metrics
    And the "Hardening" scheduler is active

  @Positive @Interleaving
  Scenario Outline: Interleaving Related Concepts
    Given the user has recently engaged with node "<Primary_Node>"
    When the system selects the next Gold Atom
    Then it should prioritize an atom from node "<Interleaved_Node>" which is structurally related but distinct
    And the CLI should display a "Context_Switch" notification to prepare the learner's cognitive state.

    Examples:
      | Primary_Node         | Interleaved_Node        | Relation Type       |
      | Calculus_Chain_Rule  | Calculus_Product_Rule   | Procedural Variation|
      | C#_Async_Await       | C#_Thread_Pooling       | Resource Management |
      | Linear_Algebra_Basis | Linear_Algebra_Span     | Structural Dependency|

  @Positive @SpacedRepetition
  Scenario: Spaced Retrieval for Hardening
    Given a node "Memory_Management" has a mastery level of "Silver"
    And the last retrieval was 3 days ago
    When the user starts a session
    Then the "Spaced_Repetition_Engine" should inject a "High_Difficulty" retrieval atom for "Memory_Management"
    And the Right Pane should be hidden to enforce "Effortful Retrieval."
