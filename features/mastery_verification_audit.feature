@Sanity @Domain:Cognitive @Logic:Audit @Status:Stable
Feature: Mastery Verification & Agent Audit
  Goal: Ensure high-fidelity mastery verification via the Spivak and Gøtzsche Agents using Roslyn-based analysis and Socratic dialogue.

  Background:
    Given a Gold-Layer Atom is active in the cortex-cli
    And the Spivak Agent (Socratic Logic) is initialized
    And the Gøtzsche Agent (Expert Model) is connected to the Roslyn Analyzer

  Scenario Outline: Auditing user submission for Structural Alignment
    When the user submits a "<SubmissionType>" solution
    Then the Gøtzsche Agent shall perform a Roslyn-based semantic comparison against the Expert Model
    And the Spivak Agent shall verify the "Structural Mapping" of the solution
    And the system shall record the "Verification_Result" in the PostgreSQL Master Ledger

    Examples:
      | SubmissionType | AnalysisMethod      | SuccessCriteria                  |
      | CODE_BLOCK     | Roslyn Dataflow     | No side-effect violations         |
      | LOGIC_PROOF    | Symbolic Solver     | Axiomatic consistency            |
      | ANALOGY_MAP    | AToM Bridge Matcher | Base-to-Target mapping accuracy  |

  Scenario: Detecting a "Partial Match" with Socratic Intervention
    Given the user provides a solution that is syntactically correct but logically "Brittle"
    When the Gøtzsche Agent identifies a "Lack of Generalization"
    Then the Spivak Agent shall intercept the "Pass" signal
    And the system shall trigger a "Depth-Check" dialogue
    And the user must answer a "Why" question before the Node is marked as "Mastered"
