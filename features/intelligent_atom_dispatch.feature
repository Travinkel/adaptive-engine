@Smoke @Domain:CLI @Domain:Cognitive @Logic:Positive @Status:Stable
Feature: Adaptive Atom Dispatcher
  Goal: Manage Cognitive Load (CLT) by matching the TUI complexity to the user's current mastery level.

  Background:
    Given a PostgreSQL Master Ledger connection is established
    And the user's NCDE (Normalized Cognitive Diagnostic Estimate) is retrieved via MCP
    And the current terminal session is initiated in "Dual-Pane" mode

  Scenario Outline: Dispatching UI components based on Atom Type and Mastery
    When the system retrieves a Gold-Layer Atom of type "<AtomType>"
    Then the cortex-cli shall render the "<PaneLeft>" component for the Source material
    And the cortex-cli shall render the "<PaneRight>" component for the Interactive Workspace
    And the system shall set ICAP_Mode to "<ICAP>"

    Examples:
      | AtomType      | PaneLeft       | PaneRight         | ICAP         | Description                     |
      | BRIDGE_ATOM   | Base_Domain    | Target_Mapping    | INTERACTIVE  | Analogical mapping (AToM)       |
      | SYNTAX_ATOM   | Documentation  | REPL_Buffer       | CONSTRUCTIVE | Active coding exercise          |
      | PROOF_ATOM    | Axiom_Set      | Logic_Constructor | CONSTRUCTIVE | Formal proof generation         |
      | CONCEPT_MAP   | Node_Graph     | Graph_Editor      | ACTIVE       | Manipulating architectural nodes|
