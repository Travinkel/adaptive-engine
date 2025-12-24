@Sanity @Domain:Cognitive @Logic:Adaptive @Status:Stable
Feature: Dynamic Zone of Proximal Development (ZPD) Scaling
  Goal: Implement Vygotsky's ZPD by ensuring the learner is always challenged just beyond their current mastery, preventing both boredom and cognitive overload.

  Background:
    Given the NCDE Engine is calculating the "Mastery Frontier"
    And the Gøtzsche Agent is monitoring "Error_Magnitude" (how far the user's logic is from the Expert Model)

  Scenario Outline: Scaling Atom Complexity on Success Streaks
    Given the user is working within the "<Domain>"
    When the user completes a "Level N" Atom with "High Efficiency" (low RetrievalLatency)
    Then the system shall bypass "Level N+1" and present a "Level N+2" Challenge Atom
    And the system shall record a "ZPD_Expansion" event in the Master Ledger
    And the cortex-cli shall update the Roadmap visualization to reflect the accelerated path.

    Examples:
      | Domain               | Level N         | Level N+2 Challenge          |
      | C#_Basics            | Variables       | Control_Flow_Complex         |
      | Design_Patterns      | Singleton       | Abstract_Factory_Composition |
      | Systems_Programming  | Stack_Memory    | Heap_Fragmentation_Analysis  |

  Scenario: Intelligent Scaffolding Re-entry on ZPD Exit
    Given the user is attempting a "Frontier Atom"
    When the user's "Struggle_Signal" exceeds the "Cognitive_Overload_Threshold"
    Then the system shall dynamically inject a "Bridge Atom" (AToM)
    And the system shall provide "Temporary Scaffolding" in the Left Pane (The Book)
    And the system shall mark the current ZPD limit at the current complexity level.

  Scenario Outline: Adjusting Socratic Depth based on ZPD Position
    When the user is "<Distance>" from the "Mastery Target"
    Then the Spivak Agent shall adjust the "Question_Abstraction_Level" to "<Abstraction>"
    And the system shall set the ICAP required interaction to "<MinICAP>"

    Examples:
      | Distance | Abstraction | MinICAP      |
      | FAR      | CONCRETE    | ACTIVE       |
      | MID      | STRUCTURAL  | INTERACTIVE  |
      | NEAR     | PHILOSOPHICAL| CONSTRUCTIVE |
