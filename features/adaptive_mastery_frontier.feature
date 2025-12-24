@Sanity @Domain:Cognitive @Logic:Adaptive @Status:Stable
Feature: Adaptive Mastery Frontier & ZPD Scaling
  Goal: Dynamically adjust the learner's "Mastery Frontier" using Vygotsky's Zone of Proximal Development (ZPD) to ensure optimal challenge without overload.

  Background:
    Given the PostgreSQL Master Ledger retrieves the user's "Efficiency_Slope"
    And the current NCDE Mastery_Score for the "Target_Domain" is calculated
    And the system is in "Frontier Discovery" mode

  Scenario Outline: Accelerated Pathing on High Automaticity
    Given the user's "Retrieval Latency" is consistently below "<LatencyThreshold>"
    When the user completes a "Gold-Layer" Atom with "<Accuracy>" accuracy
    Then the system shall trigger an "Acceleration Event"
    And the roadmap shall "Bypass" the next "<NodesToSkip>" intermediate nodes
    And the system shall inject a "Platinum-tier" Challenge to verify the leap.

    Examples:
      | LatencyThreshold | Accuracy | NodesToSkip | Description                      |
      | 3s               | 100%     | 3           | Expert-tier jump                 |
      | 5s               | 95%      | 1           | Fast-track advancement           |

  Scenario: Dynamic ZPD Re-scaling on Persistent Struggle
    Given the learner's "Struggle_Count" exceeds 3 for a specific "Knowledge_Node"
    When the system detects "Cognitive Dissonance" via Keystroke Entropy
    Then the system shall "Down-scale" the ZPD by injecting "Foundational Atoms"
    And the "Mastery Frontier" shall be reset to the last "High-Confidence" node
    And the Spivak Agent shall initiate a "Conceptual Reset" dialogue.

  Scenario Outline: Socratic Depth Scaling based on Mastery
    When the user engages with an "Interactive" Atom
    Then the system shall set the "Socratic_Depth" to "<DepthLevel>"
    And the Agent shall use "<LinguisticStyle>" in its scaffolding.

    Examples:
      | MasteryScore | DepthLevel | LinguisticStyle | Description                     |
      | < 0.4        | CONCRETE   | Procedural      | Step-by-step guidance           |
      | 0.4 - 0.7    | ANALOGICAL | Comparative     | Mapping to known concepts (AToM)|
      | > 0.7        | ABSTRACT   | Philosophical   | First-principles reasoning      |

  Scenario: Mastery Hardening via Interleaved "Decay" Checks
    Given the SM-2 algorithm identifies a "Stable" concept has reached a "Decay Threshold"
    When the user is performing a "High-Flow" session in a different domain
    Then the system shall "Interleave" a "Hardening Atom" from the decaying domain
    And the "Context Switch" penalty shall be logged to NCDE.cognitive_flexibility
    And the concept's "Stability" shall be updated based on the recovery performance.