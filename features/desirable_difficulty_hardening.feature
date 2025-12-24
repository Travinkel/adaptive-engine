@Sanity @Domain:Cognitive @Logic:Adaptive @Status:Stable
Feature: Desirable Difficulty & Mastery Hardening
  Goal: Implement Bjork's "Desirable Difficulty" by introducing productive struggle through interleaved practice and spacing to ensure long-term retention.

  Background:
    Given the PostgreSQL Master Ledger contains the user's historical performance
    And the SM-2+ Algorithm is monitoring concept stability
    And the NCDE (Normalized Cognitive Diagnostic Estimate) tracks "Struggle Efficiency"

  Scenario Outline: Injecting Interleaved Practice to prevent "Illusion of Competence"
    Given the user has achieved a Mastery_Score > 0.9 in a "Blocked Practice" session for "<PrimaryConcept>"
    When the system selects the next Gold-Layer Atom
    Then the system shall inject a "Challenge Atom" for "<InterleavedConcept>"
    And the cortex-cli shall display a "Context Switch" notification
    And the system shall measure the "Recovery Latency" to assess structural robustness.

    Examples:
      | PrimaryConcept      | InterleavedConcept | Reason                                     |
      | Async_Await         | Thread_Pool_Safety | Force retrieval of underlying concurrency   |
      | DependencyInjection | Interface_Segregation| Break the pattern-matching habit          |
      | Memory_Management   | Pointer_Arithmetic | Re-anchor abstraction in physical logic    |

  Scenario: Dynamic Spacing Adjustment based on "Desirable Difficulty"
    When a user fails an atom due to "Conceptual Decay"
    But the "Struggle_Signal" indicates "High Cognitive Effort"
    Then the system shall NOT reset the concept stability to zero
    But the system shall apply a "Hardening Multiplier" to the next spacing interval
    And the Spivak Agent shall provide a "Metacognitive Prompt" instead of a structural hint.

  Scenario: High-Speed Retrieval Hardening
    Given the user is in a "Flow State" (low KeystrokeEntropy, low Latency)
    When the system detects 5 consecutive correct atoms
    Then the system shall decrease the "Allowed_Response_Window" by 15%
    And the system shall switch the Right Pane to "Restricted_Buffer" (disabling autocomplete)
    And the ICAP mode shall transition to "CONSTRUCTIVE_HARDENED".
