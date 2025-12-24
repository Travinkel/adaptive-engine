@Sanity @Domain:CLI @Domain:Cognitive @Logic:Adaptive @Status:Stable
Feature: Mastery-Driven Pane Fading
  Goal: Eliminate Extraneous Load by dynamically removing the "Safety Net" as the user gains automaticity.

  Background:
    Given the user is in a "Harden Node" cycle
    And the NCDE Mastery_Score for the current concept is > 0.85

  Scenario: Retrieval Latency indicates automaticity
    When the user completes 3 consecutive Atoms with a latency < 5s
    Then the cortex-cli shall trigger the "Fade Effect" on the Left Pane (The Book)
    And the opacity of the documentation shall decrease by 25%
    And the system shall transition from "Active" to "Constructive" mode (Zero-Docs).

  Scenario Outline: Re-asserting the Pane on Decay Detection
    Given the SM-2 algorithm identifies a "<Concept>" has decayed
    When the user attempts a "Challenge Atom" for that concept
    And the user's "Struggle_Signal" is detected via cortex-cli
    Then the system shall restore the Left Pane to "<Visibility>"
    And the roadmap shall inject an "Interleaved Practice" task.

    Examples:
      | Concept            | Visibility | Reason                          |
      | DependencyInjection| 100%       | Fundamental Misconception       |
      | AsyncStreams       | 50%        | Minor Retrieval Delay           |
      | MemoryLayout       | 100%       | High Cognitive Load detected    |
