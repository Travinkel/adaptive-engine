@Node-AdaptiveEngine @Edge-Adaptive-KB @Data-ItemParameters @Algorithm-3PL_IRT @Science-Psychometrics @DARPA-DigitalTutor
Feature: 3-Parameter IRT Difficulty Calibration
  As a Learning Scientist
  I want to weight every Gold Atom using a 3-parameter logistic model (3PL)
  So that the "Desirable Difficulty" is perfectly aligned with the user's latent ability.

  @Positive @Calibration @Math-Formalism
  Scenario Outline: Selecting the "Perfect" Atom
    Given the user's current ability (theta) is retrieved from the Master Ledger
    When the Dispatcher scans the Gold Atom table for the next interaction
    Then it must select an atom where Difficulty (b) is <UserTheta> + 0.3
    And Discrimination (a) is high enough to differentiate "Hardened" vs "Raw" states
    And Guessing (c) is near zero to ensure "Ramanujan-tier" integrity.

    Examples:
      | AtomType             | Difficulty (b) | Discrimination (a) | Guessing (c) |
      | Symbolic_Proof       | 2.8            | 1.5                | 0.01         |
      | Logic_Trap           | 2.2            | 2.1                | 0.05         |
      | Cross_Domain_Map     | 3.0            | 1.8                | 0.00         |
