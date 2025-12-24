@Node-AdaptiveEngine @Edge-Adaptive-KB @Data-Theta @Algorithm-IRT @Science-Psychometrics @DARPA-DigitalTutor
Feature: IRT Parameter Estimation & Ability Tracking
  As an Adaptive Engine
  I want to estimate the user's latent ability (Theta) using Item Response Theory
  In order to provide the optimal "Zone of Proximal Development".

  Background:
    Given the 3-parameter IRT model is active via `@Algorithm-IRT`:
      "P(Theta) = c + (1 - c) / (1 + e^(-a(Theta - b)))"
    And "a" is Discrimination, "b" is Difficulty, "c" is Guessing

  @Positive @Math-Formalism @State-ThetaUpdate
  Scenario: Updating Theta after Correct Response
    Given a user with current Theta "0.5"
    And they answer a question with Difficulty b="1.0" correctly
    When the Bayesian update is performed
    Then the new Theta should be "> 0.5"
    And the "Standard Error" of estimation should decrease

  @Negative @Math-Formalism @Statistics-Guessing
  Scenario: Detecting Lucky Guesses (The 'c' Parameter)
    Given a user with Theta "-2.0" (Novice)
    And they answer a question with Difficulty b="3.0" (Expert) correctly
    And the Guessing parameter c="0.25"
    When the engine calculates the likelihood
    Then the system should interpret this as a probable "Guess"
    And Theta should increase only marginally
