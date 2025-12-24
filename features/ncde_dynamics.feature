@Node-AdaptiveEngine @Edge-Adaptive-KB @Data-Theta @Algorithm-NCDE @Science-NeuralCDE @DARPA-DigitalTutor
Feature: Neural Controlled Differential Equation Dynamics
  As a Mastery Gradient Monitor
  I want to model the continuous evolution of cognitive state with NCDEs
  In order to predict Lattice Collapse before it happens.

  Background:
    Given the NCDE model "dTheta_t = f(Theta_t, Metric_t) dX_t" is loaded
    And "X_t" represents the behavioral event stream (Latency, Precision)

  @Positive @State-SolidState @Fading
  Scenario: Detecting Steady State for Scaffold Fading
    Given a stream of 50 user interactions
    When the derivative "dTheta/dt" approaches 0 despite increasing task difficulty
    Then the system flags the state as "Solid-State"
    And triggers the "Adaptive Fading" event (Opacity -> 0%)

  @Negative @State-LatticeCollapse @Intervention
  Scenario: Predicting Lattice Collapse via Trajectory Divergence
    Given the user is on a "Gold-Layer" trajectory
    When "Action Precision" drops and "Retrieval Latency" spikes > 5000ms
    Then the NCDE should predict a "Lattice Collapse" event within 3 steps
    And the system should preemptively trigger "MAARTA Recruitment"
