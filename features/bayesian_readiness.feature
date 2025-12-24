@Node-AdaptiveEngine @Edge-Adaptive-KB @Data-ReadinessProb @Algorithm-BayesianNetwork @Science-InformationTheory @DARPA-DigitalTutor
Feature: Bayesian Topic Readiness & Knowledge Tracing
  As a Lesson Planner
  I want to calculate "Topic Readiness" using Bayesian Networks
  In order to ensure prerequisites are truly mastered before advancing.

  Background:
    Given the Knowledge Graph `G` contains Nodes `N` and Edges `E`
    And `P(Ready_B | Mastered_A)` represents the conditional probability

  @Positive @Graph-Traversal @Prerequisites @State-Unlocked
  Scenario: Propagating Readiness Scores
    Given Topic A is a prerequisite for Topic B
    And the user has mastered Topic A (P(Mastered_A) = 0.95)
    When the Bayesian inference runs
    Then P(Ready_B) should increase significantly
    And if P(Ready_B) > 0.8, Topic B becomes "Unlocked"

  @Logic-Negative @Backtracking @State-Intervention
  Scenario: Interleaved Practice Trigger
    Given Topic B is unlocked
    When the user fails Topic B repeatedly (P(Mastered_B) drops)
    Then the system should re-evaluate P(Ready_B)
    And if it drops below threshold, trigger "Interleaved Practice" for Topic A
