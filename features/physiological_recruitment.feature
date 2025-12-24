@Node-AdaptiveEngine @Edge-Adaptive-BioMetrics @Data-HRV @Algorithm-FatigueCalibration @Science-BioCognition @MAARTA @Sovereignty
Feature: Physiological Resource Recruitment
  As a Digital Tutor
  I want to adjust the "Desirable Difficulty" based on bio-metric constraints
  So that I avoid cognitive collapse when my neural hardware is under stress.

  @Positive @Bio-Feedback @Safety-Critical
  Scenario: REM-Deficit Session Adaptation
    Given the "services/domains/sleep/right-cycles" agent reports REM < 90m
    And the "services/domains/fitness/right-metrics" reports HRV in "Recovery" zone
    When a "Spivak Tier 3" session is requested
    Then the "Adaptive-Engine" must recruit the "Psychology-Agent"
    And the system must override the session to "Guidance-Heavy" mode
    And the NCDE friction threshold must be lowered by 40% to prevent schema burnout.
