@Adaptive @IRT @Priority-Critical @Regression
@Node-AdaptiveEngine @DARPA-DigitalTutor @Algorithm-3PL
@Phase4-Mastery @Linking-IRT @DesirableDifficulty
Feature: 3-Parameter Item Response Theory Difficulty Calibration
  """
  IRT (Item Response Theory) provides the mathematical foundation for
  calibrating atom difficulty to learner ability. The 3-parameter logistic
  model (3PL) accounts for:

  - a (Discrimination): How well the item differentiates ability levels
  - b (Difficulty): The ability level at which P(correct)=0.5
  - c (Guessing): Lower asymptote (pseudo-chance level)

  The 3PL probability model:
  P(θ) = c + (1-c) / (1 + e^(-Da(θ-b)))

  Where:
  - θ (theta) = learner ability estimate (latent trait)
  - D = scaling constant (1.7 for normal ogive approximation)

  For DARPA Digital Tutor effect sizes (2.0σ to 4.6σ), we need:
  - Precise ability estimation (SE < 0.3)
  - Optimal difficulty targeting (b ≈ θ + 0.3 for maximal information)
  - High discrimination items (a > 1.0) for hardened mastery verification

  Scientific Foundation:
  - Lord & Novick (1968). Statistical Theories of Mental Test Scores
  - Birnbaum (1968). Three-parameter logistic model
  - Baker & Kim (2017). Item Response Theory
  - van der Linden (2016). Handbook of IRT

  Key Metrics:
  - Fisher Information: I(θ) = a²(P-c)²Q / ((1-c)²P)
  - Standard Error: SE(θ) = 1/√ΣI(θ)
  - Test Information Function for optimal item selection
  """

  Background:
    Given the IRT calibration engine is initialized
    And item parameters are loaded from the Gold Atom table
    And learner ability estimates are in the Master Ledger

  # ============================================================================
  # ABILITY ESTIMATION
  # ============================================================================

  @Smoke @AbilityEstimation @MLE
  Scenario: Estimate learner ability using Maximum Likelihood
    """
    MLE estimates θ by finding the ability level that maximizes
    the likelihood of observed response patterns.
    """
    Given learner "L1" has response history:
      | atom_id    | correct | item_a | item_b | item_c |
      | atom-001   | true    | 1.2    | -0.5   | 0.10   |
      | atom-002   | false   | 1.5    | 0.8    | 0.05   |
      | atom-003   | true    | 1.8    | 0.3    | 0.02   |
      | atom-004   | true    | 1.3    | 1.2    | 0.08   |
      | atom-005   | false   | 2.0    | 1.5    | 0.03   |
    When MLE ability estimation is performed
    Then the ability estimate is:
      | metric      | value |
      | theta_hat   | 0.72  |
      | standard_error | 0.28 |
      | confidence_95 | [0.17, 1.27] |

  @Regression @AbilityEstimation @Bayesian @EAP
  Scenario: Bayesian ability estimation with prior
    """
    Expected A Posteriori (EAP) estimation incorporates prior
    distribution, useful for new learners with few responses.

    Prior: θ ~ N(0, 1) for population-level assumption
    """
    Given learner "L2" is new with only 3 responses
    When Bayesian EAP estimation is performed
    Then the estimate incorporates the prior:
      | metric         | value                    |
      | prior          | N(0, 1)                  |
      | posterior_mean | 0.45                     |
      | posterior_sd   | 0.52                     |
      | shrinkage      | toward population mean   |
    And the estimate is more stable than MLE for sparse data

  @Regression @AbilityEstimation @RealTime
  Scenario: Real-time ability update after each response
    """
    Update ability estimate after each atom interaction for
    truly adaptive difficulty selection.
    """
    Given current ability estimate θ = 0.8 (SE = 0.25)
    When learner responds correctly to atom with b = 1.2, a = 1.5
    Then ability is updated:
      | metric      | before | after |
      | theta       | 0.80   | 0.92  |
      | SE          | 0.25   | 0.22  |
    And the update follows Fisher scoring algorithm

  # ============================================================================
  # OPTIMAL ITEM SELECTION
  # ============================================================================

  @Smoke @ItemSelection @MaximalInformation
  Scenario Outline: Selecting the "Perfect" Atom for maximal learning
    """
    The dispatcher selects atoms where:
    - Difficulty (b) is approximately θ + 0.3 (desirable difficulty)
    - Discrimination (a) is high (a > 1.0) for precise measurement
    - Guessing (c) is near zero for integrity

    This ensures the "Zone of Proximal Development" is targeted precisely.
    """
    Given the user's current ability (theta) is <UserTheta>
    When the Dispatcher scans the Gold Atom table for the next interaction
    Then it must select an atom where:
      | parameter | constraint                   |
      | b         | within [θ, θ+0.5]            |
      | a         | >= 1.0                       |
      | c         | <= 0.15                      |
    And the selected atom maximizes Fisher Information at θ

    Examples:
      | UserTheta | AtomType         | Difficulty (b) | Discrimination (a) | Guessing (c) |
      | 0.5       | Symbolic_Proof   | 0.8            | 1.5                | 0.01         |
      | 1.2       | Logic_Trap       | 1.5            | 2.1                | 0.05         |
      | 2.0       | Cross_Domain_Map | 2.3            | 1.8                | 0.00         |
      | -0.3      | Factual_Recall   | 0.0            | 1.2                | 0.10         |

  @Regression @ItemSelection @InformationFunction
  Scenario: Select item with maximum information at current θ
    """
    Fisher Information I(θ) measures how much information an item
    provides about ability. We select items maximizing I(θ̂).

    I(θ) = D²a²(P-c)²(1-P) / ((1-c)²P)
    """
    Given learner ability θ = 1.0
    And candidate atoms:
      | atom_id  | a    | b    | c    | I(θ=1.0) |
      | atom-A   | 1.2  | 0.5  | 0.10 | 0.28     |
      | atom-B   | 1.8  | 1.0  | 0.05 | 0.72     |
      | atom-C   | 2.0  | 1.5  | 0.02 | 0.51     |
      | atom-D   | 1.5  | 1.2  | 0.08 | 0.48     |
    When maximum information selection is applied
    Then atom-B is selected (highest I(θ) = 0.72)
    And the selection is logged for audit

  @Regression @ItemSelection @ContentBalancing
  Scenario: Balance information maximization with content coverage
    """
    Pure information maximization can over-sample certain content.
    Weighted deviation model balances information with coverage goals.
    """
    Given learner needs coverage of concepts:
      | concept      | target_proportion | current_proportion |
      | subnetting   | 0.30              | 0.45               |
      | routing      | 0.25              | 0.15               |
      | switching    | 0.25              | 0.25               |
      | security     | 0.20              | 0.15               |
    When balanced item selection is applied
    Then items from under-represented concepts get priority boost
    And selection optimizes: max(I(θ)) + λ×coverage_deviation

  # ============================================================================
  # ITEM PARAMETER ESTIMATION
  # ============================================================================

  @Smoke @ParameterEstimation @Calibration
  Scenario: Calibrate new atom parameters from response data
    """
    When new atoms are introduced, we calibrate their IRT parameters
    using marginal maximum likelihood (MML) or joint MLE.
    """
    Given a new atom with 500+ learner responses
    When MML parameter estimation is performed
    Then item parameters are estimated:
      | parameter | estimate | SE    |
      | a         | 1.45     | 0.12  |
      | b         | 0.82     | 0.08  |
      | c         | 0.06     | 0.03  |
    And the atom is added to the calibrated item pool

  @Regression @ParameterEstimation @LinkingScale
  Scenario: Link new items to existing scale
    """
    When calibrating new items, we must link them to the existing
    ability scale to maintain comparability.
    """
    Given existing item pool on scale with mean=0, SD=1
    And new items calibrated on separate sample
    When scale linking is performed using anchor items
    Then new item parameters are transformed to common scale
    And ability estimates remain comparable across versions

  # ============================================================================
  # MASTERY DECISION RULES
  # ============================================================================

  @Smoke @MasteryDecision @CutScore
  Scenario: Determine mastery using IRT-based cut score
    """
    Mastery is determined by whether θ exceeds a criterion θ_c
    with sufficient precision (SE constraint).

    For "Hardened" mastery: θ > 1.5 AND SE(θ) < 0.3
    """
    Given concept "subnet_calculation" has mastery criterion θ_c = 1.5
    And learner has ability estimate:
      | metric | value |
      | theta  | 1.72  |
      | SE     | 0.24  |
    When mastery decision is evaluated
    Then mastery status is:
      | field            | value    |
      | theta_above_cut  | true     |
      | SE_acceptable    | true     |
      | mastery_status   | HARDENED |
      | confidence       | 0.95     |

  @Regression @MasteryDecision @PosteriorProbability
  Scenario: Bayesian mastery probability
    """
    Instead of binary mastery, compute P(θ > θ_c | responses).
    This gives a continuous mastery probability.
    """
    Given mastery criterion θ_c = 1.5
    And learner's posterior distribution is N(1.3, 0.35)
    When Bayesian mastery probability is computed
    Then:
      | metric                  | value |
      | P(mastery)              | 0.28  |
      | expected_items_to_95%   | 4     |
    And the learner is recommended 4 more items before mastery decision

  # ============================================================================
  # ADAPTIVE TESTING TERMINATION
  # ============================================================================

  @Regression @Termination @PrecisionBased
  Scenario: Terminate when ability is estimated with sufficient precision
    """
    Stop testing when SE(θ) falls below threshold (precision-based).
    This is more efficient than fixed-length testing.
    """
    Given SE threshold = 0.30
    And learner has completed 8 items with SE = 0.32
    When learner completes item 9 with SE = 0.28
    Then testing terminates
    And final ability estimate is reported

  @Regression @Termination @SequentialProbabilityRatio
  Scenario: SPRT for mastery/non-mastery classification
    """
    Sequential Probability Ratio Test terminates when we're
    confident about mastery classification.

    H0: θ < θ_c (non-mastery)
    H1: θ > θ_c (mastery)
    """
    Given mastery criterion θ_c = 1.5
    And indifference region [1.3, 1.7]
    And Type I/II error rates α = β = 0.05
    When SPRT is applied after each item
    Then testing terminates when:
      | condition                     | decision    |
      | likelihood_ratio > (1-β)/α    | MASTERY     |
      | likelihood_ratio < β/(1-α)    | NON-MASTERY |
    And expected test length is 40% shorter than fixed-length

  # ============================================================================
  # INTEGRATION WITH NCDE AND MASTERY ENGINE
  # ============================================================================

  @Integration @NCDE @StateVector
  Scenario: IRT ability feeds into NCDE state vector
    """
    IRT θ estimates become components of the NCDE cognitive state vector,
    enabling continuous-time mastery trajectory modeling.
    """
    Given learner has IRT ability estimates per concept:
      | concept      | theta | SE   |
      | subnetting   | 1.2   | 0.25 |
      | routing      | 0.8   | 0.30 |
      | switching    | 1.5   | 0.22 |
    When NCDE state vector is constructed
    Then the state vector h(t) includes:
      | dimension        | value | source |
      | theta_subnetting | 1.2   | IRT    |
      | theta_routing    | 0.8   | IRT    |
      | theta_switching  | 1.5   | IRT    |
      | uncertainty_agg  | 0.26  | SE     |
    And NCDE can model trajectory through ability space

  @Integration @Scheduling @ZScore
  Scenario: IRT parameters inform scheduling priority
    """
    Atoms with high discrimination (a) and appropriate difficulty (b)
    are prioritized for mastery verification.
    """
    Given learner θ = 1.0 for concept "routing"
    And candidate atoms:
      | atom_id | a    | b    | scheduling_weight |
      | R1      | 2.0  | 1.3  | 0.85              |
      | R2      | 1.2  | 0.5  | 0.45              |
      | R3      | 1.5  | 1.0  | 0.72              |
    When scheduling integrates IRT parameters
    Then atom R1 gets highest priority (high a, appropriate b)
    And scheduling weight contributes to Z-score priority
