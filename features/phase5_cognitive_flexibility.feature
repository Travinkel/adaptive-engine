@Adaptive @CFT @Priority-High @Regression
@Node-AdaptiveEngine @DARPA-DigitalTutor @Phase5-Advanced
@CognitiveFlexibility @Spiro @IllStructuredDomains
Feature: Cognitive Flexibility Theory - Criss-Crossing the Conceptual Landscape
  """
  CFT (Cognitive Flexibility Theory) addresses learning in ill-structured
  domains where oversimplification leads to misconceptions. Complex domains
  like mathematics (Spivak), algorithms (Knuth), and evidence-based medicine
  (Gøtzsche) cannot be learned through single perspectives.

  The Solution: "Criss-Cross" the conceptual landscape from multiple entry
  points, through multiple cases, revealing the multidimensional nature
  of complex concepts.

  Scientific Foundation:
  - Spiro et al. (1988). Cognitive Flexibility and Hypertext
  - Spiro et al. (1991). Knowledge Representation and Instruction
  - Jacobson & Spiro (1995). Hypertext Learning Environments

  Key Principles:
  1. Multiple Representations: Same concept, different expert lenses
  2. Case-Based Learning: Abstract concepts through concrete cases
  3. Non-Linear Navigation: No single "correct" path through knowledge
  4. Conceptual Interrelatedness: Highlight connections between ideas

  Implementation:
  - Agents represent different expert perspectives (Spivak, Richter, Gøtzsche)
  - Same node rendered through different "lenses"
  - Cross-domain case studies reveal structural commonalities

  Effect Size Target: 4.6σ → 5.5σ through multi-perspective integration
  """

  Background:
    Given the learner has achieved "Base Logic" mastery on a Gold node
    And multiple expert agents are available:
      | agent    | perspective                | domain_expertise            |
      | Spivak   | formal_axiomatic           | pure mathematics            |
      | Richter  | functional_implementation  | systems programming         |
      | Knuth    | algorithmic_analysis       | computer science            |
      | Gøtzsche | evidence_based             | critical appraisal          |
      | King     | narrative_rhythm           | prose construction          |

  # ============================================================================
  # MULTI-PERSPECTIVE CRISS-CROSSING
  # ============================================================================

  @Smoke @CrissCrossing @MultiPerspective
  Scenario Outline: Multi-Perspective Criss-Crossing through expert lenses
    """
    After mastering a concept through one lens, the system forces
    re-engagement through a fundamentally different perspective,
    revealing dimensions invisible from the first viewpoint.
    """
    Given I have mastered a node's "Base Logic" via the "<InitialAgent>"
    When the "Landscape-Explorer" mode is activated
    Then the CLI should re-render the node using the "<SecondaryAgent>" lens
    And the Right Pane should display a "Case-Study" from "<CaseStudyDomain>"
    And the Left Pane should require a "Re-Representation" of the logic
    And the node achieves "CFT-Verified" only when both perspectives align

    Examples:
      | InitialAgent | SecondaryAgent | CaseStudyDomain             |
      | Spivak       | Richter        | Memory Allocation Bounds    |
      | Knuth        | Gøtzsche       | Algorithmic Bias in Med AI  |
      | Richter      | Spivak         | Functional Lambda Calculus  |
      | Gøtzsche     | King           | Narrative Epidemiology      |
      | King         | Knuth          | Story Structure Algorithms  |

  @Regression @CrissCrossing @StructuralInvariance
  Scenario: Identify structural invariants across perspectives
    """
    The deepest learning happens when the learner identifies what
    remains constant across all perspectives (the structural invariant).
    """
    Given concept "limits" has been viewed through:
      | agent    | rendering                           |
      | Spivak   | ε-δ formal definition               |
      | Richter  | convergence in numerical computing  |
      | King     | tension approaching climax          |
    When the learner is prompted for structural invariant
    Then they must identify:
      | invariant_component | description                        |
      | approach_behavior   | value gets arbitrarily close       |
      | boundary_condition  | within specified tolerance          |
      | asymptotic_property | never quite reaches, infinitely near|
    And this invariant is stored as "meta-schema"

  @Regression @CrissCrossing @IllStructuredDomain
  Scenario: Handle ill-structured domain complexity
    """
    In ill-structured domains, cases differ in important ways.
    CFT ensures learners see the "irregularity" as features, not bugs.
    """
    Given domain "debugging complex systems" is ill-structured
    When multiple cases are presented:
      | case_id | surface_features     | deep_structure          |
      | C1      | null pointer         | state corruption        |
      | C2      | race condition       | state corruption        |
      | C3      | memory leak          | resource lifecycle      |
    Then learner must recognize:
      | insight                      | cases   |
      | C1 and C2 share deep structure | true  |
      | C3 has different root cause    | true  |
    And over-generalization is explicitly challenged

  # ============================================================================
  # CASE-BASED REASONING
  # ============================================================================

  @Smoke @CaseBasedReasoning @ConcreteToAbstract
  Scenario: Ground abstract concepts in multiple concrete cases
    """
    Abstract concepts become meaningful through concrete instantiation
    across diverse cases that share structural properties.
    """
    Given abstract concept "recursion" to be learned
    When case-based presentation is activated
    Then multiple cases are presented sequentially:
      | case_domain       | concrete_example              |
      | file_systems      | directory traversal           |
      | natural_language  | parsing nested sentences      |
      | mathematics       | factorial computation         |
      | art               | fractal self-similarity       |
    And learner must identify recursive structure in each
    And abstraction emerges from case comparison

  @Regression @CaseBasedReasoning @ThematicVariation
  Scenario: Thematic variation reveals concept boundaries
    """
    By varying surface features while preserving structure,
    learner discovers what is essential vs incidental.
    """
    Given concept "binary search" mastered in array context
    When thematic variations are presented:
      | variation           | surface_change        | structure_preserved |
      | git bisect          | commits not numbers   | yes                 |
      | troubleshooting     | network not array     | yes                 |
      | linear_search       | different algorithm   | no                  |
    Then learner must correctly classify each variation
    And explain why linear_search breaks the structural pattern

  # ============================================================================
  # AGENT PERSPECTIVE SWITCHING
  # ============================================================================

  @Smoke @AgentSwitching @RealTime
  Scenario: Real-time perspective switching during struggle
    """
    When a learner struggles with one perspective, switch to another
    that may illuminate the concept from a more accessible angle.
    """
    Given learner is struggling with Spivak's ε-δ definition (friction > 0.7)
    When perspective switch is triggered
    Then Richter agent provides alternative:
      | original_perspective | alternative_perspective           |
      | |f(x) - L| < ε       | assert(abs(result - expected) < tolerance) |
    And the computational metaphor may unlock understanding
    And learner returns to formal definition with new insight

  @Regression @AgentSwitching @DebateMode
  Scenario: Agents debate to reveal concept facets
    """
    Different experts would emphasize different aspects of the same concept.
    Simulating their debate exposes the full conceptual richness.
    """
    Given concept "algorithmic complexity" is being studied
    When agent debate mode is activated
    Then perspectives clash productively:
      | agent    | emphasis                          | potential_criticism        |
      | Knuth    | asymptotic analysis (Big-O)       | "Ignores constants"        |
      | Richter  | practical performance             | "Cache behavior matters"   |
      | Gøtzsche | benchmark methodology rigor       | "How was it measured?"     |
    And learner must synthesize: "When do constants matter?"

  # ============================================================================
  # NON-LINEAR NAVIGATION
  # ============================================================================

  @Smoke @NonLinear @MultiplePathways
  Scenario: Multiple valid learning pathways through concept space
    """
    CFT rejects single "correct" curriculum sequence. Learners
    can approach concepts from multiple valid entry points.
    """
    Given target concept "monads" in functional programming
    When learning pathways are generated
    Then multiple valid paths exist:
      | pathway_id | entry_point      | intermediate_concepts      |
      | P1         | functors         | applicatives → monads      |
      | P2         | state_management | mutation → monads          |
      | P3         | promise_chains   | async → monads             |
    And learner can choose based on background knowledge
    And all paths converge on equivalent understanding

  @Regression @NonLinear @Hypertext
  Scenario: Hypertext-style exploration of concept space
    """
    Original CFT used hypertext for non-linear navigation.
    We implement this through concept graph exploration.
    """
    Given learner is viewing node "memory management"
    When hypertext links are displayed
    Then navigation options include:
      | link_type        | destination               | relationship    |
      | prerequisite     | stack vs heap             | required_for    |
      | application      | garbage collection        | applies_to      |
      | contrast         | manual vs automatic       | differs_from    |
      | analogy          | library book checkout     | similar_to      |
    And navigation history enables backtracking
    And "criss-cross" paths are encouraged
