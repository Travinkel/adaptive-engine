@Domain-Cognitive @DARPA-DigitalTutor @AToM-Reasoning @SchemaInduction @Priority-Regression
Feature: Conclusion-Verified Schema Induction (CASI)
  As a Neuro-symbolic Reasoning Engine
  I want to verify that learners induce transferable schemas via structural mapping
  So that learning genuinely transfers across domains (not just surface pattern matching).

  Background:
    Given the CASI verification logic is active
    And the Structure-Mapping Engine (SME) computes alignment scores
    And the Conclusion Predicate is withheld during induction tasks
    And Systematicity scoring prioritizes higher-order relations (w=0.8)
    And the SAGE engine bonds verified schemas to Solid-State Molecules

  # ─────────────────────────────────────────────────────────────────────────
  # Core CASI Algorithm
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Smoke @CASI @Core
  Scenario: Verifying genuine schema induction
    Given the learner is presented with:
      | Role        | Domain              | Example                    |
      | Base        | Atomic Structure    | Nucleus, electrons orbit   |
      | Target      | Solar System        | Sun, planets orbit         |
      | Hidden_Goal | Derive force type   | (withheld from learner)    |
    When the learner proposes structural mapping:
      | Base_Element      | Target_Element    | Relation_Type       |
      | Nucleus           | Sun               | Central_Body        |
      | Electrons         | Planets           | Orbiting_Objects    |
      | Electrostatic     | Gravitational     | Binding_Force       |
    And generates candidate conclusion: "Gravitational force binds planets to sun"
    Then the CASI engine must verify:
      | Check                        | Status | Details                        |
      | Structural_Mapping_Complete  | PASS   | All relations aligned          |
      | Conclusion_Derivable         | PASS   | Target conclusion follows      |
      | Surface_Independent          | PASS   | Not based on "both are round"  |
    And the schema should be bonded as "Solid-State Molecule"
    And competency upgraded: "Novice" → "Intermediate"

  @Positive @CASI @MappingCompleteness
  Scenario: Evaluating structural mapping completeness
    Given Base domain "Water Flow" with structure:
      | Element        | Relations                          |
      | Pressure_Diff  | CAUSES → Water_Flow                |
      | Pipe_Diameter  | MODULATES → Flow_Rate              |
      | Viscosity      | OPPOSES → Flow                     |
    And Target domain "Electrical Circuit" to map
    When the learner provides partial mapping:
      | Base           | Target         | Mapped |
      | Pressure_Diff  | Voltage        | ✓      |
      | Pipe_Diameter  | Wire_Gauge     | ✓      |
      | Viscosity      | (unmapped)     | ✗      |
    Then CASI should return:
      | Metric               | Value | Feedback                         |
      | Mapping_Completeness | 67%   | "What resists electrical flow?"  |
      | Schema_Valid         | false | Incomplete structural alignment  |

  # ─────────────────────────────────────────────────────────────────────────
  # SME (Structure-Mapping Engine) Integration
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @SME @Scoring
  Scenario: Structure-Mapping Engine evaluation scoring
    Given a candidate mapping between domains
    When SME computes alignment score
    Then it should calculate:
      | Component              | Weight | Description                       |
      | Attribute_Matches      | 0.1    | Surface features (low priority)   |
      | First_Order_Relations  | 0.3    | Direct relationships              |
      | Higher_Order_Relations | 0.8    | Nested/systematic relations       |
    And total score: S_total = Σs_i + (w × S_higher_order)
    And scores above threshold (0.75) indicate valid structural alignment

  @Positive @SME @Systematicity
  Scenario: Systematicity principle in mapping evaluation
    Given two candidate mappings for same Base→Target:
      | Mapping | Attribute_Score | Higher_Order_Score | Total |
      | A       | 0.8             | 0.3                | 0.44  |
      | B       | 0.3             | 0.9                | 0.75  |
    When SME ranks mappings by systematicity
    Then Mapping B should be preferred despite lower attribute match
    And feedback should emphasize: "Deep structure matters more than surface"

  # ─────────────────────────────────────────────────────────────────────────
  # Pattern Matching Detection (Anti-Pattern)
  # ─────────────────────────────────────────────────────────────────────────

  @Negative @CASI @PatternMatching
  Scenario: Detecting surface pattern matching without schema induction
    Given a learner provides correct Target conclusion "Gravitational force"
    When CASI audits the structural mapping trace
    And detects the learner:
      - Matched "force" keyword in both domains (surface match)
      - Skipped "orbits" structural correspondence
      - Did not establish CAUSES relation
    Then the system should:
      | Action                    | Reason                           |
      | Block_Competency_Upgrade  | Pattern matching, not induction  |
      | Inject_Perturbation_Atom  | Break surface-level pattern      |
      | Require_Explicit_Mapping  | Force structural alignment       |
    And log: "Surface pattern match detected - schema induction incomplete"

  @Negative @CASI @ShallowTransfer
  Scenario: Identifying shallow transfer attempts
    Given a learner encounters new domain "Heat Flow" after mastering "Water Flow"
    When learner immediately maps:
      | Base (Water)  | Target (Heat)       | Quality        |
      | Water         | Heat                | Surface match  |
      | Pipe          | Conductor           | Surface match  |
      | Flow_Rate     | Temperature         | INCORRECT      |
    Then CASI should detect shallow transfer:
      | Issue              | Details                              |
      | Flow_Rate mismatch | Temperature ≠ Heat_Flow_Rate         |
      | Missing relation   | Thermal_Gradient not mapped          |
    And require deeper structural analysis before accepting

  # ─────────────────────────────────────────────────────────────────────────
  # Cross-Domain Transfer Measurement
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Transfer @Efficiency
  Scenario: Cross-domain transfer efficiency measurement
    Given learner has "Solid-State" mastery in "Linear_Regression" (Base)
    When system presents "Neural_Network_Gradient_Descent" (Target)
    Then the Transfer_Efficiency score should measure:
      | Metric                  | Formula                           | Value |
      | Mapping_Completeness    | Aligned_Relations / Total         | 0.85  |
      | Time_to_Insight         | Seconds to correct mapping        | 180   |
      | Hint_Independence       | 1 - (Hints_Used / Max_Hints)      | 0.70  |
      | First_Attempt_Success   | Binary                            | true  |
    And composite efficiency: E = w1×MC + w2×(1-TTI/max) + w3×HI

  @Positive @Transfer @FarTransfer
  Scenario: Far transfer detection and celebration
    Given learner successfully maps between distant domains:
      | Base_Domain      | Target_Domain        | Distance_Rating |
      | Musical_Harmony  | Color_Theory         | Far             |
    And structural mapping reveals:
      - Complementary relationships preserved
      - Tension/Resolution → Contrast/Balance
    When CASI verifies the mapping
    Then it should:
      | Action                    | Reason                        |
      | Award_Far_Transfer_Badge  | Rare achievement              |
      | Log_To_Portfolio          | Evidence of deep learning     |
      | Increase_Schema_Weight    | Strengthen neural pathway     |

  # ─────────────────────────────────────────────────────────────────────────
  # Schema Consolidation
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @SAGE @Bonding
  Scenario: SAGE engine bonding verified schema to Solid-State
    Given CASI has verified a structural mapping
    When the SAGE (Schema Acquisition & Generalization Engine) processes it
    Then it should:
      | Action                      | Details                           |
      | Create_Schema_Node          | Abstract structure in KB          |
      | Link_To_Base_Domain         | Bidirectional edge                |
      | Link_To_Target_Domain       | Bidirectional edge                |
      | Set_Abstraction_Level       | Higher than source domains        |
    And the schema should be queryable for future analogical retrieval
    And the learner's competency record should be updated

  @Positive @SAGE @Generalization
  Scenario: Schema generalization from multiple instances
    Given learner has verified mappings:
      | Instance | Base          | Target        | Schema_Fragment      |
      | 1        | Water_Flow    | Electricity   | Flow_Through_Medium  |
      | 2        | Heat_Transfer | Diffusion     | Gradient_Driven_Flow |
      | 3        | Air_Pressure  | Sound_Waves   | Pressure_Propagation |
    When SAGE analyzes common structure
    Then it should induce generalized schema:
      | Abstract_Schema     | Components                        |
      | Gradient_Flow       | Source, Sink, Medium, Rate, Resistance |
    And this schema should facilitate future near-transfer learning

  # ─────────────────────────────────────────────────────────────────────────
  # Scaffolding for Schema Induction
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Scaffolding @Hints
  Scenario: Providing hints without revealing conclusion
    Given learner is stuck on Target conclusion derivation
    When hint system activates
    Then hints should follow hierarchy:
      | Level | Hint_Type          | Example                           | Reveals_Answer |
      | 1     | Structural_Prompt  | "What corresponds to [Base_X]?"   | No             |
      | 2     | Relation_Prompt    | "How are these elements related?" | No             |
      | 3     | Near_Miss_Warning  | "Check the CAUSES relationship"   | No             |
    And no hint should directly state the Target conclusion
    And the Socratic agent should guide without telling

  @Positive @Scaffolding @FadingWithMastery
  Scenario: Fading scaffolding as schema induction skill develops
    Given learner has completed 10 successful CASI tasks
    When presenting new analogical mapping task
    Then scaffolding should be reduced:
      | Mastery_Level | Scaffolding_Provided              |
      | Novice        | Full structure template           |
      | Intermediate  | Partial template, some blanks     |
      | Advanced      | Minimal prompts only              |
      | Expert        | No scaffolding (pure induction)   |
