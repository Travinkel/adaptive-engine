@Domain-Cognitive @DualCoding @Paivio @Mayer @Multimodal @Priority-Regression @6Sigma-Sovereign
Feature: Dual-Coding and Multimodal Learning Integration
  As a Learning Scientist
  I want to ensure verbal content is paired with non-verbal representations
  So that learners build bi-modal (and tri-modal) mental models for maximum retention.

  Background:
    Given Dual-Coding Theory (Paivio) and Multimedia Learning (Mayer) are active
    And the Visual-Spatial Contiguity principle (Sweller) is enforced
    And multimodal content generators are available:
      | Mode          | Generator                  | Output_Type         |
      | Verbal        | Text_Engine                | Prose, Code         |
      | Visual        | Diagram_Generator          | Flowcharts, UML     |
      | Tactile       | Interactive_Sim            | Code execution      |
    And cross-mode verification is required for mastery

  # ─────────────────────────────────────────────────────────────────────────
  # Symmetrical Dual-Coding
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Smoke @DualCoding @VerbalVisual
  Scenario: Visual-verbal synthesis for code concepts
    Given a "Constructive" Text Atom is active: "C# Generics Syntax"
    When the learner completes a logical sub-goal
    Then the Right Pane should dynamically generate a "Visual Topology"
    And the learner must perform "Cross-Mode Sync":
      | Task                                    | Mode_Cross        |
      | "Match the visual node to the code line"| Visual → Verbal   |
      | "Label the type parameter in diagram"   | Verbal → Visual   |
    And mastery is only confirmed when both traces are verified
    And either trace can retrieve the other in future

  @Positive @DualCoding @AutomaticPairing
  Scenario: Automatic visual representation generation
    Given a purely verbal atom about "Heap Data Structure"
    When dual-coding engine processes the atom
    Then it should generate:
      | Generated_Visual    | Type            | Alignment_To_Text      |
      | Heap_Tree_Diagram   | Tree structure  | Each node labeled      |
      | Array_Mapping       | Index mapping   | Parent/child formulas  |
      | Insert_Animation    | Step sequence   | Algorithm phases       |
    And visuals should be spatially contiguous with text
    And no split-attention between panes

  @Positive @DualCoding @CodeVisualization
  Scenario: Visualizing code execution states
    Given a code atom for "Quicksort Partition"
    When visualization mode is active
    Then it should render:
      | Visualization_Type  | Content                          |
      | Array_State         | Current array with pointers      |
      | Pivot_Highlight     | Current pivot element            |
      | Swap_Animation      | Element exchanges                |
      | Partition_Boundary  | Left/right region markers        |
    And learner can step through visually
    And verbal explanation synchronized with visual state

  # ─────────────────────────────────────────────────────────────────────────
  # Triple-Coding (Triangulated Encoding)
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @TripleCoding @Triangulation
  Scenario: Triangulated symmetrical encoding
    Given a concept is in "Silver-Hardened" state
    When the "Triangulation Pass" begins
    Then the learner must demonstrate mastery in all three modes:
      | Mode       | Task                                    | Verification      |
      | Verbal     | Define the axiom in Socratic Terminal   | Text accuracy     |
      | Visual     | Draw the relational topology            | Structural match  |
      | Functional | Implement Minimum Viable Proof in Code  | Execution correct |
    And the concept only achieves "Gold-Sovereign" if all align
    And three neural anchors are nearly impossible to forget

  @Positive @TripleCoding @CrossModalRetrieval
  Scenario: Testing cross-modal retrieval pathways
    Given learner has triple-coded "Binary Search"
    When cross-modal retrieval test runs
    Then it should test all directions:
      | Stimulus_Mode | Response_Mode | Prompt                          |
      | Visual        | Verbal        | "Describe this diagram"         |
      | Verbal        | Visual        | "Draw this algorithm"           |
      | Verbal        | Functional    | "Implement this description"    |
      | Visual        | Functional    | "Code what you see"             |
      | Functional    | Verbal        | "Explain this code"             |
      | Functional    | Visual        | "Diagram this code's logic"     |
    And all 6 pathways should work for true mastery

  # ─────────────────────────────────────────────────────────────────────────
  # Visual-Spatial Contiguity (Sweller)
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Contiguity @SplitAttention
  Scenario: Eliminating split-attention effect
    Given the Richter Agent is explaining "Heap Fragmentation"
    When the learner hovers over the "Fragmentation" term in Right Pane
    Then the Left Pane should automatically render the Visual State Atom
    And the visual should:
      | Integration          | Details                          |
      | Highlight_Term       | Same term highlighted in both    |
      | Spatial_Proximity    | Related info physically close    |
      | Synchronized_Scroll  | Panes move together              |
    And no cognitive load wasted on visual search

  @Positive @Contiguity @IntegratedFormat
  Scenario: Integrated vs separated format comparison
    Given content can be presented two ways:
      | Format              | Description                      |
      | Separated           | Text on left, diagram on right   |
      | Integrated          | Labels embedded in diagram       |
    When contiguity optimizer evaluates
    Then it should prefer integrated format:
      | Content_Type        | Preferred_Format                 |
      | Diagram + Labels    | Integrated (labels on diagram)   |
      | Code + Explanation  | Integrated (inline comments)     |
      | Process + Steps     | Integrated (numbered in diagram) |
    And integrated format reduces extraneous load by ~30%

  # ─────────────────────────────────────────────────────────────────────────
  # Dynamic Diagram Injection
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Dynamic @ContextAwareDiagrams
  Scenario: Context-aware diagram generation
    Given learner is reading about "Depth-First Search"
    When they reach the "backtracking" section
    Then diagram engine should:
      | Action                    | Content                          |
      | Generate_Tree             | Example tree structure           |
      | Animate_Traversal         | Show DFS path with backtracking  |
      | Highlight_Backtrack       | Color change on backtrack steps  |
      | Sync_With_Text            | Current step matches prose       |
    And diagram updates as learner progresses through text

  @Positive @Dynamic @InteractiveDiagrams
  Scenario: Interactive diagram manipulation for learning
    Given a diagram of "Red-Black Tree" is displayed
    When learner interacts with diagram
    Then they should be able to:
      | Interaction           | Learning_Outcome                 |
      | Insert_Node           | See rebalancing animation        |
      | Delete_Node           | See rotation sequence            |
      | Click_Node            | See properties and constraints   |
      | Step_Through          | Replay operations step by step   |
    And active manipulation enhances encoding

  # ─────────────────────────────────────────────────────────────────────────
  # Modality Matching
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Modality @ContentTypeMatching
  Scenario: Matching modality to content type
    Given different content types require different primary modalities
    When modality selector evaluates content
    Then it should choose:
      | Content_Type          | Primary_Mode | Secondary_Mode |
      | Algorithm_Logic       | Visual       | Verbal         |
      | Syntax_Rules          | Verbal       | Code           |
      | Data_Structure        | Visual       | Functional     |
      | Mathematical_Proof    | Verbal       | Symbolic       |
      | System_Architecture   | Visual       | Verbal         |
    And inappropriate modality should be flagged

  @Positive @Modality @LearnerPreference
  Scenario: Adapting to learner modality preferences
    Given learner profile indicates:
      | Preference            | Value                            |
      | Visual_Preference     | High (0.8)                       |
      | Verbal_Preference     | Medium (0.5)                     |
      | Kinesthetic_Preference| High (0.7)                       |
    When content is presented
    Then it should weight:
      | Adaptation            | Details                          |
      | Increase_Diagrams     | More visual representations      |
      | Add_Simulations       | Interactive code execution       |
      | Maintain_Text         | Don't eliminate verbal           |
    And preferences should NOT eliminate other modalities (dual-coding still required)

  # ─────────────────────────────────────────────────────────────────────────
  # Multimodal Assessment
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Assessment @CrossModalVerification
  Scenario: Cross-modal mastery verification
    Given learner claims mastery of "Sorting Algorithms"
    When multimodal assessment runs
    Then it should test:
      | Assessment_Type       | Modality              | Weight |
      | Explain_Algorithm     | Verbal                | 0.25   |
      | Trace_Execution       | Visual + Verbal       | 0.25   |
      | Implement_Code        | Functional            | 0.25   |
      | Debug_Visual          | Visual + Functional   | 0.25   |
    And mastery requires ≥ 80% across all modalities
    And single-modality proficiency is insufficient

  @Positive @Assessment @ModalityGap
  Scenario: Identifying modality gaps in understanding
    Given learner shows uneven modality performance:
      | Modality      | Score |
      | Verbal        | 90%   |
      | Visual        | 55%   |
      | Functional    | 85%   |
    When modality gap analysis runs
    Then it should identify:
      | Gap                   | Details                          |
      | Visual_Weakness       | Can explain but not visualize    |
      | Remediation_Needed    | Diagram interpretation practice  |
    And inject visual-focused atoms for the weak modality

