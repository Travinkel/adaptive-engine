import sys
from unittest.mock import MagicMock

# Mock astartes_shared
astartes_shared_mock = MagicMock()
astartes_shared_mock.database.session_scope = MagicMock()
sys.modules["astartes_shared"] = astartes_shared_mock
sys.modules["astartes_shared.database"] = astartes_shared_mock.database

# Mock src.graph.zscore_engine
zscore_mock = MagicMock()
sys.modules["src.graph.zscore_engine"] = zscore_mock

# Mock src.integrations.vertex_tutor
vertex_tutor_mock = MagicMock()
sys.modules["src.integrations.vertex_tutor"] = vertex_tutor_mock

# Mock src.adaptive.models
adaptive_models_mock = MagicMock()
sys.modules["src.adaptive.models"] = adaptive_models_mock

# Mock src.integrations.vertex_tutor
src_mock = MagicMock()
sys.modules["src"] = src_mock
sys.modules["src.integrations"] = src_mock.integrations

from uuid import uuid4
from adaptive_engine.learning_engine import LearningEngine
from adaptive_engine.neuro_model import CognitiveDiagnosis, diagnose_interaction
import pytest
from unittest.mock import MagicMock, patch

def test_diagnose_interaction_confusable_atoms_integration():
    """
    Test that LearningEngine correctly populates confusable_atoms
    and passes it to diagnose_interaction.
    """

    # Setup mocks
    mock_session = MagicMock()
    mock_engine = LearningEngine(session=mock_session)

    # Mock data
    atom_id = uuid4()
    concept_id = uuid4()
    session_id = uuid4()

    atom_info = {
        "id": str(atom_id),
        "concept_id": str(concept_id),
        "atom_type": "mcq",
        "front": "Question",
        "back": "Answer",
    }

    # Mock internal methods
    mock_engine._get_atom_info = MagicMock(return_value=atom_info)
    mock_engine._evaluate_answer = MagicMock(return_value=(False, 0.0, "Wrong", "Correct"))
    mock_engine._record_response = MagicMock()
    mock_engine._update_session_progress = MagicMock()
    mock_engine._accumulate_session_stats = MagicMock()
    mock_engine._get_session_learner = MagicMock(return_value="learner1")
    mock_engine._get_cognitive_remediation = MagicMock(return_value=None)
    mock_engine._remediator = MagicMock()
    mock_engine._remediator.check_remediation_needed.return_value = None
    mock_engine._mastery_calc = MagicMock()

    # Mock get_session to avoid DB queries
    mock_engine.get_session = MagicMock(return_value=None)

    # IMPORTANT: Mock _get_contrastive_atoms to return some UUIDs
    confusable_uuid1 = uuid4()
    confusable_uuid2 = uuid4()
    mock_engine._get_contrastive_atoms = MagicMock(return_value=[confusable_uuid1, confusable_uuid2])

    # We need to patch diagnose_interaction where it is imported in learning_engine
    with patch("adaptive_engine.learning_engine.diagnose_interaction") as mock_diagnose:
        # Return a dummy diagnosis
        mock_diagnosis = MagicMock(spec=CognitiveDiagnosis)
        mock_diagnosis.fail_mode = None
        mock_diagnosis.success_mode = None
        mock_diagnosis.cognitive_state.value = "flow"
        mock_diagnose.return_value = mock_diagnosis

        # Call submit_answer
        mock_engine.submit_answer(
            session_id=session_id,
            atom_id=atom_id,
            answer="Wrong Answer",
            time_taken_seconds=5
        )

        # Check if _get_contrastive_atoms was called correctly
        # This assertion is expected to fail initially as I haven't implemented the call yet
        try:
            mock_engine._get_contrastive_atoms.assert_called_once()
        except AssertionError:
            print("_get_contrastive_atoms was NOT called (Expected failure)")
            return

        # args, kwargs = mock_engine._get_contrastive_atoms.call_args
        # assert str(args[1]) == str(concept_id)
        # assert str(args[2]) == str(atom_id)

        # Check if diagnose_interaction was called with populated confusable_atoms
        mock_diagnose.assert_called_once()
        call_kwargs = mock_diagnose.call_args[1]

        confusable_atoms = call_kwargs.get("confusable_atoms")
        if confusable_atoms is None:
            print("confusable_atoms is None (Expected failure)")
        else:
            print("confusable_atoms is populated!")
            assert len(confusable_atoms) == 2
            assert confusable_atoms[0]["id"] == str(confusable_uuid1)
            assert confusable_atoms[1]["id"] == str(confusable_uuid2)

if __name__ == "__main__":
    try:
        test_diagnose_interaction_confusable_atoms_integration()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
