# tests/test_ncde_service.py

import pytest
import torch
from unittest.mock import MagicMock, patch

from adaptive_engine.ncde_service import NCDE_Service

@patch('adaptive_engine.ncde_service.StateManager')
@patch('adaptive_engine.ncde_service.NCDESolver')
@patch('adaptive_engine.ncde_service.NCDE')
def test_ncde_service_process_learning_event(MockNCDE, MockNCDESolver, MockStateManager):
    """
    Tests the process_learning_event method of the NCDE_Service.
    """
    # Setup mocks
    mock_state_manager = MockStateManager.return_value
    mock_solver = MockNCDESolver.return_value
    mock_model = MockNCDE.return_value
    mock_model.vector_field.hidden_dim = 8

    # Mock the return values
    mock_state_manager.get_state.return_value = [0.1] * 8
    mock_solver.solve.return_value = torch.randn(1, 10, 8)

    # Initialize the service inside the test function
    service = NCDE_Service(db_config={}, input_dim=3, hidden_dim=8, output_dim=1)

    # Define a sample learning event
    learning_event = [[0.1, 0.2, 0.3]] * 10

    # Process the event
    new_state = service.process_learning_event('test_learner', learning_event)

    # Assertions
    mock_state_manager.get_state.assert_called_with('test_learner')
    mock_solver.solve.assert_called_once()
    mock_state_manager.save_state.assert_called_with('test_learner', new_state)
    assert isinstance(new_state, list)
    assert len(new_state) == 8

@patch('adaptive_engine.ncde_service.StateManager')
@patch('adaptive_engine.ncde_service.NCDESolver')
@patch('adaptive_engine.ncde_service.NCDE')
def test_ncde_service_initial_state(MockNCDE, MockNCDESolver, MockStateManager):
    """
    Tests the process_learning_event method for a new learner with no prior state.
    """
    # Setup mocks
    mock_state_manager = MockStateManager.return_value
    mock_solver = MockNCDESolver.return_value
    mock_model = MockNCDE.return_value
    mock_model.vector_field.hidden_dim = 8

    # Mock the return values
    mock_state_manager.get_state.return_value = None # No prior state
    mock_solver.solve.return_value = torch.randn(1, 10, 8)

    # Initialize the service
    service = NCDE_Service(db_config={}, input_dim=3, hidden_dim=8, output_dim=1)

    # Define a sample learning event
    learning_event = [[0.1, 0.2, 0.3]] * 10

    # Process the event
    new_state = service.process_learning_event('new_learner', learning_event)

    # Assertions
    mock_state_manager.get_state.assert_called_with('new_learner')
    # Verify that the initial state passed to the solver is a zero tensor
    initial_state_arg = mock_solver.solve.call_args[0][0]
    assert torch.all(initial_state_arg == 0)
    mock_state_manager.save_state.assert_called_with('new_learner', new_state)
