# tests/test_ncde.py

import pytest
import torch
from unittest.mock import MagicMock, patch

from adaptive_engine.ncde.model import NCDE, VectorField
from adaptive_engine.ncde.solver import NCDESolver
from adaptive_engine.ncde.state_manager import StateManager
from adaptive_engine.ncde.friction import CognitiveFriction

def test_vector_field_forward_pass():
    """Tests the forward pass of the VectorField."""
    vector_field = VectorField(input_dim=3, hidden_dim=8)
    h = torch.randn(1, 8)
    output = vector_field(t=0, h=h)
    assert output.shape == (1, 8, 3)

def test_ncde_solver_solve():
    """Tests the solve method of the NCDESolver."""
    model = NCDE(input_dim=3, hidden_dim=8, output_dim=1)
    solver = NCDESolver(model)

    h0 = torch.randn(1, 8)
    X = torch.randn(1, 10, 3) # (batch, time, input_dim)

    ht = solver.solve(h0, X)
    assert ht.shape == (1, 10, 8)

def test_ncde_solver_predict():
    """Tests the predict method of the NCDESolver."""
    model = NCDE(input_dim=3, hidden_dim=8, output_dim=1)
    solver = NCDESolver(model)

    h0 = torch.randn(1, 8)
    X = torch.randn(1, 10, 3) # (batch, time, input_dim)

    output = solver.predict(h0, X)
    assert output.shape == (1, 1)

@patch('psycopg2.pool.SimpleConnectionPool')
def test_state_manager_get_state(mock_pool):
    """Tests retrieving a state from the StateManager."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_pool.return_value.getconn.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    mock_cursor.fetchone.return_value = ([0.1, 0.2, 0.3],)

    manager = StateManager({})
    state = manager.get_state(learner_id='test_learner')

    assert state == [0.1, 0.2, 0.3]
    mock_cursor.execute.assert_called_with("SELECT state_vector FROM cognitive_states WHERE learner_id = %s ORDER BY timestamp DESC LIMIT 1", ('test_learner',))

@patch('psycopg2.pool.SimpleConnectionPool')
def test_state_manager_save_state(mock_pool):
    """Tests saving a state with the StateManager."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_pool.return_value.getconn.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    manager = StateManager({})
    manager.save_state(learner_id='test_learner', state_vector=[0.4, 0.5, 0.6])

    mock_cursor.execute.assert_called_with("INSERT INTO cognitive_states (learner_id, state_vector) VALUES (%s, %s)", ('test_learner', [0.4, 0.5, 0.6]))
    mock_conn.commit.assert_called_once()

def test_cognitive_friction_compute():
    """Tests the cognitive friction computation."""
    mock_trajectory = MagicMock()
    mock_trajectory.derivative.return_value = [0.1, 0.2, 0.3]

    friction = CognitiveFriction(mock_trajectory)
    result = friction.compute(t=0.5)

    # Expected: sqrt(0.1^2 + 0.2^2 + 0.3^2) = sqrt(0.01 + 0.04 + 0.09) = sqrt(0.14)
    assert abs(result - 0.3741657) < 1e-6
