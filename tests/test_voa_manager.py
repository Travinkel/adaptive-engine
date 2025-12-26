import pytest
from adaptive_engine.voa.manager import VOAManager
from unittest.mock import MagicMock
from uuid import uuid4

def test_voa_manager_initialization():
    manager = VOAManager()
    assert manager.classification == "pending"
    assert manager.feedback == "No feedback yet."

def test_voa_manager_telemetry():
    manager = VOAManager()
    manager.add_telemetry(is_correct=True, response_time_ms=1000)
    manager.add_telemetry(is_correct=False, response_time_ms=2000)
    assert len(manager.collector.telemetry_data) == 2

def test_voa_manager_process_session():
    manager = VOAManager()
    manager.add_telemetry(is_correct=True, response_time_ms=300)
    manager.add_telemetry(is_correct=True, response_time_ms=400)
    manager.process_session()
    assert manager.classification in ["expert", "novice"]
    assert "Focus on improving" in manager.feedback

def test_learning_engine_voa_integration():
    # Mock the LearningEngine to avoid database dependencies
    learning_engine = MagicMock()
    learning_engine._voa_managers = {}
    session_id = uuid4()

    # Simulate session creation
    learning_engine._voa_managers[session_id] = VOAManager()

    # Simulate telemetry
    learning_engine._voa_managers[session_id].add_telemetry(is_correct=True, response_time_ms=300)
    learning_engine._voa_managers[session_id].add_telemetry(is_correct=False, response_time_ms=1500)

    # Simulate session end
    learning_engine._voa_managers[session_id].process_session()

    # Check results
    voa_manager = learning_engine._voa_managers[session_id]
    assert voa_manager.classification == "novice"
    assert "Focus on improving" in voa_manager.feedback
