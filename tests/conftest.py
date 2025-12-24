"""
Pytest configuration for adaptive_engine tests.
"""
import sys
from pathlib import Path

import pytest

# Add the adaptive_engine package to the path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def sample_interactions():
    """Generate sample interaction history for ZPD testing."""
    interactions = []

    # Create a progression of interactions at various difficulties
    for i in range(50):
        difficulty = 0.3 + (i * 0.02)  # Range 0.3 to 1.3

        # Success probability decreases with difficulty
        success_prob = max(0.1, 1.0 - (difficulty - 0.3) * 1.5)
        is_correct = (i % 10) < int(success_prob * 10)

        interactions.append({
            "difficulty": difficulty,
            "is_correct": is_correct,
            "had_scaffold": difficulty > 0.7 and is_correct,
            "response_time_ms": 3000 + int(difficulty * 5000),
        })

    return interactions


@pytest.fixture
def zpd_engine():
    """Create a ZPDEngine instance for testing."""
    from adaptive_engine.zpd import ZPDEngine
    return ZPDEngine()


@pytest.fixture
def friction_vector():
    """Create a sample friction vector."""
    from adaptive_engine.zpd import NCDEFrictionVector
    return NCDEFrictionVector(
        retrieval_friction=1.5,
        integration_friction=2.0,
        execution_friction=1.0,
        metacognitive_friction=0.5,
    )
