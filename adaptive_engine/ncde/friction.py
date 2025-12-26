# adaptive_engine/ncde/friction.py

import numpy as np

class CognitiveFriction:
    """
    Computes cognitive friction based on the cognitive state trajectory.
    """
    def __init__(self, trajectory):
        self.trajectory = trajectory

    def compute(self, t):
        """
        Calculates the cognitive friction at a given time t.
        """
        # Example: Friction as the norm of the state derivative
        state_derivative = self.trajectory.derivative(t)
        return np.linalg.norm(state_derivative)
