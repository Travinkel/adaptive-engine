from typing import List, Tuple, Dict
import math

class NCDEDynamics:
    """
    Neural Controlled Differential Equation (NCDE) Solver for Mastery Tracking.
    Models the continuous evolution of learner state (Z_t) driven by interaction events (X_t).
    
    Equation: dZ(t) = f(Z(t)) dX(t)
    where X_t is the control path (latency, correctness).
    """

    def __init__(self, friction_threshold: float = 2.5):
        self.state_z = 0.5  # Initial Mastery State (0.0 to 1.0)
        self.friction_threshold = friction_threshold
        self.lattice_stability = 1.0 # 1.0 = Stable, 0.0 = Collaspe

    def process_interaction(self, latency_ms: int, is_correct: bool, difficulty: float) -> Dict[str, float]:
        """
        Updates the continuous state based on a discrete discrete interaction point.
        Uses a theoretical approximation of the Neural CDE update step.
        """
        
        # 1. Calculate Control Signal (dX)
        # Higher latency on easy tasks = High Friction
        normalized_latency = min(latency_ms / 10000.0, 1.0) 
        dx_val = (1.0 if is_correct else -1.0) * difficulty
        
        # 2. Calculate Cognitive Friction (Velocity of degradation)
        # If correct but slow, friction increases.
        # If incorrect and fast, friction increases (rushing).
        friction = 0.0
        if is_correct:
             friction = normalized_latency * 0.5 # Efficient thought has low latency
        else:
             friction = (1.0 - normalized_latency) + 1.0 # Fast failure is bad
             
        # 3. Update State (dZ) - Euler Step approximation
        # dZ = tanh(CurrentState * Difficulty) * dX
        # A simple non-linear update dynamics
        learning_rate = 0.1
        dz = math.tanh(self.state_z * difficulty) * dx_val * learning_rate
        
        self.state_z = max(0.0, min(1.0, self.state_z + dz))
        
        # 4. Predict Lattice Collapse
        # If Friction exceeds threshold repeatedly, stability drops
        if friction > self.friction_threshold:
            self.lattice_stability *= 0.8
        else:
            self.lattice_stability = min(1.0, self.lattice_stability * 1.1)

        result = {
            "new_theta": self.state_z,
            "friction_vector": friction,
            "lattice_stability": self.lattice_stability,
            "predicted_collapse": self.lattice_stability < 0.4
        }
        
        return result

    def get_current_trajectory(self) -> float:
        """
        Returns the instantaneous velocity of mastery (dZ/dt).
        """
        return self.state_z
