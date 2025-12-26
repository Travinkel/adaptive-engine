# adaptive_engine/ncde/solver.py

import torch
import torchcde

class NCDESolver:
    """
    Wraps the torchcde solver to integrate the NCDE model.
    """
    def __init__(self, model, method='rk4'):
        self.model = model
        self.method = method

    def solve(self, h0, X):
        """
        Solves the NCDE initial value problem using the provided control path.

        Args:
            h0: The initial state of the system.
            X: A tensor representing the control path.
        """
        # Create the control path object required by torchcde
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X)
        X_path = torchcde.CubicSpline(coeffs)

        # Define the time points for the output, matching the input's time dimension
        t_points = torch.linspace(X_path.interval[0], X_path.interval[1], X.size(1))

        # Solve the CDE
        ht = torchcde.cdeint(X=X_path, z0=h0, func=self.model.vector_field, t=t_points, method=self.method)
        return ht

    def predict(self, h0, X):
        """
        Makes a prediction from the final state of the trajectory.

        Args:
            h0: The initial state of the system.
            X: A tensor representing the control path.
        """
        ht = self.solve(h0, X)
        # Get the final state and apply readout
        output = self.model.readout(ht[:, -1])
        return output
