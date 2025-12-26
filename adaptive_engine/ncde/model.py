# adaptive_engine/ncde/model.py

import torch
import torch.nn as nn

class VectorField(nn.Module):
    """
    Defines the vector field for the NCDE.
    """
    def __init__(self, input_dim, hidden_dim):
        super(VectorField, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, hidden_dim * input_dim)

    def forward(self, t, h):
        """
        Computes the vector field at a given time and state.
        """
        return self.linear(h).view(-1, self.hidden_dim, self.input_dim)

class NCDE(nn.Module):
    """
    Neural Controlled Differential Equation model for cognitive state tracking.
    This class holds the vector field and the final readout layer.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NCDE, self).__init__()
        self.vector_field = VectorField(input_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, output_dim)
