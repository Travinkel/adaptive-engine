# adaptive_engine/ncde_service.py

import torch
from .ncde.model import NCDE
from .ncde.solver import NCDESolver
from .ncde.state_manager import StateManager

class NCDE_Service:
    """
    Orchestrates the NCDE components to track cognitive states.
    """
    def __init__(self, db_config, input_dim, hidden_dim, output_dim):
        self.state_manager = StateManager(db_config)
        self.model = NCDE(input_dim, hidden_dim, output_dim)
        self.solver = NCDESolver(self.model)

    def process_learning_event(self, learner_id, learning_event_data):
        """
        Processes a learning event, updates the cognitive state, and persists it.

        Args:
            learner_id: The ID of the learner.
            learning_event_data: A tensor representing the learning event.
        """
        # 1. Retrieve the last known state
        last_state = self.state_manager.get_state(learner_id)
        if last_state is None:
            # Initialize with a zero vector if no state exists
            last_state = torch.zeros(1, self.model.vector_field.hidden_dim)
        else:
            last_state = torch.tensor(last_state).unsqueeze(0)

        # 2. Update the state with the new event
        # The learning_event_data is expected to be a sequence, e.g., (time, features)
        X = torch.tensor(learning_event_data).unsqueeze(0) # Add batch dimension

        # 3. Solve for the new state trajectory
        new_trajectory = self.solver.solve(last_state, X)

        # 4. Extract the final state from the trajectory
        new_state = new_trajectory[:, -1, :].squeeze().tolist()

        # 5. Save the new state
        self.state_manager.save_state(learner_id, new_state)

        return new_state
