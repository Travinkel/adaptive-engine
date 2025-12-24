import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ItemParameters:
    difficulty_b: float
    discrimination_a: float
    guessing_c: float

class IRTEngine:
    """
    3-Parameter Logistic (3PL) Item Response Theory Engine.
    Used for calibrating Item Difficulty and estimating User Ability (Theta).
    
    P(theta) = c + (1 - c) / (1 + e^(-a(theta - b)))
    """

    def __init__(self):
        pass

    def calculate_probability(self, theta: float, item: ItemParameters) -> float:
        """
        Calculates the probability of a user with ability `theta` answering 
        correctly on an item with parameters (a, b, c).
        """
        exponent = -item.discrimination_a * (theta - item.difficulty_b)
        
        # Avoid overflow
        if exponent > 20: 
            return item.guessing_c
        if exponent < -20: 
            return 1.0
            
        probability = item.guessing_c + (1.0 - item.guessing_c) / (1.0 + math.exp(exponent))
        return probability

    def estimate_theta(self, current_theta: float, item: ItemParameters, is_correct: bool) -> float:
        """
        Bayesian update of theta based on a single response.
        """
        # Simplified update step (Logic: Move theta towards item difficulty if correct, away if wrong)
        # In a full system, this would use Newton-Raphson on the Likelihood function.
        
        prob = self.calculate_probability(current_theta, item)
        
        # Gradient approximation
        step_size = 0.5 * item.discrimination_a
        
        if is_correct:
            # If correct but low prob -> Big jump up
            delta = step_size * (1 - prob)
        else:
            # If wrong but high prob -> Big jump down
            delta = step_size * (0 - prob)
            
        return current_theta + delta

    def select_next_item(self, theta: float, available_items: list[ItemParameters]) -> Optional[ItemParameters]:
        """
        Selection Strategy: Information Maximization (Fisher Information).
        Selects the item where P(theta) is closest to 0.5 (Maximum uncertainty).
        """
        best_item = None
        min_diff = float("inf")
        
        for item in available_items:
            # Target probability = 0.5 (Desirable Difficulty / Vygotsky Zone)
            prob = self.calculate_probability(theta, item)
            diff = abs(prob - 0.5)
            
            if diff < min_diff:
                min_diff = diff
                best_item = item
                
        return best_item
