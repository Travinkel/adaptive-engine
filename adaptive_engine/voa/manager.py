from .collector import VOACollector
from .classifier import VOAClassifier
from .selector import select_metrics
from .interpreter import interpret_weights
import numpy as np

class VOAManager:
    """
    Manages the VOA pipeline for a single learning session.
    """
    def __init__(self):
        self.collector = VOACollector()
        self.classifier = VOAClassifier()
        self.classification = "pending"
        self.feedback = "No feedback yet."

    def add_telemetry(self, is_correct: bool, response_time_ms: int):
        """Adds telemetry data to the collector."""
        self.collector.add_telemetry(is_correct, response_time_ms)

    def _get_mock_training_data(self):
        """
        Generates a mock dataset of novice and expert performance metrics.
        """
        # Experts are accurate and fast
        expert_metrics = np.random.rand(20, 2) * np.array([0.2, 500]) + np.array([0.8, 200])
        # Novices are less accurate and slower
        novice_metrics = np.random.rand(20, 2) * np.array([0.4, 1000]) + np.array([0.3, 800])

        X = np.vstack([expert_metrics, novice_metrics])
        y = np.array(["expert"] * 20 + ["novice"] * 20)
        feature_names = ["accuracy", "avg_response_time_ms"]

        return X, y, feature_names

    def process_session(self):
        """
        Processes the session data to classify performance and generate feedback.
        """
        X_train, y_train, feature_names = self._get_mock_training_data()

        # 1. Train the classifier with all features
        self.classifier.train(X_train, y_train, feature_names)

        # 2. Select the most important features
        # For this example, we'll just use all features
        selected_features = feature_names

        # 3. Get the current session's metrics
        current_metrics = self.collector.get_metrics()

        # 4. Classify the current session's performance
        self.classification = self.classifier.classify(current_metrics)

        # 5. Generate feedback
        weights = self.classifier.get_weights()
        self.feedback = interpret_weights(weights)
