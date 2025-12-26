from typing import Dict
import numpy as np
from sklearn.svm import LinearSVC

class VOAClassifier:
    """
    Classifies skill level based on VOA metrics using a Linear SVM.
    """
    def __init__(self):
        self._model = LinearSVC(random_state=0, dual="auto")
        self._feature_names = []

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list):
        """
        Trains the SVM classifier.

        Args:
            X: The training data.
            y: The target values.
            feature_names: The names of the features.
        """
        self._feature_names = feature_names
        self._model.fit(X, y)

    def classify(self, metrics: Dict[str, float]) -> str:
        """
        Classifies the skill level based on the provided metrics.
        """
        if not self._feature_names:
            return "untrained"

        X = np.array([metrics[name] for name in self._feature_names]).reshape(1, -1)
        prediction = self._model.predict(X)
        return prediction[0]

    def get_weights(self) -> Dict[str, float]:
        """
        Returns the feature weights (coefficients) from the trained model.
        """
        if not self._feature_names:
            return {}
        return dict(zip(self._feature_names, self._model.coef_[0]))
