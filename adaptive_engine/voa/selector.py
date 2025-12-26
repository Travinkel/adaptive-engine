from typing import List
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC

def select_metrics(X: np.ndarray, y: np.ndarray, feature_names: List[str], n_features_to_select: int = 5) -> List[str]:
    """
    Selects the most discriminative metrics using Recursive Feature Elimination (RFE).

    Args:
        X: The training data.
        y: The target values.
        feature_names: The names of the features.
        n_features_to_select: The number of features to select.

    Returns:
        A list of the most discriminative metric names.
    """
    estimator = LinearSVC(random_state=0, dual="auto")
    selector = RFE(estimator, n_features_to_select=n_features_to_select, step=1)
    selector = selector.fit(X, y)

    selected_features = []
    for i, feature_name in enumerate(feature_names):
        if selector.support_[i]:
            selected_features.append(feature_name)

    return selected_features
