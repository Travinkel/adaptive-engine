import pytest
import numpy as np
from adaptive_engine.voa.collector import VOACollector
from adaptive_engine.voa.metrics import get_behavioral_metrics
from adaptive_engine.voa.selector import select_metrics
from adaptive_engine.voa.classifier import VOAClassifier
from adaptive_engine.voa.interpreter import interpret_weights

def test_get_behavioral_metrics():
    telemetry = [
        {'is_correct': True, 'response_time_ms': 1000},
        {'is_correct': False, 'response_time_ms': 2000},
        {'is_correct': True, 'response_time_ms': 3000},
    ]
    metrics = get_behavioral_metrics(telemetry)
    assert metrics['total_atoms'] == 3
    assert metrics['accuracy'] == 2 / 3
    assert metrics['avg_response_time_ms'] == 2000

def test_voa_collector():
    collector = VOACollector()
    collector.add_telemetry(is_correct=True, response_time_ms=1000)
    collector.add_telemetry(is_correct=False, response_time_ms=2000)
    collector.add_telemetry(is_correct=True, response_time_ms=3000)
    metrics = collector.get_metrics()
    assert metrics['total_atoms'] == 3
    assert metrics['accuracy'] == 2 / 3
    assert metrics['avg_response_time_ms'] == 2000

def test_select_metrics():
    # Use a deterministic dataset where metric1 is the only perfect predictor
    X = np.array([
        [10, 5, 0, 0, 0],  # y=1
        [0, 5, 0, 0, 0],   # y=0
        [10, 5, 1, 0, 0],  # y=1
        [0, 5, 1, 0, 0],   # y=0
        [10, 5, 0, 1, 0],  # y=1
        [0, 5, 0, 1, 0],   # y=0
    ])
    y = np.array([1, 0, 1, 0, 1, 0])
    feature_names = ["metric1", "metric2", "metric3", "metric4", "metric5"]

    # metric1 is perfectly correlated with the target. metric2 has no predictive power.
    selected = select_metrics(X, y, feature_names, n_features_to_select=1)
    assert selected == ["metric1"]

def test_voa_classifier():
    classifier = VOAClassifier()
    X = np.array([[1, 1], [1, 1], [0, 0], [0, 0]])
    y = np.array(["expert", "expert", "novice", "novice"])
    feature_names = ["metric1", "metric2"]
    classifier.train(X, y, feature_names)

    # Test classification
    metrics = {"metric1": 1, "metric2": 1}
    assert classifier.classify(metrics) == "expert"
    metrics = {"metric1": 0, "metric2": 0}
    assert classifier.classify(metrics) == "novice"

    # Test weights
    weights = classifier.get_weights()
    assert "metric1" in weights
    assert "metric2" in weights

def test_interpret_weights():
    weights = {'a': 0.1, 'b': 0.9}
    feedback = interpret_weights(weights)
    assert feedback == "Focus on improving b."
