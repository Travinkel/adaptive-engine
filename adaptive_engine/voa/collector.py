from typing import List, Dict, Any
from . import metrics

class VOACollector:
    """
    Collects VOA metrics from telemetry data.
    """
    def __init__(self):
        self.telemetry_data = []

    def add_telemetry(self, is_correct: bool, response_time_ms: int):
        """
        Adds a single telemetry data point to the collector.
        """
        self.telemetry_data.append({
            'is_correct': is_correct,
            'response_time_ms': response_time_ms,
        })

    def get_metrics(self) -> Dict[str, float]:
        """

        Calculates and returns the VOA metrics.
        """
        return metrics.get_behavioral_metrics(self.telemetry_data)
