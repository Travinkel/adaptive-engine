from typing import List, Dict, Any

def get_behavioral_metrics(telemetry: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculates behavioral metrics from telemetry data.

    Args:
        telemetry: A list of telemetry data points.

    Returns:
        A dictionary of calculated metrics.
    """
    if not telemetry:
        return {}

    total_atoms = len(telemetry)
    accuracy = sum(1 for t in telemetry if t['is_correct']) / total_atoms
    avg_response_time = sum(t['response_time_ms'] for t in telemetry) / total_atoms

    metrics = {
        'total_atoms': total_atoms,
        'accuracy': accuracy,
        'avg_response_time_ms': avg_response_time
    }

    return metrics
