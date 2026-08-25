from __future__ import annotations

from typing import Any

MATRIX_METRICS: dict[str, str] = {"duration": "duration_s", "distance": "distance_m"}

TSP_METRIC_ALIASES: dict[str, str] = {
    "time": "duration",
    "travel_time": "duration",
    "duration_s": "duration",
    "seconds": "duration",
    "distance_m": "distance",
    "metres": "distance",
    "meters": "distance",
}


def normalize_tsp_metric(value: Any) -> Any:
    """Canonicalize exact objective/unit synonyms without inferring an objective."""

    if not isinstance(value, str):
        return value
    normalized = value.strip().lower()
    return TSP_METRIC_ALIASES.get(normalized, normalized)
