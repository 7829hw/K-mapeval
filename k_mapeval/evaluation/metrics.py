from __future__ import annotations

from collections import defaultdict
from typing import Any


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    correct = sum(bool(record["correct"]) for record in records)
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_class[str(record["classification"])].append(record)
    return {
        "questions": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "classification_accuracy": {
            classification: {
                "questions": len(rows),
                "correct": sum(bool(row["correct"]) for row in rows),
                "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
            }
            for classification, rows in sorted(by_class.items())
        },
        "tool_calls": sum(int(record["tool_calls"]) for record in records),
        "api_calls": sum(int(record["api_calls"]) for record in records),
        "cache_hits": sum(int(record.get("cache_hits", 0)) for record in records),
        "cache_misses": sum(int(record.get("cache_misses", 0)) for record in records),
        "reasoning_steps": sum(int(record["reasoning_steps"]) for record in records),
        "latency_ms": sum(float(record["latency_ms"]) for record in records),
        "failures": sum(record.get("failure_type") is not None for record in records),
    }
