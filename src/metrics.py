from __future__ import annotations

from collections import defaultdict
from typing import Any


def _accuracy_by(records: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get(field) or "unknown")].append(record)
    return {
        label: {
            "questions": len(rows),
            "correct": sum(bool(row["correct"]) for row in rows),
            "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
        }
        for label, rows in sorted(grouped.items())
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll a record list up along the three dataset label axes.

    The axes are independent properties of the question, not coarsenings of one another:
    `mapeval_class` is MapEval-API's task category (its `unanswerable` value is this port's own
    addition and belongs in its own row, not averaged into the paper's four), `classification`
    is what gets measured, and `template_id` is the generator family error analysis works at.

    Nothing in `src/` or `main.py` calls this today -- the axis a run actually reports comes from
    `evaluator.calculate_statistics`, which computes the same three over its own row shape.
    """

    total = len(records)
    correct = sum(bool(record["correct"]) for record in records)
    return {
        "questions": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "mapeval_class_accuracy": _accuracy_by(records, "mapeval_class"),
        "classification_accuracy": _accuracy_by(records, "classification"),
        "template_accuracy": _accuracy_by(records, "template_id"),
        "tool_calls": sum(int(record["tool_calls"]) for record in records),
        "api_calls": sum(int(record["api_calls"]) for record in records),
        "cache_hits": sum(int(record.get("cache_hits", 0)) for record in records),
        "cache_misses": sum(int(record.get("cache_misses", 0)) for record in records),
        "reasoning_steps": sum(int(record["reasoning_steps"]) for record in records),
        "latency_ms": sum(float(record["latency_ms"]) for record in records),
        "failures": sum(record.get("failure_type") is not None for record in records),
    }
