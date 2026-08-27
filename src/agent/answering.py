"""Grounded answer generation, independent of multiple-choice options."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GroundedAnswer:
    value: Any
    text: str
    confidence: float | None = None
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "text": self.text,
            "confidence": self.confidence,
            "reason": self.reason,
        }


def grounded_answer_from_payload(payload: Mapping[str, Any]) -> GroundedAnswer:
    """Accept the paper-facing answer wire format and the previous response during migration."""

    value = payload.get("value", payload.get("grounded_value", payload.get("predicted_answer")))
    text = payload.get("text", payload.get("grounded_answer", payload.get("predicted_answer", "")))
    confidence = payload.get("confidence")
    try:
        confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None
    return GroundedAnswer(value, str(text or ""), confidence, str(payload.get("reason") or ""))
