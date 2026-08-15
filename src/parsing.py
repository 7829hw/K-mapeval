from __future__ import annotations

import json
import re
from typing import Any


def parse_answer(text: str, *, option_count: int) -> int | None:
    """Parse MapEval's ^^N^^ format, then conservative JSON/plain fallbacks."""

    if not text:
        return None
    patterns = (
        r"\^\^\s*(?:option[\s_-]*)?(\d+)\s*\^\^",
        r'"predicted_(?:option|answer)"\s*:\s*"?(\d+)',
        r"(?:option|답|정답)\s*[:：]?\s*(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = int(match.group(1))
            return value if 1 <= value <= option_count else None
    if re.fullmatch(r"\s*\d+\s*", text):
        value = int(text.strip())
        return value if 1 <= value <= option_count else None
    return None


def parse_json_object(text: str) -> dict[str, Any]:
    if not text:
        return {}
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE)
    try:
        value = json.loads(stripped)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(stripped[start : end + 1])
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}
