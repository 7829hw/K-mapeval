"""K-MapEval multiple-choice adaptation outside the spatial reasoning core."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.agent.answering import GroundedAnswer


@dataclass(frozen=True)
class MCQSelection:
    index: int | None
    method: str


class MCQAdapter:
    def select(
        self,
        answer: GroundedAnswer,
        options: Sequence[str],
    ) -> MCQSelection:
        exact = _match(answer.text, options, strict=True)
        if exact is not None:
            return MCQSelection(exact, "exact_grounded_text")
        contained = _match(answer.text, options, strict=False)
        if contained is not None:
            return MCQSelection(contained, "grounded_text_containment")
        value_match = _match(str(answer.value), options, strict=True)
        if value_match is not None:
            return MCQSelection(value_match, "exact_grounded_value")
        return MCQSelection(None, "unresolved")


def _match(value: str, options: Sequence[str], *, strict: bool) -> int | None:
    key = _key(value)
    if not key:
        return None
    matches = [
        index
        for index, option in enumerate(options)
        if (option_key := _key(option))
        and (option_key == key or (not strict and (key in option_key or option_key in key)))
    ]
    return matches[0] if len(matches) == 1 else None


def _key(value: str) -> str:
    return "".join(str(value).split()).casefold()
