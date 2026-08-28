"""K-MapEval multiple-choice adaptation outside the spatial reasoning core.

Reconciliation, never reasoning. Everything here reads an answer the GeoFlow core already
produced and decides which offered option says the same thing; nothing here looks at the
question, the evidence, or the map. The core is deliberately blind to the options -- it answers
in metres, in seconds and in counts -- while the options are written the way a person would say
it, so an answer of `1518.07 m` and an option of `약 1.5km` are the same answer in two notations
and matching them by text cannot work.

The rule for a quantity is rounding, not nearness: an answer matches an option when, written at
the precision that option is written to, it *is* that option. `1518.07 m` written to the nearest
0.1 km is `1.5 km`. An answer that rounds to nothing on offer resolves to nothing -- there is no
least-bad match here, and a question the core answered wrongly has to stay wrong.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from src.agent.answering import GroundedAnswer


@dataclass(frozen=True)
class MCQSelection:
    index: int | None
    method: str


#: Unit tokens by dimension, each with what one of it is worth in that dimension's base unit.
#: Length is metres, duration is seconds, and a count is itself.
_UNITS: dict[str, tuple[str, float]] = {
    "km": ("length", 1000.0),
    "킬로미터": ("length", 1000.0),
    "m": ("length", 1.0),
    "미터": ("length", 1.0),
    "시간": ("duration", 3600.0),
    "분": ("duration", 60.0),
    "초": ("duration", 1.0),
    "곳": ("count", 1.0),
    "개": ("count", 1.0),
    "군데": ("count", 1.0),
}

#: The counting numerals a Korean option writes instead of a digit: `세 곳` is 3.
_NUMERALS: dict[str, int] = {
    "한": 1, "하나": 1, "두": 2, "둘": 2, "세": 3, "셋": 3, "서": 3, "네": 4, "넷": 4,
    "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10,
}

_NUMBER = re.compile(r"(-?\d+(?:[.,]\d+)?)\s*(km|m|킬로미터|미터|시간|분|초|곳|개|군데)?", re.I)
_NUMERAL_COUNT = re.compile(
    r"(" + "|".join(sorted(_NUMERALS, key=len, reverse=True)) + r")\s*(곳|개|군데)"
)


@dataclass(frozen=True)
class Quantity:
    """A written amount, in its dimension's base unit, with what its notation resolves."""

    value: float
    dimension: str
    #: The step the written form distinguishes: `1.5km` resolves 0.1 km, so 100 m.
    resolution: float
    #: What one of the unit it was written in is worth in the base unit: 1000.0 for `km`.
    unit_scale: float = 1.0


def dimensions_named(text: str) -> frozenset[str]:
    """Which dimensions a string names units of: `약 7.26km` names length, `3곳` names a count.

    What keeps the bare value from being read in a unit the answer never used. Without it a
    count of 3 read as kilometres matched an option of `약 3km`.
    """

    return frozenset(
        _UNITS[match.group(2).lower()][0]
        for match in _NUMBER.finditer(text.replace(",", ""))
        if match.group(2)
    ) | frozenset({"count"} if _NUMERAL_COUNT.search(text) else ())


def parse_quantity(text: str) -> Quantity | None:
    """The one amount a string states, or nothing when it states none or several."""

    numeral = _NUMERAL_COUNT.search(text)
    if numeral is not None:
        return Quantity(float(_NUMERALS[numeral.group(1)]), "count", 1.0, 1.0)
    found = [
        match
        for match in _NUMBER.finditer(text.replace(",", ""))
        if match.group(1) not in ("", "-")
    ]
    if len(found) != 1:
        # Two amounts in one string name no single amount, and picking either would be the
        # least-bad match this module exists not to make.
        return None
    digits, unit = found[0].group(1), (found[0].group(2) or "").lower()
    dimension, scale = _UNITS.get(unit, ("count", 1.0) if not unit else (None, 0.0))
    if dimension is None:
        return None
    written = digits.split(".")
    decimals = len(written[1]) if len(written) > 1 else 0
    return Quantity(float(digits) * scale, dimension, scale * (10.0**-decimals), scale)


#: How this port's `unanswerable` families write the option that says the map cannot answer.
_CANNOT_KNOW = "알 수 없"


class MCQAdapter:
    def select(
        self,
        answer: GroundedAnswer,
        options: Sequence[str],
        *,
        execution_errors: int = 0,
    ) -> MCQSelection:
        """Which option the grounded answer says, or none.

        `execution_errors` is how many steps raised. It is not evidence about the question; it is
        what separates "the map does not carry price levels" from "my route computation broke",
        and only the first of those is an answer -- see `_match_declined`.
        """

        exact = _match(answer.text, options, strict=True)
        if exact is not None:
            return MCQSelection(exact, "exact_grounded_text")
        contained = _match(answer.text, options, strict=False)
        if contained is not None:
            return MCQSelection(contained, "grounded_text_containment")
        value_match = _match(str(answer.value), options, strict=True)
        if value_match is not None:
            return MCQSelection(value_match, "exact_grounded_value")
        # The value is a name as often as it is a number, and a name arrives decorated:
        # `명이비인후과 (서울 광진구 용마산로 5)` holds the option `명이비인후과` and was tried only
        # for equality, so the option went unmatched while the answer named it. Never for a
        # numeric value: a count of `3` is a substring of `약 3km` and of nothing else, which is
        # a unique match and a wrong one. Numbers are reconciled by quantity, below.
        if not isinstance(answer.value, (int, float, bool)):
            value_contained = _match(str(answer.value), options, strict=False)
            if value_contained is not None:
                return MCQSelection(value_contained, "grounded_value_containment")
        quantity = _match_quantity(answer, options)
        if quantity is not None:
            return MCQSelection(quantity, "grounded_quantity")
        ordering = _match_ordering(answer.text, options)
        if ordering is not None:
            return MCQSelection(ordering, "grounded_ordering")
        declined = _match_declined(answer, options, execution_errors)
        if declined is not None:
            return MCQSelection(declined, "grounded_decline")
        return MCQSelection(None, "unresolved")


def _match_quantity(answer: GroundedAnswer, options: Sequence[str]) -> int | None:
    """Which option the answer's amount rounds to, when exactly one of them it does."""

    stated = answer.value if isinstance(answer.value, (int, float)) else None
    # The core writes its value as a string carrying its unit as often as it writes a number:
    # `12490.86541977818 m`, `9.474132586329167 km`, `29523 m`. That is a quantity, and reading
    # it is more reliable than reading the prose, which holds every other number the answer
    # mentions -- `신길역 5호선` among them.
    valued = parse_quantity(str(answer.value)) if stated is None else None
    written = parse_quantity(answer.text)
    named = dimensions_named(answer.text) | (
        frozenset({valued.dimension}) if valued is not None else frozenset()
    )
    matched: list[int] = []
    for index, option in enumerate(options):
        offered = parse_quantity(option)
        if offered is None:
            continue
        candidates = [value for value in (written, valued) if value is not None]
        if valued is not None and valued.unit_scale != 1.0:
            # The unit the core writes is not reliable: one answer read `약 14474.72 km` for a
            # distance of 14.5 km, having labelled metres as kilometres. The bare number is the
            # measurement; the label is prose. Read it in the base unit too, and let the
            # single-match rule below decide whether that reading means anything.
            candidates.append(
                Quantity(
                    valued.value / valued.unit_scale, valued.dimension, offered.resolution
                )
            )
        if stated is not None:
            # The core writes its value in whichever unit its evidence carried -- 7.26 for
            # kilometres in one question and 6552.89 for metres in the next -- so the bare number
            # is read both as the dimension's base unit and as the unit this option is written
            # in. Reading it two ways can only add candidates, and the single-match rule below is
            # what keeps an added candidate from deciding anything on its own.
            if not named or offered.dimension in named:
                candidates.append(Quantity(float(stated), offered.dimension, offered.resolution))
                if named:
                    candidates.append(
                        Quantity(
                            float(stated) * offered.unit_scale,
                            offered.dimension,
                            offered.resolution,
                        )
                    )
        if any(
            candidate.dimension == offered.dimension
            and abs(candidate.value - offered.value) <= offered.resolution / 2
            for candidate in candidates
        ):
            matched.append(index)
    return matched[0] if len(matched) == 1 else None


#: How an option writes an itinerary. The answer writes the same order as prose.
_ORDER_SEPARATOR = re.compile(r"\s*(?:→|->|=>)\s*")


def _match_declined(
    answer: GroundedAnswer, options: Sequence[str], execution_errors: int
) -> int | None:
    """The `알 수 없음` option, when the core produced no value and nothing broke.

    This port adds an `unanswerable` category to MapEval's four, and its gold answer *is* that
    option. The core is blind to the options, so it says "the evidence carries no rating or price
    information" in prose and there was nothing to match it to: those families read 0 of 7 in
    every run of this stack.

    `AGENTS.md` records why this has to be narrow. A generation-stage "insufficient evidence"
    escape hatch once cost 5.8 points and took 50 points off the very families it was meant to
    help. This is not that hatch: it invents no decline and licenses none. The core has already
    declined on its own, and all that happens here is that its decline is read against an option
    that says the same thing. Three guards keep it from becoming the hatch -- no other method
    matched, the core produced *no value at all* rather than one that failed to match, and no
    step raised, which is what separates a map that does not carry the answer from a run that
    broke. Measured over three recorded passes: fires 13 times, right 10, and every one of the
    three it gets wrong was already unresolved, so it cost nothing.
    """

    if execution_errors or not _declined(answer.value):
        return None
    offered = [index for index, option in enumerate(options) if _CANNOT_KNOW in option]
    return offered[0] if len(offered) == 1 else None


def _declined(value: object) -> bool:
    """No measurement at all -- not a measurement that failed to match an option."""

    if value is None:
        return True
    return isinstance(value, str) and value.strip() in {"", "없음", "알 수 없음", "N/A", "null"}


def _match_ordering(text: str, options: Sequence[str]) -> int | None:
    """The option whose sequence the answer names in that order, when exactly one does.

    An ordering question offers permutations of the same places, so which one an answer means is
    decided by the order it names them in and nothing else. The core states it as prose --
    `A, B, C를 순서대로 방문한 후` -- against options written `A → B → C`, and no amount of
    text containment bridges the separator. Permutations discriminate themselves: only the
    option whose places appear in increasing position matches.
    """

    matched: list[int] = []
    for index, option in enumerate(options):
        parts = [part.strip() for part in _ORDER_SEPARATOR.split(option) if part.strip()]
        if len(parts) < 2:
            continue
        cursor = -1
        for part in parts:
            found = text.find(part, cursor + 1)
            if found <= cursor:
                cursor = -1
                break
            cursor = found
        if cursor > -1:
            matched.append(index)
    return matched[0] if len(matched) == 1 else None


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
