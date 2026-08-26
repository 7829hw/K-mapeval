from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# The second axis: what is *measured*. Two questions with the same MapEval task category are
# still different measurements -- a nearest-of-a-kind and a count-within-600m are both `nearby`
# upstream -- and this is the label that separates them.
BenchmarkClassification = Literal[
    "nearby",
    "poi",
    "routing",
    "trip",
    "type",
    "direction",
    "distance",
    "radius",
]

# The first axis: MapEval-API's own four task categories, plus the one this port added. It is not
# a coarsening of `classification` and cannot be recovered from it -- the `unanswerable_*`
# families are written as `nearby` questions and are told apart only by this field -- so the
# label travels in the dataset row and `resolve_mapeval_class` reads it there first.
MapEvalClass = Literal["nearby", "poi", "routing", "trip", "unanswerable"]

# What each measurement type is a measurement *of*, for the rows built before the label existed:
# `v1`, `v2` and `v3` carry no `mapeval_class` at all. Observed over every row in `dataset/` that
# carries both, this map reproduces the stored label exactly, with the single exception noted
# above -- so it is a fallback for old rows, never an override for a row that states its own.
_CLASSIFICATION_TO_MAPEVAL: dict[str, str] = {
    "nearby": "nearby",
    # A count or a set inside a stated radius is a Nearby search with a count measure.
    "radius": "nearby",
    "poi": "poi",
    # A pairwise comparison reports a property of the places, not a search around one.
    "distance": "poi",
    # Absorbed into `distance` by the v6 builder; still present in v1-v5 rows.
    "direction": "poi",
    # v1 only.
    "type": "poi",
    "routing": "routing",
    "trip": "trip",
}


class BenchmarkItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    options: list[str] = Field(min_length=2, max_length=4)
    answer: int = Field(ge=0)
    classification: BenchmarkClassification
    # Declared rather than left to `extra="allow"`: both have been written into every row since
    # the v4 builder, and until now neither was type-checked. `None` is what a v1/v2/v3 row has.
    mapeval_class: MapEvalClass | None = None
    template_id: str | None = None
    # Retained only so legacy MapEval-Textual rows round-trip as typed metadata. Runtime evidence
    # always comes from Kakao, and `agent_input` never returns this field.
    context: str | None = None
    region: str | None = None
    difficulty: Literal["easy", "medium", "hard"] | None = None
    verified_at: str | None = None

    @model_validator(mode="after")
    def answer_is_valid_option(self) -> BenchmarkItem:
        if self.answer >= len(self.options):
            raise ValueError("answer must be a 0-based index into options")
        if any(not option.strip() for option in self.options):
            raise ValueError("sample options must not be blank")
        return self

    def agent_input(self) -> tuple[str, list[str]]:
        """Return only benchmark-visible data; gold and metadata stay evaluation-only."""

        return self.question, list(self.options)


def load_dataset(path: str | Path) -> list[BenchmarkItem]:
    source = Path(path)
    items: list[BenchmarkItem] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                items.append(BenchmarkItem.model_validate_json(line))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Invalid dataset row at {source}:{line_number}: {exc}") from exc
    if not items:
        raise ValueError(f"Dataset is empty: {source}")
    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("Dataset IDs must be unique")
    return items


def resolve_mapeval_class(item: BenchmarkItem) -> str:
    """Which MapEval-API task category a question belongs to.

    The stored label wins whenever there is one. Deriving it from `classification` instead would
    silently fold the 21 `unanswerable_*` rows of a 300-question draw into `nearby`, which is the
    classification they are written under -- and those are the rows whose gold answer is
    "알 수 없음", so losing them loses the only families this port added to the paper's four.
    """

    if item.mapeval_class:
        return item.mapeval_class
    return _CLASSIFICATION_TO_MAPEVAL[item.classification]
