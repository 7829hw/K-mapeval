from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class BenchmarkItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    options: list[str] = Field(min_length=2, max_length=4)
    answer: int = Field(ge=0)
    classification: BenchmarkClassification
    # Retrieval results the dataset ships for this question, in MapEval's context format. It is
    # provider evidence, not agent input: the tool layer serves it, `agent_input` never returns it.
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
