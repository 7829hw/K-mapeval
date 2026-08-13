from __future__ import annotations

from pathlib import Path

import pytest

from k_mapeval.evaluation.dataset import BenchmarkItem, load_dataset
from k_mapeval.evaluation.parsing import parse_answer, parse_json_object


def test_answer_parser_is_conservative_and_one_based() -> None:
    assert parse_answer("근거를 확인했습니다. ^^Option 2^^", option_count=4) == 2
    assert parse_answer('{"predicted_option": 3}', option_count=4) == 3
    assert parse_answer("정답: 1", option_count=4) == 1
    assert parse_answer("2026년에 확인", option_count=4) is None
    assert parse_answer("^^0^^", option_count=4) is None


def test_json_parser_handles_fenced_output() -> None:
    assert parse_json_object('```json\n{"intent":"poi"}\n```') == {"intent": "poi"}


def test_sample_dataset_is_valid_and_balanced() -> None:
    items = load_dataset(Path("dataset/sample.jsonl"))
    assert len(items) == 8
    assert {item.classification for item in items} == {"nearby", "poi", "routing", "trip"}
    assert all(1 <= item.answer <= len(item.options) for item in items)


def test_invalid_gold_index_is_rejected() -> None:
    with pytest.raises(ValueError):
        BenchmarkItem(
            id="bad",
            question="q",
            options=["a", "b"],
            answer=3,
            classification="poi",
        )

