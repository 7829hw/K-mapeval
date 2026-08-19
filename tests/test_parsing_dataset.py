from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from src.dataset import BenchmarkItem, load_dataset
from src.parsing import parse_answer, parse_json_object


def test_answer_parser_is_conservative_and_zero_based() -> None:
    assert parse_answer("근거를 확인했습니다. ^^Option 2^^", option_count=4) == 2
    assert parse_answer("^^Option_1^^", option_count=4) == 1
    assert parse_answer("^^Option-3^^", option_count=4) == 3
    assert parse_answer('{"predicted_option": 3}', option_count=4) == 3
    assert parse_answer("정답: 1", option_count=4) == 1
    assert parse_answer("2026년에 확인", option_count=4) is None
    assert parse_answer("^^0^^", option_count=4) == 0
    assert parse_answer("^^4^^", option_count=4) is None


def test_json_parser_handles_fenced_output() -> None:
    assert parse_json_object('```json\n{"intent":"poi"}\n```') == {"intent": "poi"}


def test_context_benchmark_is_valid_and_carries_its_own_evidence() -> None:
    items = load_dataset(Path("dataset/seoul_mapeval_v1_mcq_100.jsonl"))
    assert len(items) == 100
    assert {item.classification for item in items} == {
        "nearby",
        "radius",
        "type",
        "direction",
        "distance",
        "routing",
    }
    assert all(0 <= item.answer < len(item.options) for item in items)
    assert all(item.context for item in items)


def test_the_context_never_reaches_the_agent() -> None:
    items = load_dataset(Path("dataset/seoul_mapeval_v1_mcq_100.jsonl"))
    question, options = items[0].agent_input()
    assert question == items[0].question
    assert options == items[0].options
    assert items[0].context not in question
    assert all(items[0].context not in option for option in options)


REPRODUCTION_BENCHMARK = Path("dataset/seoul_kmapeval_v2_mcq_100.jsonl")


def test_reproduction_benchmark_mirrors_mapeval_class_mix() -> None:
    """The multi-hop families are the point of this benchmark; a regression that drops them
    would leave a dataset that cannot tell the two architectures apart."""

    items = load_dataset(REPRODUCTION_BENCHMARK)
    assert len(items) == 100
    counts = Counter(item.classification for item in items)
    assert counts["trip"] == 24
    assert counts["routing"] == 23
    assert counts["poi"] == 23
    assert counts["nearby"] + counts["direction"] + counts["radius"] == 30
    assert all(0 <= item.answer < len(item.options) for item in items)
    assert all(len(item.options) == len(set(item.options)) for item in items)
    assert len({item.id for item in items}) == 100
    # It is graded against live Kakao, so it must not carry context evidence of its own.
    assert not any(item.context for item in items)


def test_gold_evidence_is_evaluation_only() -> None:
    """`gold_evidence` records why an answer is the answer. It must stay with `answer`."""

    items = load_dataset(REPRODUCTION_BENCHMARK)
    carrying = [item for item in items if getattr(item, "gold_evidence", None)]
    assert len(carrying) == 100
    for item in carrying[:20]:
        question, options = item.agent_input()
        assert question == item.question
        assert options == item.options
        rendered = repr(item.gold_evidence)
        assert rendered not in question
        assert all(rendered not in option for option in options)


def test_reproduction_benchmark_option_position_is_not_evidence() -> None:
    items = load_dataset(REPRODUCTION_BENCHMARK)
    spread = Counter(item.answer for item in items)
    assert set(spread) == {0, 1, 2, 3}
    # No index may carry so much of the gold that guessing it beats answering.
    assert max(spread.values()) <= 40


COMPOSITIONAL_BENCHMARK = Path("dataset/seoul_kmapeval_v3_mcq_100.jsonl")

COMPOSITIONAL_FAMILIES = {
    "trip_finish_time",
    "trip_latest_departure",
    "multisegment_total",
    "poi_brand_share",
    "routing_turns_before_road",
    "poi_bearing_and_distance",
    "nearby_from_need",
}


def test_compositional_benchmark_covers_the_families_v2_left_out() -> None:
    """Time-Window-Reverse, Multi-Segment-Aggregate and Object-Field-Measure are why v3 exists.

    Losing one of them would quietly return the benchmark to measuring pipeline reliability
    instead of composition.
    """

    items = load_dataset(COMPOSITIONAL_BENCHMARK)
    assert len(items) == 100
    families = {item.template_id for item in items}
    assert families == COMPOSITIONAL_FAMILIES
    # `classification` is the intent the agent routes on, not the Appendix E family name.
    assert {item.classification for item in items} <= {
        "trip",
        "routing",
        "radius",
        "direction",
        "nearby",
        "poi",
    }
    assert all(0 <= item.answer < len(item.options) for item in items)
    assert all(len(item.options) == len(set(item.options)) for item in items)
    assert len({item.question for item in items}) == 100


def test_compositional_gold_positions_are_balanced_within_every_family() -> None:
    """Per-family accuracy is reported, so no family may reward guessing one index."""

    items = load_dataset(COMPOSITIONAL_BENCHMARK)
    by_family: dict[str, Counter[int]] = {}
    for item in items:
        by_family.setdefault(item.template_id, Counter())[item.answer] += 1
    for family, spread in by_family.items():
        assert set(spread) == {0, 1, 2, 3}, family
        assert max(spread.values()) - min(spread.values()) <= 1, family


def test_the_need_questions_never_name_the_category_they_want() -> None:
    """The inference is the point: naming 편의점 would restore exactly what v2 gave away."""

    items = load_dataset(COMPOSITIONAL_BENCHMARK)
    needs = [item for item in items if item.template_id == "nearby_from_need"]
    assert needs
    for item in needs:
        assert item.gold_evidence["need_noun"] not in item.question
        # And a closer place of the wrong kind is present, so guessing the category is punished.
        assert item.gold_evidence["nearer_wrong_kind_m"]
        assert min(item.gold_evidence["nearer_wrong_kind_m"]) < item.gold_evidence[
            "gold_distance_m"
        ]


def test_invalid_gold_index_is_rejected() -> None:
    with pytest.raises(ValueError):
        BenchmarkItem(
            id="bad",
            question="q",
            options=["a", "b"],
            answer=3,
            classification="poi",
        )


@pytest.mark.parametrize("classification", ["type", "direction", "distance", "radius"])
def test_extended_benchmark_classifications_are_supported(classification: str) -> None:
    item = BenchmarkItem(
        id=f"{classification}_001",
        question="확장 유형 질문",
        options=["A", "B", "C", "D"],
        answer=3,
        classification=classification,
    )
    assert item.classification == classification
    assert item.answer == 3
