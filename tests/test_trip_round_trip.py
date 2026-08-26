"""`_returns_to_start` was defined twice, and nothing in the suite could tell which one ran.

`src/agent/spatial.py` carried two definitions of the predicate 600 lines apart -- an
`any(re.search(...))` over a `_RETURN_PATTERNS` tuple, and a single compiled `_RETURNS_TO_START`
regex that shadowed it. Only the second was ever reachable, so the tuple and its first definition
were dead code, but the suite exercised the predicate through one call site and would have passed
either way.

The two agreed on every question in `dataset/` -- 223 rows say 돌아옵니다 or 돌아오게, and the
deleted patterns matched and missed exactly the same rows -- so removing the unreachable one
changed no answer on any benchmark here. These cases pin that, and record the one phrasing where
the two genuinely differed: "…로 돌아갑니다" was matched only by the deleted tuple and is *not*
recognised today. No question in `dataset/` uses it, so it costs nothing measured; it is written
down here so the gap is a known limit rather than a rediscovery.

Closure is worth an option: a plan that drops the return arrives one drive early, which is why
`_ground_graph_literals` binds `return_to_start` from the question rather than leaving it to the
planner.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agent.spatial import _returns_to_start

_DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"


@pytest.mark.parametrize(
    ("question", "closed"),
    [
        # The phrasing every v3-and-later trip generator writes.
        (
            "오전 10시 00분에 가예에서 자동차로 출발해 가산로데오거리를 1시간 둘러본 뒤 "
            "가예로 돌아옵니다. 도착 시각은 언제인가요?",
            True,
        ),
        ("세 곳을 차례로 둘러본 뒤 다시 제일모텔로 돌아옵니다.", True),
        ("모든 일정을 마치고 출발지로 오면 몇 시인가요?", True),
        ("한 바퀴 돌고 돌아온다.", True),
        ("일정을 마치고 돌아와 저녁을 먹습니다.", True),
        ("A trip returning to the start.", True),
        # One-way trips and non-trip questions must stay open.
        ("A에서 출발해 B, C를 들른 뒤 C에서 일정을 마칩니다.", False),
        ("자동차 총 주행거리가 가장 짧은 방문 순서는 다음 중 무엇인가요?", False),
        ("서울역에서 가장 가까운 편의점은 어디인가요?", False),
        ("반경 600m 이내에 있는 약국은 몇 곳인가요?", False),
    ],
)
def test_the_question_says_whether_the_trip_comes_home(question: str, closed: bool) -> None:
    assert _returns_to_start(question) is closed


def test_the_phrasing_only_the_deleted_definition_matched_is_a_recorded_gap() -> None:
    """"…로 돌아갑니다" means the same thing and is not recognised. No dataset row uses it."""

    unrecognised = "가예에서 출발해 두 곳을 둘러본 뒤 가예로 돌아갑니다."
    assert _returns_to_start(unrecognised) is False
    assert not [
        row["id"]
        for path in sorted(_DATASET_DIR.glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in (json.loads(line),)
        if "돌아갑" in row["question"] or "돌아가" in row["question"]
    ]
