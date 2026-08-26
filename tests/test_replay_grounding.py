"""The replay harness has to read the logs it claims to read.

Grounding is a pure function of the planner's graph, the Analysis output, the question and the
options, all four of which a recorded run already wrote down. That makes the footprint of a
grounding change measurable offline, which is the check AGENTS.md asks for before shipping one.
The harness is only worth as much as its log parsing, and its first version silently dropped
every `trip_*` question -- the class such a change touches most -- because the logger elides a
long question with a trailing ellipsis and the match was written as an equality.
"""

from __future__ import annotations

import sys
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
if str(DATA) not in sys.path:
    sys.path.insert(0, str(DATA))

from replay_grounding import _read_log, _same_question  # noqa: E402

# One entry per line, exactly as the per-question logger writes them: the whole JSON payload
# sits on the line, which is what makes an offline replay possible at all.
_LOG = "\n".join(
    [
        "14:02:32 [INFO ] FAST WORKFLOW STARTED | AGENT: spatial_agent | ID: q1 | OPTIONS: 4",
        "14:02:32 [INFO ] Question: 서울역에서 가장 가까운 편의점은 어디인가요?",
        '14:02:42 [INFO ] [ANALYZE] {"stage": "analyze", "intent": "nearby",'
        ' "target_type": "편의점"}',
        '14:03:04 [INFO ] [COMPOSE] {"stage": "compose", "graph": [{"id": "a",'
        ' "operator": "batch_geocode", "arguments": {"place_names": ["서울역"]}}]}',
        '14:03:05 [INFO ] [FACTORIZE] {"stage": "factorize"}',
        "",
    ]
)


def test_a_log_yields_the_question_the_analysis_and_the_planner_graph(tmp_path: Path) -> None:
    path = tmp_path / "run.log"
    path.write_text(_LOG, encoding="utf-8")
    found = _read_log(path)

    assert found is not None
    assert found["question"] == "서울역에서 가장 가까운 편의점은 어디인가요?"
    assert found["analyze"]["intent"] == "nearby"
    assert found["graph"][0]["operator"] == "batch_geocode"


def test_a_log_without_a_planner_graph_is_not_replayable(tmp_path: Path) -> None:
    path = tmp_path / "half.log"
    path.write_text("\n".join(_LOG.splitlines()[:3]) + "\n", encoding="utf-8")

    assert _read_log(path) is None


def test_an_elided_question_still_matches_the_row_it_came_from() -> None:
    asked = "보광여관에서 출발해 천장산 하늘길을 1.5시간 둘러본 뒤 다시 보광여관로 돌아옵니다."

    assert _same_question(asked, asked)
    assert _same_question("보광여관에서 출발해 천장산 하늘길을 1.5시간 ...", asked)
    assert not _same_question("다른여관에서 출발해 ...", asked)
    assert not _same_question("서울역에서 가장 가까운 편의점은?", asked)
