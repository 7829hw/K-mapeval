"""A planner may fill a name slot with something that is not a name.

`kmapeval_196` composed `batch_geocode(place_names=[{"cost": "$cost_0", "index": 0}, ...])`,
using the geocoder to build a list of records. The registry refuses that with a validation
error the repair round can read, but grounding ran first and crashed on `.split()`, so the
question was thrown away with `AttributeError: 'dict' object has no attribute 'split'` instead
of being refused. Two of 843 Spatial-Agent runs on the v7d draw died that way.
"""

from __future__ import annotations

import pytest

from src.agent.spatial import _ground_graph_literals, _is_shortened_name, extract_facts
from src.tools.registry import ToolRegistry


class _StubProvider:
    def search_places(self, *args, **kwargs):
        return []

    def nearby_search(self, *args, **kwargs):
        return []

    def geocode(self, *args, **kwargs):
        return None

    def directions(self, *args, **kwargs):
        return None


@pytest.mark.parametrize(
    "candidate",
    [{"cost": "$cost_0", "index": 0}, ["우방스테이"], 3, None],
)
def test_a_non_name_is_not_a_shortened_name(candidate) -> None:
    assert _is_shortened_name(candidate, "우방스테이") is False
    assert _is_shortened_name("우방", candidate) is False


def test_a_real_shortening_still_reads_as_one() -> None:
    assert _is_shortened_name("문래", "빈칸 문래") is True
    assert _is_shortened_name("빈칸 문래", "빈칸 문래") is False


def test_grounding_survives_object_valued_place_names() -> None:
    """The graph is grounded, not crashed; the operator is left to refuse it."""

    graph = [
        {
            "id": "options_list",
            "operator": "batch_geocode",
            "arguments": {
                "place_names": [
                    {"cost": "$cost_0", "index": 0},
                    {"cost": "$cost_1", "index": 1},
                ]
            },
            "depends_on": ["cost_0", "cost_1"],
            "output_type": "object",
            "role": "condition",
        }
    ]
    question = "우방스테이에서 출발해 후암동전망대를 1.5시간 둘러본 뒤 돌아옵니다. 총 거리는?"
    grounded = _ground_graph_literals(
        graph,
        question=question,
        options=["약 20.0km", "약 25.0km", "약 30.0km", "약 35.0km"],
        facts=extract_facts({}, question),
    )
    assert len(grounded) == 1


def test_the_operator_is_what_refuses_object_valued_place_names() -> None:
    execution = ToolRegistry(_StubProvider()).invoke(
        "batch_geocode", {"place_names": [{"cost": 1, "index": 0}]}
    )
    assert execution.status == "error"
    assert "valid string" in (execution.error or "")
