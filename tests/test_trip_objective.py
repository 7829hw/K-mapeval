"""`tsp_tw` ranked every itinerary by seconds, whatever the question asked for.

"자동차 총 주행거리가 가장 짧은 방문 순서" is a question about metres, and the four orders it
chooses between are separated by a median of 2.07% of the tour, so ranking them by seconds is not
an approximation of ranking them by metres. The tour was also left open when the question closes
it — "…둘러본 뒤 다시 제일모텔로 돌아옵니다" — and the cheapest path out is not the cheapest loop.

Replayed on 114 `trip_optimal_order` rows against real cached Kakao legs:

    distance + closed tour   93/114   81.6%      <- what the question asks
    distance, open           53/114   46.5%
    duration + closed        41/114   36.0%
    duration, open           35/114   30.7%      <- what the operator did

The 21 rows the distance-closed reading still misses are legs whose cached length has drifted
since the set was built, not a disagreement about the objective.
"""

from __future__ import annotations

import pytest

from src.agent.spatial import _asks_for_distance, _ground_graph_literals, _returns_to_start
from src.tools.spatial import SpatialOperatorRegistry, build_duration_matrix

# Four stops whose two metrics disagree: B is near but slow to reach, C and D are far but fast.
# The shortest loop by metres is base→B→C→D→base; by seconds it is base→C→B→D→base.
_LEGS = {
    ("base", "B"): (1000.0, 1800.0),
    ("base", "C"): (2000.0, 300.0),
    ("base", "D"): (3000.0, 400.0),
    ("B", "C"): (500.0, 1500.0),
    ("B", "D"): (4000.0, 200.0),
    ("C", "D"): (600.0, 1700.0),
}
_ROUTES = [
    {
        "origin": origin,
        "destination": destination,
        "distance_m": metres,
        "duration_s": seconds,
        "status": "ok",
    }
    for (first, second), (metres, seconds) in _LEGS.items()
    for origin, destination in ((first, second), (second, first))
]
_NODES = [{"name": name} for name in ("base", "B", "C", "D")]


def test_the_matrix_builder_reads_whichever_metric_was_asked_for() -> None:
    by_time = build_duration_matrix(_ROUTES)
    by_length = build_duration_matrix(_ROUTES, "distance")
    assert by_time["metric"] == "duration_s"
    assert by_length["metric"] == "distance_m"
    base, other = by_time["nodes"].index("base"), by_time["nodes"].index("B")
    assert by_time["matrix"][base][other] == 1800.0
    assert by_length["matrix"][base][other] == 1000.0


def test_an_unknown_metric_is_refused_rather_than_defaulted() -> None:
    with pytest.raises(ValueError, match="metric"):
        build_duration_matrix(_ROUTES, "cost")


def test_the_two_metrics_choose_different_tours() -> None:
    """The whole reason the argument exists: same legs, different answer."""

    by_length = SpatialOperatorRegistry.tsp_tw(
        nodes=_NODES, distance_matrix={"routes": _ROUTES}, metric="distance", return_to_start=True
    )
    by_time = SpatialOperatorRegistry.tsp_tw(
        nodes=_NODES, distance_matrix={"routes": _ROUTES}, metric="duration", return_to_start=True
    )
    assert by_length["metric"] == "distance_m"
    assert by_time["metric"] == "duration_s"
    assert by_length["order"] != by_time["order"]


def test_closing_the_tour_counts_the_drive_home() -> None:
    open_tour = SpatialOperatorRegistry.tsp_tw(
        nodes=_NODES, distance_matrix={"routes": _ROUTES}, metric="distance"
    )
    closed = SpatialOperatorRegistry.tsp_tw(
        nodes=_NODES, distance_matrix={"routes": _ROUTES}, metric="distance", return_to_start=True
    )
    assert closed["order"][-1] == 0
    assert open_tour["order"][-1] != 0
    assert closed["total_cost"] > open_tour["total_cost"]
    # The drive home is not a visit.
    assert closed["visited_count"] == open_tour["visited_count"] == 3


def test_seconds_are_refused_beside_a_matrix_of_metres() -> None:
    for clock in ({"time_budget": 3600.0}, {"service_times": [0.0, 60.0, 60.0, 60.0]}):
        with pytest.raises(ValueError, match="metres"):
            SpatialOperatorRegistry.tsp_tw(
                nodes=_NODES, distance_matrix={"routes": _ROUTES}, metric="distance", **clock
            )


def test_a_tour_cannot_both_come_home_and_end_elsewhere() -> None:
    with pytest.raises(ValueError, match="return to the start"):
        SpatialOperatorRegistry.tsp_tw(
            nodes=_NODES,
            distance_matrix={"routes": _ROUTES},
            metric="distance",
            return_to_start=True,
            end_index=1,
        )


@pytest.mark.parametrize(
    ("question", "distance", "closed"),
    [
        ("자동차 총 주행거리가 가장 짧은 방문 순서는 다음 중 무엇인가요?", True, False),
        ("둘러본 뒤 다시 제일모텔로 돌아옵니다. 총 주행거리가 가장 짧은 순서는?", True, True),
        ("A → B → C 순서로 이동합니다. 전체 주행 거리는 얼마인가요?", True, False),
        ("총 3시간이 있고 자동차로 이동합니다. 몇 곳을 방문할 수 있나요?", False, False),
        ("이동 시간이 가장 짧은 방문 순서는 무엇인가요?", False, False),
    ],
)
def test_the_question_says_which_objective_and_whether_it_comes_home(
    question: str, distance: bool, closed: bool
) -> None:
    assert _asks_for_distance(question) is distance
    assert _returns_to_start(question) is closed


def test_grounding_binds_metric_and_closure_and_drops_the_decoy_clock() -> None:
    graph = [
        {
            "id": "tour",
            "operator": "tsp_tw",
            "arguments": {"nodes": "$locations", "distance_matrix": "$legs"},
            "depends_on": ["locations", "legs"],
            "output_type": "network",
            "role": "measure",
        }
    ]
    grounded = _ground_graph_literals(
        graph,
        question=(
            "제일모텔에서 출발해 난곡동 벽화마을을 1시간, 봉산무장애숲길을 1.5시간 동안 "
            "둘러본 뒤 다시 제일모텔로 돌아옵니다. "
            "자동차 총 주행거리가 가장 짧은 방문 순서는 다음 중 무엇인가요?"
        ),
        options=["A → B", "B → A", "A → A", "B → B"],
        intent="trip",
    )
    arguments = grounded[0]["arguments"]
    assert arguments["metric"] == "distance"
    assert arguments["return_to_start"] is True
    # The stays are stated but they are decoys in a distance question, and seconds beside a metre
    # matrix is what the operator refuses.
    assert "service_times" not in arguments
    assert "time_budget" not in arguments


def test_a_count_question_keeps_its_clock_and_stays_open() -> None:
    graph = [
        {
            "id": "tour",
            "operator": "tsp_tw",
            "arguments": {"nodes": "$locations", "distance_matrix": "$legs"},
            "depends_on": ["locations", "legs"],
            "output_type": "network",
            "role": "measure",
        }
    ]
    grounded = _ground_graph_literals(
        graph,
        question=(
            "지금 헤이븐스테이 종로에 있습니다. GS더프레시 성북보문점을 1.5시간 동안 "
            "적힌 순서대로 둘러보려 합니다. 총 3시간이 있고 자동차로 이동합니다. "
            "몇 곳을 방문할 수 있나요?"
        ),
        options=["한 곳", "두 곳", "세 곳", "네 곳"],
        intent="trip",
    )
    arguments = grounded[0]["arguments"]
    assert arguments.get("metric") != "distance"
    assert arguments["time_budget"] == 10800
    assert not arguments.get("return_to_start")


def _legs_node() -> dict:
    """What the `distance_matrix` tool actually returns: routes *and* a duration matrix."""

    built = build_duration_matrix(_ROUTES)
    return {
        "routes": _ROUTES,
        "route_count": len(_ROUTES),
        "nodes": built["nodes"],
        "matrix": built["matrix"],
        "matrix_metric": built["metric"],
        "matrix_complete": True,
        "missing_legs": [],
    }


def test_the_metric_is_read_off_the_routes_not_the_matrix_beside_them() -> None:
    """`$legs` carries both, and the pre-built one is always seconds.

    Preferring it made `metric="distance"` a no-op on the one input shape every trip graph here
    uses -- the operator returned a duration and called it metres.
    """

    legs = _legs_node()
    by_length = SpatialOperatorRegistry.tsp_tw(
        nodes=_NODES, distance_matrix=legs, metric="distance", return_to_start=True
    )
    by_time = SpatialOperatorRegistry.tsp_tw(
        nodes=_NODES, distance_matrix=legs, metric="duration", return_to_start=True
    )
    assert by_length["total_cost"] == 5100.0
    assert by_time["total_cost"] == 2400.0
    assert by_length["order"] != by_time["order"]


def test_a_matrix_with_no_routes_to_re_read_is_refused_not_reinterpreted() -> None:
    built = build_duration_matrix(_ROUTES)
    with pytest.raises(ValueError, match="no routes to re-read"):
        SpatialOperatorRegistry.tsp_tw(
            nodes=_NODES,
            distance_matrix={"matrix": built["matrix"], "nodes": built["nodes"]},
            metric="distance",
        )
