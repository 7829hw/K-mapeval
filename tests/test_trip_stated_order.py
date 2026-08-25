"""A trip question that states its order was being answered by reordering it.

`trip_feasible_count_five` lists its stops 적힌 순서대로 and asks how many fit a time budget.
`tsp_tw` is a travelling-salesman operator: it permutes, and when no full tour fits it fell back
to a nearest-first greedy walk. That walk reaches stops the stated order cannot, so the operator
answered a question nobody asked -- 15 of Spatial-Agent's 26 misses in that family on the v7d
draw were exactly the "one stop too many" this produces, and ReAct made the same mistake 6 times
in 7. Replayed on real cached Kakao legs, the permuting path overcounts 24 of 155 recorded
itineraries by exactly +1 and the stated-order walk gets 153.
"""

from __future__ import annotations

import pytest

from src.agent.spatial import _ground_graph_literals, _states_visiting_order
from src.tools.spatial import SpatialOperatorRegistry, _normalize_arguments


def _matrix(size: int, legs: dict[tuple[int, int], float], default: float) -> list[list[float]]:
    return [
        [0.0 if a == b else legs.get((a, b), default) for b in range(size)] for a in range(size)
    ]


# kmapeval_247, from its recorded gold evidence: three hours, five stops, and the stated order
# reaches exactly one of them. The permuting fallback reported [0, 3, 1, 5] -- two stops plus a
# forced end -- for a trip whose first leg already costs 6,320 of 10,800 seconds.
_STAYS = [0.0, 5400.0, 5400.0, 3600.0, 5400.0, 3600.0]
_STATED_LEGS = {(0, 1): 920.0, (1, 2): 3233.0, (2, 3): 2714.0, (3, 4): 2118.0, (4, 5): 3914.0}
_BUDGET = 10800.0


def test_the_stated_order_counts_travel_against_the_budget() -> None:
    result = SpatialOperatorRegistry.tsp_tw(
        nodes=[{"index": index} for index in range(6)],
        distance_matrix=_matrix(6, _STATED_LEGS, 600.0),
        service_times=_STAYS,
        time_budget=_BUDGET,
        start_index=0,
        fixed_order=True,
    )
    assert result["visited_count"] == 1
    assert result["order"] == [0, 1]
    assert result["unvisited"] == [2, 3, 4, 5]
    assert result["feasible"] is False
    assert result["objective"] == "stated_order"


def test_ignoring_travel_is_the_off_by_one_the_family_punishes() -> None:
    """The stays alone fit two stops; with travel only one does. That gap is the whole family."""

    assert _STAYS[1] + _STAYS[2] <= _BUDGET
    assert _STATED_LEGS[(0, 1)] + _STAYS[1] + _STATED_LEGS[(1, 2)] + _STAYS[2] > _BUDGET


def test_reordering_reaches_a_stop_the_stated_order_cannot() -> None:
    """Why `fixed_order` has to exist: one evidence set, two answers, depending on the question."""

    matrix = _matrix(6, {**_STATED_LEGS, (0, 3): 300.0, (3, 1): 400.0}, 9_000.0)
    stated = SpatialOperatorRegistry.tsp_tw(
        nodes=[{"index": index} for index in range(6)],
        distance_matrix=matrix,
        service_times=_STAYS,
        time_budget=_BUDGET,
        start_index=0,
        fixed_order=True,
    )
    free = SpatialOperatorRegistry.tsp_tw(
        nodes=[{"index": index} for index in range(6)],
        distance_matrix=matrix,
        service_times=_STAYS,
        time_budget=_BUDGET,
        start_index=0,
    )
    assert stated["visited_count"] == 1
    assert free["visited_count"] == 2
    assert free["objective"] == "greedy_partial"


def test_a_stated_order_that_fits_is_feasible_and_complete() -> None:
    result = SpatialOperatorRegistry.tsp_tw(
        nodes=[{"index": index} for index in range(3)],
        distance_matrix=_matrix(3, {(0, 1): 600.0, (1, 2): 600.0}, 600.0),
        service_times=[0.0, 1800.0, 1800.0],
        time_budget=9000.0,
        start_index=0,
        fixed_order=True,
    )
    assert result["order"] == [0, 1, 2]
    assert result["visited_count"] == 2
    assert result["feasible"] is True
    assert result["unvisited"] == []


def test_a_closed_stated_order_counts_the_return_leg_without_counting_a_visit() -> None:
    matrix = _matrix(3, {(0, 1): 10.0, (1, 2): 20.0, (2, 0): 70.0}, 999.0)

    open_tour = SpatialOperatorRegistry.tsp_tw(
        nodes=[{"index": index} for index in range(3)],
        distance_matrix=matrix,
        start_index=0,
        fixed_order=True,
        metric="distance",
    )
    closed_tour = SpatialOperatorRegistry.tsp_tw(
        nodes=[{"index": index} for index in range(3)],
        distance_matrix=matrix,
        start_index=0,
        fixed_order=True,
        metric="distance",
        return_to_start=True,
    )

    assert open_tour["order"] == [0, 1, 2]
    assert open_tour["total_cost"] == 30.0
    assert closed_tour["order"] == [0, 1, 2, 0]
    assert closed_tour["visited_count"] == 2
    assert closed_tour["total_cost"] == 100.0
    assert closed_tour["travel_cost"] == 100.0


def test_a_closed_stated_order_only_accepts_a_prefix_that_can_return() -> None:
    result = SpatialOperatorRegistry.tsp_tw(
        nodes=[{"index": index} for index in range(3)],
        distance_matrix=_matrix(
            3,
            {(0, 1): 10.0, (1, 0): 10.0, (1, 2): 10.0, (2, 0): 100.0},
            999.0,
        ),
        start_index=0,
        fixed_order=True,
        return_to_start=True,
        time_budget=40.0,
    )

    assert result["order"] == [0, 1, 0]
    assert result["visited_count"] == 1
    assert result["total_cost"] == 20.0
    assert result["unvisited"] == [2]
    assert result["feasible"] is False


def test_a_stop_out_of_reach_takes_the_ones_behind_it_with_it() -> None:
    """A stated order is a chain: the fourth stop is not reachable if the third was not."""

    result = SpatialOperatorRegistry.tsp_tw(
        nodes=[{"index": index} for index in range(4)],
        distance_matrix=_matrix(4, {(0, 1): 60.0, (1, 2): 60.0, (2, 3): 60.0}, 60.0),
        # The third stop is the long one; the fourth would fit on its own and must not be counted.
        service_times=[0.0, 1800.0, 9000.0, 60.0],
        time_budget=3600.0,
        start_index=0,
        fixed_order=True,
    )
    assert result["visited_count"] == 1
    assert result["unvisited"] == [2, 3]


def test_the_greedy_fallback_no_longer_appends_an_end_it_cannot_reach() -> None:
    """`end_index` was appended past the budget, naming a stop the trip never gets to."""

    result = SpatialOperatorRegistry.tsp_tw(
        nodes=[{"index": index} for index in range(4)],
        distance_matrix=_matrix(4, {}, 1200.0),
        service_times=[0.0, 1800.0, 1800.0, 1800.0],
        time_budget=4200.0,
        start_index=0,
        end_index=3,
    )
    assert 3 not in result["order"]
    assert 3 in result["unvisited"]
    assert result["total_cost"] <= 4200.0
    assert result["feasible"] is False


def test_the_greedy_fallback_reserves_and_reports_a_required_return_leg() -> None:
    result = SpatialOperatorRegistry.tsp_tw(
        nodes=[{"index": index} for index in range(3)],
        distance_matrix=_matrix(
            3,
            {
                (0, 1): 10.0,
                (1, 0): 10.0,
                (0, 2): 30.0,
                (1, 2): 10.0,
                (2, 0): 100.0,
            },
            999.0,
        ),
        start_index=0,
        return_to_start=True,
        time_budget=40.0,
    )

    assert result["fallback_used"] is True
    assert result["order"] == [0, 1, 0]
    assert result["visited_count"] == 1
    assert result["total_cost"] == 20.0
    assert result["unvisited"] == [2]


def test_tsp_rejects_runtime_shapes_that_cannot_match_the_nodes() -> None:
    nodes = [{"index": index} for index in range(2)]
    matrix = _matrix(2, {}, 60.0)

    with pytest.raises(ValueError, match="start_index must name"):
        SpatialOperatorRegistry.tsp_tw(nodes=nodes, distance_matrix=matrix, start_index=2)
    with pytest.raises(ValueError, match="service_times must have one item per node"):
        SpatialOperatorRegistry.tsp_tw(
            nodes=nodes, distance_matrix=matrix, service_times=[0.0]
        )
    with pytest.raises(ValueError, match="time_windows must have one item per node"):
        SpatialOperatorRegistry.tsp_tw(
            nodes=nodes, distance_matrix=matrix, time_windows=[[0.0, 60.0]]
        )
    with pytest.raises(ValueError, match="fixed_order cannot also set end_index"):
        SpatialOperatorRegistry.tsp_tw(
            nodes=nodes, distance_matrix=matrix, fixed_order=True, end_index=1
        )


def test_every_branch_reports_a_visited_count() -> None:
    common = {
        "nodes": [{"index": index} for index in range(3)],
        "distance_matrix": _matrix(3, {}, 60.0),
        "service_times": [0.0, 60.0, 60.0],
    }
    optimal = SpatialOperatorRegistry.tsp_tw(**common)
    stated = SpatialOperatorRegistry.tsp_tw(**common, fixed_order=True)
    partial = SpatialOperatorRegistry.tsp_tw(**common, time_budget=130.0)
    assert optimal["visited_count"] == 2 and optimal["objective"] == "optimal_order"
    assert stated["visited_count"] == 2 and stated["objective"] == "stated_order"
    assert partial["visited_count"] == 1 and partial["objective"] == "greedy_partial"


@pytest.mark.parametrize(
    "alias", ["preserve_order", "keep_order", "in_order", "sequential", "ordered"]
)
def test_the_planner_may_spell_fixed_order_its_own_way(alias: str) -> None:
    assert _normalize_arguments("tsp_tw", {alias: True})["fixed_order"] is True
    assert _normalize_arguments("tsp_tw", {alias: "true"})["fixed_order"] is True
    assert _normalize_arguments("tsp_tw", {alias: "false"})["fixed_order"] is False


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("time", "duration"),
        ("travel_time", "duration"),
        ("duration_s", "duration"),
        ("seconds", "duration"),
        ("distance_m", "distance"),
        ("metres", "distance"),
        ("meters", "distance"),
    ],
)
def test_tsp_metric_unit_and_measure_aliases_are_canonical(alias: str, canonical: str) -> None:
    assert _normalize_arguments("tsp_tw", {"metric": alias})["metric"] == canonical


def test_tsp_boolean_spellings_do_not_turn_false_into_a_truthy_string() -> None:
    assert _normalize_arguments("tsp_tw", {"return_to_start": "false"})[
        "return_to_start"
    ] is False
    assert _normalize_arguments("tsp_tw", {"return_to_start": "true"})[
        "return_to_start"
    ] is True


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("A를 1.5시간, B를 1시간 동안 적힌 순서대로 둘러봅니다. 몇 곳을 방문할 수 있나요?", True),
        ("까치산역 → 롯데시네마 신도림 → 서울식물원 순서로 자동차로 이동합니다.", True),
        ("visit the three stops in the listed order", True),
        ("자동차 총 주행거리가 가장 짧은 방문 순서는 다음 중 무엇인가요?", False),
        ("A를 들른 뒤 B로 가는 최적 경로는?", False),
    ],
)
def test_only_a_question_that_fixes_its_sequence_says_so(question: str, expected: bool) -> None:
    assert _states_visiting_order(question) is expected


def test_grounding_binds_the_stated_order_and_drops_a_fixed_end() -> None:
    graph = [
        {
            "id": "tour",
            "operator": "tsp_tw",
            "arguments": {
                "nodes": "$locations",
                "distance_matrix": "$legs",
                "end_index": 5,
                "metric": "distance",
                "service_times": [0, 1, 1],
            },
            "depends_on": ["locations", "legs"],
            "output_type": "network",
            "role": "measure",
        }
    ]
    grounded = _ground_graph_literals(
        graph,
        question=(
            "지금 헤이븐스테이 종로에 있습니다. GS더프레시 성북보문점을 1.5시간, "
            "선유도역골목형상점가를 1시간 동안 적힌 순서대로 둘러보려 합니다. "
            "총 3시간이 있고 자동차로 이동합니다. 몇 곳을 방문할 수 있나요?"
        ),
        options=["한 곳", "두 곳", "세 곳", "네 곳"],
        intent="trip",
    )
    arguments = grounded[0]["arguments"]
    assert arguments["fixed_order"] is True
    assert "end_index" not in arguments
    assert arguments["time_budget"] == 10800
    assert arguments["metric"] == "duration"


def test_grounding_leaves_an_optimal_order_question_free_to_reorder() -> None:
    graph = [
        {
            "id": "tour",
            "operator": "tsp_tw",
            "arguments": {
                "nodes": "$locations",
                "distance_matrix": "$legs",
                "metric": "distance",
                "service_times": [0, 3600, 5400],
                "time_budget": 36000,
                "end_index": 0,
            },
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
    assert not arguments.get("fixed_order")
    assert arguments["return_to_start"] is True
    assert arguments["metric"] == "distance"
    assert "end_index" not in arguments
    assert "service_times" not in arguments
