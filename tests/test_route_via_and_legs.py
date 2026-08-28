"""Two spatial relations the semantic vocabulary could not say, and the families that needed them.

`routing_turn_count_via` asks about the drive from A through B to C. The vocabulary had a route
between two places and nothing else, so `ROUTE_MEASURE` over a resolved [A, B, C] measured A to B
-- a real route, fully executed, and the wrong one -- and the turns were counted on it. That is a
confident wrong answer rather than a failure, and it is worth 69.8 points on the family.

`trip_total_distance` asks for the whole drive of a stated itinerary. `ROUTE_MATRIX` routes every
pair of the n stops and the trip drives n-1 of them, so totalling the matrix answered a question
about n^2 legs. Two of that family's 21 rows reported an errored step: the rest ran cleanly and
returned a number several times too large.

Both are missing expressivity, not missing operators: `directions` has taken waypoints all along
and the legs were all in the matrix. What was missing was a way for the graph to say which.
"""

from __future__ import annotations

from src.agent.geoflow import OPERATOR_CONTRACTS, SKELETONS, retrieve_templates
from src.agent.semantics import factorize_semantic_graph
from src.tools.spatial import SpatialOperatorRegistry

ALL = frozenset(OPERATOR_CONTRACTS)

STOPS = [
    {"id": "start", "text": "인디스타"},
    {"id": "intermediate", "text": "소설호텔"},
    {"id": "end", "text": "공작지"},
]


def _build(graph, *, concepts=(), options=("A", "B"), facts=None, available=ALL):
    return factorize_semantic_graph(
        graph, concepts=list(concepts), options=list(options), facts=facts, available=available
    )


def _route_through(via, *, concepts=STOPS, factors=None):
    step = {
        "id": "route",
        "transform": "ROUTE_MEASURE",
        "inputs": ["ends"],
        "factors": factors or {"measure": "distance"},
    }
    if via is not None:
        step["via"] = via
    return _build(
        [
            {
                "id": "ends",
                "transform": "RESOLVE_PLACES",
                "inputs": [],
                "concept_ids": [concept["id"] for concept in concepts],
            },
            step,
        ],
        concepts=concepts,
    )


# ---------------------------------------------------------------------------------------------
# Waypoints
# ---------------------------------------------------------------------------------------------


def test_a_route_through_a_waypoint_keeps_the_far_end_as_its_destination() -> None:
    built = _route_through(["intermediate"])

    route = built.graph[1]
    assert route["operator"] == "directions"
    assert route["arguments"] == {
        "origin": "$ends.0.place",
        "destination": "$ends.2.place",
        "waypoints": ["$ends.1.place"],
    }


def test_several_waypoints_are_driven_in_the_order_the_graph_lists_them() -> None:
    """Kakao drives the waypoints as given, so a reordered list is a different drive."""

    concepts = [
        {"id": "a", "text": "가"},
        {"id": "b", "text": "나"},
        {"id": "c", "text": "다"},
        {"id": "d", "text": "라"},
    ]
    built = _route_through(["c", "b"], concepts=concepts)

    assert built.graph[1]["arguments"] == {
        "origin": "$ends.0.place",
        "destination": "$ends.3.place",
        "waypoints": ["$ends.2.place", "$ends.1.place"],
    }


def test_a_waypoint_listed_among_the_inputs_is_not_also_an_end_of_the_route() -> None:
    """The shape the planner actually writes: every place an input, the middle one also `via`."""

    built = _build(
        [
            {"id": "a", "transform": "RESOLVE_PLACES", "inputs": [], "concept_ids": ["start"]},
            {
                "id": "b",
                "transform": "RESOLVE_PLACES",
                "inputs": [],
                "concept_ids": ["intermediate"],
            },
            {"id": "c", "transform": "RESOLVE_PLACES", "inputs": [], "concept_ids": ["end"]},
            {
                "id": "route",
                "transform": "ROUTE_MEASURE",
                "inputs": ["a", "b", "c"],
                "via": ["b"],
                "factors": {"measure": "distance"},
            },
        ],
        concepts=STOPS,
    )

    assert built.graph[3]["arguments"] == {
        "origin": "$a.0.place",
        "destination": "$c.0.place",
        "waypoints": ["$b.0.place"],
    }


def test_a_waypoint_may_be_a_node_of_its_own() -> None:
    """The planner resolves the three places separately as often as it resolves them together."""

    built = _build(
        [
            {"id": "a", "transform": "RESOLVE_PLACES", "inputs": [], "concept_ids": ["start"]},
            {
                "id": "b",
                "transform": "RESOLVE_PLACES",
                "inputs": [],
                "concept_ids": ["intermediate"],
            },
            {"id": "c", "transform": "RESOLVE_PLACES", "inputs": [], "concept_ids": ["end"]},
            {"id": "route", "transform": "ROUTE_MEASURE", "inputs": ["a", "c"], "via": ["b"]},
        ],
        concepts=STOPS,
    )

    route = built.graph[3]
    assert route["operator"] == "directions"
    assert route["arguments"]["waypoints"] == ["$b.0.place"]
    assert route["arguments"]["origin"] == "$a.0.place"
    assert route["arguments"]["destination"] == "$c.0.place"
    # A waypoint is something the route depends on, whether or not it was also listed as an input.
    assert set(route["depends_on"]) == {"a", "b", "c"}


def test_a_waypoint_is_never_inferred_from_where_a_place_happened_to_sit() -> None:
    """Three places with no `via` are three places. "A와 B 중 C에 더 가까운" has a middle too."""

    built = _route_through(None)

    assert "waypoints" not in built.graph[1]["arguments"]
    assert built.graph[1]["arguments"]["destination"] == "$ends.1.place"


def test_a_duration_question_still_routes_through_its_waypoints() -> None:
    """`travel_time` carries no waypoints, so asking it for a via-route measures another drive."""

    built = _route_through(["intermediate"], factors={"measure": "duration"})

    assert built.graph[1]["operator"] == "directions"
    assert built.graph[1]["arguments"]["waypoints"] == ["$ends.1.place"]


def test_a_turn_count_reads_the_guidance_of_the_route_that_has_the_waypoint() -> None:
    """The whole family: one route through the stop, one step analysis over that route."""

    built = _build(
        [
            {
                "id": "ends",
                "transform": "RESOLVE_PLACES",
                "inputs": [],
                "concept_ids": ["start", "intermediate", "end"],
                "role": "extent",
            },
            {
                "id": "route",
                "transform": "ROUTE_MEASURE",
                "inputs": ["ends"],
                "via": ["intermediate"],
                "role": "support",
            },
            {"id": "steps", "transform": "ROUTE_STEPS", "inputs": ["route"], "role": "support"},
            {"id": "answer", "transform": "MATCH_OPTIONS", "inputs": ["steps"], "role": "measure"},
        ],
        concepts=STOPS,
    )

    assert [step["operator"] for step in built.graph] == [
        "batch_geocode",
        "directions",
        "steps_analysis",
        "match_options",
    ]
    assert built.graph[1]["arguments"]["waypoints"] == ["$ends.1.place"]
    assert built.graph[2]["arguments"] == {"route": "$route"}


def test_the_turn_count_skeleton_names_its_via_point() -> None:
    route = next(node for node in SKELETONS["route_step_extract"] if node["id"] == "route")

    assert "via" in route


def test_a_turn_count_question_retrieves_the_shape_that_reads_a_step_list() -> None:
    """Question wording cannot create a benchmark-shaped template."""

    question = (
        "인디스타에서 소설호텔을 들러 공작지까지 자동차로, 거리가 가장 짧은 경로로 이동합니다. "
        "주행 안내에 따르면 좌회전을 몇 번 해야 하나요?"
    )
    analysis = {
        "intent": "trip",
        "concepts": [
            {"id": stop["id"], "text": stop["text"], "concept_type": "location", "role": "extent"}
            for stop in STOPS
        ],
        "measure": "answer choice",
        "target_type": None,
    }

    names = [template["name"] for template in retrieve_templates(analysis, question)]
    paraphrase = [template["name"] for template in retrieve_templates(analysis, "무관한 문장")]

    assert names == paraphrase
    # `ROUTE-STEP-EXTRACT` is in the macro catalogue now, so this pins *why* it is offered or
    # not: the typed concepts and factors, never the words "주행 안내" or "좌회전". This
    # analysis types every stop as a bare `location` extent and names no measure, so the
    # ranking cannot tell a turn count from anything else -- which is a fact about the
    # analysis, not about the wording.
    assert set(names) == set(paraphrase)


# ---------------------------------------------------------------------------------------------
# Route legs
# ---------------------------------------------------------------------------------------------


def _itinerary(count: int):
    return [{"id": f"s{index}", "text": f"장소{index}"} for index in range(count)]


def test_a_trip_total_covers_the_legs_it_drives_and_not_every_pair() -> None:
    stops = _itinerary(4)
    built = _build(
        [
            {
                "id": "stops",
                "transform": "RESOLVE_PLACES",
                "inputs": [],
                "concept_ids": [stop["id"] for stop in stops],
                "role": "extent",
            },
            {"id": "legs", "transform": "ROUTE_MATRIX", "inputs": ["stops"], "role": "support"},
            {
                "id": "path",
                "transform": "SELECT_LEGS",
                "inputs": ["legs"],
                "factors": {"scope": "consecutive"},
                "role": "support",
            },
            {
                "id": "total",
                "transform": "AGGREGATE",
                "inputs": ["path"],
                "factors": {"aggregate": "sum", "measure": "distance"},
                "role": "support",
            },
            {"id": "answer", "transform": "MATCH_OPTIONS", "inputs": ["total"], "role": "measure"},
        ],
        concepts=stops,
    )

    assert [step["operator"] for step in built.graph] == [
        "batch_geocode",
        "distance_matrix",
        "select_legs",
        "sum_route_metrics",
        "match_distance_options",
    ]
    assert built.graph[2]["arguments"] == {"routes": "$legs"}
    assert built.graph[3]["arguments"] == {"routes": "$path"}


def test_totalling_a_bare_matrix_composes_the_leg_selection_it_omitted() -> None:
    """The planner writes AGGREGATE straight over the matrix; the grouping is still explicit."""

    stops = _itinerary(4)
    built = _build(
        [
            {
                "id": "stops",
                "transform": "RESOLVE_PLACES",
                "inputs": [],
                "concept_ids": [stop["id"] for stop in stops],
            },
            {"id": "legs", "transform": "ROUTE_MATRIX", "inputs": ["stops"]},
            {
                "id": "total",
                "transform": "AGGREGATE",
                "inputs": ["legs"],
                "factors": {"aggregate": "sum", "scope": "groups"},
            },
        ],
        concepts=stops,
    )

    assert [step["operator"] for step in built.graph] == [
        "batch_geocode",
        "distance_matrix",
        "select_legs",
        "sum_route_metrics",
    ]
    composed = next(row for row in built.decisions if row["operator"] == "select_legs")
    assert composed["rule"] == "composed_consecutive_legs"


def test_per_option_totals_still_group_by_the_groups_the_graph_carries() -> None:
    """`aggregate_route_groups` answers "which order is shortest" and keeps its group list."""

    stops = _itinerary(2)
    built = _build(
        [
            {
                "id": "pairs",
                "transform": "RESOLVE_PLACES",
                "inputs": [],
                "concept_ids": [stop["id"] for stop in stops],
            },
            {"id": "legs", "transform": "ROUTE_MATRIX", "inputs": ["pairs"]},
            {"id": "groups", "transform": "MEASURE", "inputs": ["legs"]},
            {
                "id": "totals",
                "transform": "AGGREGATE",
                "inputs": ["legs", "groups"],
                "factors": {"aggregate": "sum", "scope": "groups"},
            },
        ],
        concepts=stops,
    )

    assert built.graph[3]["operator"] == "aggregate_route_groups"


def test_select_legs_walks_the_consecutive_pairs_of_a_square_matrix() -> None:
    matrix = {
        "routes": [
            {"pair_index": index, "distance_m": index, "duration_s": index, "status": "ok"}
            for index in range(9)
        ]
    }

    selected = SpatialOperatorRegistry.select_legs(matrix)

    assert selected["leg_count"] == 2
    assert [leg["pair_index"] for leg in selected["routes"]] == [1, 5]
    assert selected["order"] == [0, 1, 2]
    assert selected["complete"] is True


def test_select_legs_follows_a_stated_order_rather_than_the_matrix_order() -> None:
    matrix = {
        "routes": [{"pair_index": index, "distance_m": index, "status": "ok"} for index in range(9)]
    }

    selected = SpatialOperatorRegistry.select_legs(matrix, order=[2, 0, 1])

    assert [leg["pair_index"] for leg in selected["routes"]] == [6, 1]


def test_select_legs_refuses_a_grid_that_is_not_square() -> None:
    """A matrix of distinct origins and destinations has no consecutive legs to walk."""

    try:
        SpatialOperatorRegistry.select_legs({"routes": [{"distance_m": 1}, {"distance_m": 2}]})
    except ValueError as error:
        assert "square" in str(error)
    else:  # pragma: no cover - the operator must not answer this
        raise AssertionError("select_legs accepted a non-square matrix")


def test_select_legs_reports_a_leg_that_failed_rather_than_totalling_around_it() -> None:
    matrix = {
        "routes": [{"pair_index": index, "distance_m": 0, "status": "ok"} for index in range(4)]
    }
    matrix["routes"][1] = {"pair_index": 1, "status": "error", "error": "RouteNotFoundError"}

    selected = SpatialOperatorRegistry.select_legs(matrix)

    assert selected["complete"] is False
    assert selected["errors"] == ["RouteNotFoundError"]


def test_a_matrix_over_several_resolved_stops_covers_every_one_of_them() -> None:
    """Half the planner's trip graphs resolve each stop as its own node.

    Wiring only the first input made a 1x1 grid, and the leg selection then had one route and no
    leg to take: ten of twenty-one `trip_total_distance` rows raised on it in one pass.
    """

    stops = _itinerary(4)
    built = _build(
        [
            *(
                {
                    "id": f"r{index}",
                    "transform": "RESOLVE_PLACES",
                    "inputs": [],
                    "concept_ids": [stop["id"]],
                }
                for index, stop in enumerate(stops)
            ),
            {
                "id": "legs",
                "transform": "ROUTE_MATRIX",
                "inputs": [f"r{index}" for index in range(4)],
            },
        ],
        concepts=stops,
    )

    assert built.graph[4]["arguments"] == {
        "origins": ["$r0.0.place", "$r1.0.place", "$r2.0.place", "$r3.0.place"],
        "destinations": ["$r0.0.place", "$r1.0.place", "$r2.0.place", "$r3.0.place"],
    }


def test_selecting_legs_from_stops_composes_the_matrix_they_imply() -> None:
    stops = _itinerary(3)
    built = _build(
        [
            *(
                {
                    "id": f"r{index}",
                    "transform": "RESOLVE_PLACES",
                    "inputs": [],
                    "concept_ids": [stop["id"]],
                }
                for index, stop in enumerate(stops)
            ),
            {
                "id": "path",
                "transform": "SELECT_LEGS",
                "inputs": [f"r{index}" for index in range(3)],
            },
        ],
        concepts=stops,
    )

    assert [step["operator"] for step in built.graph] == [
        "batch_geocode",
        "batch_geocode",
        "batch_geocode",
        "distance_matrix",
        "select_legs",
    ]
    assert built.graph[4]["arguments"] == {"routes": "$path_matrix"}
