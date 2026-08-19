from __future__ import annotations

import pytest

from src.models import Place, Route
from src.tools.map import MapProvider
from src.tools.registry import ToolRegistry
from src.tools.spatial import SpatialOperatorRegistry, build_duration_matrix


class _MatrixProvider(MapProvider):
    """A provider that answers each pair with its own endpoints, so a matrix has distinct cells."""

    _COORDS = {"S": (37.55, 126.97), "A": (37.56, 126.98), "B": (37.57, 126.99)}

    @property
    def api_call_count(self) -> int:
        return 0

    def _place(self, name: str) -> Place:
        latitude, longitude = self._COORDS.get(name, (37.5, 127.0))
        return Place(place_id=name, name=name, latitude=latitude, longitude=longitude)

    def search_place(self, query: str, *, limit: int = 5) -> list[Place]:
        return [self._place(query)]

    def geocode(self, address: str, *, limit: int = 5) -> list[Place]:
        return [self._place(address)]

    def nearby_search(self, center, **_) -> list[Place]:  # type: ignore[no-untyped-def]
        return []

    def place_details(self, place_id: str) -> Place:
        return self._place(place_id)

    def directions(self, origin, destination, **_):  # type: ignore[no-untyped-def]
        start = origin.name if isinstance(origin, Place) else str(origin)
        end = destination.name if isinstance(destination, Place) else str(destination)
        seconds = 60 * (abs(ord(start[0]) - ord(end[0])) + 1)
        return Route(origin=start, destination=end, distance_m=seconds * 10, duration_s=seconds)


def test_haversine_and_bearing() -> None:
    ops = SpatialOperatorRegistry()
    seoul_station = {"latitude": 37.5547, "longitude": 126.9707}
    tower = {"latitude": 37.5512, "longitude": 126.9882}
    distance = ops.haversine_distance(seoul_station, tower)
    direction = ops.bearing_to_direction(seoul_station, tower)
    assert 1500 < distance["distance_m"] < 1700
    assert direction["direction"] in {"E", "SE"}
    assert direction["cardinal_direction"] == "E"
    assert direction["cardinal_direction_ko"] == "동쪽"


def test_filter_by_direction_returns_only_sector_matches_nearest_first() -> None:
    ops = SpatialOperatorRegistry()
    center = {"name": "기준", "latitude": 37.5, "longitude": 127.0}
    places = [
        {"name": "먼 북쪽", "latitude": 37.52, "longitude": 127.0},
        {"name": "동쪽", "latitude": 37.5, "longitude": 127.01},
        {"name": "가까운 북쪽", "latitude": 37.51, "longitude": 127.0},
    ]
    matches = ops.invoke(
        "filter_by_direction",
        {"center": center, "places": places, "direction": "북쪽"},
    )
    assert [place["name"] for place in matches] == ["가까운 북쪽", "먼 북쪽"]
    assert all(place["cardinal_direction_ko"] == "북쪽" for place in matches)


def test_place_shaped_operator_inputs_are_normalized_before_computation() -> None:
    ops = SpatialOperatorRegistry()
    geocoded = {
        "query": "기준",
        "place": {"name": "기준", "latitude": 37.5, "longitude": 127.0},
        "candidates": [],
    }
    target = {"place": {"name": "북쪽", "lat": 37.52, "lng": 127.0}}
    other = {"place": {"name": "동쪽", "lat": 37.5, "lng": 127.02}}

    distance = ops.invoke("haversine_distance", {"place_a": geocoded, "place_b": target})
    matches = ops.invoke(
        "filter_by_direction",
        {"center": geocoded, "places": [other, target], "direction": "북쪽"},
    )

    assert 2000 < distance["distance_m"] < 2500
    assert [place["name"] for place in matches] == ["북쪽"]


def test_unresolved_place_input_fails_as_an_explicit_place_error() -> None:
    ops = SpatialOperatorRegistry()
    center = {"name": "기준", "latitude": 37.5, "longitude": 127.0}

    with pytest.raises(ValueError, match="PlaceNotFoundError"):
        ops.invoke("haversine_distance", {"place_a": center, "place_b": None})

    resolved = {"name": "북", "latitude": 37.51, "longitude": 127.0}
    nearest = ops.invoke("nearest", {"anchor": center, "candidates": [None, resolved]})
    assert nearest["nearest"]["name"] == "북"
    assert nearest["nearest"]["candidate_index"] == 1


def test_match_distance_options_accepts_numeric_options_and_measured_records() -> None:
    result = SpatialOperatorRegistry().invoke(
        "match_distance_options",
        {"distance": {"distance_km": 1.058}, "options": [1036, 1061, 1200]},
    )

    assert result["best_option"] == 1
    assert result["computed_distance_m"] == 1058


def test_route_comparison_and_sum() -> None:
    ops = SpatialOperatorRegistry()
    routes = [
        {"distance_m": 3000, "duration_s": 900},
        {"distance_m": 2000, "duration_s": 1200},
    ]
    assert ops.compare_routes(routes)["best_index"] == 1
    assert ops.compare_routes(routes, metric="duration_s")["best_index"] == 0
    assert ops.sum_route_metrics(routes) == {"distance_m": 5000, "duration_s": 2100}


def test_planner_argument_aliases_from_recent_logs_are_supported() -> None:
    ops = SpatialOperatorRegistry()
    distance = ops.invoke(
        "haversine_distance",
        {"lat1": 37.5547, "lng1": 126.9707, "lat2": 37.5512, "lng2": 126.9882},
    )
    assert 1500 < distance["distance_m"] < 1700

    selected = ops.invoke("select_min", {"values": [300, 100, 200]})
    assert selected == {"index": 1, "value": 100}

    summed = ops.invoke(
        "sum_route_metrics",
        {"legs": [1000, 2500], "metric": "distance_m"},
    )
    assert summed == {"distance_m": 3500, "duration_s": 0}

    candidate = ops.invoke(
        "select_min",
        {
            "candidates": {
                "Option 1": {"distance_m": 3500, "duration_s": 0},
                "Option 2": {"distance_m": 5000, "duration_s": 0},
            }
        },
    )
    assert candidate["candidate"] == "Option 1"


def test_a_place_is_not_its_own_nearest_neighbour() -> None:
    """The anchor sits among the candidates often enough that 0.0 m would always win.

    A nearest-convenience-store question lists the convenience store it starts from among its
    options, and a stored retrieval heads its own block; ranked by distance the anchor answers its
    own question, which is never what "가장 가까운" asks.
    """

    ops = SpatialOperatorRegistry()
    spot = {"latitude": 37.542619, "longitude": 126.847355}
    anchor = {"place_id": "a", "name": "GS25 화곡초교점", **spot}
    twin = {"place_id": "elsewhere", "name": "GS25 화곡초교점", **spot}
    neighbour = {
        "place_id": "b",
        "name": "CU 화곡본동점",
        "latitude": 37.543215,
        "longitude": 126.848,
    }

    ranked = ops.invoke("nearest", {"anchor": anchor, "candidates": [anchor, neighbour]})
    assert ranked["nearest"]["name"] == "CU 화곡본동점"

    # Same place under another id, because the context minted one per block entry.
    ranked = ops.invoke("nearest", {"anchor": anchor, "candidates": [twin, neighbour]})
    assert ranked["nearest"]["name"] == "CU 화곡본동점"

    # An empty ranking answers nothing, so the self-match stays when it is all there is.
    ranked = ops.invoke("nearest", {"anchor": anchor, "candidates": [anchor]})
    assert ranked["nearest"]["name"] == "GS25 화곡초교점"


def test_a_direction_filter_drops_the_centre_it_measures_from() -> None:
    ops = SpatialOperatorRegistry()
    centre = {"place_id": "a", "name": "안도로메다", "latitude": 37.5620, "longitude": 126.9881}
    south = {
        "place_id": "b",
        "name": "Seoul Namsan Elementary School",
        "latitude": 37.5570,
        "longitude": 126.9880,
    }

    matches = ops.invoke(
        "filter_by_direction", {"center": centre, "places": [centre, south], "direction": "남쪽"}
    )
    assert [place["name"] for place in matches] == ["Seoul Namsan Elementary School"]


def test_tsp_tw_consumes_a_distance_matrix_node() -> None:
    """The trip path only exists if these two operators compose.

    `distance_matrix` returns `{"routes": [...]}` and `tsp_tw` reads a square matrix. While the
    two did not meet, the only matrix a planner could pass was one it made up, so the paper's
    flagship trip capability was unreachable however well the planner was prompted.
    """

    registry = ToolRegistry(_MatrixProvider())
    matrix_node = registry.invoke(
        "distance_matrix",
        {"origins": ["S", "A", "B"], "destinations": ["S", "A", "B"]},
    )
    assert matrix_node.status == "ok", matrix_node.error
    assert matrix_node.output["matrix_complete"] is True
    assert matrix_node.output["nodes"] == ["S", "A", "B"]

    operators = SpatialOperatorRegistry()
    plan = operators.invoke(
        "tsp_tw",
        {
            "nodes": [{"name": name} for name in matrix_node.output["nodes"]],
            "distance_matrix": matrix_node.output,
            "service_times": [0, 600, 600],
            "time_budget": 100_000,
            "start_index": 0,
        },
    )
    assert plan["feasible"] is True
    assert plan["order"][0] == 0
    assert sorted(plan["order"]) == [0, 1, 2]


def test_a_matrix_missing_a_leg_is_reported_rather_than_filled() -> None:
    """An absent leg is missing evidence, not a zero-cost hop."""

    built = build_duration_matrix(
        [
            {"origin": "S", "destination": "A", "duration_s": 60, "status": "ok"},
            {"origin": "A", "destination": "S", "duration_s": 60, "status": "ok"},
            {"origin": "S", "destination": "B", "status": "error", "error": "RouteNotFoundError"},
        ]
    )
    assert built["complete"] is False
    assert ["S", "B"] in built["missing_legs"]
    operators = SpatialOperatorRegistry()
    with pytest.raises(ValueError, match="square"):
        operators.invoke(
            "tsp_tw",
            {"nodes": [{"name": n} for n in built["nodes"]], "distance_matrix": built},
        )


def test_distance_matrix_accepts_a_batch_geocode_node() -> None:
    """`origins: "$places"` is the natural thing for a planner to write.

    batch_geocode hands back {query, place, candidates} records, and rejecting that shape failed
    the matrix before a single route was requested — which failed every tsp_tw downstream of it.
    """

    registry = ToolRegistry(_MatrixProvider())
    geocoded = registry.invoke("batch_geocode", {"place_names": ["S", "A", "B"]})
    assert geocoded.status == "ok", geocoded.error
    matrix_node = registry.invoke(
        "distance_matrix", {"origins": geocoded.output, "destinations": geocoded.output}
    )
    assert matrix_node.status == "ok", matrix_node.error
    assert matrix_node.output["matrix_complete"] is True
    assert len(matrix_node.output["nodes"]) == 3


def test_operator_contract_outranks_a_planner_declared_output_type() -> None:
    """A declared output_type is the planner's guess; the contract is the fact.

    Failing the graph over the disagreement threw away plans whose every leg had been looked up.
    """

    from src.agent.geoflow import normalize_and_validate_graph

    graph = {
        "graph": [
            {
                "id": "places",
                "operator": "batch_geocode",
                "arguments": {"place_names": ["A", "B"]},
                "role": "extent",
                "output_type": "object",
            },
            {
                "id": "tsp",
                "operator": "tsp_tw",
                "arguments": {"nodes": "$places", "distance_matrix": [[0, 1], [1, 0]]},
                "depends_on": ["places"],
                "role": "measure",
                "output_type": "object",  # wrong: tsp_tw outputs network
            },
        ]
    }
    steps, _ = normalize_and_validate_graph(graph, max_steps=10)
    assert [step["output_type"] for step in steps] == ["object", "network"]


@pytest.mark.parametrize(
    ("written", "expected"),
    [("fastest", "TIME"), ("SHORTEST", "DISTANCE"), ("recommend", "RECOMMEND"), ("TIME", "TIME")],
)
def test_route_priority_accepts_the_words_a_planner_reaches_for(
    written: str, expected: str
) -> None:
    from src.tools.registry import DirectionsArgs

    assert DirectionsArgs(origin="A", destination="B", priority=written).priority == expected


def test_an_unrecognized_priority_is_left_to_fail() -> None:
    """Leniency is about wording, not meaning: an unclear word must not become a silent default."""

    from src.tools.registry import DirectionsArgs

    assert DirectionsArgs(origin="A", destination="B", priority="scenic").priority == "scenic"


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("서울역 반경 600m 이내의 카페", 600),
        ("서울역에서 직선거리 600m 이내에 있는 카페", 600),
        ("서울역에서 800m 안에 있는 은행", 800),
        ("서울역 반경 1.5km 이내", 1500),
        ("서울역 근처 카페", 2000),
    ],
)
def test_radius_is_read_from_ordinary_korean(question: str, expected: int) -> None:
    """A radius is phrased several ways; recognizing one keyword silently used the default.

    Every `직선거리 Nm 이내` row in the v2 benchmark was grounded at 2000 m instead of N.
    """

    from src.agent.spatial import _extract_radius_m

    assert _extract_radius_m(question) == expected


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("오전 10시 00분", (10, 0)),
        ("오후 5시", (17, 0)),
        ("오후 5시 30분", (17, 30)),
        ("밤 11시 15분", (23, 15)),
        ("오전 12시 30분", (0, 30)),
        ("2026-08-19T09:00:00", (9, 0)),
    ],
)
def test_temporal_operators_read_the_clock_their_questions_are_written_in(
    written: str, expected: tuple[int, int]
) -> None:
    """A Korean question states a time in Korean, and a planner copies it verbatim.

    Accepting only ISO 8601 made `calculate_finish_time` raise on the very wording of the
    question it was meant to answer, and the agent then guessed the hour.
    """

    from src.tools.spatial import _parse_datetime

    moment = _parse_datetime(written, "Asia/Seoul")
    assert (moment.hour, moment.minute) == expected


def test_finish_time_accepts_a_korean_start_time() -> None:
    registry = ToolRegistry(_MatrixProvider())
    execution = registry.invoke(
        "calculate_finish_time",
        {
            "start_time": "오전 9시 00분",
            "locations": ["S", "A", "B"],
            "stay_durations_s": [0, 3600, 0],
            "timezone": "Asia/Seoul",
        },
    )
    assert execution.status == "ok", execution.error
    assert execution.output["finish_time"].endswith("+09:00")


def test_an_inferred_place_type_grounds_the_retrieval() -> None:
    """A need-shaped question names no category, so the Analysis stage supplies it.

    Without the inferred type the retrieval and the option recovery both lose their category,
    and the ranking answers "nearest of anything" — which is a closer place of the wrong kind.
    """

    from src.agent.spatial import _ground_graph_literals

    question = (
        "지금 단막극장에 있습니다. 갑자기 비가 쏟아져서 우산을 사야 합니다. 가장 가까운 곳은?"
    )
    plan = [
        {
            "id": "anchor",
            "operator": "batch_geocode",
            "arguments": {"place_names": ["단막극장"]},
            "role": "extent",
        },
        {
            "id": "near",
            "operator": "nearby_places",
            "arguments": {"center": "$anchor.0.place"},
            "depends_on": ["anchor"],
            "role": "support",
        },
        {
            "id": "recover",
            "operator": "recover_option_places",
            "arguments": {"options": [], "candidates": "$near", "anchor": "$anchor.0.place"},
            "depends_on": ["near"],
            "role": "support",
        },
    ]

    bare = _ground_graph_literals(plan, question, ["A", "B"], "nearby")
    assert all(
        step["arguments"].get("category_code") is None
        for step in bare
        if step["operator"] in {"nearby_places", "recover_option_places"}
    )

    grounded = _ground_graph_literals(
        plan, question, ["A", "B"], "nearby", inferred_type="편의점"
    )
    codes = {
        step["arguments"].get("category_code")
        for step in grounded
        if step["operator"] in {"nearby_places", "recover_option_places"}
    }
    assert codes == {"CS2"}


@pytest.mark.parametrize(
    ("question", "intent", "expected"),
    [
        ("지금 단막극장에 있습니다. 우산을 사야 합니다.", "nearby", "단막극장"),
        ("현재 서울역에 있습니다.", "nearby", "서울역"),
        ("경복궁에서 가장 가까운 카페 중", "nearby", "경복궁"),
        ("서울생활사박물관에서 직선거리 600m 이내", "radius", "서울생활사박물관"),
    ],
)
def test_the_anchor_is_found_however_the_question_states_it(
    question: str, intent: str, expected: str
) -> None:
    from src.agent.spatial import _extract_anchor

    assert _extract_anchor(question, intent) == expected


def test_steps_analysis_splits_its_counts_at_the_landmark() -> None:
    """"How many left turns before reaching X" needs a bounded count, not the route total.

    With totals alone the only number available was the whole drive's, so the answer came back
    confidently over-counted rather than as a missing capability.
    """

    operators = SpatialOperatorRegistry()
    route = {
        "steps": [
            {"instruction": "출발지", "road_name": ""},
            {"instruction": "좌회전", "road_name": "A로"},
            {"instruction": "우회전", "road_name": "B로"},
            {"instruction": "좌회전", "road_name": "왕십리로"},
            {"instruction": "좌회전", "road_name": "C로"},
            {"instruction": "좌회전", "road_name": "D로"},
        ]
    }
    analysis = operators.invoke("steps_analysis", {"route": route, "landmark": "왕십리로"})
    assert analysis["left_turn_count"] == 4
    assert analysis["landmark_index"] == 3
    assert analysis["left_turn_count_before_landmark"] == 1
    assert analysis["left_turn_count_after_landmark"] == 2
    assert analysis["right_turn_count_before_landmark"] == 1


def test_steps_analysis_without_a_landmark_reports_only_totals() -> None:
    operators = SpatialOperatorRegistry()
    route = {"steps": [{"instruction": "좌회전", "road_name": "A로"}]}
    analysis = operators.invoke("steps_analysis", {"route": route})
    assert analysis["left_turn_count"] == 1
    assert analysis["landmark_index"] is None
    assert "left_turn_count_before_landmark" not in analysis


def test_nearest_respects_the_kind_of_place_that_was_asked_for() -> None:
    """A ranking that ignores the kind returns the closest place of the wrong kind.

    Bound in grounding rather than requested in the prompt: a planner that ranks the option texts
    directly builds no retrieval to carry the category, and prose alone did not change that.
    """

    operators = SpatialOperatorRegistry()
    anchor = {"place_id": "a", "name": "앵커", "latitude": 37.5, "longitude": 127.0}
    candidates = [
        {
            "place_id": "1",
            "name": "대학로 주차장",
            "category": "교통,수송 > 주차장",
            "latitude": 37.5001,
            "longitude": 127.0,
        },
        {
            "place_id": "2",
            "name": "카페더블린",
            "category": "음식점 > 카페",
            "latitude": 37.5002,
            "longitude": 127.0,
        },
        {
            "place_id": "3",
            "name": "CU 동숭아트점",
            "category": "가정,생활 > 편의점 > CU",
            "latitude": 37.5005,
            "longitude": 127.0,
        },
    ]

    bare = operators.invoke("nearest", {"anchor": anchor, "candidates": candidates})
    assert bare["nearest"]["name"] == "대학로 주차장"

    typed = operators.invoke(
        "nearest", {"anchor": anchor, "candidates": candidates, "required_type": "편의점"}
    )
    assert typed["nearest"]["name"] == "CU 동숭아트점"

    # A category vocabulary gap must not empty the ranking: no match means no constraint.
    absent = operators.invoke(
        "nearest", {"anchor": anchor, "candidates": candidates, "required_type": "지하철역"}
    )
    assert absent["nearest"]["name"] == "대학로 주차장"


def test_grounding_binds_the_required_type_onto_nearest() -> None:
    from src.agent.spatial import _ground_graph_literals

    plan = [
        {
            "id": "options",
            "operator": "batch_geocode",
            "arguments": {"place_names": ["A", "B"]},
            "role": "extent",
        },
        {
            "id": "pick",
            "operator": "nearest",
            "arguments": {"anchor": "단막극장", "candidates": "$options"},
            "depends_on": ["options"],
            "role": "measure",
        },
    ]
    question = "지금 단막극장에 있습니다. 우산을 사야 합니다. 가장 가까운 곳은?"

    bare = _ground_graph_literals(plan, question, ["A", "B"], "nearby")
    assert all(
        step["arguments"].get("required_type") is None
        for step in bare
        if step["operator"] == "nearest"
    )

    grounded = _ground_graph_literals(
        plan, question, ["A", "B"], "nearby", inferred_type="편의점"
    )
    assert [
        step["arguments"]["required_type"] for step in grounded if step["operator"] == "nearest"
    ] == ["편의점"]


@pytest.mark.parametrize(
    ("required", "category", "matches"),
    [
        ("지하철역", "교통,수송 > 지하철,전철 > 수도권9호선", True),
        ("대형마트", "가정,생활 > 슈퍼마켓 > 대형슈퍼 > 노브랜드", True),
        ("은행", "금융,보험 > 금융서비스 > 은행 > ATM", True),
        ("편의점", "가정,생활 > 편의점 > CU", True),
        ("편의점", "음식점 > 카페", False),
        # 이마트24 is a convenience store; a loose "마트" term let it answer a 대형마트 question.
        ("대형마트", "가정,생활 > 편의점 > 이마트24", False),
        ("편의점", "가정,생활 > 편의점 > 이마트24", True),
    ],
)
def test_a_requested_kind_matches_what_kakao_calls_it(
    required: str, category: str, matches: bool
) -> None:
    """A question says 지하철역; Kakao files it under 지하철,전철.

    Matching only the question's own noun emptied the filter, which then fell back to the whole
    list — indistinguishable from having no constraint at all.
    """

    from src.tools.spatial import category_terms

    haystack = category.casefold()
    hit = any(term.casefold() in haystack for term in category_terms(required))
    assert hit is matches


def test_a_reference_is_typed_by_the_field_it_names() -> None:
    """`tsp_tw` outputs a network; `$tsp.total_cost` is the tour's duration.

    Typing the reference by the node instead of the field refused eleven correctly-composed
    plans in one run — the exact chain a "what time must I leave" question needs.
    """

    from src.agent.geoflow import normalize_and_validate_graph

    graph = {
        "graph": [
            {
                "id": "places",
                "operator": "batch_geocode",
                "arguments": {"place_names": ["A", "B", "C"]},
                "role": "extent",
            },
            {
                "id": "legs",
                "operator": "distance_matrix",
                "arguments": {"origins": ["A", "B", "C"], "destinations": ["A", "B", "C"]},
                "depends_on": ["places"],
                "role": "support",
            },
            {
                "id": "tsp",
                "operator": "tsp_tw",
                "arguments": {"nodes": "$places", "distance_matrix": "$legs"},
                "depends_on": ["places", "legs"],
                "role": "support",
            },
            {
                "id": "start",
                "operator": "calculate_start_time",
                "arguments": {
                    "arrival_time": "오후 5시",
                    "duration_s": "$tsp.total_cost",
                    "timezone": "Asia/Seoul",
                },
                "depends_on": ["tsp"],
                "role": "measure",
            },
        ]
    }
    steps, constraints = normalize_and_validate_graph(graph, max_steps=10)
    assert [step["id"] for step in steps] == ["places", "legs", "tsp", "start"]
    assert constraints["connectivity"] is True


def test_a_bare_reference_of_the_wrong_type_is_still_refused() -> None:
    """Path-awareness must not disarm G3 for references that name the whole output."""

    from src.agent.geoflow import normalize_and_validate_graph

    graph = {
        "graph": [
            {
                "id": "places",
                "operator": "batch_geocode",
                "arguments": {"place_names": ["A"]},
                "role": "extent",
            },
            {
                "id": "start",
                "operator": "calculate_start_time",
                "arguments": {
                    "arrival_time": "오후 5시",
                    "duration_s": "$places",
                    "timezone": "Asia/Seoul",
                },
                "depends_on": ["places"],
                "role": "measure",
            },
        ]
    }
    with pytest.raises(ValueError, match="Type compatibility violation"):
        normalize_and_validate_graph(graph, max_steps=10)


def test_a_node_nothing_consumes_is_pruned_not_refused() -> None:
    """An unused node is a planner leftover; the rest of the plan still answers the question."""

    from src.agent.geoflow import normalize_and_validate_graph

    graph = {
        "graph": [
            {
                "id": "ends",
                "operator": "batch_geocode",
                "arguments": {"place_names": ["A", "B"]},
                "role": "extent",
            },
            {
                "id": "orphan",
                "operator": "nearby_places",
                "arguments": {"center": "$ends.0.place", "category_code": "CE7"},
                "depends_on": ["ends"],
                "role": "support",
            },
            {
                "id": "span",
                "operator": "haversine_distance",
                "arguments": {"place_a": "$ends.0.place", "place_b": "$ends.1.place"},
                "depends_on": ["ends"],
                "role": "measure",
            },
        ]
    }
    steps, _ = normalize_and_validate_graph(graph, max_steps=10)
    assert [step["id"] for step in steps] == ["ends", "span"]


def test_an_arithmetic_expression_names_the_node_it_depends_on() -> None:
    """Planners write the sum they intend into depends_on; the node named there is still real."""

    from src.agent.geoflow import normalize_and_validate_graph

    graph = {
        "graph": [
            {
                "id": "ends",
                "operator": "batch_geocode",
                "arguments": {"place_names": ["A", "B"]},
                "role": "extent",
            },
            {
                "id": "drive_time",
                "operator": "travel_time",
                "arguments": {"origin": "$ends.0.place", "destination": "$ends.1.place"},
                "depends_on": ["ends"],
                "role": "support",
            },
            {
                "id": "leave_by",
                "operator": "calculate_start_time",
                "arguments": {
                    "arrival_time": "오후 5시",
                    "duration_s": "$drive_time.duration_s",
                    "timezone": "Asia/Seoul",
                },
                "depends_on": ["drive_time + 3600"],
                "role": "measure",
            },
        ]
    }
    steps, _ = normalize_and_validate_graph(graph, max_steps=10)
    assert steps[-1]["depends_on"] == ["drive_time"]


def test_grounding_edits_survive_the_generic_branch() -> None:
    """Every branch edits a copy of the arguments; the fall-through must append that copy.

    Appending the original step instead discarded whatever earlier branches had bound — the
    routing priority never reached the `directions` call it was bound for, and the agent read a
    different route than the question named.
    """

    from src.agent.spatial import _ground_graph_literals

    question = "A에서 B까지 자동차로, 거리가 가장 짧은 경로로 운전합니다. C 구간에 진입하기 전까지?"
    plan = [
        {
            "id": "ends",
            "operator": "batch_geocode",
            "arguments": {"place_names": ["A", "B"]},
            "role": "extent",
        },
        {
            "id": "route",
            "operator": "directions",
            "arguments": {
                "origin": "$ends.0.place",
                "destination": "$ends.1.place",
                "include_steps": True,
            },
            "depends_on": ["ends"],
            "role": "support",
        },
    ]
    grounded = _ground_graph_literals(plan, question, ["1번", "2번"], "routing")
    priorities = {
        step["operator"]: step["arguments"].get("priority") for step in grounded
    }
    assert priorities["directions"] == "DISTANCE"


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("거리가 가장 짧은 경로로 운전합니다", "DISTANCE"),
        ("가장 빠른 경로로 이동합니다", "TIME"),
        ("자동차로 이동합니다", None),
    ],
)
def test_the_question_names_the_route_it_means(question: str, expected: str | None) -> None:
    """Kakao's RECOMMEND re-optimizes against traffic, so a route-shaped answer needs the route
    named or it grades the hour it was asked in."""

    from src.agent.spatial import _extract_route_priority

    assert _extract_route_priority(question) == expected


def test_stays_are_bound_to_the_itinerary_the_plan_lists() -> None:
    """A dropped or invented stay moves the answer by a whole visit.

    Wider than the gap between two options, so the stays are bound from the question like the
    radius is — looked up by the names the plan already holds, because reading names out of the
    sentence swallowed the clause in front of the first one and gave the starting point a visit.
    """

    from src.agent.spatial import _ground_graph_literals

    question = (
        "오전 10시 00분에 구름성모텔에서 자동차로 출발해 닻올림을 1.5시간, "
        "꿈꾸는카멜레온어린이미술관을 1시간, 난우길골목형상점가를 1.5시간 동안 차례로 둘러본 뒤 "
        "구름성모텔로 돌아옵니다. 몇 시에 돌아오게 되나요?"
    )
    chain = [
        "구름성모텔",
        "닻올림",
        "꿈꾸는카멜레온어린이미술관",
        "난우길골목형상점가",
        "구름성모텔",
    ]
    plan = [
        {
            "id": "finish",
            "operator": "calculate_finish_time",
            "arguments": {
                "start_time": "오전 10시 00분",
                "locations": chain,
                "stay_durations_s": [0, 1, 2],
            },
            "role": "measure",
        }
    ]
    grounded = _ground_graph_literals(plan, question, ["오후 4시 17분"], "trip")
    assert grounded[0]["arguments"]["stay_durations_s"] == [0.0, 5400.0, 3600.0, 5400.0, 0.0]
