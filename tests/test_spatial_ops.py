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

    route_calls = 0

    def directions(self, origin, destination, **_):  # type: ignore[no-untyped-def]
        type(self).route_calls += 1
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


def test_the_itinerary_clock_runs_both_ways() -> None:
    """"When must I leave to arrive by six" is the same itinerary read backwards.

    With only a forward mode the reverse question had to be assembled by hand — sum the legs, add
    the stays, hand a scalar to `calculate_start_time` — and the planner under-counted the chain,
    answering from a single leg.
    """

    registry = ToolRegistry(_MatrixProvider())
    itinerary = {
        "locations": ["S", "A", "B"],
        "stay_durations_s": [0, 3600, 0],
        "timezone": "Asia/Seoul",
    }
    forward = registry.invoke(
        "calculate_finish_time", {"start_time": "오전 9시 00분", **itinerary}
    )
    assert forward.status == "ok", forward.error

    backward = registry.invoke(
        "calculate_finish_time",
        {"arrival_time": forward.output["finish_time"], **itinerary},
    )
    assert backward.status == "ok", backward.error
    assert backward.output["start_time"] == forward.output["start_time"]
    assert backward.output["travel_duration_s"] == forward.output["travel_duration_s"]


def test_the_itinerary_needs_exactly_one_anchor_in_time() -> None:
    registry = ToolRegistry(_MatrixProvider())
    for arguments in (
        {"locations": ["S", "A"]},
        {"locations": ["S", "A"], "start_time": "오전 9시", "arrival_time": "오후 5시"},
    ):
        execution = registry.invoke("calculate_finish_time", arguments)
        assert execution.status == "error"
        assert "exactly one" in (execution.error or "")


def test_a_consumed_measure_is_demoted_and_the_terminal_promoted() -> None:
    """A Measure is what the answer is read from, so a node another node consumes is not one."""

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
                "id": "duration1",
                "operator": "travel_time",
                "arguments": {"origin": "$ends.0.place", "destination": "$ends.1.place"},
                "depends_on": ["ends"],
                "role": "measure",
            },
            {
                "id": "answer",
                "operator": "calculate_start_time",
                "arguments": {
                    "arrival_time": "오후 5시",
                    "duration_s": "$duration1.duration_s",
                    "timezone": "Asia/Seoul",
                },
                "depends_on": ["duration1"],
                "role": "support",
            },
        ]
    }
    steps, _ = normalize_and_validate_graph(graph, max_steps=10)
    roles = {step["id"]: step["role"] for step in steps}
    assert roles == {"ends": "extent", "duration1": "support", "answer": "measure"}


def test_a_departure_time_counts_the_stops_on_the_way() -> None:
    """Travel is not the whole wait: time spent at a stop delays departure just as much."""

    operators = SpatialOperatorRegistry()
    bare = operators.invoke(
        "calculate_start_time",
        {"arrival_time": "오후 7시 00분", "duration_s": 5472, "timezone": "Asia/Seoul"},
    )
    with_stops = operators.invoke(
        "calculate_start_time",
        {
            "arrival_time": "오후 7시 00분",
            "duration_s": 5472,
            "stay_durations_s": [1800, 2700],
            "timezone": "Asia/Seoul",
        },
    )
    assert with_stops["stay_duration_s"] == 4500
    assert with_stops["total_duration_s"] == 5472 + 4500
    assert with_stops["start_time"] < bare["start_time"]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("A를 2시간, B를 1.5시간 둘러봅니다", [7200.0, 5400.0]),
        ("A에서 30분, B에서 45분씩 들러야 합니다", [1800.0, 2700.0]),
        ("A를 1시간, B에서 15분 들릅니다", [3600.0, 900.0]),
    ],
)
def test_a_stop_is_stated_in_either_shape(question: str, expected: list[float]) -> None:
    """"X를 2시간" and "X에서 30분" are the same statement; reading only the first returned
    nothing for a question full of errands and left the departure short by all of them."""

    from src.agent.spatial import _extract_trip_schedule

    stays, _ = _extract_trip_schedule(question)
    assert sorted(stays.values()) == sorted(expected)


def test_an_empty_measure_takes_the_node_it_depends_on() -> None:
    """A Measure with nothing to measure is a leftover; failing threw away gathered evidence."""

    from src.agent.spatial import _ground_graph_literals

    plan = [
        {
            "id": "span",
            "operator": "haversine_distance",
            "arguments": {"place_a": "A", "place_b": "B"},
            "role": "extent",
        },
        {
            "id": "answer",
            "operator": "identity_measure",
            "arguments": {},
            "depends_on": ["span"],
            "role": "measure",
        },
    ]
    grounded = _ground_graph_literals(plan, "A와 B 사이 거리는?", ["1km"], "distance")
    assert grounded[-1]["arguments"]["value"] == "$span"


def test_every_required_argument_in_a_contract_is_one_the_operator_demands() -> None:
    """A contract that requires an argument the tool treats as optional refuses working plans.

    `calculate_finish_time` gained an `arrival_time` mode and the contract still required
    `start_time`, so G4 rejected every plan that used the reverse direction the prompt had just
    described — five questions in one run, all with the evidence already gathered.
    """

    from src.agent.geoflow import OPERATOR_CONTRACTS
    from src.tools.registry import ToolRegistry

    registry = ToolRegistry(_MatrixProvider())
    for schema in registry.schemas():
        name = schema["function"]["name"]
        contract = OPERATOR_CONTRACTS.get(name)
        if contract is None:
            continue
        demanded = set(schema["function"]["parameters"].get("required") or [])
        claimed = set(contract.required_arguments)
        assert claimed <= demanded, (
            f"{name}: contract requires {sorted(claimed - demanded)} which the tool accepts "
            "without"
        )


def test_a_stray_dependency_is_dropped_when_the_references_are_sound() -> None:
    """`depends_on` is a declaration; the `$` references are what execution follows.

    A planner that writes the arithmetic it means into the dependency ("travel_duration + 3600")
    names nothing resolvable, but the argument beside it already points at the right node.
    """

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
                "id": "travel",
                "operator": "travel_time",
                "arguments": {"origin": "$ends.0.place", "destination": "$ends.1.place"},
                "depends_on": ["ends"],
                "role": "support",
            },
            {
                "id": "departure",
                "operator": "calculate_start_time",
                "arguments": {
                    "arrival_time": "오후 5시",
                    "duration_s": "$travel.duration_s",
                    "timezone": "Asia/Seoul",
                },
                "depends_on": ["travel_duration + 3600"],
                "role": "measure",
            },
        ]
    }
    steps, _ = normalize_and_validate_graph(graph, max_steps=10)
    assert steps[-1]["depends_on"] == ["travel"]


def test_a_node_that_depends_on_nothing_real_still_fails() -> None:
    """Dropping stray names must not turn a genuinely broken graph into a valid one."""

    from src.agent.geoflow import normalize_and_validate_graph

    graph = {
        "graph": [
            {
                "id": "a",
                "operator": "batch_geocode",
                "arguments": {"place_names": ["A"]},
                "role": "extent",
            },
            {
                "id": "b",
                "operator": "haversine_distance",
                "arguments": {"place_a": "A", "place_b": "B"},
                "depends_on": ["nonexistent"],
                "role": "measure",
            },
        ]
    }
    with pytest.raises(ValueError, match="Unknown dependency"):
        normalize_and_validate_graph(graph, max_steps=10)


def test_picking_an_extreme_works_on_whatever_the_collection_holds() -> None:
    """select_min/select_max/sort_by read a named key; the concept type is not their business.

    Restricting them refused a plan that ranked itineraries, which are events.
    """

    from src.agent.geoflow import CORE_CONCEPTS, OPERATOR_INPUT_TYPES

    for name in ("select_min", "select_max", "sort_by"):
        assert OPERATOR_INPUT_TYPES[name]["items"] == frozenset(CORE_CONCEPTS)


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("09:00", (9, 0)),
        ("10:00", (10, 0)),
        ("17:00:00", (17, 0)),
        ("오후 5:30", (17, 30)),
        ("오전 10시 00분", (10, 0)),
    ],
)
def test_a_planner_writes_the_clock_in_the_machine_form(
    written: str, expected: tuple[int, int]
) -> None:
    """The question says "오전 10시 00분"; the planner normalizes it before writing the argument.

    `datetime.fromisoformat` takes neither `10:00` nor `17:00:00` as a datetime, so the clock
    step failed outright and the generation stage answered a time-window question from prose
    arithmetic instead — a confident wrong option rather than a failure.
    """

    from src.tools.spatial import _parse_datetime

    parsed = _parse_datetime(written, "Asia/Seoul")
    assert (parsed.hour, parsed.minute) == expected


def test_a_lone_place_stands_where_a_list_of_places_is_expected() -> None:
    """The mirror of "a one-element list is the place the planner forgot to index into"."""

    from src.tools.registry import CalculateFinishTimeArgs

    record = {
        "query": "키이토",
        "place": {
            "place_id": "1",
            "name": "키이토",
            "address": "서울 노원구",
            "latitude": 37.6,
            "longitude": 127.1,
            "category": "음식점",
            "phone": "",
            "place_url": "u",
            "rating": None,
            "price_level": None,
            "opening_hours": None,
            "timezone": None,
            "is_open": None,
        },
        "candidates": [],
    }
    args = CalculateFinishTimeArgs(start_time="10:00", locations=record)
    assert [place.name for place in args.locations] == ["키이토"]


def test_a_reference_carrying_arithmetic_still_names_its_node() -> None:
    """`$travel_s + 2700` is a reference plus the stay the question states, not a broken id.

    Read undecorated it handed the validator a node id that does not exist, and the `KeyError`
    escaped the per-step isolation and lost the whole question before any tool ran.
    """

    from src.agent.geoflow import normalize_and_validate_graph, split_reference_arithmetic

    assert split_reference_arithmetic("$travel_s + 2700") == ("$travel_s", 2700.0)
    assert split_reference_arithmetic("$legs.duration_s - 600") == ("$legs.duration_s", -600.0)
    assert split_reference_arithmetic("$plain.path") == ("$plain.path", 0.0)

    graph = {
        "graph": [
            {
                "id": "ends",
                "operator": "batch_geocode",
                "arguments": {"place_names": ["A", "B"]},
                "role": "extent",
            },
            {
                "id": "travel_s",
                "operator": "travel_time",
                "arguments": {"origin": "$ends.0.place", "destination": "$ends.1.place"},
                "depends_on": ["ends"],
                "role": "support",
            },
            {
                "id": "total_s",
                "operator": "identity_measure",
                "arguments": {"value": "$travel_s.duration_s + 2700"},
                "depends_on": ["travel_s"],
                "role": "measure",
            },
        ]
    }
    steps, _ = normalize_and_validate_graph(graph, max_steps=10)
    assert [step["id"] for step in steps][-1] == "total_s"


def test_an_offset_reference_resolves_to_the_number_plus_the_offset() -> None:
    from src.agent.spatial import _resolve_references

    results = {"travel_s": {"duration_s": 1800.0}, "who": {"name": "A"}}
    assert _resolve_references({"value": "$travel_s.duration_s + 2700"}, results) == {
        "value": 4500.0
    }
    # A reference that resolves to anything but a number carries no arithmetic.
    assert _resolve_references({"value": "$who.name"}, results) == {"value": "A"}


def test_a_route_carries_its_guidance_and_a_travel_time_does_not() -> None:
    """Upstream's Directions prints every step; its TravelTime reports a duration and a distance.

    Our port had `directions` omit the guidance unless asked, which made `steps_analysis` report
    zero turns on every route -- and made the guidance a capability the port withheld from the
    baseline rather than one MapEval's design withholds. The split lives in the two args models,
    so both architectures get the same route from the same tool.
    """

    from src.tools.registry import DirectionsArgs, TravelTimeArgs

    pair = {"origin": "A", "destination": "B"}
    assert DirectionsArgs.model_validate(pair).include_steps is True
    assert TravelTimeArgs.model_validate(pair).include_steps is False


def test_an_itinerary_is_the_whole_list_the_plan_geocoded() -> None:
    """A finish time has no legs to time when the plan indexes one stop out of the itinerary."""

    from src.agent.spatial import _ground_graph_literals

    steps = [
        {
            "id": "places",
            "operator": "batch_geocode",
            "arguments": {"place_names": ["기점", "A", "B", "기점"]},
            "role": "extent",
        },
        {
            "id": "finish",
            "operator": "calculate_finish_time",
            "arguments": {"start_time": "10:00", "locations": "$places.0"},
            "depends_on": ["places"],
            "role": "measure",
        },
    ]
    grounded = _ground_graph_literals(steps, "오전 10시에 출발합니다", [], "trip")
    assert grounded[-1]["arguments"]["locations"] == "$places"


def test_a_tour_that_must_end_somewhere_is_not_free_to_end_anywhere() -> None:
    """"An appointment at X at 7pm, with errands on the way" fixes the last stop.

    Left free, the search finds a cheaper route that finishes at an errand and answers a
    departure time for a trip that never reaches the appointment.
    """

    registry = SpatialOperatorRegistry()
    # 0 start, 1 and 2 errands, 3 the appointment. The appointment sits next door to the start
    # and the errands are across town, so the cheapest free tour visits it second and finishes at
    # an errand — a departure time for a trip that never ends where the deadline is.
    matrix = [
        [0, 2000, 2500, 200],
        [2000, 0, 300, 2000],
        [2500, 300, 0, 2400],
        [200, 2000, 2400, 0],
    ]
    nodes = [{"name": name} for name in ("start", "errand1", "errand2", "appointment")]
    service = [0.0, 1800.0, 2700.0, 0.0]

    free = registry.invoke(
        "tsp_tw",
        {"nodes": nodes, "distance_matrix": matrix, "service_times": service, "start_index": 0},
    )
    fixed = registry.invoke(
        "tsp_tw",
        {
            "nodes": nodes,
            "distance_matrix": matrix,
            "service_times": service,
            "start_index": 0,
            "end_index": 3,
        },
    )
    assert free["order"][-1] != 3
    assert fixed["order"] == [0, 1, 2, 3]
    assert fixed["total_cost"] == 2000 + 300 + 2400 + 4500
    # The halves are reported so a planner cannot mistake the total for travel alone.
    assert fixed["travel_cost"] == 4700
    assert fixed["service_cost"] == 4500


def test_an_expression_over_several_nodes_names_all_of_them() -> None:
    """`$dur1 + 2700 + $dur2 + 900 + $dur3` is a three-leg errand run, not a broken id.

    Read as one name it reached `by_id` with a key that is not a node, and the `KeyError` came
    from inside validation — outside the per-step isolation — losing the whole question.
    """

    from src.agent.geoflow import reference_expression, reference_roots
    from src.agent.spatial import _resolve_references

    assert reference_expression("$dur1 + 2700 + $dur2 + 900 + $dur3") == (
        ["$dur1", "$dur2", "$dur3"],
        3600.0,
    )
    assert reference_expression("not a reference") is None
    assert reference_roots({"value": "$dur1 + 2700 + $dur2"}) == ["dur1", "dur2"]

    results = {"dur1": {"duration_s": 1000.0}, "dur2": 2000.0, "dur3": {"duration_s": 500.0}}
    assert _resolve_references(
        {"value": "$dur1.duration_s + 2700 + $dur2 + 900 + $dur3.duration_s"}, results
    ) == {"value": 7100.0}


def test_an_unresolvable_reference_root_fails_its_step_not_the_graph() -> None:
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
                "id": "total",
                "operator": "identity_measure",
                "arguments": {"value": "$nowhere * 3"},
                "depends_on": ["ends"],
                "role": "measure",
            },
        ]
    }
    steps, _ = normalize_and_validate_graph(graph, max_steps=10)
    assert [step["id"] for step in steps] == ["ends", "total"]


def test_a_ranking_answers_on_the_geometry_it_has() -> None:
    """A metric is the planner's choice of measure; it has no authority over what evidence exists.

    Asking for travel time without fetching routes is a plan that forgot a step. Failing threw
    away an anchor and a candidate set that were both resolved, and the generation stage then
    picked the nearest place of any kind — exactly the decoy these questions plant.
    """

    registry = SpatialOperatorRegistry()
    anchor = {"name": "anchor", "latitude": 37.5, "longitude": 127.0}
    candidates = [
        {"name": "far pharmacy", "latitude": 37.5009, "longitude": 127.0},
        {"name": "near cafe", "latitude": 37.5001, "longitude": 127.0},
    ]
    result = registry.invoke(
        "nearest", {"anchor": anchor, "candidates": candidates, "metric": "travel_time"}
    )
    assert result["nearest"]["name"] == "near cafe"
    # The trace must not claim a travel time nobody computed.
    assert result["metric_used"] == "haversine"
    assert result["metric_requested"] == "travel_time"


def test_a_tour_cost_is_not_topped_up_with_the_stays_it_already_counts() -> None:
    from src.agent.spatial import _ground_graph_literals

    question = (
        "천장산 하늘길에서 오후 7시 00분에 약속이 있습니다. 가는 길에 킴스클럽 강남점에서 30분, "
        "메가MGC커피 상계주공6단지점에서 45분씩 들러야 하고, 이동은 모두 자동차로 가장 빠른 "
        "경로를 이용합니다. 신용산역 4호선에서 늦어도 몇 시에 출발해야 하나요?"
    )
    names = ["신용산역 4호선", "킴스클럽 강남점", "메가MGC커피 상계주공6단지점", "천장산 하늘길"]
    steps = [
        {
            "id": "places",
            "operator": "batch_geocode",
            "arguments": {"place_names": names},
            "role": "extent",
        },
        {
            "id": "tsp",
            "operator": "tsp_tw",
            "arguments": {
                "nodes": "$places",
                "distance_matrix": "$legs",
                "service_times": [0, 1800, 2700, 0],
                "start_index": 0,
            },
            "depends_on": ["places"],
            "role": "support",
        },
        {
            "id": "dep",
            "operator": "calculate_start_time",
            "arguments": {
                "arrival_time": "2024-01-01T19:00:00",
                "duration_s": "$tsp.total_cost + 4500",
            },
            "depends_on": ["tsp"],
            "role": "measure",
        },
    ]
    grounded = _ground_graph_literals(steps, question, [], "trip")
    assert grounded[1]["arguments"]["end_index"] == 3
    assert grounded[2]["arguments"]["duration_s"] == "$tsp.total_cost"
    assert "stay_durations_s" not in grounded[2]["arguments"]

    # A duration that is genuinely travel-only still gets the stated stays bound beside it.
    steps[2]["arguments"]["duration_s"] = "$legs.total_duration_s"
    grounded = _ground_graph_literals(steps, question, [], "trip")
    assert grounded[2]["arguments"]["stay_durations_s"] == [1800.0, 2700.0]


def test_the_itinerary_carries_its_stays_even_when_it_is_a_reference() -> None:
    """The names are not in hand when `locations` is `$places`, but the geocode node lists them.

    Left unbound, the planner's own stays mismatched the resolved length and the args model
    rejected the call outright — the whole clock step lost to a length check.
    """

    from src.agent.spatial import _ground_graph_literals

    question = (
        "오전 10시 00분에 가예에서 자동차로 출발해 가산로데오거리를 1시간, "
        "용양봉저정공원 하늘전망대를 1.5시간, 메가박스 상암월드컵경기장점을 1시간 동안 차례로 "
        "둘러본 뒤 가예로 돌아옵니다. 몇 시에 돌아오게 되나요?"
    )
    names = [
        "가예",
        "가산로데오거리",
        "용양봉저정공원 하늘전망대",
        "메가박스 상암월드컵경기장점",
        "가예",
    ]
    steps = [
        {
            "id": "geo",
            "operator": "batch_geocode",
            "arguments": {"place_names": names},
            "role": "extent",
        },
        {
            "id": "finish",
            "operator": "calculate_finish_time",
            "arguments": {
                "start_time": "10:00",
                "locations": "$geo",
                "stay_durations_s": [3600, 5400, 3600, 0],
            },
            "depends_on": ["geo"],
            "role": "measure",
        },
    ]
    grounded = _ground_graph_literals(steps, question, [], "trip")
    assert grounded[-1]["arguments"]["stay_durations_s"] == [0.0, 3600.0, 5400.0, 3600.0, 0.0]


def test_a_computed_clock_outranks_the_answer_the_generation_stage_wrote() -> None:
    """When the graph produced the time, reporting it is the generation stage's whole job.

    Revising is what it did instead: a trace reading 14:40 came back as 16:33 "accounting for
    real-world traffic, parking, and navigation variations", and one reading 13:36 as 15:46 for
    an "unrecorded return trip". Both are invented evidence and both moved the answer one option.
    """

    from src.agent.spatial import _select_option

    options = ["오후 5시 41분", "오후 4시 46분", "오후 3시 46분", "오후 2시 21분"]
    # `calculate_finish_time` echoes the start it was given, so the result carries both fields;
    # the answer is the one it derived.
    results = {
        "trip": {
            "start_time": "2026-08-19T09:00:00+09:00",
            "finish_time": "2026-08-19T13:36:04+09:00",
            "derived_clock": "finish_time",
        }
    }
    assert _select_option(
        {"predicted_answer": "오후 3시 46분", "predicted_option": 2}, options, results
    ) == (3, "computed_clock")

    # `arrival_time` is the deadline the question states, never a computed value.
    reverse = {
        "start": {
            "arrival_time": "2024-01-01T17:00:00+09:00",
            "start_time": "2024-01-01T13:56:13+09:00",
            "derived_clock": "start_time",
        }
    }
    reverse_options = ["오후 1시 52분", "오후 12시 17분", "오전 10시 42분", "오후 3시 27분"]
    assert _select_option({"predicted_answer": "오후 12시 17분"}, reverse_options, reverse) == (
        0,
        "computed_clock",
    )

    # Without a computed clock, or with options that are not clocks, the old path decides.
    no_clock: dict[str, object] = {"x": {"nearest": {}}}
    assert _select_option({"predicted_answer": "오후 3시 46분"}, options, no_clock) == (
        2,
        "exact_answer_text",
    )
    assert _select_option({"predicted_answer": "2번"}, ["1번", "2번", "3번"], results) == (
        1,
        "exact_answer_text",
    )


def test_a_bare_tour_total_is_not_topped_up_with_stays_either() -> None:
    """`$tsp.total_cost` carries the stays whether or not the planner also wrote an addition."""

    from src.agent.spatial import _ground_graph_literals

    question = (
        "중계동학원가에서 오후 5시 00분에 약속이 있습니다. 가는 길에 킴스클럽 강남점에서 45분, "
        "메가MGC커피 상계주공6단지점에서 45분씩 들러야 하고, 이동은 모두 자동차로 가장 빠른 "
        "경로를 이용합니다. 미아역 4호선에서 늦어도 몇 시에 출발해야 하나요?"
    )
    names = ["미아역 4호선", "킴스클럽 강남점", "메가MGC커피 상계주공6단지점", "중계동학원가"]
    steps = [
        {
            "id": "g",
            "operator": "batch_geocode",
            "arguments": {"place_names": names},
            "role": "extent",
        },
        {
            "id": "tsp",
            "operator": "tsp_tw",
            "arguments": {
                "nodes": "$g",
                "distance_matrix": "$m",
                "service_times": [0, 2700, 2700, 0],
                "start_index": 0,
            },
            "depends_on": ["g"],
            "role": "support",
        },
        {
            "id": "st",
            "operator": "calculate_start_time",
            "arguments": {
                "arrival_time": "2024-01-01T17:00:00",
                "duration_s": "$tsp.total_cost",
            },
            "depends_on": ["tsp"],
            "role": "measure",
        },
    ]
    grounded = _ground_graph_literals(steps, question, [], "trip")
    assert grounded[1]["arguments"]["end_index"] == 3
    assert "stay_durations_s" not in grounded[2]["arguments"]


def test_a_stated_return_leg_is_part_of_the_itinerary() -> None:
    """"…둘러본 뒤 X로 돌아옵니다" is a leg, and a plan that drops it arrives one drive early."""

    from src.agent.spatial import _ground_graph_literals

    question = (
        "오전 9시 00분에 동산장모텔에서 자동차로 출발해 투모로우바이투게더숲을 1시간, "
        "경복궁 연생전을 1.5시간, 한승수 유엔홀을 1시간 동안 차례로 둘러본 뒤 동산장모텔로 "
        "돌아옵니다. 몇 시에 돌아오게 되나요?"
    )
    open_names = ["동산장모텔", "투모로우바이투게더숲", "경복궁 연생전", "한승수 유엔홀"]
    steps = [
        {
            "id": "p",
            "operator": "batch_geocode",
            "arguments": {"place_names": open_names},
            "role": "extent",
        },
        {
            "id": "t",
            "operator": "calculate_finish_time",
            "arguments": {"start_time": "09:00", "locations": "$p"},
            "depends_on": ["p"],
            "role": "measure",
        },
    ]
    grounded = _ground_graph_literals(steps, question, [], "trip")
    assert grounded[-1]["arguments"]["locations"] == [*open_names, "동산장모텔"]
    assert grounded[-1]["arguments"]["stay_durations_s"] == [0.0, 3600.0, 5400.0, 3600.0, 0.0]

    # A plan that already closed the loop is left exactly as it is.
    steps[0]["arguments"]["place_names"] = [*open_names, "동산장모텔"]
    grounded = _ground_graph_literals(steps, question, [], "trip")
    assert grounded[-1]["arguments"]["locations"] == "$p"


def test_the_derived_clock_is_the_one_the_operator_computed() -> None:
    """A clock operator reports both ends and computes one of them.

    Run forwards the start is the question's and the finish is the answer; run backwards it is
    the other way round, and nothing in the field names says which. Preferring `finish_time`
    answered "when must I leave" with the deadline the question had just handed over — six
    questions in one run, every prediction later than its gold.
    """

    from src.agent.spatial import _select_option

    reverse_options = ["오후 3시 32분", "오후 5시 52분", "오후 2시 07분", "오후 7시 17분"]
    reverse = {
        "s": {
            "start_time": "2026-08-19T15:58:14+09:00",
            "finish_time": "2026-08-19T18:00:00+09:00",
            "derived_clock": "start_time",
        }
    }
    assert _select_option({"predicted_answer": "오후 5시 52분"}, reverse_options, reverse) == (
        0,
        "computed_clock",
    )

    forward_options = ["오후 5시 41분", "오후 4시 46분", "오후 3시 46분", "오후 2시 21분"]
    forward = {
        "s": {
            "start_time": "2026-08-19T09:00:00+09:00",
            "finish_time": "2026-08-19T13:36:04+09:00",
            "derived_clock": "finish_time",
        }
    }
    assert _select_option({"predicted_answer": "오후 3시 46분"}, forward_options, forward) == (
        3,
        "computed_clock",
    )

    # A result that does not say what it derived is not a computed clock.
    silent: dict[str, object] = {"s": {"finish_time": "2026-08-19T13:36:04+09:00"}}
    assert _select_option({"predicted_answer": "오후 3시 46분"}, forward_options, silent) == (
        2,
        "exact_answer_text",
    )


def test_a_clock_operator_says_which_end_it_derived() -> None:
    from src.tools.registry import ToolRegistry

    registry = ToolRegistry(_MatrixProvider())
    forward = registry.invoke(
        "calculate_finish_time", {"start_time": "10:00", "locations": ["A", "B"]}
    )
    assert forward.output["derived_clock"] == "finish_time"
    backward = registry.invoke(
        "calculate_finish_time", {"arrival_time": "18:00", "locations": ["A", "B"]}
    )
    assert backward.output["derived_clock"] == "start_time"

    operators = SpatialOperatorRegistry()
    reverse = operators.invoke(
        "calculate_start_time",
        {"arrival_time": "18:00", "duration_s": 3600, "timezone": "Asia/Seoul"},
    )
    assert reverse["derived_clock"] == "start_time"


def test_a_round_trip_starts_and_ends_where_the_question_says() -> None:
    """Only the order of the stops between the endpoints is the plan's business.

    A plan that drops the return arrives one drive early; one that drops the departure loses its
    first leg *and* shifts every stay onto the wrong stop. Neither fails, and both answer an
    option away — the second bound every stay to zero, because a stop written as `$geo.1.place`
    is not a name any stay can be looked up by.
    """

    from src.agent.spatial import _ground_graph_literals

    question = (
        "오전 10시 00분에 키이토에서 자동차로 출발해 수락산나들길을 1.5시간, "
        "서대문형무소역사관을 1.5시간, 갤러리이서를 1시간 동안 차례로 둘러본 뒤 키이토로 "
        "돌아옵니다. 몇 시에 돌아오게 되나요?"
    )
    names = ["키이토", "수락산나들길", "서대문형무소역사관", "갤러리이서"]
    steps = [
        {
            "id": "geo",
            "operator": "batch_geocode",
            "arguments": {"place_names": names},
            "role": "extent",
        },
        {
            "id": "it",
            "operator": "calculate_finish_time",
            "arguments": {
                "start_time": "10:00",
                # The base written last, and every stop written as a reference.
                "locations": ["$geo.1.place", "$geo.2.place", "$geo.3.place", "$geo.0.place"],
                "stay_durations_s": [5400, 5400, 3600, 0],
            },
            "depends_on": ["geo"],
            "role": "measure",
        },
    ]
    grounded = _ground_graph_literals(steps, question, [], "trip")
    assert grounded[-1]["arguments"]["locations"] == [
        "키이토",
        "수락산나들길",
        "서대문형무소역사관",
        "갤러리이서",
        "키이토",
    ]
    assert grounded[-1]["arguments"]["stay_durations_s"] == [0.0, 5400.0, 5400.0, 3600.0, 0.0]

    # An itinerary that already runs base-first and closes is left exactly as it is.
    steps[1]["arguments"]["locations"] = [*names, "키이토"]
    grounded = _ground_graph_literals(steps, question, [], "trip")
    assert grounded[-1]["arguments"]["locations"] == [*names, "키이토"]


def test_a_priority_word_that_names_no_objective_is_the_ordinary_route() -> None:
    """"normal" and "traffic" are a planner filling a required field, not a fourth objective.

    Refusing them failed all twenty-five legs of a distance matrix at once, which left `tsp_tw`
    nothing square to read and the generation stage guessing the answer. A word that does name a
    different objective still fails: this stays leniency about wording, not about meaning.
    """

    from src.tools.registry import _as_priority

    for written in ("normal", "traffic", "realtime", "기본", "default", "Standard"):
        assert _as_priority(written) == "RECOMMEND"
    assert _as_priority("fastest") == "TIME"
    assert _as_priority("shortest") == "DISTANCE"
    for unknown in ("scenic", "teleport", "cheapest"):
        assert _as_priority(unknown) == unknown


def test_a_concept_something_is_built_from_is_not_the_measure() -> None:
    """The Analysis stage labelled a radius `measure`, and G2 refused the plan outright.

    A radius is a condition on a search. The operator graph already demotes a Measure that
    another node consumes; the concept graph gets the same treatment rather than losing a plan
    whose retrieval was already specified.
    """

    from src.agent.geoflow import factorize_geoflow, normalize_and_validate_graph

    graph = {
        "graph": [
            {
                "id": "anchor",
                "operator": "batch_geocode",
                "arguments": {"place_names": ["호스텔온기"]},
                "role": "extent",
                "concept_ids": ["origin"],
            },
            {
                "id": "found",
                "operator": "nearby_places",
                "arguments": {
                    "center": "$anchor.0.place",
                    "radius_m": 800,
                    "category_code": "MT1",
                },
                "depends_on": ["anchor"],
                "role": "measure",
                "concept_ids": ["radius", "derived_matches"],
            },
        ],
        "concept_graph": {
            "nodes": [
                {
                    "id": "origin",
                    "text": "호스텔온기",
                    "concept_type": "location",
                    "role": "extent",
                },
                {"id": "radius", "text": "800m", "concept_type": "amount", "role": "measure"},
                {
                    "id": "derived_matches",
                    "text": "대형마트",
                    "concept_type": "object",
                    "role": "support",
                },
            ],
            "edges": [
                {"source": "origin", "target": "radius"},
                {"source": "radius", "target": "derived_matches"},
            ],
        },
    }
    analysis = {"intent": "radius", "concepts": graph["concept_graph"]["nodes"]}
    factored = factorize_geoflow(analysis, {"graph": graph["graph"]}).as_dict()
    factored["concept_graph"] = graph["concept_graph"]
    steps, constraints = normalize_and_validate_graph(factored, max_steps=10)
    assert constraints["concept_connectivity"] is True
    assert [step["id"] for step in steps] == ["anchor", "found"]


def test_an_address_the_geocoder_cannot_place_fails_once() -> None:
    """Returned as `[]` it became a `center: []` and a cascade of validation noise.

    The empty list failed the retrieval as a pydantic type error, and the `{"error": ...}` that
    left behind was then validated as a Place by the next step, producing seven more errors
    describing fields an error message does not have — burying the one failure that happened.
    """

    from src.tools.map import MapProvider, PlaceNotFoundError
    from src.tools.registry import ToolRegistry

    class _Empty(MapProvider):
        @property
        def api_call_count(self) -> int:
            return 0

        def search_place(self, query: str, *, limit: int = 5) -> list[Place]:
            return []

        def geocode(self, address: str, *, limit: int = 5) -> list[Place]:
            return []

        def nearby_search(self, center, **_):  # type: ignore[no-untyped-def]
            return []

        def place_details(self, place_id: str) -> Place:
            raise PlaceNotFoundError(place_id)

        def directions(self, origin, destination, **_):  # type: ignore[no-untyped-def]
            raise PlaceNotFoundError("no route")

    execution = ToolRegistry(_Empty()).invoke("geocode", {"address": "미스바 프로젝트"})
    assert execution.status == "error"
    assert execution.error.startswith("PlaceNotFoundError")


def test_a_failed_step_is_not_evidence_the_next_step_can_read() -> None:
    from src.agent.spatial import _resolve_references

    results = {"anchor": {"error": "PlaceNotFoundError: nothing"}, "good": {"name": "A"}}
    with pytest.raises(ValueError, match="failed step 'anchor'"):
        _resolve_references({"center": "$anchor"}, results)
    # A result that merely carries an `error` key beside real output is still output.
    partial = {"matrix": {"error": None, "nodes": ["A", "B"]}}
    assert _resolve_references({"m": "$matrix.nodes"}, partial) == {"m": ["A", "B"]}


def test_candidates_that_carry_no_coordinates_fail_instead_of_emptying_the_sector() -> None:
    """An operator handed only names must fail, not answer "nothing lies that way".

    A planner wrote the option texts into `filter_by_direction.places` instead of the places it
    had just geocoded. The names dropped out, the empty sector read as evidence, and the
    generation stage picked an option by eyeballing latitudes out of the trace.
    """

    ops = SpatialOperatorRegistry()
    center = {"name": "미아사거리역", "latitude": 37.6132, "longitude": 127.0300}
    with pytest.raises(ValueError, match="PlaceNotFoundError"):
        ops.filter_by_direction(
            center=center,
            places=["하나로마트 미아점", "이마트 미아점"],
            direction="북쪽",
        )


def test_an_empty_candidate_list_is_still_an_empty_ranking() -> None:
    """Nothing to rank is not the same as candidates that could not be read."""

    ops = SpatialOperatorRegistry()
    anchor = {"name": "미아사거리역", "latitude": 37.6132, "longitude": 127.0300}
    assert ops.nearest(anchor=anchor, candidates=[])["nearest"] is None


def test_filter_places_reads_geocode_records_and_kakao_category_codes() -> None:
    """The filter is handed what `batch_geocode` returns, and the kind as the prompt spells it.

    A planner wrote `required_types: ["CS2"]` over the `{query, place, candidates}` records of
    its own geocoding step. Neither the code nor the wrapper was understood, the filter emptied
    the candidate list, and the ranking that followed had nothing to rank — so the generation
    stage answered with a cafe 16 m away instead of the convenience store the question needs.
    """

    ops = SpatialOperatorRegistry()
    records = [
        {
            "query": "카페더블린",
            "place": {
                "place_id": "1",
                "name": "카페더블린",
                "category": "음식점 > 카페",
                "latitude": 37.58,
                "longitude": 127.0,
            },
        },
        {
            "query": "CU 동숭아트점",
            "place": {
                "place_id": "2",
                "name": "CU 동숭아트점",
                "category": "가정,생활 > 편의점 > CU",
                "latitude": 37.581,
                "longitude": 127.001,
            },
        },
    ]
    kept = ops.filter_places(places=records, required_types=["CS2"])
    assert [place["place_id"] for place in kept] == ["2"]


def test_several_required_types_are_alternatives_not_a_conjunction() -> None:
    ops = SpatialOperatorRegistry()
    places = [
        {"place_id": "1", "name": "약국", "category": "의료 > 약국",
         "latitude": 1.0, "longitude": 1.0},
        {"place_id": "2", "name": "카페", "category": "음식점 > 카페",
         "latitude": 1.0, "longitude": 1.0},
    ]
    kept = ops.filter_places(places=places, required_types=["약국", "병원"])
    assert [place["place_id"] for place in kept] == ["1"]


def test_a_kind_the_category_vocabulary_misses_drops_the_filter() -> None:
    """An unknown kind is a gap in the lexicon, not evidence that nothing qualifies."""

    ops = SpatialOperatorRegistry()
    places = [
        {"place_id": "1", "name": "가게", "category": "가정,생활 > 잡화",
         "latitude": 1.0, "longitude": 1.0}
    ]
    assert ops.filter_places(places=places, required_types=["우산가게"]) == places


def test_a_clock_that_cannot_tell_two_options_apart_is_not_evidence_for_either() -> None:
    """A computed clock counts only when it picks an option decisively.

    A plan whose stays failed to bind computed a travel-only 12:30 for an itinerary with four
    stated hours of visits. That is 113 minutes from one option and 173 from the next, which
    tells the two apart no better than a coin — and it overruled a generation stage that had
    added the four hours itself and written the correct answer.
    """

    from src.agent.spatial import _select_option

    options = ["오후 7시 18분", "오후 4시 53분", "오후 2시 23분", "오후 3시 23분"]
    travel_only = {
        "s": {
            "start_time": "2026-08-19T10:00:00+09:00",
            "finish_time": "2026-08-19T12:30:32+09:00",
            "derived_clock": "finish_time",
        }
    }
    assert _select_option({"predicted_answer": "오후 4시 53분"}, options, travel_only) == (
        1,
        "exact_answer_text",
    )

    complete = {
        "s": {
            "start_time": "2026-08-19T10:00:00+09:00",
            "finish_time": "2026-08-19T16:53:00+09:00",
            "derived_clock": "finish_time",
        }
    }
    assert _select_option({"predicted_answer": "오후 2시 23분"}, options, complete) == (
        1,
        "computed_clock",
    )


def test_the_diagonal_of_a_matrix_costs_nothing_and_no_api_call() -> None:
    """Kakao refuses a leg whose ends are the same place, and it should never be asked.

    An `origins = destinations` matrix asked for its own diagonal, and one run spent 750 route
    calls collecting the refusals. The generation stage then read a matrix full of errors. This
    is the only leg that may be filled: an absent off-diagonal leg is still missing evidence.
    """

    provider = _MatrixProvider()
    registry = ToolRegistry(provider)
    before = provider.route_calls
    execution = registry.invoke(
        "distance_matrix", {"origins": ["S", "A"], "destinations": ["S", "A"]}
    )
    assert execution.status == "ok"
    diagonal = [
        route
        for route in execution.output["routes"]
        if route["origin"] == route["destination"]
    ]
    assert len(diagonal) == 2
    assert all(route["status"] == "ok" and route["duration_s"] == 0 for route in diagonal)
    assert execution.output["matrix_complete"] is True
    assert provider.route_calls - before == 2


def test_two_anchors_are_not_an_anchor() -> None:
    """"A와 B 양쪽 모두에서 …" reads to the splitter as one long place name."""

    from src.agent.spatial import _extract_anchor

    both = "가좌동 마을극장과 증산역 6호선 양쪽 모두에서 직선거리 1500m 이내에 있는 대형마트는?"
    assert _extract_anchor(both, "radius") is None
    one = "호스텔온기에서 직선거리 800m 이내에 있는 대형마트는 다음 중 어디인가요?"
    assert _extract_anchor(one, "radius") == "호스텔온기"


def test_a_ranking_key_is_found_through_the_wrapper_a_planner_pointed_at() -> None:
    """`select_max(items, "distance_m")` over geocode records used to find no comparable item."""

    ops = SpatialOperatorRegistry()
    items = [
        {"query": "A", "place": {"name": "A", "distance_m": 120.0}},
        {"query": "B", "place": {"name": "B", "distance_m": 900.0}},
    ]
    assert ops.select_max(items, "distance_m")["query"] == "B"
    assert ops.select_min(items, "distance")["query"] == "A"
    assert [item["query"] for item in ops.sort_by(items, "distance_m")] == ["A", "B"]


def test_an_amount_resolves_only_when_the_item_carries_one_metric() -> None:
    """`amount` names no metric, so it must never choose between two."""

    ops = SpatialOperatorRegistry()
    single = [{"label": "x", "distance_m": 10.0}, {"label": "y", "distance_m": 4.0}]
    assert ops.select_min(single, "amount")["label"] == "y"
    both = [{"label": "x", "distance_m": 10.0, "duration_s": 400.0}]
    with pytest.raises(ValueError, match="comparable key"):
        ops.select_min(both, "amount")


def test_a_unit_is_never_an_alias() -> None:
    ops = SpatialOperatorRegistry()
    with pytest.raises(ValueError, match="comparable key"):
        ops.select_min([{"distance_km": 1.2}], "distance_m")


def test_a_concept_ring_keeps_the_measure_the_analysis_stage_named() -> None:
    """Demoting a consumed Measure must not leave a concept graph with none.

    An Analysis stage that makes its concepts depend on each other in a ring leaves nothing that
    nothing is built from, so there is no terminal to promote. Taking the last Measure away
    refused a direction question whose operator graph — geocode, retrieve banks, filter south,
    rank — was correct and whose retrieval was already specified.
    """

    from src.agent.geoflow import factorize_geoflow, normalize_and_validate_graph

    analysis = {
        "intent": "direction",
        "concepts": [
            {"id": "anchor", "text": "로데오모텔", "concept_type": "location",
             "role": "extent", "depends_on": []},
            {"id": "bearing", "text": "남쪽 방향", "concept_type": "field",
             "role": "measure", "depends_on": ["anchor"]},
            {"id": "kind", "text": "은행", "concept_type": "object",
             "role": "condition", "depends_on": []},
            {"id": "answer", "text": "가장 가까운 은행", "concept_type": "location",
             "role": "extent", "depends_on": ["anchor", "bearing", "kind"]},
        ],
        "measure": "direction",
    }
    graph = {
        "graph": [
            {"id": "places", "operator": "batch_geocode",
             "arguments": {"place_names": ["로데오모텔"], "anchor": "로데오모텔"},
             "depends_on": [], "output_type": "object", "role": "extent"},
            {"id": "banks", "operator": "nearby_places",
             "arguments": {"center": "$places.0.place", "category_code": "BK9",
                           "radius_m": 20000, "limit": 45},
             "depends_on": ["places"], "output_type": "object", "role": "condition"},
            {"id": "south", "operator": "filter_by_direction",
             "arguments": {"center": "$places.0.place", "places": "$banks", "direction": "남쪽"},
             "depends_on": ["banks"], "output_type": "object", "role": "support"},
            {"id": "nearest", "operator": "nearest",
             "arguments": {"anchor": "$places.0.place", "candidates": "$south"},
             "depends_on": ["south"], "output_type": "object", "role": "measure"},
        ]
    }
    factorized = factorize_geoflow(analysis, graph).as_dict()
    steps, constraints = normalize_and_validate_graph(factorized, max_steps=8)
    assert all(constraints.values())
    assert [step["id"] for step in steps] == ["places", "banks", "south", "nearest"]


def test_a_meal_question_is_not_answered_by_a_cafe() -> None:
    """Kakao files a cafe as `음식점 > 카페`, so the type word names its neighbour too.

    A "끼니를 해결해야 합니다" question offered a cafe 220 m away and the restaurant it means at
    461 m. On the option-ranking path there is no retrieval category to keep them apart, so the
    type test has to.
    """

    from src.tools.spatial import matches_required_type

    cafe = {"name": "헬리어드", "category": "음식점 > 카페"}
    bar = {"name": "한강르네상스", "category": "음식점 > 술집"}
    assert matches_required_type(bar, "FD6") is True
    assert matches_required_type(cafe, "FD6") is False
    # The cafe is still a cafe when a cafe is what was asked for.
    assert matches_required_type(cafe, "카페") is True
    assert matches_required_type(cafe, "CE7") is True


def test_the_deepest_kind_the_vocabulary_names_is_the_kind_the_place_is() -> None:
    """The taxonomy decides which of two kinds a path names, not a list of pairs to maintain.

    `음식점 > 카페` is the instance that was observed; `의료,건강 > 약국` is one nobody wrote a
    rule for, and it has to behave the same way or the exclusion is tuned to the benchmark. A
    coarser kind above a finer one in the same path is the parent, not the answer.
    """

    from src.tools.spatial import matches_required_type

    pharmacy = {"name": "이도약국", "category": "의료,건강 > 약국"}
    clinic = {"name": "드림서울이비인후과의원", "category": "의료,건강 > 병원 > 이비인후과"}
    # 병원's own alias list carries 의료, which the pharmacy's path also leads with.
    assert matches_required_type(pharmacy, "병원") is False
    assert matches_required_type(pharmacy, "약국") is True
    assert matches_required_type(clinic, "병원") is True
    # A kind named deeper than the match is only a finer kind when the lexicon knows it: a
    # 한식 restaurant is a restaurant, because the alias table files 한식 under 음식점.
    korean = {"name": "고기리막국수", "category": "음식점 > 한식 > 국수"}
    assert matches_required_type(korean, "음식점") is True


def test_a_filter_over_a_field_the_evidence_lacks_is_dropped() -> None:
    """Kakao publishes no ratings, so `min_rating` was a way to return nothing.

    An empty candidate list is what the generation stage guesses over, so a filter the evidence
    cannot speak to has to stand down -- and stand up again the moment one place carries the
    field, where an empty result is a real answer.
    """

    ops = SpatialOperatorRegistry()
    unrated = [
        {"place_id": "1", "name": "가", "latitude": 37.5, "longitude": 127.0},
        {"place_id": "2", "name": "나", "latitude": 37.5, "longitude": 127.0},
    ]
    assert ops.filter_places(unrated, min_rating=4.0) == unrated
    assert ops.filter_places(unrated, open_now=True) == unrated
    rated = [
        {**unrated[0], "rating": 4.5},
        {**unrated[1], "rating": 3.0},
    ]
    assert [place["place_id"] for place in ops.filter_places(rated, min_rating=4.0)] == ["1"]


def test_the_type_filter_and_the_ranking_agree_on_what_a_kind_is() -> None:
    ops = SpatialOperatorRegistry()
    anchor = {"name": "서함숲", "latitude": 37.5700, "longitude": 126.8800}
    cafe = {"place_id": "1", "name": "헬리어드", "category": "음식점 > 카페",
            "latitude": 37.5702, "longitude": 126.8800}
    restaurant = {"place_id": "2", "name": "한강르네상스", "category": "음식점 > 술집",
                  "latitude": 37.5740, "longitude": 126.8800}
    kept = ops.filter_places(places=[cafe, restaurant], required_types=["FD6"])
    assert [place["place_id"] for place in kept] == ["2"]
    ranked = ops.nearest(anchor=anchor, candidates=[cafe, restaurant], required_type="음식점")
    assert ranked["nearest"]["place_id"] == "2"


def test_a_null_measurement_is_skipped_rather_than_crashing_the_ranking() -> None:
    """A key that is present and null is not a comparable value."""

    ops = SpatialOperatorRegistry()
    items = [{"label": "a", "distance_m": None}, {"label": "b", "distance_m": 900.0}]
    assert ops.select_min(items, "distance_m")["label"] == "b"
    assert [item["label"] for item in ops.sort_by(items, "distance_m")] == ["b"]
    with pytest.raises(ValueError, match="comparable key"):
        ops.select_min([{"label": "a", "distance_m": None}], "distance_m")


def test_a_distance_option_is_read_past_its_own_separator() -> None:
    """"남쪽, 약 6.6km" begins with a comma, and the old pattern matched that comma.

    `[\\d,.]+` accepted the separator as the number and then called `float("")`, so every option
    of a direction-and-distance question failed to parse — the whole family's measurement step
    reported an error while the coordinates behind it were correct.
    """

    ops = SpatialOperatorRegistry()
    options = ["남쪽, 약 6.6km", "남쪽, 약 10.6km", "북쪽, 약 10.6km", "북쪽, 약 6.6km"]
    matched = ops.match_distance_options(distance={"distance_m": 6650}, options=options)
    assert matched["best_option"] == 0
    assert matched["fits"] is True
    # A plain metre count still reads as metres.
    plain = ops.match_distance_options(
        distance={"distance_m": 1058}, options=["1036 m", "1061 m", "1.2 km"]
    )
    assert plain["best_option"] == 1


def test_a_clock_reads_the_duration_out_of_whatever_carries_it() -> None:
    """A planner hands the operator the route it measured, not the route's duration."""

    ops = SpatialOperatorRegistry()
    from_number = ops.calculate_start_time(
        arrival_time="오후 6시 00분", duration_s=3600, timezone="Asia/Seoul"
    )
    from_route = ops.calculate_start_time(
        arrival_time="오후 6시 00분",
        duration_s={"origin": "A", "destination": "B", "distance_m": 12000, "duration_s": 3600},
        timezone="Asia/Seoul",
    )
    assert from_route["start_time"] == from_number["start_time"]
    from_tour = ops.calculate_start_time(
        arrival_time="오후 6시 00분",
        duration_s={"order": [0, 1], "total_cost": 3600.0, "feasible": True},
        timezone="Asia/Seoul",
    )
    assert from_tour["start_time"] == from_number["start_time"]
    with pytest.raises(ValueError, match="no duration"):
        ops.calculate_start_time(
            arrival_time="오후 6시 00분", duration_s={"name": "A"}, timezone="Asia/Seoul"
        )


def test_sum_route_metrics_takes_the_matrix_node_a_planner_referenced() -> None:
    """`$legs` is the whole `distance_matrix` node, which carries its routes under "routes".

    Refusing the node lost the leg sum and every clock built on it. A leg that failed carries no
    measurement and is not a zero-length one, so it is left out of the total rather than counted
    as nothing.
    """

    ops = SpatialOperatorRegistry()
    node = {
        "routes": [
            {"distance_m": 100, "duration_s": 10, "status": "ok"},
            {"distance_m": 200, "duration_s": 20, "status": "ok"},
            {"status": "error", "error": "RouteNotFoundError: ..."},
        ],
        "matrix": None,
    }
    assert ops.invoke("sum_route_metrics", {"routes": node}) == {
        "distance_m": 300,
        "duration_s": 30,
    }


def test_a_ranking_takes_the_node_that_carries_the_places() -> None:
    """`match_options` reports its places under `retrieved_places`, `nearest` under `ranked`.

    A planner passes the node rather than the field, and the whole record then read as one
    unresolvable candidate — which now raises rather than ranking nothing, so the unwrap has to
    know where an operator keeps its places.
    """

    ops = SpatialOperatorRegistry()
    anchor = {"name": "기준", "latitude": 37.50, "longitude": 127.00}
    matched = {
        "mode": "name",
        "retrieved_places": [
            {"place_id": "1", "name": "가까운곳", "latitude": 37.501, "longitude": 127.00},
            {"place_id": "2", "name": "먼곳", "latitude": 37.520, "longitude": 127.00},
        ],
        "best_option": 0,
    }
    assert ops.nearest(anchor=anchor, candidates=matched)["nearest"]["place_id"] == "1"


def test_option_totals_take_the_matrix_node_a_planner_referenced() -> None:
    ops = SpatialOperatorRegistry()
    node = {
        "routes": [
            {"distance_m": 10, "duration_s": 1, "status": "ok"},
            {"distance_m": 20, "duration_s": 2, "status": "ok"},
        ],
        "matrix": None,
    }
    totals = ops.aggregate_route_groups(routes=node, groups=[[0], [1]])
    assert [entry["duration_s"] for entry in totals["option_totals"]] == [1, 2]


def test_the_farthest_of_three_plan_validates_and_runs() -> None:
    """The plan v6 lost four rows to, end to end: pick the largest distance, then match options.

    `select_max` returns the winning *record*, and `match_distance_options` reads metres off a
    record perfectly well -- so the only thing that ever refused this graph was the declared-type
    table saying "object" where the operator says "whatever carries a measurement". Upstream has
    no such table at all.
    """

    from src.agent.geoflow import normalize_and_validate_graph
    from src.tools import SpatialOperatorRegistry

    graph = {
        "graph": [
            {
                "id": "places",
                "operator": "batch_geocode",
                "arguments": {"place_names": ["A", "B", "C", "D"], "anchor": "A"},
                "role": "extent",
            },
            {
                "id": "d1",
                "operator": "haversine_distance",
                "arguments": {"place_a": "$places.0.place", "place_b": "$places.1.place"},
                "depends_on": ["places"],
                "role": "condition",
            },
            {
                "id": "d2",
                "operator": "haversine_distance",
                "arguments": {"place_a": "$places.0.place", "place_b": "$places.2.place"},
                "depends_on": ["places"],
                "role": "condition",
            },
            {
                "id": "farthest",
                "operator": "select_max",
                "arguments": {"items": ["$d1", "$d2"], "key": "distance_m"},
                "depends_on": ["d1", "d2"],
                "role": "condition",
            },
            {
                "id": "answer",
                "operator": "match_distance_options",
                "arguments": {"distance": "$farthest", "options": ["약 1.0km", "약 9.0km"]},
                "depends_on": ["farthest"],
                "role": "measure",
            },
        ]
    }

    steps, _ = normalize_and_validate_graph(graph, max_steps=10)
    assert [step["id"] for step in steps][-1] == "answer"

    ops = SpatialOperatorRegistry()
    farthest = ops.invoke(
        "select_max",
        {"items": [{"distance_m": 1000.0}, {"distance_m": 9100.0}], "key": "distance_m"},
    )
    matched = ops.invoke(
        "match_distance_options",
        {"distance": farthest, "options": ["약 1.0km", "약 9.0km"]},
    )
    assert matched["best_option"] == 1
    assert matched["computed_distance_m"] == 9100.0


def test_a_lenient_pass_skips_our_own_rules_and_keeps_every_structural_one() -> None:
    """The two rules upstream does not have must not be able to lose a graph that would run.

    Everything about whether the graph *can* execute -- unknown operators, dependencies that are
    not nodes, cycles, a graph with no Measure -- still refuses it, leniently or not.
    """

    from src.agent.geoflow import normalize_and_validate_graph

    typed_wrong = {
        "graph": [
            {
                "id": "distance",
                "operator": "haversine_distance",
                "arguments": {
                    "place_a": {"latitude": 37.0, "longitude": 127.0},
                    "place_b": {"latitude": 37.1, "longitude": 127.1},
                },
                "role": "extent",
            },
            {
                "id": "totals",
                "operator": "sum_route_metrics",
                "arguments": {"routes": "$distance"},
                "depends_on": ["distance"],
                "role": "measure",
            },
        ]
    }
    with pytest.raises(ValueError, match="Type compatibility"):
        normalize_and_validate_graph(typed_wrong, max_steps=8)
    steps, _ = normalize_and_validate_graph(typed_wrong, max_steps=8, strict_types=False)
    assert [step["id"] for step in steps] == ["distance", "totals"]

    unknown_operator = {
        "graph": [
            {
                "id": "a",
                "operator": "teleport",
                "arguments": {},
                "role": "extent",
            },
            {
                "id": "b",
                "operator": "match_options",
                "arguments": {"options": [], "places": "$a"},
                "depends_on": ["a"],
                "role": "measure",
            },
        ]
    }
    with pytest.raises(ValueError):
        normalize_and_validate_graph(unknown_operator, max_steps=8, strict_types=False)

    no_measure = {
        "graph": [
            {
                "id": "places",
                "operator": "batch_geocode",
                "arguments": {"place_names": ["A"]},
                "role": "extent",
            }
        ]
    }
    with pytest.raises(ValueError, match="Measure"):
        normalize_and_validate_graph(no_measure, max_steps=8, strict_types=False)


def test_the_concept_role_rule_is_skipped_leniently_like_the_node_one() -> None:
    """One rule applied to two graphs; both halves have to be skippable or neither is.

    Real analysis and plan from `seoul_kmapeval_v6_004`, one of three questions the first run on
    the new model still lost to `Concept role ordering violation` -- raised during factorization,
    where the node-level flag never reached. The plan runs perfectly well: grounding is what
    introduces the offending edge, and upstream annotates functional roles without enforcing any
    ordering between them.
    """

    from src.agent.spatial import _factorize_validate_plan

    analysis = {
            "intent": "nearby",
            "concepts": [
                    {
                            "id": "anchor",
                            "text": "노량진만나로 골목형상점가",
                            "concept_type": "location",
                            "role": "extent",
                            "attributes": {},
                            "depends_on": []
                    },
                    {
                            "id": "target_type",
                            "text": "내과",
                            "concept_type": "object",
                            "role": "support",
                            "attributes": {},
                            "depends_on": [
                                    "anchor"
                            ]
                    },
                    {
                            "id": "candidates",
                            "text": (
                                "['동작고려의원', '김영내과의원', "
                                "'이용국내과의원', '기쁨준내과의원']"
                            ),
                            "concept_type": "location",
                            "role": "measure",
                            "attributes": {
                                    "rank": "third_nearest"
                            },
                            "depends_on": [
                                    "anchor",
                                    "target_type"
                            ]
                    }
            ],
            "measure": "nearby",
            "target_type": None
    }

    graph = [
            {
                    "id": "all_locations",
                    "operator": "batch_geocode",
                    "arguments": {
                            "place_names": [
                                    "동작고려의원",
                                    "김영내과의원",
                                    "이용국내과의원",
                                    "기쁨준내과의원"
                            ],
                            "anchor": "노량진만나로 골목형상점가"
                    },
                    "depends_on": [],
                    "output_type": "object",
                    "role": "extent",
                    "concept_ids": [
                            "anchor",
                            "candidates"
                    ]
            },
            {
                    "id": "distances",
                    "operator": "pairwise_distances",
                    "arguments": {
                            "pairs": [
                                    {
                                            "place_a": "$all_locations.0.place",
                                            "place_b": "$all_locations.1.place",
                                            "label": "동작고려의원"
                                    },
                                    {
                                            "place_a": "$all_locations.0.place",
                                            "place_b": "$all_locations.2.place",
                                            "label": "김영내과의원"
                                    },
                                    {
                                            "place_a": "$all_locations.0.place",
                                            "place_b": "$all_locations.3.place",
                                            "label": "이용국내과의원"
                                    },
                                    {
                                            "place_a": "$all_locations.0.place",
                                            "place_b": "$all_locations.4.place",
                                            "label": "기쁨준내과의원"
                                    }
                            ]
                    },
                    "depends_on": [
                            "all_locations"
                    ],
                    "output_type": "field",
                    "role": "support",
                    "concept_ids": [
                            "anchor",
                            "candidates"
                    ]
            },
            {
                    "id": "third_nearest",
                    "operator": "identity_measure",
                    "arguments": {
                            "value": "third_nearest_candidate_by_distance"
                    },
                    "depends_on": [
                            "distances"
                    ],
                    "output_type": "object",
                    "role": "measure",
                    "concept_ids": [
                            "candidates"
                    ]
            }
    ]

    question = (
        "지금 노량진만나로 골목형상점가에 있습니다. 어제부터 배탈이 나서 계속 속이 안 좋습니다. "
        "여기서 세 번째로 가까운 내과는 다음 중 어디인가요?"
    )
    options = ["동작고려의원", "김영내과의원", "이용국내과의원", "기쁨준내과의원"]

    with pytest.raises(ValueError, match="Concept role ordering violation"):
        _factorize_validate_plan(analysis, {"graph": graph}, question, options, "nearby", 15)

    _, steps, _ = _factorize_validate_plan(
        analysis, {"graph": graph}, question, options, "nearby", 15, strict_types=False
    )
    assert [step["id"] for step in steps] == ["all_locations", "distances", "third_nearest"]


def test_a_node_that_names_no_operator_is_the_planners_failure_not_a_crash() -> None:
    """Seen once in 900 questions: `"operator": ["extract_distance", "extract_distance", ...]`.

    Grounding looked that list up in a set of operator names and came back as
    `TypeError: unhashable type: 'list'` -- a crash in our own file, recorded against the agent as
    an `agent_reasoning_failure` it had to be read to understand.
    """

    from src.agent.spatial import _factorize_validate_plan

    analysis = {
        "intent": "distance",
        "concepts": [
            {
                "id": "anchor",
                "text": "서울역",
                "concept_type": "location",
                "role": "extent",
                "attributes": {},
                "depends_on": [],
            }
        ],
        "measure": "distance",
    }
    graph = [
        {
            "id": "places",
            "operator": "batch_geocode",
            "arguments": {"place_names": ["서울역", "경복궁"]},
            "role": "extent",
            "concept_ids": ["anchor"],
        },
        {
            "id": "legs",
            "operator": ["extract_distance", "extract_distance"],
            "arguments": {"routes": "$places"},
            "depends_on": ["places"],
            "role": "measure",
        },
    ]

    with pytest.raises(ValueError, match="names no operator"):
        _factorize_validate_plan(
            analysis, {"graph": graph}, "서울역에서 경복궁까지?", ["1km", "2km"], "distance", 15
        )


def test_the_second_closest_plan_validates_and_runs() -> None:
    """The ordinal plan the planner kept writing against an operator set that had no k-th.

    Replayed from `seoul_kmapeval_v7h_010`'s compose stage: geocode the anchor and the four
    options, measure each anchor-to-option distance, sort them, and take index 1. Every node of
    it existed except the last, and `Unknown GeoFlow operator: select_by_index` ended the
    question. Six questions across these runs died on that name and three more on
    `select_second_closest`, `select_second_nearest` and `select_second_min`.
    """

    from src.agent.geoflow import normalize_and_validate_graph

    graph = {
        "graph": [
            {
                "id": "all_places",
                "operator": "batch_geocode",
                "arguments": {"place_names": ["anchor", "A", "B", "C"], "anchor": "anchor"},
                "role": "extent",
            },
            {
                "id": "all_distances",
                "operator": "pairwise_distances",
                "arguments": {
                    "pairs": [
                        {"place_a": "$all_places.0.place", "place_b": "$all_places.1.place"},
                        {"place_a": "$all_places.0.place", "place_b": "$all_places.2.place"},
                        {"place_a": "$all_places.0.place", "place_b": "$all_places.3.place"},
                    ]
                },
                "depends_on": ["all_places"],
                "role": "support",
            },
            {
                "id": "sorted_distances",
                "operator": "sort_by",
                "arguments": {"items": "$all_distances", "key": "distance_m"},
                "depends_on": ["all_distances"],
                "role": "support",
            },
            {
                "id": "second_closest",
                "operator": "select_by_index",
                "arguments": {"items": "$sorted_distances", "index": 1},
                "depends_on": ["sorted_distances"],
                "role": "support",
            },
            {
                "id": "measure",
                "operator": "identity_measure",
                "arguments": {"value": "$second_closest"},
                "depends_on": ["second_closest"],
                "role": "measure",
            },
        ]
    }

    steps, _ = normalize_and_validate_graph(graph, max_steps=15)
    assert [step["id"] for step in steps][-1] == "measure"

    ops = SpatialOperatorRegistry()
    measured = [
        {"pair_index": 0, "label": "A", "distance_m": 900.0},
        {"pair_index": 1, "label": "B", "distance_m": 300.0},
        {"pair_index": 2, "label": "C", "distance_m": 600.0},
    ]
    ordered = ops.invoke("sort_by", {"items": measured, "key": "distance_m"})
    second = ops.invoke("select_by_index", {"items": ordered, "index": 1})
    # 0-based: the nearest is index 0, so "두 번째로 가까운" is index 1.
    assert second["label"] == "C"
    assert second["selected_index"] == 1
    assert ops.invoke("select_by_index", {"items": ordered, "index": 0})["label"] == "B"


def test_an_ordinal_past_the_end_of_the_list_fails_instead_of_clamping() -> None:
    """The nearest place is not the second nearest, and answering as if it were invents evidence."""

    ops = SpatialOperatorRegistry()
    with pytest.raises(ValueError, match="0-based"):
        ops.invoke("select_by_index", {"items": [{"distance_m": 5.0}], "index": 1})


def test_two_measured_legs_add_up_to_the_via_route_total() -> None:
    """`sum_amounts` closes the gap between measuring legs and answering with their total.

    `sum_route_metrics` needs a route list and `aggregate_route_groups` needs route indexes, so a
    graph that had already reduced each leg to an amount with `extract_distance` had nothing left
    that could add two numbers. This is the plan from `seoul_kmapeval_v7h_052`, whose planner
    wrote `sum_amounts(items=[$leg1_distance_m, $leg2_distance_m])`.
    """

    ops = SpatialOperatorRegistry()
    total = ops.invoke("sum_amounts", {"items": [{"distance_m": 4200.0}, {"distance_m": 3400.0}]})
    assert total["distance_m"] == 7600.0
    assert total["value"] == 7600.0

    matched = ops.invoke(
        "match_distance_options",
        {"distance": total, "options": ["약 5.0km", "약 7.6km", "약 9.9km"]},
    )
    assert matched["best_option"] == 1


def test_a_route_shaped_sum_keeps_both_metrics_and_a_duration_sum_is_not_a_distance() -> None:
    """Seconds are not metres, and the result must not let a plan spend them as though they were."""

    ops = SpatialOperatorRegistry()
    both = ops.invoke(
        "sum_amounts",
        {
            "routes": [
                {"distance_m": 1200, "duration_s": 300},
                {"distance_m": 800, "duration_s": 200},
            ]
        },
    )
    assert (both["distance_m"], both["duration_s"]) == (2000.0, 500.0)

    seconds = ops.invoke("sum_amounts", {"amounts": [{"duration_s": 300}, {"duration_s": 200}]})
    assert seconds["total"] == 500.0
    assert "value" not in seconds and "distance_m" not in seconds
    with pytest.raises(ValueError, match="measured distance"):
        ops.invoke("match_distance_options", {"distance": seconds, "options": ["약 1.0km"]})


def test_a_detour_cost_is_a_subtraction_the_operator_set_could_not_express() -> None:
    """`difference` keeps the sign and reports the magnitude, which is what an option carries."""

    ops = SpatialOperatorRegistry()
    detour = ops.invoke("difference", {"a": {"distance_m": 9100.0}, "b": {"distance_m": 5900.0}})
    assert detour["difference"] == 3200.0
    assert detour["value"] == 3200.0

    # Subtracted the other way round, the option it matches is the same one.
    reversed_order = ops.invoke("difference", {"values": [{"distance_m": 5900.0}, 9100.0]})
    assert reversed_order["difference"] == -3200.0
    assert reversed_order["value"] == 3200.0
    matched = ops.invoke(
        "match_distance_options",
        {"distance": reversed_order, "options": ["약 1.2km", "약 3.2km"]},
    )
    assert matched["best_option"] == 1


def test_an_unresolved_reference_is_not_quietly_summed_as_zero() -> None:
    """One planner wrote `sum_amounts(amounts=["dist_A_C", "dist_C_B"])` -- names, not values.

    Those are node ids that were never marked with `$`, so nothing resolved them and they arrive
    as text. Adding them as zeroes would answer the question with a total of nothing.
    """

    ops = SpatialOperatorRegistry()
    with pytest.raises(ValueError, match="never resolved"):
        ops.invoke("sum_amounts", {"amounts": ["dist_A_C", "dist_C_B"]})


def test_a_synonym_for_a_now_existing_operator_resolves_to_it() -> None:
    """`calculate_difference` and `subtraction` are the same operation under another name.

    Both were written by planners before `difference` existed. The rewrite happens once, in
    `normalize_and_validate_graph`, so the executor is handed the canonical name and there is no
    second table to keep in step. Names that would need an ordinal read out of them --
    `select_second_closest` -- are deliberately not here.
    """

    from src.agent.geoflow import OPERATOR_SYNONYMS, normalize_and_validate_graph

    graph = {
        "graph": [
            {
                "id": "where",
                "operator": "batch_geocode",
                "arguments": {"place_names": ["X", "Y", "Z"], "anchor": "X"},
                "role": "extent",
            },
            {
                "id": "a",
                "operator": "haversine_distance",
                "arguments": {"place_a": "$where.0.place", "place_b": "$where.1.place"},
                "depends_on": ["where"],
                "role": "condition",
            },
            {
                "id": "b",
                "operator": "haversine_distance",
                "arguments": {"place_a": "$where.0.place", "place_b": "$where.2.place"},
                "depends_on": ["where"],
                "role": "condition",
            },
            {
                "id": "gap",
                "operator": "calculate_difference",
                "arguments": {"a": "$a", "b": "$b"},
                "depends_on": ["a", "b"],
                "role": "measure",
            },
        ]
    }
    steps, _ = normalize_and_validate_graph(graph, max_steps=10)
    assert steps[-1]["operator"] == "difference"
    assert "select_second_closest" not in OPERATOR_SYNONYMS


def test_the_required_argument_check_accepts_every_spelling_the_operator_does() -> None:
    """A plan the executor would have run must not be refused for how it spelled a slot.

    `sum_amounts(items=[...])` is what the planner wrote in `seoul_kmapeval_v7h_052`, and the
    canonical name is `amounts`. The same defect was already sitting under `select_min`, whose
    contract required a `key` the implementation defaults and an `items` the implementation
    accepts as `values`, `inputs`, `list` or `candidates`.
    """

    from src.agent.geoflow import normalize_and_validate_graph

    def measure(operator: str, arguments: dict) -> dict:
        return {
            "graph": [
                {
                    "id": "where",
                    "operator": "batch_geocode",
                    "arguments": {"place_names": ["X", "Y"], "anchor": "X"},
                    "role": "extent",
                },
                {
                    "id": "m",
                    "operator": operator,
                    "arguments": arguments,
                    "depends_on": ["where"],
                    "role": "measure",
                },
            ]
        }

    for operator, arguments in (
        ("sum_amounts", {"items": [1, 2]}),
        ("sum_amounts", {"legs": [1, 2]}),
        ("select_min", {"values": [3, 1]}),
        ("select_by_index", {"candidates": [1, 2], "i": 1}),
        ("difference", {"a": 5, "b": 2}),
    ):
        steps, _ = normalize_and_validate_graph(measure(operator, arguments), max_steps=5)
        assert steps[-1]["operator"] in {operator, "difference"}

    ops = SpatialOperatorRegistry()
    assert ops.invoke("sum_amounts", {"items": [1, 2]})["total"] == 3.0
    assert ops.invoke("select_min", {"values": [3, 1]})["value"] == 1


def test_a_ranking_without_a_named_key_ranks_by_the_measurement_it_was_given() -> None:
    """Relaxing the contract must not turn a repairable plan into an executed failure.

    `select_max`'s contract used to require a `key` the implementation defaults, so a plan that
    omitted it was refused at validation and went to the repair round. Now that the required
    check accepts what the implementation accepts, the implementation has to actually handle the
    omission -- otherwise the same plan reaches execution and dies there instead, which is worse.
    Forty-five calls across `logs/` rank measured distances with no key spelled out.
    """

    ops = SpatialOperatorRegistry()
    measured = [{"distance_m": 900.0}, {"distance_m": 2400.0}, {"distance_m": 1500.0}]
    assert ops.invoke("select_max", {"items": measured})["distance_m"] == 2400.0
    assert ops.invoke("select_min", {"items": measured})["distance_m"] == 900.0
    # An explicit key still wins over the inferred one.
    rated = [{"distance_m": 900.0, "rating": 4.8}, {"distance_m": 2400.0, "rating": 2.1}]
    assert ops.invoke("select_max", {"items": rated, "key": "rating"})["rating"] == 4.8


def test_a_minimum_asked_for_by_index_is_an_ordinal_not_a_minimum() -> None:
    """Both `seoul_kmapeval_v7h_001` and `_010` wrote `select_min(items=..., index=1)`.

    The old contract refused those for a missing `key`, which sent them to the repair round.
    Accepting them without honouring `index` would be worse than either: the plan would run and
    return the *nearest* candidate for a question asking which is second nearest, and a wrong
    answer given confidently is not a failure anybody can see.
    """

    from src.agent.geoflow import normalize_and_validate_graph

    graph = {
        "graph": [
            {
                "id": "geo",
                "operator": "batch_geocode",
                "arguments": {"place_names": ["anchor", "A", "B"], "anchor": "anchor"},
                "role": "extent",
            },
            {
                "id": "ranked",
                "operator": "nearest",
                "arguments": {"anchor": "$geo.0.place", "candidates": "$geo"},
                "depends_on": ["geo"],
                "role": "support",
            },
            {
                "id": "final_selection",
                "operator": "select_min",
                "arguments": {"items": "$ranked.candidates", "index": 1},
                "depends_on": ["ranked"],
                "role": "measure",
            },
        ]
    }
    steps, _ = normalize_and_validate_graph(graph, max_steps=10)
    assert steps[-1]["operator"] == "select_by_index"
    assert steps[-1]["arguments"]["index"] == 1

    ops = SpatialOperatorRegistry()
    ranked = [{"distance_m": 300.0}, {"distance_m": 600.0}, {"distance_m": 900.0}]
    assert ops.invoke("select_by_index", {"items": ranked, "index": 1})["distance_m"] == 600.0


def test_an_ordinal_nearby_question_retrieves_before_it_ranks() -> None:
    """The k-th nearest place is the k-th of the neighbourhood, not of the option list.

    `nearby_kth_nearest` draws the gold as rank k of everything within 1800 m and the three
    decoys from ranks 1 to 6, so the options are a subset of the ranking and the nearest place is
    only sometimes among them. Ranking the four options against each other therefore answers a
    different question. Across the v7 runs, 130 composed plans for the two ordinal families
    retrieved anything four times: `Geocode-Batch-Compare` won on "가까운" and its example is
    exactly "geocode the anchor and the options, then take the nearest".
    """

    from src.agent.geoflow import retrieve_templates

    ordinal = retrieve_templates("nearby", "삼성출판박물관에서 두 번째로 가까운 편의점은?")
    assert ordinal[0]["name"] == "Retrieve-Rank-Ordinal"
    assert "nearby_places" in ordinal[0]["pattern"]

    # A superlative is not an ordinal: there the options *are* the candidates, and the template
    # that ranks them stays in front.
    superlative = retrieve_templates("nearby", "서울역에서 가장 가까운 편의점은 어디인가요?")
    assert superlative[0]["name"] == "Geocode-Batch-Compare"

    # And a radius question is neither.
    radius = retrieve_templates("nearby", "서울역 반경 500m 이내에 있는 약국은 몇 곳인가요?")
    assert radius[0]["name"] == "Filter-Aggregate-Measure"


def test_the_ordinal_nearby_chain_lands_on_the_option_the_ranking_names() -> None:
    """Retrieve, rank, take index k-1, ground it back to an option -- on real numbers.

    These are `seoul_kmapeval_v7h2_011`'s: the second nearest convenience store to 양천문화원 is
    GS25 목동파크점, which the question offers as option 3. Ranking only the three named options
    would have put it first.
    """

    ops = SpatialOperatorRegistry()
    anchor = {"name": "양천문화원", "latitude": 37.55, "longitude": 126.97}
    neighbourhood = [
        {"name": "CU 뉴목동14단지점", "latitude": 37.5505, "longitude": 126.9705},
        {"name": "GS25 목동파크점", "latitude": 37.5512, "longitude": 126.9712},
        {"name": "GS25 목동본점", "latitude": 37.5525, "longitude": 126.9725},
    ]
    ranking = ops.invoke("nearest", {"anchor": anchor, "candidates": neighbourhood})
    second = ops.invoke("select_by_index", {"items": ranking["ranked"], "index": 1})
    assert second["name"] == "GS25 목동파크점"

    matched = ops.invoke(
        "match_options",
        {
            "options": [
                "주어진 지도 정보로는 알 수 없음",
                "CU 뉴목동14단지점",
                "GS25 목동본점",
                "GS25 목동파크점",
            ],
            "places": [second],
        },
    )
    assert matched["best_option"] == 3


def test_the_ordinal_template_is_offered_whichever_intent_the_analysis_guessed() -> None:
    """Gating a template on one intent label gates it on the Analysis stage guessing right.

    Of 54 ordinal questions the stage called 14 of them `poi` and the rest `nearby`, and the 14
    were handed `Geocode-Batch-Compare` instead -- which spans four intents and shows exactly the
    option-ranking shape the ordinal question must not use.
    """

    from src.agent.geoflow import retrieve_templates

    for intent in ("nearby", "poi"):
        names = [t["name"] for t in retrieve_templates(intent, "여기서 세 번째로 가까운 내과는?")]
        assert names[0] == "Retrieve-Rank-Ordinal", intent


def test_a_superseded_template_is_not_offered_beside_the_one_that_beat_it() -> None:
    """A worked example that answers a different question is worse than no second example.

    27 of the 40 plans that were offered the ordinal template still built
    `Geocode-Batch-Compare`'s shape -- geocode the four options, take the nearest -- because it
    was sitting right there beside it. Suppression only runs downward: on a superlative question
    `Geocode-Batch-Compare` wins and keeps its place.
    """

    from src.agent.geoflow import retrieve_templates

    ordinal = [t["name"] for t in retrieve_templates("nearby", "두 번째로 가까운 편의점은?")]
    assert "Geocode-Batch-Compare" not in ordinal

    superlative = [t["name"] for t in retrieve_templates("nearby", "가장 가까운 편의점은?")]
    assert superlative[0] == "Geocode-Batch-Compare"
    assert "Retrieve-Rank-Ordinal" in superlative

    # A question neither template is about keeps whatever it had.
    radius = [t["name"] for t in retrieve_templates("nearby", "반경 500m 이내에 약국은 몇 곳?")]
    assert radius[0] == "Filter-Aggregate-Measure"
