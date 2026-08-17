from __future__ import annotations

from typing import Any

import pytest

from src.agent import ReactAgent, SpatialAgent
from src.agent.geoflow import (
    CORE_CONCEPTS,
    OPERATOR_CONTRACTS,
    TEMPLATES,
    build_concept_graph,
    factorize_geoflow,
    normalize_analysis,
    normalize_and_validate_graph,
)
from src.agent.spatial import (
    _bind_prevalidated_template,
    _ground_graph_literals,
    _heuristic_intent,
    _resolve_output_binding,
    _resolve_references,
)
from src.llm import LLMResponse, LLMToolCall
from src.models import Place, Route
from src.tools import MapProvider, SpatialOperatorRegistry, ToolRegistry


class FakeProvider(MapProvider):
    def __init__(self) -> None:
        self._api_calls = 0
        self.place = Place(
            place_id="1",
            name="경복궁",
            address="서울 종로구",
            latitude=37.5796,
            longitude=126.977,
            category="관광명소",
        )

    @property
    def api_call_count(self) -> int:
        return self._api_calls

    def search_place(self, query: str, *, limit: int = 5) -> list[Place]:
        self._api_calls += 1
        return [self.place]

    def geocode(self, address: str, *, limit: int = 5) -> list[Place]:
        self._api_calls += 1
        return [self.place]

    def nearby_search(self, center: str | Place, **_: Any) -> list[Place]:
        self._api_calls += 1
        return [self.place]

    def place_details(self, place_id: str) -> Place:
        return self.place

    def directions(self, origin: str | Place, destination: str | Place, **_: Any) -> Route:
        self._api_calls += 1
        return Route(origin="A", destination="B", distance_m=100, duration_s=20)


class QueuedLLM:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = responses
        self.messages: list[list[dict[str, Any]]] = []

    def chat(self, messages: list[dict[str, Any]], *, tools=None) -> LLMResponse:
        self.messages.append(messages)
        return self.responses.pop(0)


def test_react_executes_common_tool_then_parses_answer() -> None:
    llm = QueuedLLM(
        [
            LLMResponse("", (LLMToolCall("call-1", "place_search", {"query": "경복궁"}),)),
            LLMResponse("^^2^^"),
        ]
    )
    agent = ReactAgent(llm, ToolRegistry(FakeProvider()), max_steps=3)
    result = agent.answer("질문", ["A", "B", "C", "D"])
    assert result.predicted_answer == 2
    assert result.tool_calls == 1
    assert result.api_calls == 1
    assert result.failure_type is None


def test_spatial_agent_runs_paper_aligned_pipeline() -> None:
    llm = QueuedLLM(
        [
            LLMResponse('{"intent":"poi"}'),
            LLMResponse(
                '{"steps":[{"id":"s1","operator":"place_search",'
                '"arguments":{"query":"경복궁","limit":1}}]}'
            ),
            LLMResponse('{"predicted_option":1,"confidence":0.9,"reason":"evidence"}'),
        ]
    )
    agent = SpatialAgent(llm, ToolRegistry(FakeProvider()), max_steps=3)
    result = agent.answer("질문", ["A", "B"])
    assert result.predicted_answer == 1
    assert result.response == "^^1^^"
    assert [entry["stage"] for entry in result.trace] == [
        "analyze",
        "retrieve_templates",
        "compose",
        "factorize",
        "validate",
        "execute",
        "evaluate",
        "generate",
    ]
    assert result.tool_calls == 1
    assert result.api_calls == 1


def test_registry_returns_error_observation_instead_of_raising() -> None:
    registry = ToolRegistry(FakeProvider())
    execution = registry.invoke("nearby_places", {"center": "경복궁"})
    assert execution.status == "error"
    assert "requires query or category_code" in (execution.error or "")


def test_registry_clamps_llm_generated_kakao_limits() -> None:
    registry = ToolRegistry(FakeProvider())
    search = registry.invoke("place_search", {"query": "경복궁", "limit": 20})
    nearby = registry.invoke(
        "nearby_places",
        {"center": "경복궁", "query": "식당", "radius_m": 50_000, "limit": 50},
    )
    assert search.status == "ok"
    assert search.arguments["limit"] == 15
    assert nearby.status == "ok"
    assert nearby.arguments["radius_m"] == 20_000
    assert nearby.arguments["limit"] == 45


def test_registry_infers_exact_kakao_category_for_station_search() -> None:
    execution = ToolRegistry(FakeProvider()).invoke(
        "nearby_places", {"center": "경복궁", "query": "역", "radius_m": 300}
    )
    assert execution.status == "ok"
    assert execution.arguments["category_code"] == "SW8"


def test_batch_geocode_preserves_input_order_in_one_geoflow_tool_call() -> None:
    registry = ToolRegistry(FakeProvider())
    execution = registry.invoke("batch_geocode", {"place_names": ["경복궁", "광화문"], "limit": 1})
    assert execution.status == "ok"
    assert [item["query"] for item in execution.output] == ["경복궁", "광화문"]
    assert all(item["place"]["name"] == "경복궁" for item in execution.output)
    assert registry.tool_call_count == 1
    assert registry.provider.api_call_count == 2


def test_batch_geocode_progressively_retries_a_business_section_suffix() -> None:
    class VariantProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__()
            self.queries: list[str] = []

        def search_place(self, query: str, *, limit: int = 5) -> list[Place]:
            self.queries.append(query)
            self._api_calls += 1
            return [] if query == "더현대서울식품관" else [self.place]

    provider = VariantProvider()
    execution = ToolRegistry(provider).invoke(
        "batch_geocode", {"place_names": ["더현대서울식품관"]}
    )
    assert execution.status == "ok"
    assert execution.output[0]["place"] is not None
    assert provider.queries[:2] == ["더현대서울식품관", "더현대서울"]


def test_batch_geocode_strips_branch_suffix_after_exact_search_misses() -> None:
    class BranchProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__()
            self.queries: list[str] = []

        def search_place(self, query: str, *, limit: int = 5) -> list[Place]:
            self.queries.append(query)
            self._api_calls += 1
            if query != "힘난다버거":
                return []
            return [self.place.model_copy(update={"name": "힘난다버거 용두점"})]

    provider = BranchProvider()
    execution = ToolRegistry(provider).invoke(
        "batch_geocode", {"place_names": ["힘난다버거 용두점"]}
    )
    assert execution.status == "ok"
    assert execution.output[0]["place"] is not None
    assert provider.queries == ["힘난다버거 용두점", "힘난다버거"]


def test_batch_geocode_prefers_bank_branch_over_same_brand_atm() -> None:
    class BankProvider(FakeProvider):
        def search_place(self, query: str, *, limit: int = 5) -> list[Place]:
            self._api_calls += 1
            return [self.place]

        def nearby_search(self, center: str | Place, **_: Any) -> list[Place]:
            self._api_calls += 1
            return [
                Place(
                    place_id="atm",
                    name="하나은행365 아파트상가 ATM",
                    address="서울",
                    latitude=37.58,
                    longitude=126.98,
                    category="금융,보험 > 금융서비스 > 은행 > ATM",
                ),
                Place(
                    place_id="branch",
                    name="하나은행 광운대출장소",
                    address="서울",
                    latitude=37.59,
                    longitude=126.99,
                    category="금융,보험 > 금융서비스 > 은행 > 하나은행",
                ),
            ]

    execution = ToolRegistry(BankProvider()).invoke(
        "batch_geocode",
        {"place_names": ["GS25 장위뉴타운점", "하나은행"], "anchor": "GS25 장위뉴타운점"},
    )

    assert execution.status == "ok"
    assert execution.output[1]["place"]["place_id"] == "branch"


def test_recover_option_places_queries_only_missing_option_evidence() -> None:
    class OptionProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__()
            self.queries: list[str] = []

        def nearby_search(self, center: str | Place, **kwargs: Any) -> list[Place]:
            query = kwargs.get("query")
            self.queries.append(query)
            self._api_calls += 1
            return [self.place.model_copy(update={"place_id": query, "name": query})]

    provider = OptionProvider()
    existing = provider.place.model_copy(update={"place_id": "a", "name": "하랑갤러리"})
    execution = ToolRegistry(provider).invoke(
        "recover_option_places",
        {
            "options": ["하랑갤러리", "페이지룸8"],
            "candidates": [existing.model_dump()],
            "anchor": provider.place.model_dump(),
        },
    )

    assert execution.status == "ok"
    assert provider.queries == ["페이지룸8"]
    assert [place["name"] for place in execution.output] == ["하랑갤러리", "페이지룸8"]


def test_distance_matrix_and_multi_segment_aggregate_keep_option_mapping() -> None:
    registry = ToolRegistry(FakeProvider())
    execution = registry.invoke(
        "distance_matrix",
        {
            "pairs": [
                {"origin": "S", "destination": "A"},
                {"origin": "A", "destination": "B"},
                {"origin": "S", "destination": "C"},
                {"origin": "C", "destination": "D"},
            ],
            "priority": "DISTANCE",
        },
    )
    assert execution.status == "ok"
    aggregate = SpatialOperatorRegistry().invoke(
        "aggregate_route_groups",
        {"routes": execution.output["routes"], "groups": [[0, 1], [2, 3]]},
    )
    assert [item["option_index"] for item in aggregate["option_totals"]] == [0, 1]
    assert aggregate["option_totals"][0]["distance_m"] == 200


def test_distance_matrix_isolates_unresolved_pairs_and_keeps_valid_routes() -> None:
    registry = ToolRegistry(FakeProvider())
    execution = registry.invoke(
        "distance_matrix",
        {
            "pairs": [
                {"origin": "S", "destination": "A", "label": "0"},
                {"origin": "S", "destination": None, "label": "1"},
            ]
        },
    )
    assert execution.status == "ok"
    assert [route["status"] for route in execution.output["routes"]] == ["ok", "error"]
    best = SpatialOperatorRegistry().invoke(
        "select_min", {"items": execution.output["routes"], "key": "distance_m"}
    )
    assert best["label"] == "0"


def test_geoflow_validation_topologically_orders_and_checks_all_constraints() -> None:
    graph = {
        "graph": [
            {
                "id": "nearest",
                "operator": "nearest",
                "arguments": {"anchor": "$places.0.place", "candidates": []},
                "depends_on": ["places"],
                "output_type": "object",
                "role": "measure",
            },
            {
                "id": "places",
                "operator": "batch_geocode",
                "arguments": {"place_names": ["경복궁"]},
                "depends_on": [],
                "output_type": "object",
                "role": "extent",
            },
        ]
    }
    ordered, constraints = normalize_and_validate_graph(graph, max_steps=8)
    assert [step["id"] for step in ordered] == ["places", "nearest"]
    assert all(constraints.values())


def test_geoflow_validation_rejects_cycles_instead_of_truncating() -> None:
    graph = {
        "graph": [
            {
                "id": "a",
                "operator": "place_search",
                "arguments": {"query": "$b"},
                "role": "extent",
            },
            {
                "id": "b",
                "operator": "place_search",
                "arguments": {"query": "$a"},
                "role": "extent",
            },
        ]
    }
    with pytest.raises(ValueError, match="acyclicity"):
        normalize_and_validate_graph(graph, max_steps=8)


def test_geoflow_g5_rejects_node_not_reachable_from_context() -> None:
    graph = {
        "graph": [
            {
                "id": "context",
                "operator": "place_search",
                "arguments": {"query": "서울역"},
                "role": "extent",
            },
            {
                "id": "orphan",
                "operator": "place_search",
                "arguments": {"query": "경복궁"},
                "role": "support",
            },
            {
                "id": "measure",
                "operator": "merge_places",
                "arguments": {"items": ["$context", "$orphan"]},
                "depends_on": ["context", "orphan"],
                "role": "measure",
            },
        ]
    }

    with pytest.raises(ValueError, match="not reachable from EXTENT"):
        normalize_and_validate_graph(graph, max_steps=8)


def test_factorization_binds_every_analysis_concept_and_validates() -> None:
    analysis = normalize_analysis(
        {
            "intent": "nearby",
            "concepts": [
                {
                    "id": "anchor",
                    "text": "서울역",
                    "concept_type": "location",
                    "role": "extent",
                },
                {
                    "id": "candidate_set",
                    "text": "카페",
                    "concept_type": "object",
                    "role": "sub_condition",
                    "depends_on": ["anchor"],
                },
                {
                    "id": "answer",
                    "text": "nearest cafe",
                    "concept_type": "object",
                    "role": "measure",
                    "depends_on": ["candidate_set"],
                },
            ],
        },
        "서울역에서 가장 가까운 카페는?",
        "nearby",
    )
    factorized = factorize_geoflow(
        analysis,
        {
            "graph": [
                {
                    "id": "anchor_lookup",
                    "operator": "batch_geocode",
                    "arguments": {"place_names": ["서울역"]},
                },
                {
                    "id": "candidates",
                    "operator": "nearby_places",
                    "arguments": {"center": "$anchor_lookup.0.place", "query": "카페"},
                },
                {
                    "id": "answer_measure",
                    "operator": "identity_measure",
                    "arguments": {"value": "$candidates"},
                },
            ]
        },
    )

    ordered, constraints = normalize_and_validate_graph(factorized.as_dict(), max_steps=8)
    bound = {concept_id for step in ordered for concept_id in step["concept_ids"]}
    expected = {node["id"] for node in factorized.as_dict()["concept_graph"]["nodes"]}
    assert expected <= bound
    assert constraints["contextual_connectivity"]
    assert constraints["concept_factorization"]


def test_concept_graph_g5_rejects_disconnected_bound_concept() -> None:
    payload = {
        "concept_graph": {
            "nodes": [
                {"id": "ctx", "role": "extent"},
                {"id": "orphan", "role": "support"},
                {"id": "answer", "role": "measure"},
            ],
            "edges": [{"source": "ctx", "target": "answer"}],
        },
        "graph": [
            {
                "id": "context",
                "operator": "place_search",
                "arguments": {"query": "서울역"},
                "role": "extent",
                "concept_ids": ["ctx", "orphan"],
                "output_bindings": [
                    {"concept_id": "ctx", "path": "$"},
                    {"concept_id": "orphan", "path": "$"},
                ],
            },
            {
                "id": "measure",
                "operator": "identity_measure",
                "arguments": {"value": "$context"},
                "role": "measure",
                "concept_ids": ["answer"],
                "output_bindings": [{"concept_id": "answer", "path": "$"}],
            },
        ],
    }

    with pytest.raises(ValueError, match="Concept is not reachable"):
        normalize_and_validate_graph(payload, max_steps=8)


def test_multiple_concepts_can_bind_to_distinct_operator_output_paths() -> None:
    output = [{"place": {"name": "서울역"}}, {"place": {"name": "숭례문"}}]
    assert _resolve_output_binding(output, "$.0.place")["name"] == "서울역"
    assert _resolve_output_binding(output, "$.1.place")["name"] == "숭례문"


def test_operator_contracts_cover_all_scientific_core_concepts() -> None:
    assert CORE_CONCEPTS <= {contract.output_type for contract in OPERATOR_CONTRACTS.values()}


def test_template_catalog_covers_appendix_e_macro_families() -> None:
    assert {template["name"] for template in TEMPLATES.values()} == {
        "Filter-Aggregate-Measure",
        "Object-Field-Measure",
        "Route-Optimize",
        "Geocode-Batch-Compare",
        "Location-Bearing-Classify",
        "Route-Step-Extract",
        "Multi-Route-Compare",
        "Place-Attribute-Query",
        "Multi-Segment-Aggregate",
        "Time-Window-Reverse",
    }


@pytest.mark.parametrize("template_key", sorted(TEMPLATES))
def test_each_appendix_e_template_is_executable(template_key: str) -> None:
    provider_tools = ToolRegistry(FakeProvider())
    operators = SpatialOperatorRegistry()
    provider_names = {
        schema["function"]["name"] for schema in provider_tools.schemas()
    }
    ordered, constraints = normalize_and_validate_graph(
        TEMPLATES[template_key]["example"], max_steps=20
    )
    results: dict[str, Any] = {}
    for step in ordered:
        arguments = _resolve_references(step["arguments"], results)
        if step["operator"] in provider_names:
            execution = provider_tools.invoke(step["operator"], arguments)
            assert execution.status == "ok", execution.error
            results[step["id"]] = execution.output
        else:
            results[step["id"]] = operators.invoke(step["operator"], arguments)
    assert constraints["connectivity"] is True
    assert results[ordered[-1]["id"]] is not None


def test_concept_edges_are_not_inferred_from_role_levels() -> None:
    analysis = normalize_analysis(
        {
            "intent": "poi",
            "concepts": [
                {"id": "extent", "concept_type": "object", "role": "extent"},
                {"id": "support", "concept_type": "field", "role": "support"},
                {"id": "measure", "concept_type": "amount", "role": "measure"},
            ],
        },
        "질문",
        "poi",
    )
    assert build_concept_graph(analysis).edges == ()


def test_appendix_c_operator_semantics() -> None:
    operators = SpatialOperatorRegistry()
    places = [
        {"name": "A", "latitude": 37.0, "longitude": 127.0},
        {"name": "B", "latitude": 37.01, "longitude": 127.0},
        {"name": "C", "latitude": 37.1, "longitude": 127.0},
    ]
    extremes = operators.invoke("pairwise_extremes", {"locations": places})
    assert extremes["farthest_pair"]["indexes"] == [0, 2]
    routes = [
        {"distance_m": 10, "duration_s": 10, "steps": [{"instruction": "유료도로 진입"}]},
        {"distance_m": 20, "duration_s": 5, "steps": [{"instruction": "일반도로"}]},
    ]
    filtered = operators.invoke(
        "filter_routes", {"routes": routes, "keyword": "유료도로", "include": False}
    )
    assert filtered["route_indexes"] == [1]
    nearest = operators.invoke(
        "nearest",
        {
            "anchor": places[0],
            "candidates": places[1:],
            "metric": "travel_time",
            "routes": [
                {"pair_index": 0, "distance_m": 10, "duration_s": 20, "status": "ok"},
                {"pair_index": 1, "distance_m": 20, "duration_s": 5, "status": "ok"},
            ],
        },
    )
    assert nearest["nearest"]["candidate_index"] == 1
    open_status = operators.invoke(
        "open_at_time",
        {
            "schedule": {"sunday": {"open": "23:00", "close": "02:00"}},
            "local_time": "2026-08-17T01:00:00",
            "timezone": "Asia/Seoul",
        },
    )
    assert open_status["is_open"] is True
    fallback = operators.invoke(
        "tsp_tw",
        {
            "nodes": places,
            "distance_matrix": [[0, 2, 5], [2, 0, 2], [5, 2, 0]],
            "time_windows": [[0, 0], [0, 3], [0, 3]],
            "time_budget": 3,
        },
    )
    assert fallback["fallback_used"] is True
    assert fallback["order"] == [0, 1]


def test_calculate_finish_time_uses_each_route_and_stay() -> None:
    execution = ToolRegistry(FakeProvider()).invoke(
        "calculate_finish_time",
        {
            "start_time": "2026-08-17T09:00:00",
            "locations": ["A", "B", "C"],
            "stay_durations_s": [10, 20, 30],
            "timezone": "Asia/Seoul",
        },
    )
    assert execution.status == "ok"
    assert execution.output["travel_duration_s"] == 40
    assert execution.output["stay_duration_s"] == 60
    assert execution.output["finish_time"].endswith("09:01:40+09:00")


def test_contextual_roles_are_not_part_of_g2_precedence() -> None:
    payload = {
        "graph": [
            {
                "id": "extent",
                "operator": "place_search",
                "arguments": {"query": "서울역"},
                "role": "extent",
            },
            {
                "id": "support",
                "operator": "identity_measure",
                "arguments": {"value": "$extent"},
                "role": "support",
            },
            {
                "id": "temporal_context",
                "operator": "identity_measure",
                "arguments": {"value": "$support"},
                "role": "temporal_extent",
            },
            {
                "id": "measure",
                "operator": "identity_measure",
                "arguments": {"value": "$temporal_context"},
                "role": "measure",
            },
        ]
    }
    ordered, constraints = normalize_and_validate_graph(payload, max_steps=8)
    assert [step["id"] for step in ordered] == [
        "extent",
        "support",
        "temporal_context",
        "measure",
    ]
    assert constraints["role_ordering"] is True


def test_llm_generation_is_not_overwritten_by_rule_matcher() -> None:
    llm = QueuedLLM(
        [
            LLMResponse('{"intent":"type"}'),
            LLMResponse(
                '{"graph":['
                '{"id":"place","operator":"place_search",'
                '"arguments":{"query":"경복궁","limit":1},"role":"extent"},'
                '{"id":"match","operator":"match_type_options",'
                '"arguments":{"place":"$place.0","options":["관광명소","은행"]},'
                '"role":"measure"}]}'
            ),
            LLMResponse('{"predicted_option":1,"confidence":0.2,"reason":"generated"}'),
        ]
    )
    result = SpatialAgent(llm, ToolRegistry(FakeProvider()), max_steps=4).answer(
        "경복궁 유형은?", ["관광명소", "은행"]
    )
    assert result.predicted_answer == 1
    assert result.trace[-2]["reason"] == "generated"


def test_geoflow_validation_rejects_incompatible_dependency_types() -> None:
    graph = {
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
                "id": "route_totals",
                "operator": "sum_route_metrics",
                "arguments": {"routes": "$distance"},
                "role": "measure",
            },
        ]
    }
    with pytest.raises(ValueError, match="Type compatibility"):
        normalize_and_validate_graph(graph, max_steps=8)


def test_geoflow_accepts_braced_references_and_execution_resolves_them() -> None:
    graph = {
        "graph": [
            {
                "id": "places",
                "operator": "batch_geocode",
                "arguments": {"place_names": ["경복궁"]},
                "role": "extent",
            },
            {
                "id": "nearby",
                "operator": "nearby_places",
                "arguments": {"center": "${places[0].place}", "query": "카페"},
                "role": "measure",
            },
        ]
    }
    ordered, _ = normalize_and_validate_graph(graph, max_steps=8)
    assert ordered[1]["depends_on"] == ["places"]
    place = FakeProvider().place.model_dump()
    assert _resolve_references("${places[0].place}", {"places": [{"place": place}]}) == place


def test_geoflow_repairs_numeric_dependencies_and_placeholder_node_references() -> None:
    graph = {
        "graph": [
            {
                "id": "ref",
                "operator": "batch_geocode",
                "arguments": {"place_names": ["회기"]},
                "role": "extent",
            },
            {
                "id": "measure",
                "operator": "nearby_places",
                "arguments": {
                    "center": "$node.0.place",
                    "category_code": "SC4",
                    "radius_m": 500,
                },
                "depends_on": [0],
                "role": "measure",
            },
        ]
    }
    ordered, _ = normalize_and_validate_graph(graph, max_steps=8)
    assert ordered[1]["depends_on"] == ["ref"]
    assert ordered[1]["arguments"]["center"] == "$ref.0.place"


def test_graph_grounding_restores_verbatim_anchor_and_options() -> None:
    steps = [
        {
            "id": "places",
            "operator": "batch_geocode",
            "arguments": {
                "place_names": ["양꼬치", "A", "B"],
                "anchor": "양꼬치",
            },
            "depends_on": [],
            "output_type": "object",
            "role": "support",
        }
    ]
    grounded = _ground_graph_literals(
        steps,
        "뜻밖에 양꼬치에서 남쪽에 있는 가장 가까운 카페 중 어디인가요?",
        ["포근한 다락방 카페", "코티지블루"],
        "direction",
    )
    assert grounded[0]["arguments"]["anchor"] == "뜻밖에 양꼬치"
    assert grounded[0]["arguments"]["place_names"] == [
        "뜻밖에 양꼬치",
        "포근한 다락방 카페",
        "코티지블루",
    ]


def test_radius_grounding_builds_candidate_set_graph() -> None:
    grounded = _bind_prevalidated_template(
        "radius",
        "기준점 반경 500m 안에 있는 패스트푸드점 목록은 무엇인가요?",
        ["롯데리아 | 윤토스트", "롯데리아 | 노브랜드버거"],
        "기준점",
    )

    assert grounded is not None
    assert grounded[0]["operator"] == "batch_geocode"
    assert grounded[0]["arguments"]["place_names"] == ["기준점"]
    assert [step["operator"] for step in grounded] == [
        "batch_geocode",
        "nearby_places",
        "nearby_places",
        "merge_places",
        "match_options",
    ]
    assert grounded[1]["arguments"]["query"] == "패스트푸드"
    assert grounded[2]["arguments"]["category_code"] == "FD6"
    assert grounded[-1]["arguments"]["mode"] == "radius_set"


def test_radius_literals_are_factors_not_synthetic_output_references() -> None:
    steps = [
        {
            "id": "anchor",
            "operator": "batch_geocode",
            "arguments": {"place_names": ["기준"]},
        },
        {
            "id": "constraint",
            "operator": "identity_measure",
            "arguments": {"value": 1000},
            "depends_on": ["anchor"],
        },
        {
            "id": "nearby",
            "operator": "nearby_places",
            "arguments": {
                "center": "$anchor.0.place",
                "query": "음식점",
                "radius_m": "$constraint.0",
            },
            "depends_on": ["anchor", "constraint"],
        },
        {
            "id": "match",
            "operator": "match_options",
            "arguments": {"options": ["wrong"], "places": "$nearby"},
            "depends_on": ["nearby"],
        },
    ]
    grounded = _ground_graph_literals(
        steps,
        "기준점 반경 500m 안에 있는 카페 목록은 무엇인가요?",
        ["A | B", "A | B | C"],
        "radius",
    )

    assert grounded[0]["arguments"]["place_names"] == ["기준점"]
    assert grounded[2]["arguments"]["category_code"] == "CE7"
    assert grounded[2]["arguments"]["radius_m"] == 500
    assert grounded[3]["arguments"]["mode"] == "radius_set"
    assert grounded[3]["arguments"]["options"] == ["A | B", "A | B | C"]


def test_factorization_keeps_radius_and_category_as_operator_factors() -> None:
    analysis = normalize_analysis(
        {
            "intent": "radius",
            "concepts": [
                {
                    "id": "anchor",
                    "text": "기준점",
                    "concept_type": "location",
                    "role": "extent",
                    "depends_on": [],
                },
                {
                    "id": "radius",
                    "text": "500m",
                    "concept_type": "amount",
                    "role": "condition",
                    "depends_on": ["anchor"],
                },
                {
                    "id": "target",
                    "text": "카페",
                    "concept_type": "object",
                    "role": "condition",
                    "depends_on": ["anchor"],
                },
                {
                    "id": "answer",
                    "text": "목록",
                    "concept_type": "object",
                    "role": "measure",
                    "depends_on": ["target"],
                },
            ],
        },
        "기준점 반경 500m 안에 있는 카페 목록은 무엇인가요?",
        "radius",
    )
    factorized = factorize_geoflow(
        analysis,
        {
            "graph": [
                {
                    "id": "anchor",
                    "operator": "batch_geocode",
                    "arguments": {"place_names": ["기준점"]},
                    "concept_ids": ["anchor"],
                },
                {
                    "id": "nearby",
                    "operator": "nearby_places",
                    "arguments": {
                        "center": "$anchor.0.place",
                        "category_code": "CE7",
                        "radius_m": 500,
                    },
                },
                {
                    "id": "match",
                    "operator": "match_options",
                    "arguments": {"options": ["A"], "places": "$nearby"},
                },
            ]
        },
    )
    ordered, constraints = normalize_and_validate_graph(factorized.as_dict(), max_steps=8)
    nearby = next(step for step in ordered if step["id"] == "nearby")

    assert {"radius", "target"} <= set(nearby["input_concepts"])
    assert not ({"radius", "target"} & set(nearby["concept_ids"]))
    assert constraints["concept_factorization"]


def test_match_options_recovers_minor_historical_name_changes() -> None:
    anchor = FakeProvider().place.model_dump()
    places = [
        {
            **anchor,
            "place_id": "toast",
            "name": "토스티드클럽",
            "latitude": 37.58,
        },
        {
            **anchor,
            "place_id": "burger",
            "name": "회기버거 경희궁자이점",
            "latitude": 37.60,
        },
    ]

    result = SpatialOperatorRegistry().invoke(
        "match_options",
        {
            "anchor": anchor,
            "places": places,
            "options": ["지미존스", "Olive Chicken Cafe", "토스티트 클럽", "회기버거"],
            "mode": "nearest",
        },
    )

    assert result["best_option"] == 2
    assert result["option_matches"][2]["matched"]["name"] == "토스티드클럽"


def test_match_options_rejects_unrelated_names_with_shared_category_suffix() -> None:
    anchor = FakeProvider().place.model_dump()
    places = [
        {
            **anchor,
            "place_id": "art-center",
            "name": "정수아트센터",
            "latitude": 37.58,
        }
    ]

    result = SpatialOperatorRegistry().invoke(
        "match_options",
        {"options": ["김희수아트센터"], "places": places, "anchor": anchor},
    )

    assert result["best_option"] is None
    assert result["option_matches"][0]["matched"] is None


def test_match_distance_options_uses_computed_measure() -> None:
    result = SpatialOperatorRegistry().invoke(
        "match_distance_options",
        {"distance": {"distance_m": 1058}, "options": ["1036 m", "1061 m", "1.2 km"]},
    )

    assert result["best_option"] == 1


def test_event_network_proportion_temporal_and_tsp_operators_execute() -> None:
    operators = SpatialOperatorRegistry()
    events = operators.invoke(
        "events_from_objects",
        {"objects": [{"kind": "crime"}, {"kind": "other"}], "event_type": "incident"},
    )
    filtered = operators.invoke(
        "filter_events",
        {"events": events, "field": "object.kind", "operator": "eq", "value": "crime"},
    )
    proportion = operators.invoke(
        "calculate_proportion", {"numerator": filtered, "denominator": events}
    )
    network = operators.invoke(
        "build_route_network",
        {"nodes": [{"id": "a"}, {"id": "b"}], "edges": [{"source": "a", "target": "b"}]},
    )
    converted = operators.invoke(
        "timezone_convert",
        {
            "local_time": "2026-01-01T09:00:00",
            "from_timezone": "Asia/Seoul",
            "to_timezone": "UTC",
        },
    )
    route = operators.invoke(
        "tsp_tw",
        {
            "nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            "distance_matrix": [[0, 10, 20], [10, 0, 5], [20, 5, 0]],
            "start_index": 0,
        },
    )

    assert len(filtered) == 1
    assert proportion["proportion"] == 0.5
    assert network["edge_count"] == 1
    assert converted["converted_time"].endswith("+00:00")
    assert route["order"] == [0, 1, 2]


def test_spatial_operators_skip_unresolved_candidates() -> None:
    anchor = FakeProvider().place.model_dump()
    nearby = {**anchor, "place_id": "2", "latitude": 37.58}
    nearest = SpatialOperatorRegistry().invoke(
        "nearest", {"anchor": anchor, "candidates": [None, nearby]}
    )
    assert nearest["nearest"]["candidate_index"] == 1


def test_directions_accepts_a_normalized_place_from_a_plan_reference() -> None:
    registry = ToolRegistry(FakeProvider())
    place = FakeProvider().place.model_dump()
    execution = registry.invoke(
        "directions",
        {"origin": place, "destination": place, "mode": "driving"},
    )
    assert execution.status == "ok"
    assert execution.output["distance_m"] == 100


def test_spatial_agent_handles_google_style_references_without_aborting() -> None:
    llm = QueuedLLM(
        [
            LLMResponse('{"intent":"nearby"}'),
            LLMResponse(
                '{"steps":['
                '{"id":"s1","operator":"place_search",'
                '"arguments":{"query":"경복궁","limit":1}},'
                '{"id":"s2","operator":"place_search",'
                '"arguments":{"query":"광화문","limit":1}},'
                '{"id":"s3","operator":"haversine_distance",'
                '"arguments":{"lat1":"$s1.0.geometry.lat",'
                '"lng1":"$s1.0.geometry.lng","lat2":"$s2.0.lat",'
                '"lng2":"$s2.0.lng"}}]}'
            ),
            LLMResponse('{"predicted_option":1,"confidence":0.9,"reason":"distance"}'),
        ]
    )
    agent = SpatialAgent(llm, ToolRegistry(FakeProvider()), max_steps=3)
    result = agent.answer("질문", ["A", "B"])
    execute = next(stage for stage in result.trace if stage["stage"] == "execute")
    assert [step["status"] for step in execute["steps"]] == ["ok", "ok", "ok"]
    assert result.predicted_answer == 1
    assert result.failure_type is None


def test_bad_plan_reference_is_isolated_and_evaluation_still_runs() -> None:
    llm = QueuedLLM(
        [
            LLMResponse('{"intent":"poi"}'),
            LLMResponse(
                '{"steps":['
                '{"id":"s1","operator":"place_search",'
                '"arguments":{"query":"경복궁","limit":1}},'
                '{"id":"s2","operator":"haversine_distance",'
                '"arguments":{"place_a":"$s1.0.missing",'
                '"place_b":"$s1.0"}}]}'
            ),
            LLMResponse('{"predicted_option":1,"confidence":0.3,"reason":"partial evidence"}'),
        ]
    )
    result = SpatialAgent(llm, ToolRegistry(FakeProvider()), max_steps=2).answer("질문", ["A", "B"])
    execute = next(stage for stage in result.trace if stage["stage"] == "execute")
    assert execute["steps"][1]["status"] == "error"
    assert "Missing field" in execute["steps"][1]["error"]
    assert result.predicted_answer == 1
    assert result.failure_type is None


def test_spatial_router_heuristics_cover_extended_intents() -> None:
    questions = {
        "type": "농협은행 불암지점의 장소 유형은 무엇인가요?",
        "direction": "GS25 언주제일에서 북쪽에 있는 가장 가까운 카페는?",
        "distance": "서울역과 숭례문 사이의 직선거리는 약 얼마인가요?",
        "radius": "서울역 반경 500m 안에 있는 편의점 목록은 무엇인가요?",
    }
    assert {intent: _heuristic_intent(question) for intent, question in questions.items()} == {
        intent: intent for intent in questions
    }


def test_spatial_router_prioritizes_trip_and_routing_over_nearest_wording() -> None:
    assert (
        _heuristic_intent(
            "에슬로우서울역점에서 출발해 두 곳을 차례로 방문할 때 "
            "총 자동차 이동거리가 가장 짧은 일정은?"
        )
        == "trip"
    )
    assert _heuristic_intent("서울역에서 자동차 최단거리 경로로 가장 가까운 목적지는?") == "routing"
