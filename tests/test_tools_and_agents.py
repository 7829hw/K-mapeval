from __future__ import annotations

from typing import Any

import pytest

from src.agent import ReactAgent, SpatialAgent
from src.agent.geoflow import normalize_and_validate_graph
from src.agent.spatial import _heuristic_intent
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
        {"center": "경복궁", "query": "식당", "radius_m": 50_000, "limit": 20},
    )
    assert search.status == "ok"
    assert search.arguments["limit"] == 15
    assert nearby.status == "ok"
    assert nearby.arguments["radius_m"] == 20_000
    assert nearby.arguments["limit"] == 15


def test_batch_geocode_preserves_input_order_in_one_geoflow_tool_call() -> None:
    registry = ToolRegistry(FakeProvider())
    execution = registry.invoke("batch_geocode", {"place_names": ["경복궁", "광화문"], "limit": 1})
    assert execution.status == "ok"
    assert [item["query"] for item in execution.output] == ["경복궁", "광화문"]
    assert all(item["place"]["name"] == "경복궁" for item in execution.output)
    assert registry.tool_call_count == 1
    assert registry.provider.api_call_count == 2


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
                "role": "support",
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
                "role": "support",
            },
            {
                "id": "b",
                "operator": "place_search",
                "arguments": {"query": "$a"},
                "role": "support",
            },
        ]
    }
    with pytest.raises(ValueError, match="acyclicity"):
        normalize_and_validate_graph(graph, max_steps=8)


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
                "role": "support",
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
