from __future__ import annotations

from typing import Any

from src.agent import ReactAgent, SpatialAgent
from src.agent.spatial import _heuristic_intent
from src.llm import LLMResponse, LLMToolCall
from src.models import Place, Route
from src.tools import MapProvider, ToolRegistry


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


def test_spatial_agent_preserves_all_five_stages() -> None:
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
        "route",
        "plan",
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
