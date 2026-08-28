from __future__ import annotations

import json
from typing import Any

import pytest

from src.agent import ReactAgent, SpatialAgent
from src.agent.geoflow import (
    CORE_CONCEPTS,
    OPERATOR_CONTRACTS,
    SKELETONS,
    TEMPLATES,
    build_concept_graph,
    factorize_geoflow,
    normalize_analysis,
    normalize_and_validate_graph,
)
from src.agent.spatial import (
    RETRIEVAL_LIMIT,
    _bind_named_places,
    _compact_evaluation_evidence,
    _ground_graph_literals,
    _resolve_output_binding,
    _resolve_references,
    extract_facts,
)
from src.llm import LLMResponse, LLMToolCall
from src.models import Place, Route
from src.tools import MapProvider, SpatialOperatorRegistry, ToolRegistry
from src.tools.kakao import retrieval_specs as kakao_retrieval_specs


class FakeProvider(MapProvider):
    def __init__(self) -> None:
        self._api_calls = 0
        self._issued: dict[str, Place] = {}
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

    def dereference(self, value: str | Place) -> Place | None:
        """A fake provider still has to take back what it handed out."""

        if isinstance(value, Place):
            return value
        return self._issued.get(value) or super().dereference(value)

    def _named(self, query: str) -> Place:
        """A geocoder answers with a place that carries the name it was asked for.

        Distinct names stand on distinct spots. They used to share one coordinate, which was
        harmless while the tools passed names through to the provider — once the aggregations
        resolve a name before routing, two names on the same spot are one place, and every leg of
        a matrix became a zero-cost self-route.
        """

        offset = (sum(ord(character) for character in query) % 97) * 0.0011
        place = self.place.model_copy(
            update={
                "place_id": query,
                "name": query,
                "latitude": round(self.place.latitude + offset, 6),
                "longitude": round(self.place.longitude + offset / 2, 6),
            }
        )
        self._issued[place.place_id] = place
        return place

    def search_place(self, query: str, *, limit: int = 5) -> list[Place]:
        self._api_calls += 1
        return [self._named(query)]

    def geocode(self, address: str, *, limit: int = 5) -> list[Place]:
        self._api_calls += 1
        return [self._named(address)]

    def nearby_search(self, center: str | Place, **kwargs: Any) -> list[Place]:
        self._api_calls += 1
        query = kwargs.get("query")
        first = self._named(query) if query else self.place
        # A neighbourhood of one cannot exercise a ranking, and an ordinal question indexes into
        # one -- `select_by_index(index=1)` has nothing to select from. Two more distinct spots,
        # so `nearest` has something to order and the ordinal template is actually executed.
        stem = query or "장소"
        return [first, self._named(f"{stem} 2"), self._named(f"{stem} 3")]

    def place_details(self, place_id: str) -> Place:
        return self._issued.get(place_id, self.place)

    def directions(self, origin: str | Place, destination: str | Place, **_: Any) -> Route:
        self._api_calls += 1
        # Echo the endpoints asked for. A double that answers every pair with the same two names
        # cannot exercise anything that keys legs by endpoint, such as the duration matrix
        # `tsp_tw` consumes — it collapses a whole matrix into one cell.
        start = origin.name if isinstance(origin, Place) else str(origin)
        end = destination.name if isinstance(destination, Place) else str(destination)
        span = 100 + 10 * (len(start) + len(end))
        return Route(origin=start, destination=end, distance_m=span, duration_s=span // 5)


class QueuedLLM:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = responses
        self.messages: list[list[dict[str, Any]]] = []

    def chat(self, messages: list[dict[str, Any]], *, tools=None) -> LLMResponse:
        self.messages.append(messages)
        return self.responses.pop(0)


PAPER_PLAN = json.dumps(
    {
        "concept_nodes": [
            {
                "id": "anchor",
                "text": "경복궁",
                "core_concept": "location",
                "functional_role": "extent",
                "attributes": {},
            },
            {
                "id": "answer",
                "text": "grounded place",
                "core_concept": "object",
                "functional_role": "measure",
                "attributes": {},
            },
        ],
        "factor_nodes": [],
        "transformation_edges": [
            {
                "id": "resolve",
                "transformation": "RESOLVE_PLACES",
                "input_concepts": [],
                "output_concepts": ["anchor"],
                "factor_nodes": [],
            },
            {
                "id": "measure",
                "transformation": "MEASURE",
                "input_concepts": ["anchor"],
                "output_concepts": ["answer"],
                "factor_nodes": [],
            },
        ],
    },
    ensure_ascii=False,
)


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


def test_react_preserves_a_tool_error_even_when_it_recovers_an_answer() -> None:
    llm = QueuedLLM(
        [
            LLMResponse("", (LLMToolCall("bad-call", "place_details", {}),)),
            LLMResponse("^^1^^"),
        ]
    )

    result = ReactAgent(llm, ToolRegistry(FakeProvider()), max_steps=3).answer("질문", ["A", "B"])

    assert result.predicted_answer == 1
    assert result.failure_type is None
    assert len(result.execution_errors) == 1
    assert result.execution_errors[0]["step_id"] == "bad-call"
    assert result.execution_errors[0]["operator"] == "place_details"
    assert "place_id" in result.execution_errors[0]["error"]


def test_a_question_records_what_it_cost_at_the_endpoint() -> None:
    """Tokens are summed over every completion the question asked for, thinking counted apart.

    Two agents that score the same are not the same result if one spent three times the tokens,
    and on a reasoning model most of a completion is text the parser never sees. `reasoning_tokens`
    is whatever the server reported and nothing else: the deployment this runs against returns
    `completion_tokens_details: null` beside a populated `message.reasoning`, and an estimate
    printed next to three measured counts would read as a fourth.
    """

    from src.llm import TokenUsage

    llm = QueuedLLM(
        [
            LLMResponse(
                "",
                (LLMToolCall("call-1", "place_search", {"query": "경복궁"}),),
                TokenUsage(
                    prompt_tokens=120,
                    completion_tokens=64,
                    total_tokens=184,
                    reasoning_tokens=40,
                    reasoning_chars=310,
                ),
            ),
            LLMResponse(
                "^^2^^",
                (),
                TokenUsage(
                    prompt_tokens=200,
                    completion_tokens=12,
                    total_tokens=212,
                    reasoning_tokens=5,
                    reasoning_chars=44,
                ),
            ),
        ]
    )
    agent = ReactAgent(llm, ToolRegistry(FakeProvider()), max_steps=5)
    result = agent.answer("질문", ["A", "B", "C", "D"])

    assert result.llm_calls == 2
    assert result.prompt_tokens == 320
    assert result.completion_tokens == 76
    assert result.total_tokens == 396
    assert result.reasoning_tokens == 45
    assert result.reasoning_chars == 354


def test_an_unsplit_completion_reports_no_reasoning_token_count() -> None:
    """A server that does not break the completion down leaves the field empty, not zero.

    Zero would read as "it did no thinking", which is the opposite of what a null
    `completion_tokens_details` means beside 441 completion tokens and a paragraph of
    `message.reasoning`.
    """

    from src.llm import TokenUsage

    llm = QueuedLLM(
        [
            LLMResponse(
                "^^1^^",
                (),
                TokenUsage(
                    prompt_tokens=90, completion_tokens=441, total_tokens=531, reasoning_chars=900
                ),
            )
        ]
    )
    result = ReactAgent(llm, ToolRegistry(FakeProvider()), max_steps=3).answer(
        "질문", ["A", "B", "C", "D"]
    )
    assert result.reasoning_tokens is None
    assert result.reasoning_chars == 900
    assert result.completion_tokens == 441


def test_react_takes_one_action_per_iteration_like_the_structured_chat_parser() -> None:
    """Upstream's agent cannot emit two actions in one step, so ours must not either.

    `AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION` parses a single JSON action blob out of
    each response. A native tool channel returns as many calls as the model likes, and executing
    all of them collapses several of upstream's iterations into one: measured on the v5 run, one
    question executed 24 tool calls across 6 LLM rounds against a nominal budget of 30.
    """

    llm = QueuedLLM(
        [
            LLMResponse(
                "",
                (
                    LLMToolCall("call-1", "place_search", {"query": "경복궁"}),
                    LLMToolCall("call-2", "place_search", {"query": "남산"}),
                    LLMToolCall("call-3", "place_search", {"query": "서울역"}),
                ),
            ),
            LLMResponse("^^2^^"),
        ]
    )
    agent = ReactAgent(llm, ToolRegistry(FakeProvider()), max_steps=5, single_action=True)
    result = agent.answer("질문", ["A", "B", "C", "D"])
    assert result.tool_calls == 1
    assert result.predicted_answer == 2
    # The dropped calls leave no orphan tool_call_id behind, or the next request is malformed.
    assistant = [
        message
        for message in llm.messages[-1]
        if message.get("role") == "assistant" and message.get("tool_calls")
    ]
    assert [len(message["tool_calls"]) for message in assistant] == [1]
    replies = [message for message in llm.messages[-1] if message.get("role") == "tool"]
    assert [message["tool_call_id"] for message in replies] == ["call-1"]


def test_react_running_out_of_iterations_answers_nothing() -> None:
    """`early_stopping_method="force"` is langchain's default, and it does not ask again.

    It returns "Agent stopped due to iteration limit or time limit." as the output, and
    `Evaluator2.py` finds no `^^N^^` in that string. An extra call to force an answer is a turn the
    paper's baseline never gets, and it converts an exhausted budget into a scored guess.
    """

    llm = QueuedLLM(
        [
            LLMResponse("", (LLMToolCall("call-1", "place_search", {"query": "경복궁"}),)),
            LLMResponse("", (LLMToolCall("call-2", "place_search", {"query": "남산"}),)),
        ]
    )
    agent = ReactAgent(
        llm,
        ToolRegistry(FakeProvider()),
        max_steps=2,
        single_action=True,
        force_final_answer=False,
    )
    result = agent.answer("질문", ["A", "B", "C", "D"])
    assert result.predicted_answer is None
    # Its own failure type: a miss, the way upstream counts it, but not an unreadable answer.
    assert result.failure_type == "iteration_limit"
    assert result.response == ReactAgent.ITERATION_LIMIT_OUTPUT
    # Two iterations, two LLM calls: the stop costs none.
    assert result.reasoning_steps == 2
    assert not llm.responses


def test_running_out_of_steps_is_not_reported_as_the_map_failing() -> None:
    """One failed lookup among many observations does not make the budget stop a provider failure.

    Measured on v6: `trip_optimal_order_four` needs five route legs per candidate order and five
    place ids before that, so a 15-iteration budget cannot finish one -- and a provider error
    somewhere in those fifteen observations would have relabelled the whole question as the map
    being unable to answer it.
    """

    llm = QueuedLLM(
        [
            LLMResponse("", (LLMToolCall("call-1", "directions", {"origin_id": "p1"}),)),
            LLMResponse("", (LLMToolCall("call-2", "place_search", {"place_name": "경복궁"}),)),
        ]
    )
    agent = ReactAgent(
        llm,
        ToolRegistry(FakeProvider()),
        max_steps=2,
        single_action=True,
        force_final_answer=False,
    )

    result = agent.answer("질문", ["A", "B", "C", "D"])

    assert any(entry.get("status") == "error" for entry in result.trace)
    assert result.failure_type == "iteration_limit"


def test_the_native_loop_stays_available_as_the_ablation_it_is() -> None:
    """The stronger loop is not deleted, it is named. Reports record which one ran."""

    llm = QueuedLLM(
        [
            LLMResponse(
                "",
                (
                    LLMToolCall("call-1", "place_search", {"query": "경복궁"}),
                    LLMToolCall("call-2", "place_search", {"query": "남산"}),
                ),
            ),
            LLMResponse("^^1^^"),
        ]
    )
    agent = ReactAgent(llm, ToolRegistry(FakeProvider()), max_steps=5, single_action=False)
    result = agent.answer("질문", ["A", "B", "C", "D"])
    assert result.tool_calls == 2
    assert result.predicted_answer == 1


def test_spatial_agent_runs_paper_aligned_pipeline() -> None:
    llm = QueuedLLM(
        [
            LLMResponse("{}"),
            LLMResponse(PAPER_PLAN),
            LLMResponse('{"value":"B","text":"B","confidence":0.9,"reason":"evidence"}'),
        ]
    )
    agent = SpatialAgent(llm, ToolRegistry(FakeProvider()), max_steps=3)
    result = agent.answer("질문", ["A", "B"])
    assert result.predicted_answer == 1
    assert result.response == "^^1^^"
    # The paper's pipeline, one trace entry per stage. `transform` is Concept Transformation:
    # the planner's semantic graph mapped onto executable operators, deterministically, between
    # composition and factorization.
    assert [entry["stage"] for entry in result.trace] == [
        "analyze",
        "retrieve_templates",
        "compose",
        "construct_geoflow",
        "transform",
        "factorize",
        "validate",
        "execute",
        "grounded_answer",
        "mcq_adapt",
        "generate",
    ]
    assert result.tool_calls == 1
    assert result.api_calls == 1
    assert result.predicted_intent is None
    assert '"intent"' not in llm.messages[1][1]["content"]
    assert "Intent:" not in llm.messages[-1][1]["content"]


def test_normalized_measure_does_not_fall_back_to_predicted_intent() -> None:
    analysis = normalize_analysis({"intent": "trip"}, "질문")

    assert "intent" not in analysis
    assert analysis["measure"] == "answer choice"
    assert analysis["concepts"][-1]["text"] == "answer choice"


def test_spatial_evaluation_prompt_compacts_repeated_large_operator_state() -> None:
    """The auditable trace stays complete while the final LLM sees bounded evidence.

    A neighbourhood result is copied into ranking arguments/results and concept bindings.  With
    45 places that exceeded the model's context window even though the Measure result was tiny.
    """

    places = [
        {
            "place_id": str(index),
            "name": f"장소 {index}",
            "address": f"서울시 긴 주소 {index}",
            "latitude": 37.5 + index / 10_000,
            "longitude": 127.0 + index / 10_000,
            "category": "의료,건강 > 병원 > 내과",
            "phone": "02-0000-0000",
            "place_url": f"https://example.test/{index}",
        }
        for index in range(45)
    ]
    answer = {
        "best_option": 3,
        "confidence": 0.95,
        "option_matches": [{"option_index": 3, "similarity": 1.0}],
    }
    execution_log = [
        {
            "id": "nearby",
            "operator": "nearby_places",
            "role": "support",
            "status": "ok",
            "arguments": {"query": "내과", "candidates": places},
            "result": places,
        },
        {
            "id": "ranking",
            "operator": "nearest",
            "role": "support",
            "status": "ok",
            "arguments": {"candidates": places},
            "result": {"ranked": places},
        },
        {
            "id": "answer",
            "operator": "match_options",
            "role": "measure",
            "status": "ok",
            "arguments": {"places": places, "options": ["A", "B", "C", "D"]},
            "result": answer,
        },
    ]
    concept_state = {"neighbourhood": places, "ranking": places, "answer": answer}
    original = json.dumps(
        {"steps": execution_log, "final_state": concept_state}, ensure_ascii=False
    )

    compacted = _compact_evaluation_evidence(execution_log, concept_state)
    encoded = json.dumps(compacted, ensure_ascii=False)

    assert len(encoded) < len(original) / 4
    assert len(encoded) < 50_000
    assert '"_omitted_items"' in encoded
    assert '"best_option": 3' in encoded
    assert "장소 44" not in encoded
    assert '"phone": "02-0000-0000"' in encoded
    # Prompt projection must not mutate the full trace that reports and per-query logs retain.
    assert (
        json.dumps({"steps": execution_log, "final_state": concept_state}, ensure_ascii=False)
        == original
    )


def test_a_plan_only_our_own_rules_reject_still_gets_executed() -> None:
    """Last resort before giving a question up: run it the way upstream would.

    Upstream has no output-type check and no role-ordering rule, so a graph they refuse is a graph
    upstream would have executed. On v6 that lost Spatial-Agent five questions outright -- four of
    them a correct `select_max` plan -- and each was recorded as the architecture reasoning badly.
    Structural rules still refuse; these two only get to inform the repair round.
    """

    typed_wrong = (
        '{"graph":[{"id":"s1","operator":"place_search","arguments":{"query":"경복궁"},'
        '"role":"extent"},'
        '{"id":"s2","operator":"sum_route_metrics","arguments":{"routes":"$s1"},'
        '"depends_on":["s1"],"role":"measure"}]}'
    )
    structurally_invalid_repair = (
        '{"graph":[{"id":"broken","operator":"operator_that_does_not_exist",'
        '"arguments":{},"role":"measure"}]}'
    )
    llm = QueuedLLM(
        [
            LLMResponse('{"intent":"poi"}'),
            LLMResponse(typed_wrong),
            LLMResponse(structurally_invalid_repair),
            LLMResponse('{"predicted_option":1,"confidence":0.9,"reason":"evidence"}'),
        ]
    )

    result = SpatialAgent(llm, ToolRegistry(FakeProvider()), max_steps=4).answer("질문", ["A", "B"])

    assert result.failure_type == "graph_validation_failure"
    assert result.predicted_answer is None
    stages = [entry["stage"] for entry in result.trace]
    assert "repair" in stages and "execute" not in stages
    assert all(entry.get("status") != "lenient" for entry in result.trace)


def test_lenient_validation_executes_a_structurally_repaired_graph_first() -> None:
    """A repaired graph rejected only by local typing must not be replaced by the broken draft."""

    invalid_original = (
        '{"graph":['
        '{"id":"places","operator":"batch_geocode",'
        '"arguments":{"place_names":["경복궁"]},"role":"extent"},'
        '{"id":"answer","operator":"identity_measure",'
        '"arguments":{"value":"$places.anchor_place"},'
        '"depends_on":["places"],"role":"measure"}]}'
    )
    repaired_but_typed_wrong = (
        '{"graph":['
        '{"id":"repaired_place","operator":"place_search",'
        '"arguments":{"query":"경복궁"},"role":"extent"},'
        '{"id":"repaired_measure","operator":"sum_route_metrics",'
        '"arguments":{"routes":"$repaired_place"},'
        '"depends_on":["repaired_place"],"role":"measure"}]}'
    )
    llm = QueuedLLM(
        [
            LLMResponse('{"intent":"poi"}'),
            LLMResponse(invalid_original),
            LLMResponse(repaired_but_typed_wrong),
            LLMResponse('{"predicted_option":1,"confidence":0.5,"reason":"evidence"}'),
        ]
    )

    result = SpatialAgent(llm, ToolRegistry(FakeProvider()), max_steps=4).answer("질문", ["A", "B"])

    assert result.failure_type == "graph_validation_failure"
    assert result.predicted_answer is None
    assert not any(entry["stage"] == "execute" for entry in result.trace)


def test_spatial_agent_preserves_recovered_operator_errors() -> None:
    llm = QueuedLLM(
        [
            LLMResponse('{"intent":"poi"}'),
            LLMResponse(
                '{"graph":[{"id":"empty_rank","operator":"select_by_index",'
                '"arguments":{"items":[],"index":0},"role":"measure"}]}'
            ),
            LLMResponse('{"predicted_option":0,"confidence":0.5,"reason":"fallback"}'),
        ]
    )

    result = SpatialAgent(llm, ToolRegistry(FakeProvider()), max_steps=4).answer("질문", ["A", "B"])

    assert result.predicted_answer is None
    assert result.failure_type == "graph_validation_failure"
    assert result.execution_errors == []


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
        {
            "center": search.output[0]["place_id"],
            "query": "식당",
            "radius_m": 50_000,
            "limit": 50,
        },
    )
    assert search.status == "ok"
    assert search.arguments["limit"] == 15
    assert nearby.status == "ok"
    assert nearby.arguments["radius_m"] == 20_000
    assert nearby.arguments["limit"] == 45


def test_registry_infers_exact_kakao_category_for_station_search() -> None:
    registry = ToolRegistry(FakeProvider())
    centre = registry.invoke("place_search", {"query": "경복궁"}).output[0]["place_id"]
    execution = registry.invoke("nearby_places", {"center": centre, "query": "역", "radius_m": 300})
    assert execution.status == "ok"
    assert execution.arguments["category_code"] == "SW8"


def test_batch_geocode_preserves_input_order_in_one_geoflow_tool_call() -> None:
    registry = ToolRegistry(FakeProvider())
    execution = registry.invoke("batch_geocode", {"place_names": ["경복궁", "광화문"], "limit": 1})
    assert execution.status == "ok"
    assert [item["query"] for item in execution.output] == ["경복궁", "광화문"]
    assert [item["place"]["name"] for item in execution.output] == ["경복궁", "광화문"]
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
            return [] if query == "더현대서울식품관" else [self._named(query)]

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


def test_batch_geocode_prefers_the_branch_nearest_the_anchor_over_the_shortest_name() -> None:
    """A bare brand name matches every branch equally, so proximity has to break the tie."""

    class BrandProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__()
            self.place = Place(
                place_id="anchor",
                name="GS25 신림골드점",
                address="서울 관악구",
                latitude=37.4832,
                longitude=126.9061,
            )

        def search_place(self, query: str, *, limit: int = 5) -> list[Place]:
            self._api_calls += 1
            return [self.place]

        def nearby_search(self, center: str | Place, **_: Any) -> list[Place]:
            self._api_calls += 1
            return [
                Place(
                    place_id="near",
                    name="맘스터치 구로디지털단지역점",
                    address="서울",
                    latitude=37.4835,
                    longitude=126.9020,
                ),
                Place(
                    place_id="far",
                    name="맘스터치 신림역점",  # shorter name, so text similarity alone picks it
                    address="서울",
                    latitude=37.4844,
                    longitude=126.9283,
                ),
            ]

    execution = ToolRegistry(BrandProvider()).invoke(
        "batch_geocode",
        {"place_names": ["GS25 신림골드점", "맘스터치"], "anchor": "GS25 신림골드점"},
    )

    assert execution.status == "ok"
    assert execution.output[1]["place"]["place_id"] == "near"


def test_batch_geocode_repicks_an_anchor_that_landed_in_another_city() -> None:
    """The anchor is the one name nothing disambiguates, so it can land far from its own batch."""

    seoul_peer = Place(
        place_id="peer",
        name="자양빵공장",
        address="서울 광진구",
        latitude=37.5350,
        longitude=127.0820,
    )
    seoul_anchor = Place(
        place_id="right",
        name="진주약국",
        address="서울 광진구",
        latitude=37.5355,
        longitude=127.0825,
    )

    class AmbiguousProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__()
            self.anchored_queries: list[str] = []

        def search_place(self, query: str, *, limit: int = 5) -> list[Place]:
            self._api_calls += 1
            if "진주약" in query:
                # A nationwide keyword search fills every slot with the far-away city.
                return [
                    Place(
                        place_id=f"jinju{index}",
                        name=f"진주약국{index}",
                        address="경남 진주시",
                        latitude=35.18,
                        longitude=128.10,
                    )
                    for index in range(3)
                ]
            return [seoul_peer]

        def nearby_search(self, center: str | Place, **kwargs: Any) -> list[Place]:
            self._api_calls += 1
            query = str(kwargs.get("query") or "")
            self.anchored_queries.append(query)
            if "진주약" in query and isinstance(center, Place) and center.latitude > 37:
                return [seoul_anchor]
            if "자양" in query:
                return [seoul_peer]
            return []

    provider = AmbiguousProvider()
    execution = ToolRegistry(provider).invoke(
        "batch_geocode",
        {"place_names": ["진주약", "자양빵공장"], "anchor": "진주약", "radius_m": 20000},
    )

    assert execution.status == "ok"
    assert execution.output[0]["place"]["place_id"] == "right"
    assert execution.output[1]["place"]["place_id"] == "peer"
    assert "진주약" in provider.anchored_queries


def test_unresolved_place_arguments_fail_as_provider_errors_not_validation_errors() -> None:
    execution = ToolRegistry(FakeProvider()).invoke(
        "nearby_places", {"center": None, "query": "도서관", "radius_m": 200}
    )

    assert execution.status == "error"
    assert execution.error is not None
    assert execution.error.startswith("PlaceNotFoundError")


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
    assert aggregate["option_totals"][0]["distance_m"] == 240


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


@pytest.mark.parametrize("strict_types", [True, False])
def test_geoflow_validation_rejects_out_of_range_batch_geocode_reference(
    strict_types: bool,
) -> None:
    """Known list cardinality is a structural rule, including on the lenient last attempt."""

    graph = {
        "graph": [
            {
                "id": "places",
                "operator": "batch_geocode",
                "arguments": {"place_names": ["기준", "후보1", "후보2"]},
                "role": "extent",
            },
            {
                "id": "distance",
                "operator": "haversine_distance",
                "arguments": {
                    "place_a": "$places.0.place",
                    "place_b": "$places.3.place",
                },
                "role": "measure",
            },
        ]
    }

    with pytest.raises(ValueError, match="has 3 place_names"):
        normalize_and_validate_graph(graph, max_steps=8, strict_types=strict_types)


def test_geoflow_validation_rejects_a_field_projection_from_batch_geocode_list() -> None:
    graph = {
        "graph": [
            {
                "id": "places",
                "operator": "batch_geocode",
                "arguments": {"place_names": ["기준", "후보"]},
                "role": "extent",
            },
            {
                "id": "distance",
                "operator": "haversine_distance",
                "arguments": {
                    "place_a": "$places.anchor_place",
                    "place_b": "$places.1.place",
                },
                "role": "measure",
            },
        ]
    }

    with pytest.raises(ValueError, match="use a numeric record index first"):
        normalize_and_validate_graph(graph, max_steps=8)


def test_grounding_prepends_anchor_only_when_plan_references_the_extra_record() -> None:
    graph = [
        {
            "id": "places",
            "operator": "batch_geocode",
            "arguments": {
                "place_names": ["후보1", "후보2", "후보3"],
                "anchor": "기준점",
            },
        },
        {
            "id": "distance",
            "operator": "haversine_distance",
            "arguments": {
                "place_a": "$places.0.place",
                "place_b": "$places.3.place",
            },
        },
    ]

    grounded = _ground_graph_literals(
        graph,
        "기준점에서 세 후보 가운데 가장 먼 곳까지의 직선거리는?",
        ["1km", "2km", "3km", "알 수 없음"],
        extract_facts({}, "기준점에서 세 후보 가운데 가장 먼 곳까지의 직선거리는?"),
    )

    assert grounded[0]["arguments"]["place_names"] == [
        "기준점",
        "후보1",
        "후보2",
        "후보3",
    ]

    no_extra_reference = [
        graph[0],
        {
            **graph[1],
            "arguments": {
                "place_a": "$places.0.place",
                "place_b": "$places.2.place",
            },
        },
    ]
    unchanged = _ground_graph_literals(
        no_extra_reference,
        "기준점에서 후보까지의 직선거리는?",
        ["1km", "2km", "3km", "알 수 없음"],
        extract_facts({}, "기준점에서 후보까지의 직선거리는?"),
    )
    assert unchanged[0]["arguments"]["place_names"] == ["후보1", "후보2", "후보3"]


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
        facts=extract_facts({}, "서울역에서 가장 가까운 카페는?"),
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
    names = {template["name"] for template in TEMPLATES.values()}
    appendix_e = {
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
    assert appendix_e <= names
    # And nothing else. `Retrieve-Rank-Ordinal` was this port's one addition, and it was a
    # benchmark family wearing a template's clothes: the ordinal is a factor on a selection now,
    # so the composition RESOLVE_PLACES -> PLACE_SEARCH -> SORT -> ORDINAL_SELECT -> MATCH_OPTIONS
    # reproduces its example graph exactly. A template is what the retrieval stage hands the
    # planner as a worked example, so adding one changes what every question of that shape gets
    # composed from -- keep the catalogue at Appendix E's ten unless there is a reason recorded
    # in docs/REFERENCE_MAPPING.md.
    # Two additions, both recorded in docs/REFERENCE_MAPPING.md. They are *shapes*, not task
    # types: neither carries an operator recipe, the factorizer still picks every operator, and
    # `Search-Rank-Ordinal`'s k is a factor. They exist because deleting the 163-line planner
    # prompt deleted the question-shape knowledge with it and cost 31 points -- a
    # "네 번째로 가까운 은행" question retrieved `Geocode-Batch-Compare`, whose pattern says to
    # resolve the candidates and rank them, and copied it.
    assert names - appendix_e == {
        "Search-Rank-Ordinal",
        "Pairwise-Difference",
        # A question that names its own candidates and one that names a narrowed kind are
        # different shapes from the retrieval they resemble, and each was answering the other's
        # question: a count over the neighbourhood instead of over the four names offered, and a
        # ranking of every restaurant instead of the 중식 ones.
        "Listed-Measure-Filter-Count",
        "Search-Narrow-Rank",
    }
    assert appendix_e <= names


@pytest.mark.parametrize(
    "template_key", sorted(key for key in TEMPLATES if TEMPLATES[key]["example"]["graph"])
)
def test_each_appendix_e_template_is_executable(template_key: str) -> None:
    """The worked operator graphs, for the templates that still carry one.

    `Search-Rank-Ordinal` and `Pairwise-Difference` carry only a skeleton -- they were written in
    the semantic vocabulary and never had a concrete example. The test below validates those.
    """

    provider_tools = ToolRegistry(FakeProvider())
    operators = SpatialOperatorRegistry()
    provider_names = {schema["function"]["name"] for schema in provider_tools.schemas()}
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


@pytest.mark.parametrize("template_key", sorted(SKELETONS))
def test_every_skeleton_the_planner_is_shown_factorizes_and_validates(template_key: str) -> None:
    """A skeleton is what a planner copies, so one that cannot be built teaches a broken shape.

    Run in the pipeline's own order -- factorize, ground, validate -- with the facts a question
    of that shape supplies. The placeholders in `concept_ids` are prose for the planner to
    replace; the name fallback chain covers them here.
    """

    from src.agent.semantics import factorize_semantic_graph
    from src.agent.spatial import GroundingFacts, _ground_graph_literals

    facts = GroundingFacts(anchor="기준점", target_type="약국", radius_m=600)
    options = ["가", "나", "다", "라"]
    built = factorize_semantic_graph(
        SKELETONS[template_key],
        concepts=[],
        options=options,
        facts=facts,
        available=frozenset(OPERATOR_CONTRACTS),
    )
    assert built.concrete_nodes == (), "a skeleton must name no operator"

    grounded = _ground_graph_literals(built.graph, "질문", options, facts)
    ordered, constraints = normalize_and_validate_graph(
        {"graph": grounded}, max_steps=20, strict_types=False
    )
    assert constraints["connectivity"] is True
    assert ordered[-1]["role"] == "measure"


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
        facts=extract_facts({}, "질문"),
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
    assert execution.output["travel_duration_s"] == 48
    assert execution.output["stay_duration_s"] == 60
    assert execution.output["finish_time"].endswith("09:01:48+09:00")


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
            LLMResponse("{}"),
            LLMResponse(PAPER_PLAN),
            LLMResponse('{"value":"은행","text":"은행","confidence":0.2,"reason":"generated"}'),
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
    """The batch is [anchor, *option texts] when the graph goes on to rank those options.

    "Goes on to rank them" is the condition, read off the dataflow. It used to be
    `intent in {"nearby", "direction", "routing"}`, whose real content was "not a trip" -- and a
    trip is exactly the plan whose `batch_geocode` must keep its stops, because overwriting them
    with the option texts answers a different itinerary. So the `match_options` node below is not
    decoration: it is what says these names are candidates. Replayed over 2,577 recorded planner
    graphs the two rules splice the same 61 nodes, neither more nor fewer.
    """

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
        },
        {
            "id": "chosen",
            "operator": "match_options",
            "arguments": {"options": [], "places": "$places"},
            "depends_on": ["places"],
            "output_type": "object",
            "role": "measure",
        },
    ]
    question = "뜻밖에 양꼬치에서 남쪽에 있는 가장 가까운 카페 중 어디인가요?"
    grounded = _ground_graph_literals(
        steps,
        question,
        ["포근한 다락방 카페", "코티지블루"],
        extract_facts({}, question),
    )
    assert grounded[0]["arguments"]["anchor"] == "뜻밖에 양꼬치"
    assert grounded[0]["arguments"]["place_names"] == [
        "뜻밖에 양꼬치",
        "포근한 다락방 카페",
        "코티지블루",
    ]


def test_an_itinerary_batch_keeps_its_stops_however_many_options_there_are() -> None:
    """The case the intent set was really excluding, now stated as the graph states it."""

    steps = [
        {
            "id": "places",
            "operator": "batch_geocode",
            "arguments": {"place_names": ["가예", "A", "B"]},
            "depends_on": [],
            "output_type": "object",
            "role": "extent",
        },
        {
            "id": "tour",
            "operator": "tsp_tw",
            "arguments": {"nodes": "$places", "distance_matrix": "$legs"},
            "depends_on": ["places"],
            "output_type": "network",
            "role": "measure",
        },
    ]
    question = "가예에서 출발해 A를 1시간, B를 1시간 둘러본 뒤 가예로 돌아옵니다. 순서는?"
    grounded = _ground_graph_literals(
        steps, question, ["A → B", "B → A"], extract_facts({}, question)
    )

    assert grounded[0]["arguments"]["place_names"] == ["가예", "A", "B"]


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
        extract_facts({}, "기준점 반경 500m 안에 있는 카페 목록은 무엇인가요?"),
        retrieval_specs=kakao_retrieval_specs,
    )

    assert grounded[0]["arguments"]["place_names"] == ["기준점"]
    assert grounded[2]["arguments"]["category_code"] == "CE7"
    assert grounded[2]["arguments"]["radius_m"] == 500
    assert grounded[3]["arguments"]["mode"] == "radius_set"
    assert grounded[3]["arguments"]["options"] == ["A | B", "A | B | C"]


def test_retrieval_grounding_fans_out_over_kakao_place_type_synonyms() -> None:
    steps = [
        {
            "id": "anchor",
            "operator": "batch_geocode",
            "arguments": {"place_names": ["기준"]},
            "depends_on": [],
            "role": "extent",
        },
        {
            "id": "nearby",
            "operator": "nearby_places",
            "arguments": {"center": "$anchor.0.place", "query": "경찰서", "limit": 5},
            "depends_on": ["anchor"],
            "role": "support",
        },
        {
            "id": "match",
            "operator": "match_options",
            "arguments": {"options": ["틀림"], "places": "$nearby"},
            "depends_on": ["nearby"],
            "role": "measure",
        },
    ]

    grounded = _ground_graph_literals(
        steps,
        "기준점에서 가장 가까운 경찰서 중 어디인가요?",
        ["동묘파출소", "안임지구대"],
        extract_facts({}, "기준점에서 가장 가까운 경찰서 중 어디인가요?"),
        retrieval_specs=kakao_retrieval_specs,
    )

    retrievals = [step for step in grounded if step["operator"] == "nearby_places"]
    merged = next(step for step in grounded if step["operator"] == "merge_places")
    assert [step["arguments"]["query"] for step in retrievals] == [
        "경찰서",
        "파출소",
        "지구대",
        "치안센터",
    ]
    assert all(step["arguments"]["limit"] == RETRIEVAL_LIMIT for step in retrievals)
    # The merge keeps the planner's node id so downstream references stay valid.
    assert merged["id"] == "nearby"
    assert merged["arguments"]["items"] == [f"${step['id']}" for step in retrievals]
    assert grounded[-1]["arguments"]["options"] == ["동묘파출소", "안임지구대"]


def test_distance_measure_grounding_restores_verbatim_option_texts() -> None:
    steps = [
        {
            "id": "places",
            "operator": "batch_geocode",
            "arguments": {"place_names": ["A", "B"]},
            "depends_on": [],
            "role": "extent",
        },
        {
            "id": "distance",
            "operator": "haversine_distance",
            "arguments": {"place_a": "$places.0.place", "place_b": "$places.1.place"},
            "depends_on": ["places"],
            "role": "support",
        },
        {
            "id": "match",
            "operator": "match_distance_options",
            "arguments": {"distance": "$distance", "options": [349, 358, 340, 342]},
            "depends_on": ["distance"],
            "role": "measure",
        },
    ]

    grounded = _ground_graph_literals(
        steps,
        "A 및 B 사이의 직선거리는 몇 m인가요?",
        ["349 m", "358 m", "340 m", "342 m"],
        extract_facts({}, "A 및 B 사이의 직선거리는 몇 m인가요?"),
    )

    assert grounded[-1]["arguments"]["options"] == ["349 m", "358 m", "340 m", "342 m"]


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
        facts=extract_facts({}, "기준점 반경 500m 안에 있는 카페 목록은 무엇인가요?"),
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
    origin = FakeProvider().place.model_dump()
    destination = {**origin, "place_id": "2", "name": "남산", "latitude": 37.55}
    execution = registry.invoke(
        "directions",
        {"origin": origin, "destination": destination, "mode": "driving"},
    )
    assert execution.status == "ok"
    # The fake echoes the endpoints it was given, so the span follows from their names.
    assert execution.output["origin"] == "경복궁"
    assert execution.output["destination"] == "남산"
    assert execution.output["distance_m"] > 0


def test_a_leg_from_a_place_to_itself_is_answered_for_both_architectures() -> None:
    """Kakao refuses to route it, and the refusal was being collected by the hundred.

    The evidence below the tools has to be the same for both agents, so the zero-cost leg is
    answered in `travel_time` and `directions` — which is all ReAct has — exactly as it is inside
    `distance_matrix`, which only Spatial-Agent reaches.
    """

    provider = FakeProvider()
    registry = ToolRegistry(provider)
    palace = registry.invoke("place_search", {"query": "경복궁"}).output[0]["place_id"]
    before = provider.api_call_count
    for tool in ("travel_time", "directions"):
        execution = registry.invoke(tool, {"origin": palace, "destination": palace})
        assert execution.status == "ok"
        assert execution.output["distance_m"] == 0
        assert execution.output["duration_s"] == 0
    assert provider.api_call_count == before


def test_spatial_agent_handles_google_style_references_without_aborting() -> None:
    llm = QueuedLLM(
        [
            LLMResponse('{"intent":"nearby"}'),
            LLMResponse(
                '{"steps":['
                '{"id":"s1","operator":"batch_geocode",'
                '"arguments":{"place_names":["경복궁"],"limit":1}},'
                '{"id":"s2","operator":"batch_geocode",'
                '"arguments":{"place_names":["광화문"],"limit":1}},'
                '{"id":"s3","operator":"haversine_distance",'
                '"arguments":{"place_a":"$s1.0.geometry.location",'
                '"place_b":"$s2.0.geometry.location"}}]}'
            ),
            LLMResponse('{"predicted_option":1,"confidence":0.9,"reason":"distance"}'),
        ]
    )
    agent = SpatialAgent(llm, ToolRegistry(FakeProvider()), max_steps=3)
    result = agent.answer("질문", ["A", "B"])
    assert result.failure_type == "graph_validation_failure"
    assert not any(stage["stage"] == "execute" for stage in result.trace)


def test_overspecified_plan_reference_degrades_to_the_closest_resolvable_object() -> None:
    llm = QueuedLLM(
        [
            LLMResponse('{"intent":"poi"}'),
            LLMResponse(
                '{"steps":['
                '{"id":"s1","operator":"batch_geocode",'
                '"arguments":{"place_names":["경복궁"],"limit":1}},'
                '{"id":"s2","operator":"haversine_distance",'
                '"arguments":{"place_a":"$s1.0.place",'
                '"place_b":"$s1.0"}}]}'
            ),
            LLMResponse('{"predicted_option":1,"confidence":0.3,"reason":"partial evidence"}'),
        ]
    )
    result = SpatialAgent(llm, ToolRegistry(FakeProvider()), max_steps=2).answer("질문", ["A", "B"])
    assert result.failure_type == "graph_validation_failure"
    assert result.predicted_answer is None


def test_unresolvable_plan_reference_is_isolated_and_evaluation_still_runs() -> None:
    llm = QueuedLLM(
        [
            LLMResponse('{"intent":"poi"}'),
            LLMResponse(
                '{"steps":['
                '{"id":"s1","operator":"place_search",'
                '"arguments":{"query":"경복궁","limit":1}},'
                '{"id":"s2","operator":"haversine_distance",'
                '"arguments":{"place_a":"한번도찾지못한장소",'
                '"place_b":"$s1.0"}}]}'
            ),
            LLMResponse('{"predicted_option":1,"confidence":0.3,"reason":"partial evidence"}'),
        ]
    )
    result = SpatialAgent(llm, ToolRegistry(FakeProvider()), max_steps=2).answer("질문", ["A", "B"])
    assert result.failure_type == "graph_validation_failure"
    assert result.predicted_answer is None


def test_generation_prefers_exact_answer_text_over_a_miscounted_index() -> None:
    llm = QueuedLLM(
        [
            LLMResponse("{}"),
            LLMResponse(PAPER_PLAN),
            LLMResponse(
                '{"predicted_answer":"경복궁","predicted_option":0,'
                '"confidence":0.9,"reason":"category evidence"}'
            ),
        ]
    )

    result = SpatialAgent(llm, ToolRegistry(FakeProvider()), max_steps=2).answer(
        "질문", ["창덕궁", "경복궁"]
    )

    evaluate = next(stage for stage in result.trace if stage["stage"] == "mcq_adapt")
    assert result.predicted_answer == 1
    assert evaluate["selection_method"] == "exact_grounded_text"
    assert result.response == "^^1^^"


def test_generation_does_not_accept_a_declared_index_without_matching_text() -> None:
    llm = QueuedLLM(
        [
            LLMResponse("{}"),
            LLMResponse(PAPER_PLAN),
            LLMResponse(
                '{"predicted_answer":"알 수 없음","predicted_option":1,'
                '"confidence":0.4,"reason":"weak evidence"}'
            ),
        ]
    )

    result = SpatialAgent(llm, ToolRegistry(FakeProvider()), max_steps=2).answer(
        "질문", ["창덕궁", "경복궁"]
    )

    evaluate = next(stage for stage in result.trace if stage["stage"] == "mcq_adapt")
    assert result.predicted_answer is None
    assert result.failure_type == "answer_parse_failure"
    assert evaluate["selection_method"] == "unresolved"


def test_batch_geocode_prefers_the_right_brand_over_an_unrelated_namesake() -> None:
    """Branch-suffix mismatch must not outrank being the wrong kind of business entirely."""

    class BrandProvider(FakeProvider):
        def search_place(self, query: str, *, limit: int = 5) -> list[Place]:
            self._api_calls += 1
            return [
                # No branch suffix at all, so the branch term scores it 0 while every real CU
                # branch scores -2; only the category term separates them.
                Place(
                    place_id="tower",
                    name="센트럴타워",
                    address="서울",
                    latitude=37.50,
                    longitude=127.11,
                    category="부동산 > 건물",
                ),
                Place(
                    place_id="store",
                    name="CU 가락센타점",
                    address="서울",
                    latitude=37.49,
                    longitude=127.12,
                    category="가정,생활 > 편의점 > CU",
                ),
            ]

    execution = ToolRegistry(BrandProvider()).invoke(
        "batch_geocode", {"place_names": ["CU 가락센트럴점"], "limit": 1}
    )

    assert execution.status == "ok"
    assert execution.output[0]["place"]["place_id"] == "store"


def test_batch_reconciliation_leaves_a_multi_name_anchor_where_it_resolved() -> None:
    """Scattered option brands must not out-vote the anchor and move the whole batch."""

    anchor = Place(
        place_id="anchor",
        name="GS25 오류행복점",
        address="서울 구로구",
        latitude=37.4940,
        longitude=126.8420,
    )
    near = Place(
        place_id="near",
        name="파리바게뜨 오류역점",
        address="서울 구로구",
        latitude=37.4960,
        longitude=126.8450,
    )
    far = Place(
        place_id="far",
        name="뚜레쥬르 영등포도림점",
        address="서울 영등포구",
        latitude=37.5100,
        longitude=126.9000,
    )

    class ScatteredProvider(FakeProvider):
        def search_place(self, query: str, *, limit: int = 5) -> list[Place]:
            self._api_calls += 1
            return [anchor if "GS25" in query else far]

        def nearby_search(self, center: str | Place, **kwargs: Any) -> list[Place]:
            self._api_calls += 1
            query = str(kwargs.get("query") or "")
            if "파리바게뜨" in query:
                return [near]
            if "뚜레쥬르" in query:
                return [far]
            return []

    execution = ToolRegistry(ScatteredProvider()).invoke(
        "batch_geocode",
        {
            "place_names": ["GS25 오류행복점", "파리바게뜨", "뚜레쥬르"],
            "anchor": "GS25 오류행복점",
            "limit": 1,
        },
    )

    assert execution.status == "ok"
    assert execution.output[0]["place"]["place_id"] == "anchor"
    assert execution.output[1]["place"]["place_id"] == "near"


def test_option_recovery_excludes_the_anchor_and_honours_the_asked_for_category() -> None:
    anchor = Place(
        place_id="bookstore",
        name="교보문고 목동점",
        address="서울 양천구",
        latitude=37.5280,
        longitude=126.8750,
    )
    station = Place(
        place_id="station",
        name="오목교역 5호선",
        address="서울 양천구",
        latitude=37.5245,
        longitude=126.8752,
        category="교통,수송 > 지하철 > 수도권",
    )

    class StationProvider(FakeProvider):
        def nearby_search(self, center: str | Place, **kwargs: Any) -> list[Place]:
            self._api_calls += 1
            if kwargs.get("category_code") == "SW8":
                return [station]
            # Uncategorised, the anchor's own name satisfies the option "목동".
            return [anchor, station]

    execution = ToolRegistry(StationProvider()).invoke(
        "recover_option_places",
        {
            "options": ["오목교", "목동", "까치산"],
            "candidates": [station.model_dump()],
            "anchor": anchor.model_dump(),
            "radius_m": 500,
            "category_code": "SW8",
        },
    )

    assert execution.status == "ok"
    assert [place["name"] for place in execution.output] == ["오목교역 5호선"]


def test_grounding_gives_option_recovery_the_questions_radius_and_category() -> None:
    graph = [
        {
            "id": "recover",
            "operator": "recover_option_places",
            "arguments": {"options": ["오목교"], "candidates": "$n", "anchor": "$a"},
            "depends_on": [],
            "output_type": "object",
            "role": "support",
        }
    ]
    grounded = _ground_graph_literals(
        graph,
        "교보문고 목동점 반경 500m 안에 있는 역 목록은 무엇인가요?",
        ["오목교", "목동", "까치산", "오목교 | 목동"],
        extract_facts({}, "교보문고 목동점 반경 500m 안에 있는 역 목록은 무엇인가요?"),
        retrieval_specs=kakao_retrieval_specs,
    )

    arguments = grounded[0]["arguments"]
    assert arguments["options"] == ["오목교", "목동", "까치산", "오목교 | 목동"]
    assert arguments["radius_m"] == 500
    assert arguments["category_code"] == "SW8"


def test_a_sibling_branch_of_the_same_brand_is_not_the_queried_branch() -> None:
    """The bare-brand retry finds a different shop; only the branch part tells them apart."""

    class BrandOnlyProvider(FakeProvider):
        def search_place(self, query: str, *, limit: int = 5) -> list[Place]:
            self._api_calls += 1
            # Kakao has no CU 구로소담점, so the exact query is empty and the brand-only retry
            # answers with whichever CU sits nearest the region prior's centre.
            if query.strip() != "CU":
                return []
            return [
                Place(
                    place_id="other",
                    name="CU 중구세종대로점",
                    address="서울 중구",
                    latitude=37.5665,
                    longitude=126.9780,
                    category="가정,생활 > 편의점 > CU",
                )
            ]

        def nearby_search(self, center: str | Place, **kwargs: Any) -> list[Place]:
            self._api_calls += 1
            return []

    execution = ToolRegistry(BrandOnlyProvider()).invoke(
        "batch_geocode", {"place_names": ["CU 구로소담점"], "limit": 1}
    )

    assert execution.status == "ok"
    assert execution.output[0]["place"] is None
    assert "PlaceNotFoundError" in execution.output[0]["error"]


def test_a_kakao_spelling_of_the_same_branch_still_resolves() -> None:
    """The distinguishing residue must be long enough to distinguish before it can reject."""

    class SpellingProvider(FakeProvider):
        def search_place(self, query: str, *, limit: int = 5) -> list[Place]:
            self._api_calls += 1
            return [
                Place(
                    place_id="store",
                    name="CU 가락센타점",
                    address="서울 송파구",
                    latitude=37.4950,
                    longitude=127.1180,
                    category="가정,생활 > 편의점 > CU",
                )
            ]

    execution = ToolRegistry(SpellingProvider()).invoke(
        "batch_geocode", {"place_names": ["CU 가락센트럴점"], "limit": 1}
    )

    assert execution.status == "ok"
    assert execution.output[0]["place"]["place_id"] == "store"


def test_option_recovery_drops_a_namesake_outside_the_asked_radius() -> None:
    """The nationwide fallback answers "anywhere", which a proximity question never asks."""

    anchor = Place(
        place_id="anchor",
        name="먹골",
        address="서울 중랑구",
        latitude=37.6100,
        longitude=127.0770,
    )

    class NationwideProvider(FakeProvider):
        def nearby_search(self, center: str | Place, **kwargs: Any) -> list[Place]:
            self._api_calls += 1
            return []

        def search_place(self, query: str, *, limit: int = 5) -> list[Place]:
            self._api_calls += 1
            return [
                Place(
                    place_id="faraway",
                    name="꽃담공방",
                    address="전남 순천시",
                    latitude=34.9500,
                    longitude=127.4870,
                )
            ]

    execution = ToolRegistry(NationwideProvider()).invoke(
        "recover_option_places",
        {"anchor": anchor, "candidates": [], "options": ["꽃담공방"], "radius_m": 20000},
    )

    assert execution.status == "ok"
    assert execution.output == []


def test_one_retrieved_place_cannot_support_two_options() -> None:
    """A single POI is one place, so it answers at most one candidate option."""

    anchor = {"name": "유진마트", "latitude": 37.6200, "longitude": 127.0700}
    school = {"name": "서울공릉초등학교", "latitude": 37.6210, "longitude": 127.0710}

    result = SpatialOperatorRegistry.match_options(
        options=["서울오륜초등학교", "서울평화초등학교"],
        places=[school],
        anchor=anchor,
    )

    matched = [match for match in result["option_matches"] if match["matched"]]
    assert len(matched) <= 1


def test_a_shared_generic_suffix_is_not_evidence_of_the_same_place() -> None:
    """서울오륜초등학교 and 서울공릉초등학교 differ exactly where the name identifies a school."""

    anchor = {"name": "유진마트", "latitude": 37.6200, "longitude": 127.0700}
    school = {"name": "서울공릉초등학교", "latitude": 37.6210, "longitude": 127.0710}

    result = SpatialOperatorRegistry.match_options(
        options=["서울오륜초등학교"], places=[school], anchor=anchor
    )

    assert result["best_option"] is None


def test_coordinates_are_usable_where_a_place_name_is_expected() -> None:
    """An agent holding a POI's coordinates asks about them, not about a place with that name."""

    from src.tools.kakao import _parse_coordinate_literal

    assert _parse_coordinate_literal("37.5771637987289,126.96943884968") == (
        37.5771637987289,
        126.96943884968,
    )
    assert _parse_coordinate_literal(" 37.5771 , 126.9694 ") == (37.5771, 126.9694)
    assert _parse_coordinate_literal("경복궁") is None
    assert _parse_coordinate_literal("서울 종로구 1,2") is None
    assert _parse_coordinate_literal("991.0,126.9") is None


def test_a_neighbour_of_the_same_kind_is_not_the_named_place() -> None:
    """Being near the anchor is not evidence of being the place the question named."""

    class NeighbourhoodProvider(FakeProvider):
        def search_place(self, query: str, *, limit: int = 5) -> list[Place]:
            self._api_calls += 1
            if query.strip() == "CU 성내천오금점":
                return [self._named(query)]
            return []  # Kakao has no 신사정육점 under any spelling

        def nearby_search(self, center: str | Place, **kwargs: Any) -> list[Place]:
            self._api_calls += 1
            # Kakao's keyword search is tolerant: asked for a butcher it does not have, it
            # answers with the butchers it does.
            return [
                Place(
                    place_id="butcher",
                    name="한아름축산",
                    address="서울 강남구",
                    latitude=37.5240,
                    longitude=127.0230,
                    category="가정,생활 > 정육점",
                )
            ]

    execution = ToolRegistry(NeighbourhoodProvider()).invoke(
        "batch_geocode", {"place_names": ["CU 성내천오금점", "신사정육점"], "limit": 1}
    )

    assert execution.status == "ok"
    assert execution.output[1]["place"] is None
    assert "PlaceNotFoundError" in execution.output[1]["error"]


def test_a_transliterated_brand_resolves_inside_the_anchors_neighbourhood() -> None:
    """Characters cannot testify across scripts, so the bounded search speaks instead."""

    class TransliteratedProvider(FakeProvider):
        def search_place(self, query: str, *, limit: int = 5) -> list[Place]:
            self._api_calls += 1
            if query.strip() == "CU 성내천오금점":
                return [self._named(query)]
            return []

        def nearby_search(self, center: str | Place, **kwargs: Any) -> list[Place]:
            self._api_calls += 1
            return [
                Place(
                    place_id="cafe",
                    name="투썸플레이스 장안점",
                    address="서울 동대문구",
                    latitude=37.5680,
                    longitude=127.0700,
                    category="음식점 > 카페",
                )
            ]

    execution = ToolRegistry(TransliteratedProvider()).invoke(
        "batch_geocode", {"place_names": ["CU 성내천오금점", "A TWOSOME PLACE"], "limit": 1}
    )

    assert execution.output[1]["place"]["name"] == "투썸플레이스 장안점"


def test_strict_names_refuses_even_a_cross_script_neighbour() -> None:
    """A question that states both POIs precisely gets no transliteration licence."""

    class TransliteratedProvider(FakeProvider):
        def search_place(self, query: str, *, limit: int = 5) -> list[Place]:
            self._api_calls += 1
            if query.strip() == "CU 성내천오금점":
                return [self._named(query)]
            return []

        def nearby_search(self, center: str | Place, **kwargs: Any) -> list[Place]:
            self._api_calls += 1
            return [
                Place(
                    place_id="cafe",
                    name="투썸플레이스 장안점",
                    address="서울 동대문구",
                    latitude=37.5680,
                    longitude=127.0700,
                    category="음식점 > 카페",
                )
            ]

    execution = ToolRegistry(TransliteratedProvider()).invoke(
        "batch_geocode",
        {"place_names": ["CU 성내천오금점", "A TWOSOME PLACE"], "strict_names": True, "limit": 1},
    )

    assert execution.output[1]["place"] is None


def test_an_option_carrying_its_address_resolves_by_its_name() -> None:
    """A dataset separates namesakes with an address; Kakao indexes names."""

    class NameOnlyProvider(FakeProvider):
        def search_place(self, query: str, *, limit: int = 5) -> list[Place]:
            self._api_calls += 1
            if query.strip() != "서울난곡우체국":
                return []
            return [self._named("서울난곡우체국")]

        def nearby_search(self, center: str | Place, **kwargs: Any) -> list[Place]:
            self._api_calls += 1
            return []

    execution = ToolRegistry(NameOnlyProvider()).invoke(
        "batch_geocode",
        {"place_names": ["서울난곡우체국 - 서울특별시 관악구 신림동 난곡로 275"], "limit": 1},
    )

    assert execution.output[0]["place"]["name"] == "서울난곡우체국"


def test_a_short_name_buried_in_a_long_one_is_not_evidence() -> None:
    """압구정 sits inside a riverboat ramp's name while naming somewhere else entirely."""

    from src.tools.registry import _containment_is_evidence

    assert not _containment_is_evidence("압구정", "해피냠냠라면가게한강버스압구정선착장점")
    assert _containment_is_evidence("올리브영", "올리브영거여역점")
    assert _containment_is_evidence("진주리", "카페진주리")


def test_one_institution_under_two_names_is_one_place() -> None:
    """OSM writes 연남치안센터 where Kakao lists 연남파출소; 수유6 is still not 쌍문1."""

    from src.tools.registry import _search_key

    assert _search_key("연남치안센터") == _search_key("연남파출소")
    assert _search_key("쌍문1치안센터") != _search_key("수유6치안센터")


def test_the_mapeval_baseline_withholds_the_aggregation_tools() -> None:
    """The paper compares its graph against ReAct over map API *primitives*.

    MapEval's own baseline is the five tools `mapeval-api/Evaluator2.py` (35d481a, line 33)
    instantiates: PlaceSearch, PlaceDetails, NearbyPlaces, TravelTime, Directions. Everything
    beyond that here — batch geocoding, a distance matrix, a multi-stop finish time — is an
    aggregation over those primitives, which is what GeoFlow's operator graph exists to express.
    Handing them to ReAct answers a different question: a trip the paper's baseline must
    orchestrate over a dozen turns becomes two calls.

    The address tools are withheld for a different reason. Upstream reaches every place through
    a place id and never converts between an address and coordinates, so `geocode` and
    `reverse_geocode` are capabilities the baseline was measured without — and in practice
    `geocode` was answering bare place names, duplicating PlaceSearch through a second index.
    """

    full = ToolRegistry(FakeProvider())
    baseline = ToolRegistry(FakeProvider(), allowed=ToolRegistry.MAPEVAL_BASELINE_TOOLS)

    full_names = {schema["function"]["name"] for schema in full.schemas()}
    baseline_names = {schema["function"]["name"] for schema in baseline.schemas()}

    assert baseline_names < full_names
    # One name per tool `Evaluator2.py` constructs, and nothing else.
    assert baseline_names == {
        "place_search",
        "place_details",
        "nearby_places",
        "travel_time",
        "directions",
    }
    assert full_names - baseline_names == {
        "batch_geocode",
        "batch_place_details",
        "distance_matrix",
        "calculate_finish_time",
        "recover_option_places",
        "geocode",
        "reverse_geocode",
    }


def test_restricting_to_an_unknown_tool_is_refused() -> None:
    with pytest.raises(ValueError, match="Unknown tools"):
        ToolRegistry(FakeProvider(), allowed={"place_search", "teleport"})


def test_the_baseline_gets_the_papers_tool_surface_by_default() -> None:
    """A tool surface is part of an architecture, so the two agents do not share one.

    Making them identical did not remove a confound — it deleted the difference the paper
    measures. Upstream carries `get_distance_matrix` in `spatial-agent/src/tools/google_maps.py`
    and `mapeval-api/Evaluator2.py` hands its baseline nothing of the kind. `full` remains
    reachable as an explicit ablation, which is why the flag still exists.
    """

    from main import build_parser

    parser = build_parser()
    assert parser.parse_args([]).react_tools == "reference"
    assert parser.parse_args(["--react-tools", "native"]).react_tools == "native"
    assert parser.parse_args(["--react-tools", "full"]).react_tools == "full"


def test_the_reference_surface_carries_upstreams_argument_contracts() -> None:
    """Restricting the tool *names* is not restricting the tool surface.

    Our five accepted arguments upstream's five do not, and an argument is a capability:
    `directions` took up to 30 waypoints where `mapeval-api/Tools.py` has
    `Directions(originId, destinationId, travelMode)`, so a detour upstream must assemble from two
    routes and an addition was one call here. Measured on the v5 run, ReAct issued a waypointed
    call on all 8 `routing_distance_via` rows. This pins the contract, not the roster.
    """

    reference = ToolRegistry(
        FakeProvider(), allowed=ToolRegistry.MAPEVAL_BASELINE_TOOLS, contract="reference"
    )
    fields = {
        schema["function"]["name"]: set(schema["function"]["parameters"]["properties"])
        for schema in reference.schemas()
    }
    assert fields["place_search"] == {"place_name"}
    assert fields["place_details"] == {"place_id"}
    assert fields["nearby_places"] == {"place_id", "type", "rankby", "radius"}
    assert fields["travel_time"] == {"origin_id", "destination_id", "travel_mode"}
    assert fields["directions"] == {"origin_id", "destination_id", "travel_mode"}
    for name in ("directions", "travel_time"):
        assert "waypoints" not in fields[name]
        assert "priority" not in fields[name]

    native = ToolRegistry(
        FakeProvider(), allowed=ToolRegistry.MAPEVAL_BASELINE_TOOLS, contract="native"
    )
    native_fields = {
        schema["function"]["name"]: set(schema["function"]["parameters"]["properties"])
        for schema in native.schemas()
    }
    assert "waypoints" in native_fields["directions"]
    assert fields.keys() == native_fields.keys()


def test_the_reference_nearby_refuses_a_radius_when_it_ranks_by_distance() -> None:
    """Upstream will not do both in one call, and that refusal is what costs its baseline a turn.

    "The nearest pharmacy within 500 m" is two calls and a comparison for MapEval's agent. Our
    `nearby_places` answered it in one, which is a capability the paper's baseline never had.
    """

    registry = ToolRegistry(
        FakeProvider(), allowed=ToolRegistry.MAPEVAL_BASELINE_TOOLS, contract="reference"
    )
    refused = registry.invoke(
        "nearby_places",
        {"place_id": "p1", "type": "pharmacy", "rankby": "distance", "radius": 500},
    )
    assert refused.status == "ok"
    assert "radius is disallowed" in str(refused.output)

    needs_radius = registry.invoke(
        "nearby_places", {"place_id": "p1", "type": "pharmacy", "rankby": "prominence"}
    )
    assert "radius parameter is required" in str(needs_radius.output)


def test_the_react_prompt_carries_no_tool_strategy() -> None:
    """MapEval's baseline gets the question, the options and the answer format, and no plan.

    Naming the question taxonomy and which tool each shape wants is planning handed to the
    baseline in prose — the same error as handing it the aggregation tools, in another currency.
    """

    from src.agent.react import REACT_SYSTEM_PROMPT

    for planted in ("nearby_places", "directions only", "coordinates for direction", "radius"):
        assert planted not in REACT_SYSTEM_PROMPT
    # What MapEval's own prompt does carry stays.
    assert "^^Option_Number^^" in REACT_SYSTEM_PROMPT
    assert "0-based" in REACT_SYSTEM_PROMPT


def test_a_place_argument_is_a_reference_the_provider_issued() -> None:
    """A name is not a place, and the baseline tools no longer search behind the call.

    `mapeval-api/FormattedTools.py` gives its baseline one way to turn a name into a place —
    `PlaceSearchTool`, which returns a `place_id` — and `PlaceDetails`, `NearbyPlaces`, `TravelTime`
    and `Directions` all consume that id. Threading it is part of the task upstream measures.
    Resolving names inside `nearby_search` and `directions` excused our port from it: on the v4
    run ReAct passed a bare name in about two thirds of its place arguments (`travel_time` 123
    against 53 ids) while Spatial-Agent passed none, so the convenience was worth nothing to the
    architecture under test and a whole error class to the baseline.
    """

    provider = FakeProvider()
    registry = ToolRegistry(provider, allowed=ToolRegistry.MAPEVAL_BASELINE_TOOLS)

    named = registry.invoke("travel_time", {"origin": "서울역", "destination": "경복궁"})
    assert named.status == "error"
    assert "place_search" in (named.error or "")

    # What the tool does accept is what `place_search` handed back.
    found = registry.invoke("place_search", {"query": "서울역"})
    assert found.status == "ok"
    reference = found.output[0]["place_id"]
    assert "latitude" not in found.output[0]  # upstream returns an id, not a place
    threaded = registry.invoke(
        "travel_time",
        {"origin": reference, "destination": reference},
    )
    assert threaded.status == "ok"


def test_an_aggregation_tool_still_resolves_the_names_a_plan_holds() -> None:
    """The discipline is the baseline's surface, not a rule about evidence.

    `batch_geocode`, `distance_matrix` and `calculate_finish_time` exist to take the names a plan
    is holding and resolve them in one step — that is what makes them aggregations over
    PlaceSearch. They are Spatial-Agent's tools and they keep doing it, through the same matcher.
    """

    registry = ToolRegistry(FakeProvider())
    matrix = registry.invoke(
        "distance_matrix", {"pairs": [{"origin": "서울역", "destination": "경복궁"}]}
    )
    assert matrix.status == "ok"
    assert matrix.output["routes"][0]["status"] == "ok"


def test_the_react_baseline_is_the_upstream_port_and_stays_frozen() -> None:
    """Pins the port so a future edit to the baseline has to be deliberate.

    `mapeval-api/Evaluator2.py` constructs five tools and prompts with the question, the options
    and the answer format. Everything this repo learns from a benchmark run belongs on the
    Spatial-Agent side; an accuracy gap ReAct shows is the finding, and closing one here would
    make the baseline a function of the test set.
    """

    from src.agent.react import REACT_SYSTEM_PROMPT
    from src.tools.registry import DirectionsArgs, ToolRegistry

    assert ToolRegistry.MAPEVAL_BASELINE_TOOLS == frozenset(
        {"place_search", "place_details", "nearby_places", "travel_time", "directions"}
    )
    assert REACT_SYSTEM_PROMPT == (
        "You are the MapEval-style ReAct baseline for Korean spatial questions.\n"
        "Use the map tools to gather evidence and reason over only the question and candidate "
        "options.\nSelect one 0-based option. Never invent a place ID. When you have enough "
        "evidence, answer exactly as\n^^Option_Number^^. You are not given and must not ask for "
        "the gold answer."
    )
    # A parameter is documented by its accepted values, the way upstream documents travelMode.
    # Glossing what each priority optimizes was written after watching ReAct read a
    # shortest-route question as RECOMMEND -- which priority a question asks for is grounding,
    # and grounding is a Spatial-Agent stage under measurement.
    priority = DirectionsArgs.model_fields["priority"]
    assert priority.description == "RECOMMEND, TIME, or DISTANCE"


def test_a_place_named_in_an_argument_is_the_place_the_plan_already_resolved() -> None:
    """Planners reference option texts where the geocoded places belong.

    The local operators spend no API call, so a name they cannot look up drops out of the
    candidate list -- and a direction filter then answers with an empty sector. Binding the
    name back to the plan's own geocoding grants no evidence the run did not already gather.
    """

    results = {
        "places": [
            {
                "query": "하나로마트 미아점",
                "place": {
                    "place_id": "1",
                    "name": "하나로마트 미아점",
                    "latitude": 37.6215,
                    "longitude": 127.0269,
                },
            },
            {
                "query": "이마트 미아점 - 서울특별시 성북구 도봉로 17",
                "place": {
                    "place_id": "2",
                    "name": "이마트 미아점",
                    "latitude": 37.6108,
                    "longitude": 127.0298,
                },
            },
        ]
    }
    bound = _bind_named_places(
        {
            "center": "$places.0.place",
            "places": ["하나로마트 미아점", "이마트 미아점", "한번도찾지못한장소"],
            "direction": "북쪽",
        },
        results,
    )
    assert bound["places"][0]["place_id"] == "1"
    assert bound["places"][1]["place_id"] == "2"
    # A name the plan never resolved is left alone, so it still fails as a missing place.
    assert bound["places"][2] == "한번도찾지못한장소"
    # Only place-valued arguments are bound; a reference and a direction pass through.
    assert bound["center"] == "$places.0.place"
    assert bound["direction"] == "북쪽"


def test_place_names_a_geocode_answered_under_another_spelling_still_bind() -> None:
    """The query text is a name of the place as much as the name Kakao stores."""

    results = {
        "geo": [
            {
                "query": "이마트 미아점 - 서울특별시 성북구 도봉로 17",
                "place": {
                    "place_id": "2",
                    "name": "이마트 미아점",
                    "latitude": 37.6108,
                    "longitude": 127.0298,
                },
            }
        ]
    }
    bound = _bind_named_places({"anchor": "이마트 미아점 - 서울특별시 성북구 도봉로 17"}, results)
    assert bound["anchor"]["place_id"] == "2"


def test_a_coordinate_literal_is_never_mistaken_for_a_name() -> None:
    results = {
        "geo": [
            {
                "query": "37.5,127.0",
                "place": {"place_id": "9", "name": "x", "latitude": 1.0, "longitude": 2.0},
            }
        ]
    }
    assert _bind_named_places({"center": "37.5,127.0"}, results)["center"] == "37.5,127.0"


def test_an_unresolved_geocode_row_is_dropped_rather_than_failing_the_call() -> None:
    """`batch_geocode` answers every name, including the ones it could not resolve.

    A planner passes the whole list on. One `{"query": …, "place": None}` among good places used
    to fail the call with seven validation errors about fields an unresolved row does not have,
    and the tool never ran — while `recover_option_places` exists precisely to look up what is
    missing.
    """

    from src.tools.registry import RecoverOptionPlacesArgs

    args = RecoverOptionPlacesArgs.model_validate(
        {
            "options": ["A"],
            "candidates": [
                {"query": "없는곳", "place": None, "candidates": []},
                {"place_id": "1", "name": "A", "latitude": 37.5, "longitude": 127.0},
                {"error": "ValueError: an earlier step failed"},
            ],
            "anchor": "X",
            "radius_m": 1000,
        }
    )
    assert [place.name for place in args.candidates] == ["A"]


def test_a_name_the_planner_copied_out_wrong_is_restored_to_the_question_s_spelling() -> None:
    """A name that is nearly a literal the question wrote is that literal.

    `잠원한강공원 눈쌨매장` matched nothing at Kakao, and the step that needed it — plus every
    step downstream — was lost. A name resembling nothing in the question is left exactly as the
    planner wrote it, so this can only restore evidence.
    """

    from src.agent.spatial import _verbatim_place_names

    question = (
        "잠원한강공원 눈썰매장에서 오후 5시 00분에 약속이 있습니다. 가는 길에 "
        "킴스클럽 강남점에서 30분 들러야 하고, 더월호텔에서 늦어도 몇 시에 출발해야 하나요?"
    )
    corrected = _verbatim_place_names(
        {"place_names": ["더월호텔", "잠원한강공원 눈쌨매장", "전혀다른어떤장소이름"]},
        question,
        ["오후 1시", "오후 2시"],
    )
    assert corrected["place_names"] == [
        "더월호텔",
        "잠원한강공원 눈썰매장",
        "전혀다른어떤장소이름",
    ]


def test_a_name_the_planner_wrote_short_is_restored_from_the_question() -> None:
    """`빈칸 문래` came through as `문래`, which resolves — to a different place.

    A truncated name is the worst kind of copying error: the lookup succeeds, the route is a real
    route, and every stage reports success while measuring somewhere else. The question states
    the origin of a routing question outright, so grounding restores it.
    """

    from src.agent.spatial import _ground_graph_literals

    question = (
        "빈칸 문래에서 훈련원공원 야외극장까지 자동차로, 거리가 가장 짧은 경로로 운전합니다. "
        "마포대로 구간에 진입하기 전까지 좌회전을 몇 번 하게 되나요?"
    )
    steps = [
        {
            "id": "geo",
            "operator": "batch_geocode",
            "arguments": {"place_names": ["문래", "훈련원공원 야외극장"]},
            "depends_on": [],
            "output_type": "object",
            "role": "extent",
        }
    ]
    grounded = _ground_graph_literals(steps, question, ["1번", "2번"], extract_facts({}, question))
    assert grounded[0]["arguments"]["place_names"] == ["빈칸 문래", "훈련원공원 야외극장"]


def test_pair_endpoints_written_as_names_bind_like_any_other_place() -> None:
    """`pairwise_distances` keeps its endpoints one level down, inside `pairs`."""

    from src.agent.spatial import _bind_named_pairs

    results = {
        "geo": [
            {
                "query": "보라믹",
                "place": {"place_id": "1", "name": "보라믹", "latitude": 37.5, "longitude": 127.0},
            },
            {
                "query": "봉제산",
                "place": {"place_id": "2", "name": "봉제산", "latitude": 37.6, "longitude": 126.9},
            },
        ]
    }
    bound = _bind_named_pairs(
        {"pairs": [{"place_a": "보라믹", "place_b": "봉제산", "label": "A"}]}, results
    )
    pair = bound["pairs"][0]
    assert pair["place_a"]["place_id"] == "1"
    assert pair["place_b"]["place_id"] == "2"
    assert pair["label"] == "A"


def test_grounding_does_not_overwrite_an_option_with_the_anchor() -> None:
    """Only a node that has an anchor slot gets the anchor written into it.

    A plan may geocode the anchor in one node and the four option texts in another. Replacing the
    head of the second deleted an option — the gold one — from a radius question whose every
    other stage worked.
    """

    from src.agent.spatial import _ground_graph_literals

    question = "강북솔밭국악당에서 직선거리 600m 이내에 있는 대형마트는 다음 중 어디인가요?"
    options = [
        "이마트에브리데이 쌍문동점",
        "GS더프레시 수유중앙점",
        "홈플러스 메가푸드마켓 방학점",
        "북서울농협하나로마트",
    ]
    steps = [
        {
            "id": "anchor_geocode",
            "operator": "batch_geocode",
            "arguments": {"place_names": ["강북솔밭국악당"]},
            "depends_on": [],
            "output_type": "object",
            "role": "extent",
        },
        {
            "id": "option_geocodes",
            "operator": "batch_geocode",
            "arguments": {"place_names": list(options)},
            "depends_on": [],
            "output_type": "object",
            "role": "support",
        },
    ]
    grounded = _ground_graph_literals(steps, question, options, extract_facts({}, question))
    assert grounded[0]["arguments"]["place_names"] == ["강북솔밭국악당"]
    assert grounded[1]["arguments"]["place_names"] == options
    # The anchor still biases both lookups; it just does not replace a name.
    assert grounded[1]["arguments"]["anchor"] == "강북솔밭국악당"


def test_a_place_is_not_within_its_own_radius() -> None:
    from src.tools import SpatialOperatorRegistry

    ops = SpatialOperatorRegistry()
    center = {"place_id": "1", "name": "강북솔밭국악당", "latitude": 37.6546, "longitude": 127.0127}
    other = {"place_id": "2", "name": "이마트", "latitude": 37.6560, "longitude": 127.0130}
    inside = ops.within_radius(center=center, candidates=[dict(center), other], radius_m=600)
    assert [place["place_id"] for place in inside] == ["2"]


def test_option_recovery_stays_in_the_sector_the_question_asks_about() -> None:
    """Recovery adds places the direction filter never saw.

    A "which mart north of here" question ranked a recovered mart 271 m *south* of the anchor
    above the northern one the filter had correctly found at 961 m: the recovered option carried
    no constraint at all, and the direction disappeared from the answer.
    """

    class _Provider(FakeProvider):
        def nearby_search(self, center: Any, **kwargs: Any) -> list[Place]:
            self._api_calls += 1
            query = str(kwargs.get("query") or "")
            offsets = {
                "북쪽마트": (0.01, 0.0),
                "남쪽마트": (-0.01, 0.0),
                "대각마트": (0.01, 0.01),
            }
            if query not in offsets:
                return []
            latitude, longitude = offsets[query]
            return [
                Place(
                    place_id=query,
                    name=query,
                    latitude=self.place.latitude + latitude,
                    longitude=self.place.longitude + longitude,
                    category="가정,생활 > 대형마트",
                )
            ]

    registry = ToolRegistry(_Provider())
    arguments = {
        "options": ["북쪽마트", "남쪽마트", "대각마트"],
        "candidates": [],
        "anchor": FakeProvider().place.model_dump(),
        "radius_m": 5000,
    }
    unconstrained = registry.invoke("recover_option_places", arguments)
    assert {place["name"] for place in unconstrained.output} == {
        "북쪽마트",
        "남쪽마트",
        "대각마트",
    }

    constrained = registry.invoke("recover_option_places", {**arguments, "direction": "북쪽"})
    assert {place["name"] for place in constrained.output} == {"북쪽마트", "대각마트"}

    diagonal = registry.invoke("recover_option_places", {**arguments, "direction": "북동쪽"})
    assert [place["name"] for place in diagonal.output] == ["대각마트"]


@pytest.mark.parametrize(
    ("question", "family", "expected"),
    [
        ("아트힐 연희에서 가장 가까운 대형마트는 다음 중 어디인가요?", "nearby", "대형마트"),
        (
            "단비갤러리에서 서쪽 방향에 있는 은행 중 가장 가까운 곳은 다음 중 어디인가요?",
            "direction",
            "은행",
        ),
        (
            "서울생활사박물관 별관동에서 직선거리 600m 이내에 있는 대형마트는 다음 중 어디인가요?",
            "radius",
            "대형마트",
        ),
        (
            "컬러풀뮤지엄과 뤄니갤러리 양쪽 모두에서 직선거리 1500m 이내에 있는 병원은 "
            "다음 중 어디인가요?",
            "radius",
            "병원",
        ),
        # The inferred-category family never states a kind, and "곳" is not one.
        (
            "지금 단막극장에 있습니다. 우산을 사야 합니다. 다음 중 걸어가기에 가장 가까운 곳은 "
            "어디인가요?",
            "nearby",
            None,
        ),
        # Phrasings no row in `dataset/` uses. The lead-in carries the intent, the ending is
        # just Korean, and a question that ends differently still states its kind.
        ("서울역에서 가장 가까운 약국은 어디인가요?", "nearby", "약국"),
        ("서울역과 가장 인접한 약국은 어디인가요?", "nearby", "약국"),
        ("경복궁 남동쪽 방향에 있는 카페 중 어느 곳이 가장 가깝나요?", "direction", "카페"),
        ("홍대입구역 반경 500m 안에 있는 서점 목록을 알려주세요", "radius", "서점"),
        ("홍대입구역으로부터 500m 내에 위치한 서점은 어디인가요?", "radius", "서점"),
        ("서울역 북쪽에 있는 가장 가까운 지하철역은 어디인가요?", "direction", "지하철역"),
    ],
)
def test_the_kind_of_place_is_read_from_the_phrasings_the_questions_use(
    question: str, family: str, expected: str | None
) -> None:
    """The lead-in carries the relation; the tail is grammar and must not be enumerated.

    An earlier revision wrote one regex per observed sentence — "북쪽에 있는 가장 가까운 X 중",
    "안에 있는 X 목록" — against benchmarks that say "북쪽 방향에 있는 X 중 가장 가까운 곳" and
    "이내에 있는 X는". Nothing matched, and a literal sitting in plain sight reached the retrieval
    only as the Analysis stage's guess. Rewriting the sentences we had seen would have made the
    extractor a function of the test set, so the split is structural: the final cases use
    endings no dataset row does.

    `family` records which kind of question each phrasing came from and is deliberately not
    passed to the extractor. The leads used to be keyed by it and only the matching key was
    tried, so a question the Analysis stage mislabelled read as stating no kind at all -- and it
    mislabelled 21 of 90 `nearby_subtype_kth` graphs on the recorded runs. All three leads are
    tried now, and they do not compete: each names a different relation.
    """

    from src.agent.spatial import _extract_target_type

    assert _extract_target_type(question) == expected, family


def test_a_geocode_node_written_where_its_places_belong_is_flattened() -> None:
    """`locations: ["$places"]` resolves to a list holding one list of four records.

    An itinerary of one four-place list is not a shape any operator can read, and the clock
    failed before a single leg was routed. The planner indexed one level too few — the mirror of
    the one-element unwrap the same normalizer already does.
    """

    from src.tools.registry import CalculateFinishTimeArgs

    records = [
        {
            "query": name,
            "place": {
                "place_id": name,
                "name": name,
                "latitude": 37.5 + index / 100,
                "longitude": 127.0,
            },
        }
        for index, name in enumerate(["출발지", "첫째", "둘째", "출발지"])
    ]
    args = CalculateFinishTimeArgs.model_validate({"start_time": "10:00", "locations": [records]})
    assert [place.name for place in args.locations] == ["출발지", "첫째", "둘째", "출발지"]


def test_a_place_serialized_back_as_json_text_is_that_place() -> None:
    """The ReAct baseline hands back the place it retrieved, as a JSON string.

    The string went to Kakao as a keyword query — twelve HTTP 400s in one run, each one a
    retrieval the agent believed it had made. The coordinates in it came from an earlier tool
    result, so reading them back adds no evidence.
    """

    import json as json_module

    from src.tools.registry import PlaceSearchArgs

    blob = json_module.dumps(
        {"place_id": "1", "name": "금호미술관", "latitude": 37.5774, "longitude": 126.9798}
    )
    args = PlaceSearchArgs.model_validate(
        {"center": blob, "category_code": "HP8", "radius_m": 1500, "limit": 20}
    )
    assert isinstance(args.center, Place)
    assert args.center.name == "금호미술관"
    # A name that merely looks like a brace is still a name.
    plain = PlaceSearchArgs.model_validate({"center": "{서울역", "query": "편의점"})
    assert plain.center == "{서울역"


def test_a_ranking_over_bare_node_ids_resolves_them() -> None:
    """`items: ["d0","d1"]` names two steps of the plan and forgets the `$`."""

    from src.agent.spatial import _bind_step_references

    results = {"d0": {"distance_m": 120.0}, "d1": {"distance_m": 900.0}}
    bound = _bind_step_references({"items": ["d0", "d1"], "key": "distance_m"}, results)
    assert bound["items"] == [{"distance_m": 120.0}, {"distance_m": 900.0}]
    # A string that is not a step the run executed is a string the planner meant literally.
    literal = _bind_step_references({"items": ["d0", "서울역"], "key": "distance_m"}, results)
    assert literal["items"] == ["d0", "서울역"]


def test_a_nested_itinerary_is_flattened_before_the_stays_are_counted() -> None:
    """The tool flattens `locations: ["$places"]`; the stays are bound in grounding.

    Counted against the unflattened list they came out length 1 against four places, and the args
    model rejected the call for the mismatch — after the flattening fix had made the shape legal.
    """

    from src.agent.spatial import _ground_graph_literals

    question = (
        "오전 10시 00분에 2C게스트하우스에서 자동차로 출발해 이지영갤러리를 1시간, "
        "낙낙별길을 1.5시간 동안 차례로 둘러본 뒤 2C게스트하우스로 돌아옵니다. "
        "몇 시에 돌아오게 되나요?"
    )
    steps = [
        {
            "id": "finish",
            "operator": "calculate_finish_time",
            "arguments": {
                "start_time": "10:00",
                "locations": [["2C게스트하우스", "이지영갤러리", "낙낙별길", "2C게스트하우스"]],
            },
            "depends_on": [],
            "output_type": "event",
            "role": "measure",
        }
    ]
    grounded = _ground_graph_literals(
        steps,
        question,
        ["오후 1시", "오후 2시"],
        extract_facts({}, question),
    )
    arguments = grounded[0]["arguments"]
    assert len(arguments["locations"]) == 4
    assert len(arguments["stay_durations_s"]) == 4


def test_geocode_falls_back_to_the_place_name_index() -> None:
    """Kakao keeps addresses and place names in two indexes.

    `대림동 우리 골목형상점가` has no address entry and one exact place entry. Failing there cost
    the whole question: the anchor never resolved, so no step after it had anything to work on.
    """

    class _AddresslessProvider(FakeProvider):
        def geocode(self, address: str, *, limit: int = 5) -> list[Place]:
            self._api_calls += 1
            return []

    registry = ToolRegistry(_AddresslessProvider())
    execution = registry.invoke("geocode", {"address": "대림동 우리 골목형상점가"})
    assert execution.status == "ok"
    assert execution.output[0]["name"] == "대림동 우리 골목형상점가"

    class _EmptyProvider(_AddresslessProvider):
        def search_place(self, query: str, *, limit: int = 5) -> list[Place]:
            self._api_calls += 1
            return []

    missing = ToolRegistry(_EmptyProvider()).invoke("geocode", {"address": "없는주소"})
    assert missing.status == "error"
    assert "PlaceNotFoundError" in missing.error


def test_a_trip_stop_the_planner_cut_short_is_restored_from_the_stays() -> None:
    """`kmapeval_211` lost this way in every run of five revisions, and it is one character.

    `백련산꿈마을숲정이를` is a name plus a particle, and the planner segmented it as
    `백련산꿈마을숲정` plus `이를`. The short form is still a substring of the question, so the
    fallback that repairs a *mis-typed* name leaves it alone; only a list of the names the
    question states can tell the two apart. A trip's stops are stated exactly like its anchor —
    `_extract_trip_schedule` already reads each one to bind its stay — so they belong in that
    list. Without it Kakao found nothing and the refusal surfaced three nodes later as
    `tsp_tw distance_matrix must be square`, naming neither the place nor the lookup.
    """

    from src.agent.spatial import _ground_graph_literals

    question = (
        "인게스트하우스에서 출발해 한남매봉공원산책길을 1시간, 백련산꿈마을숲정이를 1시간, "
        "관악아트홀을 1.5시간 동안 둘러본 뒤 다시 인게스트하우스로 돌아옵니다. "
        "자동차 총 주행거리가 가장 짧은 방문 순서는 다음 중 무엇인가요?"
    )
    options = [
        "주어진 지도 정보로는 알 수 없음",
        "백련산꿈마을숲정이 → 한남매봉공원산책길 → 관악아트홀",
        "관악아트홀 → 한남매봉공원산책길 → 백련산꿈마을숲정이",
        "한남매봉공원산책길 → 관악아트홀 → 백련산꿈마을숲정이",
    ]
    steps = [
        {
            "id": "all_locations",
            "operator": "batch_geocode",
            "arguments": {
                "place_names": [
                    "인게스트하우스",
                    "한남매봉공원산책길",
                    "백련산꿈마을숲정",
                    "관악아트홀",
                ]
            },
            "depends_on": [],
            "output_type": "object",
            "role": "extent",
        }
    ]
    grounded = _ground_graph_literals(steps, question, options, extract_facts({}, question))
    assert grounded[0]["arguments"]["place_names"] == [
        "인게스트하우스",
        "한남매봉공원산책길",
        "백련산꿈마을숲정이",
        "관악아트홀",
    ]


def test_a_stop_that_resembles_nothing_the_question_states_is_left_alone() -> None:
    """The restoration only ever puts back a name the question wrote.

    A trip's stays now feed the same list the anchor does, so the negative space is worth pinning:
    a name that is not a short form of any stated stop stays exactly as the planner wrote it.
    """

    from src.agent.spatial import _ground_graph_literals

    question = "A타워에서 출발해 B공원을 1시간 둘러본 뒤 다시 A타워로 돌아옵니다. 총 주행거리는?"
    steps = [
        {
            "id": "geo",
            "operator": "batch_geocode",
            "arguments": {"place_names": ["A타워", "B공원", "전혀다른어떤장소이름"]},
            "depends_on": [],
            "output_type": "object",
            "role": "extent",
        }
    ]
    grounded = _ground_graph_literals(
        steps,
        question,
        ["약 1km", "약 2km"],
        extract_facts({}, question),
    )
    assert grounded[0]["arguments"]["place_names"] == [
        "A타워",
        "B공원",
        "전혀다른어떤장소이름",
    ]


def test_a_field_projected_off_a_batch_list_resolves_to_the_list_and_runs() -> None:
    """The proof behind letting that spelling through: the plan the validator refused executes.

    `place_ids` is not a field of a `batch_geocode` list, so `_resolve_references` degrades the
    projection to the whole list -- the same value the legal `$geo` names -- and
    `batch_place_details` reads each row's existing id. Nothing is searched and nothing is
    invented; the refusal was costing the question a plan the executor answers.
    """

    from src.agent.spatial import _resolve_references

    registry = ToolRegistry(FakeProvider())
    rows = registry.invoke("batch_geocode", {"place_names": ["경복궁", "남산타워"]}).output
    assert _resolve_references("$geo.place_ids", {"geo": rows}) == rows

    details = registry.invoke(
        "batch_place_details", {"place_ids": _resolve_references("$geo.place_ids", {"geo": rows})}
    ).output
    assert [place["place_id"] for place in details] == ["경복궁", "남산타워"]


def test_grounding_keeps_each_geocode_step_its_own_place_when_there_are_no_options() -> None:
    """`len(names) == len(options) + 1` is the structural proof that a batch is
    [anchor, *option texts]. MCQ matching left the reasoning core, so grounding is now always
    handed an empty option list, and the test degenerated into "this batch names one place" --
    which overwrote that one name with the anchor. Every three-place question then measured the
    anchor against itself: 40 of 95 recorded graphs collapsed to a single batch, and the
    distance-difference family answered 0.0 km with every stage reporting success.
    """

    from src.agent.spatial import _ground_graph_literals, extract_facts

    question = (
        "토전김익영도자예술에서 지민숲까지의 직선거리와 "
        "토전김익영도자예술에서 CGV 여의도까지의 직선거리는 얼마나 차이가 나나요?"
    )
    analysis = {
        "concepts": [
            {"id": "anchor", "text": "토전김익영도자예술", "concept_type": "location"},
            {"id": "target1", "text": "지민숲", "concept_type": "location"},
            {"id": "target2", "text": "CGV 여의도", "concept_type": "location"},
        ]
    }
    graph = [
        {
            "id": f"resolve_{index}",
            "operator": "batch_geocode",
            "arguments": {"place_names": [name]},
        }
        for index, name in enumerate(
            ("토전김익영도자예술의 위치", "지민숲의 위치", "CGV 여의도의 위치")
        )
    ]
    grounded = _ground_graph_literals(graph, question, [], extract_facts(analysis, question))
    resolved = [step["arguments"]["place_names"] for step in grounded]
    # Each step keeps its own place, and the planner's description is repaired to the literal.
    assert resolved == [["토전김익영도자예술"], ["지민숲"], ["CGV 여의도"]]


def test_a_descriptive_tail_never_eats_what_tells_two_candidates_apart() -> None:
    """The first spelling of the repair bound any stated literal contained in the name, which
    turned `후보1` into `후보` -- the character that distinguishes the candidates."""

    from src.agent.spatial import _verbatim_name

    question = "기준점에서 후보까지의 직선거리는?"
    assert _verbatim_name("후보1", question, [], ["후보"]) == "후보1"
    assert _verbatim_name("후보의 위치", question, [], ["후보"]) == "후보"


def test_a_decorated_place_name_is_repaired_to_the_literal_the_question_wrote() -> None:
    """The decorations cannot be enumerated -- `(위치 정보)`, `(located)`, `Resolved location of`
    -- so the rule is the leftover, not the decoration: take the longest stated place the text
    contains and accept it only when what is left could not name a place.
    """

    from src.agent.spatial import _verbatim_name

    question = "토전김익영도자예술에서 지민숲까지, CGV 여의도와 호암늘솔길도 봅니다."
    stated = ["토전김익영도자예술", "지민숲", "CGV 여의도", "여의도", "호암늘솔길"]
    assert _verbatim_name("Resolved location of 토전김익영도자예술", question, [], stated) == (
        "토전김익영도자예술"
    )
    assert _verbatim_name("호암늘솔길 (located)", question, [], stated) == "호암늘솔길"
    assert _verbatim_name("지민숲의 위치", question, [], stated) == "지민숲"
    # Longest, not unique: `여의도` is stated too, and trimming to it names a different place.
    assert _verbatim_name("CGV 여의도 (located)", question, [], stated) == "CGV 여의도"
    # A leftover holding a digit or a Hangul syllable can distinguish two places, so it is not a
    # decoration and the name is left exactly as written.
    assert _verbatim_name("후보1", "기준점에서 후보까지?", [], ["후보"]) == "후보1"


def test_a_name_the_planner_translated_is_left_alone_rather_than_guessed_at() -> None:
    """`Located Noiji Gallery` holds no literal the question wrote. Repairing it would mean
    guessing which place it meant, which is the least-bad match this module refuses to make."""

    from src.agent.spatial import _verbatim_name

    question = "노이지갤러리에서 가장 가까운 곳은?"
    assert _verbatim_name("Located Noiji Gallery", question, [], ["노이지갤러리"]) == (
        "Located Noiji Gallery"
    )


def test_a_clause_the_planner_copied_with_the_name_is_trimmed_to_the_place() -> None:
    """`삼성출판박물관을 경유해서 가는 경우` passes the "is it in the question" guard precisely
    because it is in the question, and it is still not a place: the geocoder found nothing and
    the whole question was lost as a `PlaceNotFoundError`."""

    from src.agent.spatial import _verbatim_name

    question = (
        "포유모텔에서 학동역 7호선까지 자동차로 갈 때, 삼성출판박물관을 경유해서 가는 경우는?"
    )
    stated = ["포유모텔", "학동역 7호선", "삼성출판박물관"]
    assert _verbatim_name("삼성출판박물관을 경유해서 가는 경우", question, [], stated) == (
        "삼성출판박물관"
    )
    assert _verbatim_name("포유모텔에서", question, [], stated) == "포유모텔"
    # The particle has to be followed by a space: `에` here begins a syllable, not an ending.
    assert _verbatim_name("강남역에스컬레이터", "강남역에스컬레이터 근처?", [], ["강남역"]) == (
        "강남역에스컬레이터"
    )


def test_a_place_is_read_out_of_a_concept_text_that_carries_a_clause() -> None:
    """The Analysis stage copies a clause as often as a name, and then the name itself is in no
    vocabulary anything can repair against: `삼성출판박물관을 경유해서 가는 경우` was geocoded as
    written and the whole question was lost as a `PlaceNotFoundError`."""

    from src.agent.spatial import _verbatim_concept_texts

    question = "포유모텔에서 학동역 7호선까지, 삼성출판박물관을 경유해서 가는 경우의 주행거리는?"
    literals = _verbatim_concept_texts(
        {"concepts": [{"text": "삼성출판박물관을 경유해서 가는 경우"}]}, question
    )
    assert "삼성출판박물관" in literals


def test_a_name_that_merely_ends_in_a_particle_syllable_is_left_whole() -> None:
    """`중계동학원가` is a place, not `중계동학원` plus a subject marker. A clause has to
    actually follow before the ending counts as grammar."""

    from src.agent.spatial import _verbatim_concept_texts

    question = "중계동학원가에서 신설동역 1호선까지"
    literals = _verbatim_concept_texts({"concepts": [{"text": "중계동학원가"}]}, question)
    assert literals == ("중계동학원가",)


def test_a_name_the_question_states_verbatim_is_never_repaired() -> None:
    """Adding clause-carrying concept texts to the repair vocabulary made this concrete:
    `삼성출판박물관` is a substring of `삼성출판박물관을 경유해서 가는 경우`, so the shortened-name
    branch "restored" the correct name into the clause and the geocoder found nothing."""

    from src.agent.spatial import _verbatim_name

    question = "포유모텔에서 학동역 7호선까지, 삼성출판박물관을 경유해서 가는 경우의 주행거리는?"
    stated = ["삼성출판박물관", "삼성출판박물관을 경유해서 가는 경우"]
    assert _verbatim_name("삼성출판박물관", question, [], stated) == "삼성출판박물관"
