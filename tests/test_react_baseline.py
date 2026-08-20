"""The ReAct baseline adapter reproduces `Evaluator2.py`, and nothing more.

The agent is upstream's `initialize_agent` call; what this repository adds is the prompt it is
given, the parse of what comes back, and the conversion into the 0-based index the rest of the
harness speaks. Those three are exactly where a harness can accidentally change the task, so
they are pinned line-for-line against `MapEval/MapEval-API@35d481a`.
"""

from __future__ import annotations

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from openai import APIConnectionError, APIStatusError, APITimeoutError

from src.agent.react import (
    BASELINE_TOOL_TYPES,
    ReactAgent,
    build_prompt,
    parse_upstream_answer,
)
from src.kakao_maps import KakaoMapsClient
from src.llm import LLMUnavailableError
from src.mapeval_api.FormattedTools import (
    DirectionsTool,
    NearbyPlacesTool,
    PlaceDetailsTool,
    PlaceSearchTool,
    TravelTimeTool,
)

# ------------------------------------------------------------------ the prompt


def test_the_prompt_is_the_one_upstream_builds() -> None:
    """`Evaluator2.py` lines 58-76, character for character.

    The wording is what the model is actually asked, so it is part of the task and not a
    formatting detail: a rewrite here makes the baseline a function of this repository.
    """

    prompt = build_prompt("가장 가까운 편의점은?", ["GS25", "CU", "세븐일레븐", "이마트24"])

    assert prompt == (
        "가장 가까운 편의점은?"
        "Choose the answer from the following options (1/2/3/4). So, the output format will "
        'be "^^Option_Number^^". Choose the correct answer from the following options: '
        "Option1: GS25, Option2: CU, Option3: 세븐일레븐, Option4: 이마트24, "
    )


def test_the_prompt_numbers_options_from_one_as_upstream_does() -> None:
    prompt = build_prompt("q", ["A", "B"])

    assert "Option1: A" in prompt and "Option2: B" in prompt
    assert "Option0" not in prompt


def test_a_blank_option_ends_the_list_as_upstream_does() -> None:
    prompt = build_prompt("q", ["A", "B", "", "D"])

    assert "Option3" not in prompt


# ------------------------------------------------------------------ the parse


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("^^2^^", 2),
        ("The answer is ^^ Option 3 ^^.", 3),
        ("^^Option1^^", 1),
        ("답은 2번입니다", None),
        ("", None),
    ],
)
def test_the_parse_is_upstreams_caret_span_then_its_first_digit(output, expected) -> None:
    """`re.search(r"\\^\\^(.*?)\\^\\^", output)` (line 132) then `extract` (line 14)."""

    assert parse_upstream_answer(output) == expected


# ------------------------------------------------------------------ the agent


def build_agent(script: list[str]) -> ReactAgent:
    client = KakaoMapsClient(
        "k",
        client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={}))),
        cache_path="",
    )
    return ReactAgent(FakeListChatModel(responses=script), client)


def test_the_agent_is_given_the_five_tools_in_upstreams_order() -> None:
    assert BASELINE_TOOL_TYPES == (
        PlaceSearchTool,
        PlaceDetailsTool,
        NearbyPlacesTool,
        TravelTimeTool,
        DirectionsTool,
    )
    agent = build_agent(["irrelevant"])
    assert [tool.name for tool in agent.tools] == [
        "PlaceSearch",
        "PlaceDetails",
        "NearbyPlaces",
        "TravelTime",
        "Directions",
    ]


def test_a_one_based_selection_becomes_a_zero_based_answer() -> None:
    """Upstream compares against `item["answer"]["correct"] + 1`; this repository is 0-based
    everywhere. The prompt the model saw is still upstream's, and the conversion happens once,
    on the way out."""

    agent = build_agent(['{"action": "Final Answer", "action_input": "^^2^^"}'])

    result = agent.answer("q", ["A", "B", "C", "D"])

    assert result.predicted_answer == 1
    assert result.failure_type is None


def test_a_selection_outside_the_option_range_is_a_parse_failure_not_an_answer() -> None:
    """`^^0^^` is upstream's refusal channel, and these benchmarks have no index for it.

    Mapping it onto a real option would make a refusal score as a guess.
    """

    agent = build_agent(['{"action": "Final Answer", "action_input": "^^0^^"}'])

    result = agent.answer("q", ["A", "B"])

    assert result.predicted_answer is None
    assert result.failure_type == "answer_parse_failure"


def test_an_answer_with_no_selection_is_a_parse_failure() -> None:
    agent = build_agent(['{"action": "Final Answer", "action_input": "I think it is B"}'])

    result = agent.answer("q", ["A", "B"])

    assert result.predicted_answer is None
    assert result.failure_type == "answer_parse_failure"


def test_the_baseline_predicts_no_intent_because_it_has_no_classifier() -> None:
    """ReAct has no classification stage. Reporting an intent for it would measure the
    harness's guess, and the intent metric counts only questions an intent was predicted for."""

    agent = build_agent(['{"action": "Final Answer", "action_input": "^^1^^"}'])

    assert agent.answer("q", ["A", "B"]).predicted_intent is None


def test_a_dead_endpoint_is_infrastructure_not_reasoning() -> None:
    """An outage says nothing about an architecture, so the Evaluator has to be able to tell
    it apart and ask the question again."""

    class DeadModel(FakeListChatModel):
        def _call(self, *args, **kwargs):
            raise APIConnectionError(request=httpx.Request("POST", "http://x"))

    client = KakaoMapsClient(
        "k",
        client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={}))),
        cache_path="",
    )
    agent = ReactAgent(DeadModel(responses=["x"]), client)

    with pytest.raises(LLMUnavailableError):
        agent.answer("q", ["A", "B"])


def test_a_request_we_malformed_stays_the_agents_problem() -> None:
    """400/413/422 describe the request we sent, so repeating it only repeats the mistake."""

    class BadRequestModel(FakeListChatModel):
        def _call(self, *args, **kwargs):
            raise APIStatusError(
                "too long",
                response=httpx.Response(413, request=httpx.Request("POST", "http://x")),
                body=None,
            )

    client = KakaoMapsClient(
        "k",
        client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={}))),
        cache_path="",
    )
    agent = ReactAgent(BadRequestModel(responses=["x"]), client)

    with pytest.raises(APIStatusError):
        agent.answer("q", ["A", "B"])


def test_a_timeout_is_waited_out_as_an_outage_not_recorded_as_an_answer() -> None:
    class SlowModel(FakeListChatModel):
        def _call(self, *args, **kwargs):
            raise APITimeoutError(request=httpx.Request("POST", "http://x"))

    client = KakaoMapsClient(
        "k",
        client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={}))),
        cache_path="",
    )
    agent = ReactAgent(SlowModel(responses=["x"]), client)

    with pytest.raises(LLMUnavailableError):
        agent.answer("q", ["A", "B"])
