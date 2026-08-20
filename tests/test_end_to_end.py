"""One question, all the way through, with the LLM and Kakao stubbed.

The pieces are tested separately elsewhere. What this asserts is that they are actually wired
together: the Evaluator reaches the vendored agent, the vendored tools reach the Kakao client,
and the counters and the report come back describing what happened. A break in the seam
between them would otherwise surface only on a live run.
"""

from __future__ import annotations

import json
from contextlib import contextmanager

import httpx
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.agent.react import ReactAgent
from src.dataset import BenchmarkItem
from src.evaluator import Evaluator
from src.kakao_maps import KakaoMapsClient
from tests.test_kakao_maps import ROUTE, STARBUCKS, TWOSOME


def structured_chat(action: str, action_input) -> str:
    """The wire format `STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION` actually parses."""

    return "Action:\n```json\n" + json.dumps(
        {"action": action, "action_input": action_input}, ensure_ascii=False
    ) + "\n```"


# One tool call, then a final answer.
SCRIPT = [
    structured_chat("PlaceSearch", {"placeName": "스타벅스 강남점"}),
    structured_chat("Final Answer", "가장 가까운 곳은 투썸플레이스입니다. ^^2^^"),
]


def kakao_transport(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if "search/keyword" in url:
        return httpx.Response(200, json={"documents": [STARBUCKS, TWOSOME]})
    if "directions" in url:
        return httpx.Response(200, json=ROUTE)
    return httpx.Response(200, json={"documents": [], "meta": {"is_end": True}})


def test_a_question_runs_from_the_evaluator_to_kakao_and_back(tmp_path) -> None:
    item = BenchmarkItem(
        id="smoke_000",
        question="스타벅스 강남점에서 가장 가까운 카페는?",
        options=["할리스", "투썸플레이스 역삼점", "이디야", "커피빈"],
        answer=1,
        classification="nearby",
    )

    @contextmanager
    def session():
        client = KakaoMapsClient(
            "test",
            client=httpx.Client(transport=httpx.MockTransport(kakao_transport)),
            cache_path="",
        )
        try:
            yield ReactAgent(FakeListChatModel(responses=SCRIPT), client)
        finally:
            client.close()

    report = Evaluator(
        None,
        [item],
        agent_factory=session,
        max_workers=1,
        output_dir=tmp_path / "reports",
        dataset_path="tests",
        agent_type="react",
    ).run()

    row = report.results[0]
    assert row["id"] == "smoke_000"
    # `^^2^^` is upstream's one-based selection; the harness records the 0-based index.
    assert row["predicted_option"] == 1
    assert row["answer_correct"] is True
    assert row["failure_type"] is None
    # The tool actually reached Kakao, rather than the model answering from the question.
    assert row["tool_calls"] == 1
    assert row["api_calls"] >= 1
    assert report.statistics["overall_answer_accuracy"]["accuracy"] == 1.0


def test_the_report_records_which_agent_and_which_upstream_answered(tmp_path) -> None:
    """A shelf of reports whose accuracies differ for reasons no field records is not a
    result. Every run says which architecture, which model and which upstream produced it."""

    item = BenchmarkItem(
        id="smoke_001", question="q", options=["A", "B"], answer=0, classification="poi"
    )

    @contextmanager
    def session():
        client = KakaoMapsClient(
            "test",
            client=httpx.Client(transport=httpx.MockTransport(kakao_transport)),
            cache_path="",
        )
        try:
            yield ReactAgent(
                FakeListChatModel(responses=[structured_chat("Final Answer", "^^1^^")]),
                client,
            )
        finally:
            client.close()

    report = Evaluator(
        None,
        [item],
        agent_factory=session,
        max_workers=1,
        output_dir=tmp_path / "reports",
        agent_type="react",
        llm_profile={
            "llm_model": "test-model",
            "provider": "kakao",
            "upstream_mapeval_api": "MapEval/MapEval-API@35d481a",
        },
    ).run()

    assert report.metadata["agent_type"] == "react"
    assert report.metadata["provider"] == "kakao"
    assert report.metadata["upstream_mapeval_api"] == "MapEval/MapEval-API@35d481a"
