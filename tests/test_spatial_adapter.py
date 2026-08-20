"""The Spatial-Agent adapter changes nothing about the agent, and must not.

`src/spatial_agent/` is `ecerybao/Spatial-Agent@6876bba` with the Google Maps client swapped
for the Kakao one. What is tested here is the seam: that the vendored graph still builds, that
it reads the client the harness gave it, and that its result reaches the Evaluator without the
index or the failure kind being altered on the way.
"""

from __future__ import annotations

import httpx
import pytest

from src.agent.spatial import SpatialAgent, classify_error
from src.kakao_maps import KakaoMapsClient


@pytest.fixture(autouse=True)
def _upstream_env(monkeypatch):
    # The vendored agent reads these at construction time; `main.export_upstream_environment`
    # is what sets them for a real run.
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("KAKAO_REST_API_KEY", "test")


def build_client() -> KakaoMapsClient:
    return KakaoMapsClient(
        "test",
        client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={}))),
        cache_path="",
    )


def test_the_vendored_graph_still_builds() -> None:
    """A smoke test with teeth: `SpatialAgent.__init__` compiles the whole langgraph workflow
    and constructs every node, so an import or a signature broken by the API swap fails here
    rather than a hundred questions into a run."""

    agent = SpatialAgent(build_client())

    assert agent.agent_type == "spatial"
    assert agent._agent.workflow is not None


def test_the_agent_reads_the_client_the_harness_gave_it() -> None:
    """Upstream constructs its own client from the key. Swapping in the caller's is what makes
    the per-question API and cache counters readable, and what keeps two concurrent workers
    from sharing one HTTP client."""

    client = build_client()
    agent = SpatialAgent(client)

    assert agent._agent.kakao_client is client
    assert agent._agent.transformation_executor.client is client


def test_the_zero_based_option_passes_through_unchanged() -> None:
    """Upstream's own prompt says "predicted_option is zero-based" (`spatial_agent.py` line
    35), which is this repository's convention too — so unlike the ReAct baseline there is no
    conversion, and adding one would silently shift every answer by an option."""

    agent = SpatialAgent(build_client())
    agent._agent.process_question = lambda *a, **k: {
        "answer": "두 번째 옵션입니다",
        "intent": "nearby",
        "predicted_option": 1,
        "concept_flow": [{"operator": "geocode"}, {"operator": "nearest"}],
        "evaluation": {"confidence": 0.9},
        "error": None,
    }

    result = agent.answer("q", ["A", "B", "C", "D"])

    assert result.predicted_answer == 1
    assert result.predicted_intent == "nearby"
    assert result.tool_calls == 2
    assert result.failure_type is None


def test_an_intent_outside_upstreams_four_is_not_reported_as_one() -> None:
    """Upstream routes to nearby/routing/trip/poi. This repository's benchmarks classify eight
    ways, so an intent the architecture has no vocabulary for is absent, not wrong."""

    agent = SpatialAgent(build_client())
    agent._agent.process_question = lambda *a, **k: {
        "intent": "something_else",
        "predicted_option": 0,
        "answer": "",
    }

    assert agent.answer("q", ["A", "B"]).predicted_intent is None


def test_an_option_outside_the_range_is_not_recorded_as_an_answer() -> None:
    agent = SpatialAgent(build_client())
    agent._agent.process_question = lambda *a, **k: {"predicted_option": 7, "answer": ""}

    result = agent.answer("q", ["A", "B"])

    assert result.predicted_answer is None
    assert result.failure_type == "answer_parse_failure"


def test_gold_never_reaches_the_agent() -> None:
    """`process_question` accepts `correct_answer` for its own logging. Passing it would make
    the benchmark unmeasurable, so the adapter sends only the question and its options."""

    agent = SpatialAgent(build_client())
    seen: dict = {}

    def record(question, options=None, correct_answer=None, question_id=None):
        seen.update(
            {
                "question": question,
                "options": options,
                "correct_answer": correct_answer,
                "question_id": question_id,
            }
        )
        return {"predicted_option": 0, "answer": ""}

    agent._agent.process_question = record
    agent.answer("q", ["A", "B"])

    assert seen == {
        "question": "q",
        "options": ["A", "B"],
        "correct_answer": None,
        "question_id": None,
    }


# ------------------------------------------------------------------ failures


def test_a_kakao_failure_is_a_provider_failure_the_harness_can_retry() -> None:
    """`process_question` catches every exception and reports `str(e)`, which drops the class
    name. `is_transient_failure` identifies a retryable failure by the name the message starts
    with, so the marker is put back."""

    failure_type, message = classify_error("KakaoRateLimitError: Kakao API rate limit exceeded")

    assert failure_type == "provider_failure"
    assert message.startswith("KakaoRateLimitError")


def test_a_timeout_worded_by_httpx_is_still_recognised_as_transient() -> None:
    failure_type, message = classify_error("ReadTimeout: request timed out")

    assert failure_type == "provider_failure"
    assert message.startswith("KakaoTimeoutError")


def test_a_reasoning_failure_is_never_dressed_up_as_a_provider_failure() -> None:
    """A wrong plan is the result the architecture earned. Retrying it measures luck."""

    failure_type, message = classify_error("계획 실행 실패: unknown operator 'teleport'")

    assert failure_type == "agent_reasoning_failure"
    assert message == "계획 실행 실패: unknown operator 'teleport'"


def test_an_error_the_agent_reported_reaches_the_evaluator_as_transient_or_not() -> None:
    from src.evaluator import is_transient_failure

    agent = SpatialAgent(build_client())
    agent._agent.process_question = lambda *a, **k: {
        "error": "KakaoTimeoutError: Kakao API request timed out",
        "answer": "",
    }

    result = agent.answer("q", ["A", "B"])

    assert result.failure_type == "provider_failure"
    assert is_transient_failure(
        {"failure_type": "provider_failure", "error": result.failure_message}
    )
