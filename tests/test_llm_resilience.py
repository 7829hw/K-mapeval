from __future__ import annotations

from typing import Any

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, BadRequestError

from src.agent import ReactAgent, SpatialAgent
from src.agent.base import AgentResult, BenchmarkAgent
from src.config import Settings
from src.dataset import BenchmarkItem
from src.evaluator import Evaluator
from src.llm import (
    LLMContextOverflowError,
    LLMOutputTruncatedError,
    LLMUnavailableError,
    OpenAIChatClient,
    TokenUsage,
)
from src.tools import ToolRegistry
from tests.test_tools_and_agents import FakeProvider


def build_client(**overrides: Any) -> OpenAIChatClient:
    settings = Settings(
        # Never the developer's `.env`: a local setting there once made a client test fail on
        # one machine and pass on every other. A test asserts what the code does, not what the
        # machine running it is configured to do.
        _env_file=None,
        llm_api_key="test-key",
        llm_model="test-model",
        llm_base_url="http://localhost:1/v1",
        llm_retry_backoff_seconds=0.001,
        **overrides,
    )
    return OpenAIChatClient(settings)


def status_error(code: int, message: str = "boom") -> APIStatusError:
    request = httpx.Request("POST", "http://localhost:1/v1/chat/completions")
    response = httpx.Response(code, request=request, json={"error": {"message": message}})
    error_type = BadRequestError if code == 400 else APIStatusError
    return error_type(message, response=response, body=None)


class ScriptedCompletions:
    """Stands in for `client.chat.completions`, replaying a queue of errors then a response."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.calls = 0
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls += 1
        self.requests.append(kwargs)
        outcome = self.script.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def install(client: OpenAIChatClient, script: list[Any]) -> ScriptedCompletions:
    completions = ScriptedCompletions(script)
    client._client.chat.completions = completions  # type: ignore[assignment]
    return completions


def completion(content: str, *, finish_reason: str = "stop", **usage: Any) -> Any:
    message = type("Message", (), {"content": content, "tool_calls": None, **usage})()
    choice = type("Choice", (), {"message": message, "finish_reason": finish_reason})()
    return type("Completion", (), {"choices": [choice], "usage": None})()


def test_transient_endpoint_errors_are_retried_instead_of_failing_the_question() -> None:
    client = build_client()
    script = install(
        client,
        [
            status_error(502),
            status_error(503),
            APIConnectionError(request=httpx.Request("POST", "http://localhost:1/v1")),
            completion("^^1^^"),
        ],
    )

    assert client.chat([{"role": "user", "content": "q"}]).content == "^^1^^"
    assert script.calls == 4


def test_a_persistently_failing_endpoint_is_an_infrastructure_error_not_an_agent_error() -> None:
    client = build_client(llm_max_retries=2)
    script = install(client, [status_error(502)] * 3)

    with pytest.raises(LLMUnavailableError, match="after 3 attempts"):
        client.chat([{"role": "user", "content": "q"}])
    assert script.calls == 3


def test_a_model_not_found_is_retried_because_vllm_returns_it_while_reloading() -> None:
    client = build_client(llm_max_retries=2)
    script = install(client, [status_error(404), status_error(404), completion("^^0^^")])

    assert client.chat([{"role": "user", "content": "q"}]).content == "^^0^^"
    assert script.calls == 3


def test_a_model_that_never_comes_back_names_itself_in_the_failure() -> None:
    client = build_client(llm_max_retries=1)
    install(client, [status_error(404)] * 2)

    with pytest.raises(LLMUnavailableError, match="test-model"):
        client.chat([{"role": "user", "content": "q"}])


def test_a_rejected_request_is_waited_out_like_any_other_endpoint_answer() -> None:
    """One response cannot tell a wrong key from a proxy that has not loaded its config yet."""

    client = build_client(llm_max_retries=1)
    script = install(client, [status_error(401), completion("^^0^^")])

    assert client.chat([{"role": "user", "content": "q"}]).content == "^^0^^"
    assert script.calls == 2


def test_a_request_we_malformed_is_not_retried_and_stays_the_agent_s_problem() -> None:
    client = build_client()
    script = install(client, [status_error(400)])

    with pytest.raises(BadRequestError):
        client.chat([{"role": "user", "content": "q"}])
    assert script.calls == 1


class DownEndpointLLM:
    def chat(self, messages: list[dict[str, Any]], *, tools: Any = None) -> Any:
        raise LLMUnavailableError("LLM endpoint failed after 5 attempts: APIStatusError: 502")


@pytest.mark.parametrize("agent_name", ["react", "spatial"])
def test_both_agents_report_a_dead_endpoint_as_infrastructure_not_reasoning(
    agent_name: str,
) -> None:
    tools = ToolRegistry(FakeProvider())
    agent_class = ReactAgent if agent_name == "react" else SpatialAgent
    agent = agent_class(DownEndpointLLM(), tools, max_steps=3)

    result = agent.answer("질문", ["A", "B"])

    assert result.failure_type == "llm_unavailable"
    assert result.predicted_answer is None


class DownEndpointAgent(BenchmarkAgent):
    agent_type = "down"

    def answer(self, question: str, options: list[str]) -> AgentResult:
        return AgentResult(
            agent_type=self.agent_type,
            failure_type="llm_unavailable",
            failure_message="LLMUnavailableError: endpoint down",
        )


def test_a_downed_endpoint_costs_only_the_questions_it_actually_failed(tmp_path, capsys) -> None:
    """No batch-level verdict: every question is still asked, and the count is reported plainly."""

    items = [
        BenchmarkItem(
            id=f"q{index}",
            question="q",
            options=["x", "y"],
            answer=0,
            classification="poi",
        )
        for index in range(10)
    ]
    report = Evaluator(
        DownEndpointAgent(),
        items,
        output_dir=None,
        log_dir=tmp_path / "logs",
    ).run()

    performance = report.statistics["performance"]
    assert performance["failure_types"] == {"llm_unavailable": 10}
    assert performance["llm_unavailable_count"] == 10
    assert "run_validity" not in report.statistics
    assert len(report.results) == 10
    assert all(row["failure_type"] == "llm_unavailable" for row in report.results)
    assert "Never answered by the LLM endpoint: 10/10" in capsys.readouterr().out


def test_a_healthy_run_records_the_agent_it_measured(tmp_path) -> None:
    class HealthyAgent(BenchmarkAgent):
        agent_type = "healthy"

        def answer(self, question: str, options: list[str]) -> AgentResult:
            return AgentResult(
                agent_type=self.agent_type,
                predicted_intent="poi",
                predicted_answer=0,
            )

    items = [
        BenchmarkItem(id="a", question="q", options=["x", "y"], answer=0, classification="poi")
    ]
    report = Evaluator(
        HealthyAgent(),
        items,
        output_dir=None,
        log_dir=tmp_path / "logs",
        agent_type="spatial_agent",
        llm_profile={"llm_model": "test-model", "llm_base_url": "http://localhost:1/v1"},
    ).run()

    assert report.statistics["performance"]["llm_unavailable_count"] == 0
    assert report.metadata["agent_type"] == "spatial_agent"
    assert report.metadata["llm_model"] == "test-model"


class FlakyAgent(BenchmarkAgent):
    """Fails the first `failures` attempts on the endpoint, then answers."""

    agent_type = "flaky"

    def __init__(self, failures: int) -> None:
        self.remaining = failures
        self.attempts = 0

    def answer(self, question: str, options: list[str]) -> AgentResult:
        self.attempts += 1
        if self.remaining > 0:
            self.remaining -= 1
            return AgentResult(
                agent_type=self.agent_type,
                failure_type="llm_unavailable",
                failure_message="LLMUnavailableError: endpoint down",
            )
        return AgentResult(
            agent_type=self.agent_type,
            predicted_intent="poi",
            predicted_answer=0,
        )


def one_question() -> list[BenchmarkItem]:
    return [
        BenchmarkItem(id="a", question="q", options=["x", "y"], answer=0, classification="poi")
    ]


def test_a_question_the_endpoint_failed_is_asked_again(tmp_path) -> None:
    agent = FlakyAgent(failures=1)

    report = Evaluator(
        agent,
        one_question(),
        output_dir=None,
        log_dir=tmp_path / "logs",
        question_retries=2,
        question_retry_backoff_seconds=0.001,
    ).run()

    assert agent.attempts == 2
    assert report.results[0]["answer_correct"] is True
    assert report.results[0]["attempts"] == 2
    assert report.results[0]["failure_type"] is None
    assert report.statistics["performance"]["retried_question_count"] == 1
    assert report.statistics["performance"]["retry_recovered_ids"] == ["a"]


def test_retries_stop_once_the_attempts_are_spent(tmp_path) -> None:
    agent = FlakyAgent(failures=99)

    report = Evaluator(
        agent,
        one_question(),
        output_dir=None,
        log_dir=tmp_path / "logs",
        question_retries=2,
        question_retry_backoff_seconds=0.001,
    ).run()

    assert agent.attempts == 3
    assert report.results[0]["attempts"] == 3
    assert report.results[0]["failure_type"] == "llm_unavailable"
    assert report.statistics["performance"]["llm_unavailable_count"] == 1
    assert report.statistics["performance"]["retry_recovered_ids"] == []


def test_a_wrong_answer_is_never_retried(tmp_path) -> None:
    """Re-rolling the agent's own reasoning would measure luck, not architecture."""

    class WrongAgent(BenchmarkAgent):
        agent_type = "wrong"

        def __init__(self) -> None:
            self.attempts = 0

        def answer(self, question: str, options: list[str]) -> AgentResult:
            self.attempts += 1
            return AgentResult(
                agent_type=self.agent_type,
                predicted_intent="poi",
                predicted_answer=1,
            )

    agent = WrongAgent()
    report = Evaluator(
        agent,
        one_question(),
        output_dir=None,
        log_dir=tmp_path / "logs",
        question_retries=3,
        question_retry_backoff_seconds=0.001,
    ).run()

    assert agent.attempts == 1
    assert report.results[0]["answer_correct"] is False


def test_a_missing_place_is_evidence_and_a_timed_out_provider_is_not(tmp_path) -> None:
    from src.evaluator import is_transient_failure

    assert is_transient_failure(
        {"failure_type": "provider_failure", "error": "ProviderTimeoutError: Kakao timed out"}
    )
    assert is_transient_failure(
        {"failure_type": "provider_failure", "error": "ProviderRateLimitError: slow down"}
    )
    assert not is_transient_failure(
        {
            "failure_type": "provider_failure",
            "error": "PlaceNotFoundError: No place matched '만화시장'",
        }
    )
    assert not is_transient_failure(
        {"failure_type": "provider_failure", "error": "ProviderAuthError: bad key"}
    )
    assert not is_transient_failure({"failure_type": "agent_reasoning_failure", "error": "boom"})


def test_a_transient_provider_failure_is_asked_again(tmp_path) -> None:
    class RateLimitedAgent(BenchmarkAgent):
        agent_type = "limited"

        def __init__(self) -> None:
            self.attempts = 0

        def answer(self, question: str, options: list[str]) -> AgentResult:
            self.attempts += 1
            if self.attempts == 1:
                return AgentResult(
                    agent_type=self.agent_type,
                    failure_type="provider_failure",
                    failure_message="ProviderRateLimitError: Kakao rate limit",
                )
            return AgentResult(
                agent_type=self.agent_type,
                predicted_intent="poi",
                predicted_answer=0,
            )

    agent = RateLimitedAgent()
    report = Evaluator(
        agent,
        one_question(),
        output_dir=None,
        log_dir=tmp_path / "logs",
        question_retries=1,
        question_retry_backoff_seconds=0.001,
    ).run()

    assert agent.attempts == 2
    assert report.results[0]["answer_correct"] is True


def test_every_request_is_decoded_greedily_and_carries_no_ceiling_of_ours() -> None:
    """Temperature is ours to send; the output ceiling is the serving side's to enforce.

    `max_tokens` is deliberately absent: the vLLM deployment governs it, and both upstreams
    likewise construct their clients without one.
    """

    client = build_client()
    script = install(client, [completion("^^1^^")])

    client.chat([{"role": "user", "content": "q"}])

    assert script.requests[0]["temperature"] == 0.0
    assert "max_tokens" not in script.requests[0]


def test_a_completion_the_ceiling_cut_off_is_its_own_failure_not_a_bad_answer() -> None:
    """`finish_reason="length"` means the answer was never written, not that it was wrong.

    Left to flow, the agent would parse an empty or half-finished message and record an
    `answer_parse_failure`, which reads in a report as the architecture failing to answer.
    """

    client = build_client()
    install(client, [completion("", finish_reason="length", reasoning="thinking and thinking")])

    with pytest.raises(LLMOutputTruncatedError) as caught:
        client.chat([{"role": "user", "content": "q"}])

    assert "output limit" in str(caught.value)
    # The tokens it burned before the cut still belong in the question's cost.
    assert caught.value.usage.reasoning_chars == len("thinking and thinking")


class CeilingLLM:
    """An endpoint whose every completion runs into its own output limit."""

    def chat(self, messages: list[dict[str, Any]], *, tools: Any = None) -> Any:
        raise LLMOutputTruncatedError(
            "The completion was cut off at the endpoint output limit after 64 tokens",
            TokenUsage(completion_tokens=64, total_tokens=64, reasoning_chars=200),
        )


@pytest.mark.parametrize("agent_name", ["react", "spatial"])
def test_both_agents_blame_the_ceiling_rather_than_their_own_reasoning(agent_name: str) -> None:
    agent_class = ReactAgent if agent_name == "react" else SpatialAgent
    agent = agent_class(CeilingLLM(), ToolRegistry(FakeProvider()), max_steps=3)

    result = agent.answer("질문", ["A", "B"])

    assert result.failure_type == "llm_output_truncated"
    assert result.predicted_answer is None
    # The truncated call still cost tokens, and the question still has to report them.
    assert result.completion_tokens == 64
    assert result.reasoning_chars == 200


class TruncatedAgent(BenchmarkAgent):
    agent_type = "truncated"

    def __init__(self) -> None:
        self.calls = 0

    def answer(self, question: str, options: list[str]) -> AgentResult:
        self.calls += 1
        raise LLMOutputTruncatedError("cut off at the output limit", TokenUsage())


def test_a_question_the_ceiling_cut_off_is_counted_and_not_asked_again(tmp_path, capsys) -> None:
    """Retrying is pointless: the ceiling is a setting, so the second attempt hits the same one."""

    agent = TruncatedAgent()
    items = [
        BenchmarkItem(id="a", question="q", options=["x", "y"], answer=0, classification="poi")
    ]
    report = Evaluator(
        agent,
        items,
        output_dir=None,
        log_dir=tmp_path / "logs",
        question_retries=2,
        question_retry_backoff_seconds=0.001,
    ).run()

    assert agent.calls == 1
    assert report.results[0]["failure_type"] == "llm_output_truncated"
    assert report.statistics["performance"]["llm_output_truncated_count"] == 1
    assert "output limit" in capsys.readouterr().out


def test_a_prompt_too_long_for_the_window_is_its_own_failure_not_a_malformed_request() -> None:
    """The endpoint says 400 for both, and only one of them is the agent's mistake.

    A ReAct trace and a Spatial-Agent execution log both grow with the question; when one grows
    past the window the model never sees it, which is not the same finding as an agent that
    reasoned badly.
    """

    client = build_client()
    script = install(
        client,
        [
            status_error(
                400,
                "This model's maximum context length is 65536 tokens. However, you requested 0 "
                "output tokens and your prompt contains at least 65537 input tokens",
            )
        ],
    )

    with pytest.raises(LLMContextOverflowError) as caught:
        client.chat([{"role": "user", "content": "q"}])

    assert "context window" in str(caught.value)
    # The same prompt is the same length next time, so it is asked exactly once.
    assert script.calls == 1


class OverflowingLLM:
    def chat(self, messages: list[dict[str, Any]], *, tools: Any = None) -> Any:
        raise LLMContextOverflowError("The prompt was longer than the model's context window")


@pytest.mark.parametrize("agent_name", ["react", "spatial"])
def test_both_agents_blame_the_window_rather_than_their_own_reasoning(agent_name: str) -> None:
    agent_class = ReactAgent if agent_name == "react" else SpatialAgent
    agent = agent_class(OverflowingLLM(), ToolRegistry(FakeProvider()), max_steps=3)

    result = agent.answer("질문", ["A", "B"])

    assert result.failure_type == "llm_context_overflow"
    assert result.predicted_answer is None
