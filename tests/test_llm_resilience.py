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
from src.llm import LLMUnavailableError, OpenAIChatClient
from src.tools import ToolRegistry
from tests.test_tools_and_agents import FakeProvider


def build_client(**overrides: Any) -> OpenAIChatClient:
    settings = Settings(
        llm_api_key="test-key",
        llm_model="test-model",
        llm_base_url="http://localhost:1/v1",
        llm_retry_backoff_seconds=0.001,
        **overrides,
    )
    return OpenAIChatClient(settings)


def status_error(code: int) -> APIStatusError:
    request = httpx.Request("POST", "http://localhost:1/v1/chat/completions")
    response = httpx.Response(code, request=request, json={"error": {"message": "boom"}})
    error_type = BadRequestError if code == 400 else APIStatusError
    return error_type("boom", response=response, body=None)


class ScriptedCompletions:
    """Stands in for `client.chat.completions`, replaying a queue of errors then a response."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.calls = 0

    def create(self, **_: Any) -> Any:
        self.calls += 1
        outcome = self.script.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def install(client: OpenAIChatClient, script: list[Any]) -> ScriptedCompletions:
    completions = ScriptedCompletions(script)
    client._client.chat.completions = completions  # type: ignore[assignment]
    return completions


def completion(content: str) -> Any:
    message = type("Message", (), {"content": content, "tool_calls": None})()
    choice = type("Choice", (), {"message": message})()
    return type("Completion", (), {"choices": [choice]})()


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


def test_bad_credentials_fail_immediately_because_auth_is_never_transient() -> None:
    client = build_client()
    script = install(client, [status_error(401)])

    with pytest.raises(LLMUnavailableError, match="LLM_API_KEY"):
        client.chat([{"role": "user", "content": "q"}])
    assert script.calls == 1


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


def test_a_downed_endpoint_aborts_the_batch_and_marks_the_report_invalid(tmp_path, capsys) -> None:
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
        abort_after_llm_failures=3,
    ).run()

    failure_types = report.statistics["performance"]["failure_types"]
    assert failure_types["llm_unavailable"] == 3
    assert failure_types["run_aborted"] == 7
    assert report.statistics["run_validity"] == {
        "valid": False,
        "llm_unavailable_count": 10,
        "aborted": True,
    }
    assert report.statistics["overall_answer_accuracy"]["accuracy"] == 0.0
    output = capsys.readouterr().out
    assert "Aborting run" in output
    assert "INVALID RUN" in output


def test_a_healthy_run_stays_valid_and_records_the_agent_it_measured(tmp_path) -> None:
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
        abort_after_llm_failures=3,
    ).run()

    assert report.statistics["run_validity"]["valid"] is True
    assert report.metadata["agent_type"] == "spatial_agent"
    assert report.metadata["llm_model"] == "test-model"
