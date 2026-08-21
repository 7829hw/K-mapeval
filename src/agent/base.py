from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from src.tools import ToolRegistry


class AgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_type: str
    predicted_intent: str | None = None
    predicted_answer: int | None = None
    response: str = ""
    tool_calls: int = Field(default=0, ge=0)
    api_calls: int = Field(default=0, ge=0)
    cache_hits: int = Field(default=0, ge=0)
    cache_misses: int = Field(default=0, ge=0)
    reasoning_steps: int = Field(default=0, ge=0)
    # What the question cost at the endpoint, summed over every completion the agent asked for.
    # `reasoning_tokens` stays None when the server does not split the completion; see
    # `src.llm.TokenUsage`. `reasoning_chars` is the thinking text that came back and never
    # reached the parser, which is measurable on every deployment.
    llm_calls: int = Field(default=0, ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    reasoning_chars: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0)
    failure_type: str | None = None
    failure_message: str | None = None
    trace: list[dict[str, Any]] = Field(default_factory=list)


class LiveTrace(list):
    """A trace list that hands each entry to a listener the moment it is appended.

    A question can run for minutes, and its log used to appear all at once when the agent came
    back -- so a run in flight showed nothing at all about the question it was working on, and a
    question that never returned left an empty file. Appending is the only mutation either agent
    performs on its trace, so this is the whole of it.
    """

    def __init__(self, sink: Callable[[dict[str, Any]], None] | None = None) -> None:
        super().__init__()
        self._sink = sink

    def append(self, entry: dict[str, Any]) -> None:
        super().append(entry)
        if self._sink is not None:
            self._sink(entry)


class BenchmarkAgent(ABC):
    agent_type: str
    # Every benchmarked agent owns one; a test double answering from a script owns none.
    tools: ToolRegistry | None = None
    # Where trace entries go as they happen. Set by whoever is watching -- the evaluator, for the
    # length of one question -- and left None when nobody is. Worker-owned, like the agent itself.
    trace_sink: Callable[[dict[str, Any]], None] | None = None

    def new_trace(self) -> list[dict[str, Any]]:
        """The trace to build this question's answer in."""

        return LiveTrace(self.trace_sink)

    @abstractmethod
    def answer(self, question: str, options: list[str]) -> AgentResult: ...


def format_question(question: str, options: list[str]) -> str:
    rendered = "\n".join(f"Option {index}: {option}" for index, option in enumerate(options))
    return (
        f"Question:\n{question}\n\nCandidate options:\n{rendered}\n\n"
        "Option numbers are 0-based. Return the final selection exactly as "
        '"^^Option_Number^^", for example ^^1^^.'
    )


def find_provider_failure(value: Any) -> str | None:
    """Find a serialized provider failure in a nested trace without using tracebacks."""

    provider_error_names = (
        "ProviderError",
        "ProviderAuthError",
        "ProviderRateLimitError",
        "ProviderTimeoutError",
        "PlaceNotFoundError",
        "RouteNotFoundError",
        "UnsupportedTravelModeError",
    )
    if isinstance(value, str) and value.startswith(provider_error_names):
        return value
    if isinstance(value, dict):
        for item in value.values():
            found = find_provider_failure(item)
            if found:
                return found
    if isinstance(value, list):
        for item in value:
            found = find_provider_failure(item)
            if found:
                return found
    return None
