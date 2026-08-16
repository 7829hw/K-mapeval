from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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
    latency_ms: float = Field(default=0.0, ge=0)
    failure_type: str | None = None
    failure_message: str | None = None
    trace: list[dict[str, Any]] = Field(default_factory=list)


class BenchmarkAgent(ABC):
    agent_type: str

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
