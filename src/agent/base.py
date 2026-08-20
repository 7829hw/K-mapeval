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
