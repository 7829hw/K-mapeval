from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import Any, Protocol

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from src.config import Settings

# HTTP statuses where the endpoint is telling us "not now" rather than "not ever". A self-hosted
# vLLM behind a reverse proxy answers 502/503 while it reloads a model, which is exactly the window
# a long benchmark is most likely to hit. 404 belongs here too: while vLLM swaps models it reports
# "the model does not exist" for a name it serves again a minute later, so a wrong LLM_MODEL and a
# reloading one are indistinguishable from a single response — retry, then say so in the message.
RETRYABLE_STATUS_CODES = frozenset({404, 408, 409, 425, 429, 500, 502, 503, 504})
# Authentication and permission are never transient, so these fail immediately.
CONFIGURATION_STATUS_CODES = frozenset({401, 403})


class LLMUnavailableError(RuntimeError):
    """The LLM endpoint could not serve a request.

    Distinct from an agent-reasoning failure: nothing about the question or the agent's plan caused
    it, so a batch full of these is an infrastructure outage and not a benchmark result.
    """


@dataclass(frozen=True)
class LLMToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    content: str
    tool_calls: tuple[LLMToolCall, ...] = ()

    def assistant_message(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": self.content or None}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in self.tool_calls
            ]
        return message


class ChatClient(Protocol):
    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse: ...


class OpenAIChatClient:
    """Small OpenAI-compatible adapter configured entirely through `.env`."""

    def __init__(self, settings: Settings) -> None:
        settings.require_llm()
        kwargs: dict[str, Any] = {
            "api_key": settings.llm_api_key,
            "timeout": settings.llm_timeout_seconds,
            # Retries are handled here so the backoff and the failure classification stay visible
            # to the benchmark instead of being swallowed inside the SDK.
            "max_retries": 0,
        }
        if settings.llm_base_url:
            kwargs["base_url"] = settings.llm_base_url
        self._client = OpenAI(**kwargs)
        self._model = settings.llm_model
        self._max_retries = settings.llm_max_retries
        self._backoff = settings.llm_retry_backoff_seconds

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        completion = self._request_with_retries(kwargs)
        message = completion.choices[0].message
        calls: list[LLMToolCall] = []
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {"_invalid_json": call.function.arguments}
            calls.append(LLMToolCall(call.id, call.function.name, arguments))
        return LLMResponse(message.content or "", tuple(calls))

    def _request_with_retries(self, kwargs: dict[str, Any]) -> Any:
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return self._client.chat.completions.create(**kwargs)
            except (APIConnectionError, APITimeoutError) as exc:
                last_error = exc
            except APIStatusError as exc:
                if exc.status_code in CONFIGURATION_STATUS_CODES:
                    raise LLMUnavailableError(
                        f"{type(exc).__name__}: {exc} "
                        f"(model={self._model!r}; check LLM_MODEL, LLM_BASE_URL and LLM_API_KEY)"
                    ) from exc
                if exc.status_code not in RETRYABLE_STATUS_CODES:
                    # 400/413/422 and friends are caused by what we sent — a prompt that grew too
                    # long, a malformed tool schema. Those belong to the agent, not the endpoint.
                    raise
                last_error = exc
            if attempt < self._max_retries:
                time.sleep(self._retry_delay(attempt))
        attempts = self._max_retries + 1
        raise LLMUnavailableError(
            f"LLM endpoint failed after {attempts} attempt{'s' if attempts > 1 else ''} "
            f"(model={self._model!r}): {type(last_error).__name__}: {last_error}"
        ) from last_error

    def _retry_delay(self, attempt: int) -> float:
        """Exponential backoff with jitter, so concurrent workers do not retry in lockstep."""

        return self._backoff * (2**attempt) * (0.5 + random.random())

    def close(self) -> None:
        self._client.close()
