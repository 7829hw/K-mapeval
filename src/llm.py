from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import Any, Protocol

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from src.config import Settings

# The only responses we do not wait out. These describe the request we sent — a prompt that grew
# past the context window, a malformed tool schema — so they belong to the agent and repeating them
# only repeats the mistake. Every other failure is the endpoint saying "not now", and the answer to
# "not now" is to wait: a self-hosted vLLM behind a reverse proxy answers 502/503 while it reloads,
# and reports 404 for a model name it serves again a minute later. Judging which of those is fatal
# from a single response is guesswork that turns a slow endpoint into a lost question.
REQUEST_STATUS_CODES = frozenset({400, 413, 422})
# Retry delays grow exponentially but stop growing here, so a long outage keeps checking back
# instead of scheduling the next attempt an hour out.
MAX_RETRY_DELAY_SECONDS = 60.0


class LLMUnavailableError(RuntimeError):
    """The LLM endpoint did not serve a request within the time we were willing to wait.

    Distinct from an agent-reasoning failure: nothing about the question or the agent's plan caused
    it, so it says nothing about the architecture being measured.
    """


class LLMOutputTruncatedError(RuntimeError):
    """A token ceiling ended the completion before the model finished writing it.

    Its own failure type, because it is neither the endpoint being unavailable nor the agent
    reasoning badly: the question did not fit under the serving side's output limit. Nothing here
    sends `max_tokens` -- the vLLM deployment owns that ceiling, which is also what both upstreams
    do -- so this says the server stopped the answer, not that the run asked it to. On a thinking
    model the chain of thought is billed to the completion and emitted first, so the cut almost
    always lands on the answer, which would otherwise be recorded as an `answer_parse_failure` and
    read as the architecture failing to answer.

    Carries the usage of the truncated call so the question still reports what it spent.
    """

    def __init__(self, message: str, usage: TokenUsage) -> None:
        super().__init__(message)
        self.usage = usage


@dataclass(frozen=True)
class LLMToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class TokenUsage:
    """What one completion actually cost, as the endpoint reported it.

    `reasoning_tokens` is `None` rather than 0 when the server does not break the completion down.
    The deployment this repository runs against returns `completion_tokens_details: null` while
    still returning a populated `message.reasoning`, and exposes no `/tokenize` route, so the
    thinking tokens cannot be counted from here without a tokenizer that would have to agree with
    the server's. Reporting an estimate as a count is worse than reporting nothing, so the estimate
    is not made: `reasoning_chars` records how much thinking text came back, which is measurable,
    and `reasoning_tokens` fills in by itself on a server that reports it.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int | None = None
    reasoning_chars: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        reasoning = (
            None
            if self.reasoning_tokens is None and other.reasoning_tokens is None
            else (self.reasoning_tokens or 0) + (other.reasoning_tokens or 0)
        )
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            reasoning_tokens=reasoning,
            reasoning_chars=self.reasoning_chars + other.reasoning_chars,
        )


def _token_usage(completion: Any, message: Any) -> TokenUsage:
    """Read the usage block, tolerating a server that omits any part of it."""

    usage = getattr(completion, "usage", None)
    details = getattr(usage, "completion_tokens_details", None) if usage else None
    reasoning = getattr(details, "reasoning_tokens", None) if details else None
    # vLLM puts the chain of thought on `reasoning`; the OpenAI reasoning models use
    # `reasoning_content`. Either is thinking text that never reaches the parser.
    text = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
    return TokenUsage(
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
        reasoning_tokens=None if reasoning is None else int(reasoning),
        reasoning_chars=len(text) if isinstance(text, str) else 0,
    )


@dataclass(frozen=True)
class LLMResponse:
    content: str
    tool_calls: tuple[LLMToolCall, ...] = ()
    usage: TokenUsage = TokenUsage()

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
            # Retries are handled here so the backoff stays visible to the benchmark instead of
            # being swallowed inside the SDK.
            "max_retries": 0,
        }
        if settings.llm_base_url:
            kwargs["base_url"] = settings.llm_base_url
        self._client = OpenAI(**kwargs)
        self._model = settings.llm_model
        self._temperature = settings.llm_temperature
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
            "temperature": self._temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        completion = self._request_with_retries(kwargs)
        choice = completion.choices[0]
        message = choice.message
        usage = _token_usage(completion, message)
        if getattr(choice, "finish_reason", None) == "length":
            raise LLMOutputTruncatedError(
                "The completion was cut off at the endpoint's output limit after "
                f"{usage.completion_tokens} completion tokens "
                f"({usage.reasoning_chars} of them thinking text), so the answer it was writing "
                "never arrived. The ceiling is the serving side's; nothing here sends max_tokens.",
                usage,
            )
        calls: list[LLMToolCall] = []
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {"_invalid_json": call.function.arguments}
            calls.append(LLMToolCall(call.id, call.function.name, arguments))
        return LLMResponse(message.content or "", tuple(calls), usage)

    def _request_with_retries(self, kwargs: dict[str, Any]) -> Any:
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return self._client.chat.completions.create(**kwargs)
            except (APIConnectionError, APITimeoutError) as exc:
                last_error = exc
            except APIStatusError as exc:
                if exc.status_code in REQUEST_STATUS_CODES:
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

        delay = min(self._backoff * (2**attempt), MAX_RETRY_DELAY_SECONDS)
        return delay * (0.5 + random.random())

    def close(self) -> None:
        self._client.close()
