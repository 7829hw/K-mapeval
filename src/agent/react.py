from __future__ import annotations

import json
import time
from typing import Any

from src.agent.base import (
    AgentResult,
    BenchmarkAgent,
    find_provider_failure,
    format_question,
)
from src.llm import ChatClient
from src.parsing import parse_answer
from src.tools import ToolRegistry

REACT_SYSTEM_PROMPT = """You are the MapEval-style ReAct baseline for Korean spatial questions.
Use the map tools to gather evidence and reason over only the question and candidate options.
The benchmark includes nearby, POI, routing, trip, place-type, cardinal-direction,
straight-line-distance, and radius-constrained questions. Use a place's category for type,
coordinates for direction and straight-line distance, nearby_places with the exact radius for
radius questions, and directions only for road-route questions. Nearby results are nearest first.
Select one 0-based option. Never invent a place ID. When you have enough evidence, answer exactly as
^^Option_Number^^. You are not given and must not ask for the gold answer."""


class ReactAgent(BenchmarkAgent):
    agent_type = "react"

    def __init__(self, llm: ChatClient, tools: ToolRegistry, *, max_steps: int = 8) -> None:
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps

    def answer(self, question: str, options: list[str]) -> AgentResult:
        started = time.perf_counter()
        api_before = self.tools.provider.api_call_count
        cache_hits_before = self.tools.provider.cache_hit_count
        cache_misses_before = self.tools.provider.cache_miss_count
        tools_before = self.tools.tool_call_count
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {"role": "user", "content": format_question(question, options)},
        ]
        trace: list[dict[str, Any]] = []
        final_text = ""
        failure_type: str | None = None
        failure_message: str | None = None
        reasoning_steps = 0
        try:
            for _ in range(self.max_steps):
                reasoning_steps += 1
                response = self.llm.chat(messages, tools=self.tools.schemas())
                messages.append(response.assistant_message())
                if not response.tool_calls:
                    final_text = response.content
                    break
                for call in response.tool_calls:
                    execution = self.tools.invoke(call.name, call.arguments)
                    observation = execution.observation()
                    trace.append(
                        {
                            "stage": "act",
                            "tool": call.name,
                            "arguments": execution.arguments,
                            **observation,
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps(observation, ensure_ascii=False),
                        }
                    )
            if not final_text:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Tool budget is exhausted. Select the best-supported option now."
                        ),
                    }
                )
                reasoning_steps += 1
                final_text = self.llm.chat(messages).content
        except Exception as exc:
            failure_type = "agent_reasoning_failure"
            failure_message = f"{type(exc).__name__}: {exc}"
        predicted = parse_answer(final_text, option_count=len(options))
        if predicted is None and failure_type is None:
            provider_failure = find_provider_failure(trace)
            if provider_failure:
                failure_type = "provider_failure"
                failure_message = provider_failure
            else:
                failure_type = "answer_parse_failure"
                failure_message = "No valid 0-based option found in the final response"
        return AgentResult(
            agent_type=self.agent_type,
            predicted_answer=predicted,
            response=final_text,
            tool_calls=self.tools.tool_call_count - tools_before,
            api_calls=self.tools.provider.api_call_count - api_before,
            cache_hits=self.tools.provider.cache_hit_count - cache_hits_before,
            cache_misses=self.tools.provider.cache_miss_count - cache_misses_before,
            reasoning_steps=reasoning_steps,
            latency_ms=(time.perf_counter() - started) * 1000,
            failure_type=failure_type,
            failure_message=failure_message,
            trace=trace,
        )
