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
from src.llm import (
    ChatClient,
    LLMContextOverflowError,
    LLMOutputTruncatedError,
    LLMUnavailableError,
    TokenUsage,
)
from src.parsing import parse_answer
from src.tools import ToolRegistry

# This agent is a *port* of MapEval-API's baseline, and it is finished. Upstream is
# `mapeval-api/Evaluator2.py` (35d481a): a stock langchain
# `initialize_agent(tools, llm, AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
# handle_parsing_errors=True, return_intermediate_steps=True)` over the five tools
# `FormattedTools.py` defines, prompted with the question, the options, and the answer format.
# Element by element:
#
#   upstream                                   | here
#   -------------------------------------------|--------------------------------------------
#   five tools, constructed in `evaluate()`    | `ToolRegistry.MAPEVAL_BASELINE_TOOLS`
#   structured-chat JSON action blob in text   | the provider's native tool-call channel
#   `handle_tool_error=True` on every tool     | `ToolRegistry.invoke` returns errors as
#                                              |   observations, never as exceptions
#   `handle_parsing_errors=True`               | an unparsed final answer is a recorded
#                                              |   `answer_parse_failure`, not a crash
#   `max_iterations=15` (langchain default)    | `MAX_REASONING_STEPS`, default 15
#   1-based options, `^^Option_Number^^`       | 0-based options, `^^N^^` (repo-wide invariant)
#   question + options + answer format         | `REACT_SYSTEM_PROMPT` + `format_question`
#
# **Do not tune it against benchmark results.** Every accuracy gap this agent shows is the
# finding; closing one by editing its prompt, its budget, or the description of a parameter it
# reaches would make the baseline a function of the test set. The two places that stay live are
# the ones shared with the other architecture — the provider below the tools, and the tool
# contracts both agents read — and a change there has to be argued from the provider, never from
# a question ReAct got wrong. Anything MapEval's own baseline does not do belongs on the
# Spatial-Agent side, where it is an architectural stage under measurement.
#
# The prompt itself carries what MapEval's carries and nothing more: role, evidence discipline,
# and the wire format. An earlier revision named the benchmark's question taxonomy and told the
# agent which tool each shape wants, which is planning handed to the baseline in prose — the same
# mistake as handing it the aggregation tools, in another currency. Tool contracts live in the
# tool descriptions, where both agents read them.
REACT_SYSTEM_PROMPT = """You are the MapEval-style ReAct baseline for Korean spatial questions.
Use the map tools to gather evidence and reason over only the question and candidate options.
Select one 0-based option. Never invent a place ID. When you have enough evidence, answer exactly as
^^Option_Number^^. You are not given and must not ask for the gold answer."""


class ReactAgent(BenchmarkAgent):
    agent_type = "react"

    def __init__(
        self,
        llm: ChatClient,
        tools: ToolRegistry,
        *,
        max_steps: int = 15,
        single_action: bool = True,
        force_final_answer: bool = False,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        # `AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION` parses one JSON action blob out
        # of each LLM response, so upstream's baseline takes exactly one action per iteration. A
        # provider's native tool channel will happily return several, and executing all of them
        # turns one iteration into many: on the v5 run one question executed 24 tool calls in 6
        # LLM rounds. Keeping only the first is what upstream's parser would have seen.
        self.single_action = single_action
        # Upstream stops. `initialize_agent` defaults to `early_stopping_method="force"`, which
        # returns "Agent stopped due to iteration limit or time limit." as the output and makes no
        # further call; `Evaluator2.py` then finds no `^^N^^` in it and the row scores nothing.
        # Asking once more for an answer is a free extra turn the paper's baseline never gets.
        self.force_final_answer = force_final_answer

    #: What langchain returns when the iteration budget runs out under the default
    #: `early_stopping_method="force"`. It carries no option, which is the point.
    ITERATION_LIMIT_OUTPUT = "Agent stopped due to iteration limit or time limit."

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
        trace = self.new_trace()
        final_text = ""
        failure_type: str | None = None
        failure_message: str | None = None
        reasoning_steps = 0
        usage = TokenUsage()
        stopped_at_budget = False
        try:
            for _ in range(self.max_steps):
                reasoning_steps += 1
                response = self.llm.chat(messages, tools=self.tools.schemas())
                usage += response.usage
                messages.append(response.assistant_message())
                if not response.tool_calls:
                    final_text = response.content
                    break
                calls = response.tool_calls[:1] if self.single_action else response.tool_calls
                if self.single_action and len(response.tool_calls) > 1:
                    # Drop the extra calls from the message we keep, rather than answering them
                    # with a refusal: an orphan tool_call_id breaks the next request, and upstream
                    # never emitted them in the first place.
                    messages[-1] = _one_action(messages[-1], calls[0].id)
                for call in calls:
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
                if not self.force_final_answer:
                    stopped_at_budget = True
                    final_text = self.ITERATION_LIMIT_OUTPUT
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Tool budget is exhausted. Select the best-supported option now."
                            ),
                        }
                    )
                    reasoning_steps += 1
                    forced = self.llm.chat(messages)
                    usage += forced.usage
                    final_text = forced.content
        except LLMUnavailableError as exc:
            failure_type = "llm_unavailable"
            failure_message = f"{type(exc).__name__}: {exc}"
        except LLMOutputTruncatedError as exc:
            failure_type = "llm_output_truncated"
            failure_message = f"{type(exc).__name__}: {exc}"
            usage += exc.usage
        except LLMContextOverflowError as exc:
            failure_type = "llm_context_overflow"
            failure_message = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            failure_type = "agent_reasoning_failure"
            failure_message = f"{type(exc).__name__}: {exc}"
        predicted = parse_answer(final_text, option_count=len(options))
        if predicted is None and failure_type is None:
            provider_failure = find_provider_failure(trace)
            if stopped_at_budget:
                # Ahead of the provider check on purpose: the budget is what ended the question,
                # and one failed lookup among fifteen observations does not mean the map could
                # not answer it. Still a miss, exactly as upstream counts it -- but a miss for a
                # reason a report can act on, where `answer_parse_failure` said the agent wrote
                # something unreadable and 18 of v6's rows were labelled that way.
                failure_type = "iteration_limit"
                failure_message = self.ITERATION_LIMIT_OUTPUT
            elif provider_failure:
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
            llm_calls=reasoning_steps,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            reasoning_chars=usage.reasoning_chars,
            latency_ms=(time.perf_counter() - started) * 1000,
            failure_type=failure_type,
            failure_message=failure_message,
            trace=trace,
        )


def _one_action(message: dict[str, Any], keep: str) -> dict[str, Any]:
    """The assistant message with a single tool call, the way the structured-chat parser saw it."""

    kept = [
        call
        for call in (message.get("tool_calls") or [])
        if (call.get("id") if isinstance(call, dict) else getattr(call, "id", None)) == keep
    ]
    return {**message, "tool_calls": kept}
