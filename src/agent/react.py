"""The MapEval-API ReAct baseline, driven by this repository's Evaluator.

The agent itself is upstream's. `mapeval-api/Evaluator2.py` (35d481a) builds it in nine lines
and this adapter reproduces those nine, element by element, so a difference in the score is a
difference in the map and not in the harness:

- line 33, the five tools it instantiates
      -> `BASELINE_TOOL_TYPES`
- `initialize_agent(..., STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION)`, with
  `handle_parsing_errors=True` and `return_intermediate_steps=True`
      -> `ReactAgent.__init__`
- lines 58-76, the "Choose the answer ... (1/2/3/4)" prompt and its one-based option list
      -> `build_prompt`
- line 132's caret span, then line 14's first digit
      -> `parse_upstream_answer`
- `ground_truth = item["answer"]["correct"] + 1`
      -> the one-based conversion in `answer`

Two things upstream's loop does are the harness's job here and are not reproduced: it posts
each verdict to `http://localhost:5000/api/evaluation/`, and it sleeps 60 s between questions.
`src/evaluator.py` writes the report and paces the run instead.

One deliberate index change. Upstream numbers its options from 1 and stores gold as
`correct + 1`; this repository is 0-based everywhere (`AGENT.md`). The prompt the model sees
is still upstream's one-based text — changing it would change the task — and the conversion
happens once, on the way out, in `answer`.

`Option0: Unanswerable` is not prepended. Upstream adds it only on rows whose classification
is None, which announces the answer before the question is read; the benchmarks here carry
their refusal as an ordinary option instead.
"""

from __future__ import annotations

import re
import time
from typing import Any

from langchain.agents import AgentType, initialize_agent
from openai import APIConnectionError, APIStatusError, APITimeoutError

from src.agent.base import AgentResult, BenchmarkAgent
from src.kakao_maps import KakaoMapsClient
from src.llm import LLMUnavailableError
from src.mapeval_api.FormattedTools import (
    DirectionsTool,
    NearbyPlacesTool,
    PlaceDetailsTool,
    PlaceSearchTool,
    TravelTimeTool,
    set_client,
)

# The five tools `Evaluator2.py` instantiates, in its order. `Tools.py` also defines a
# `PlaceIdTool` the evaluator never constructs, and `PlaceSearchTool` is itself documented as
# "Get place ID for a given location" — the two are one primitive under two names, not a
# sixth tool. Read `Evaluator2.py`, not `Tools.py`, before adding to this list.
BASELINE_TOOL_TYPES = (
    PlaceSearchTool,
    PlaceDetailsTool,
    NearbyPlacesTool,
    TravelTimeTool,
    DirectionsTool,
)

# Upstream's own request codes stay the agent's problem; everything else is the endpoint
# saying "not now", which the Evaluator retries as a whole question.
REQUEST_STATUS_CODES = frozenset({400, 413, 422})


def build_prompt(question: str, options: list[str]) -> str:
    """`Evaluator2.py` lines 58-76, verbatim, including its one-based option numbers."""

    prompt = (
        question
        + "Choose the answer from the following options (1/2/3/4). So, the output format will be "
        '"^^Option_Number^^". Choose the correct answer from the following options: '
    )
    for i in range(len(options)):
        if options[i] == "":
            break
        prompt = prompt + "Option" + str(i + 1) + ": " + options[i] + ", "
    return prompt


def parse_upstream_answer(output: str) -> int | None:
    """`Evaluator2.py` lines 132 and 14: the `^^...^^` span, then its first digit."""

    match = re.search(r"\^\^(.*?)\^\^", output or "")
    if not match:
        return None
    for char in match.group(1):
        if char.isdigit():
            return int(char)
    return None


class ReactAgent(BenchmarkAgent):
    """MapEval-API's structured-chat ReAct agent over the five Kakao-backed primitives."""

    agent_type = "react"

    def __init__(
        self,
        llm: Any,
        kakao_client: KakaoMapsClient,
        *,
        verbose: bool = False,
        max_iterations: int = 30,
    ) -> None:
        self.llm = llm
        self.kakao_client = kakao_client
        self.tools = [tool_type() for tool_type in BASELINE_TOOL_TYPES]
        # `initialize_agent`'s own default is 15. Thirty is what this repository has always
        # given the baseline: on five primitives a four-stop itinerary needs four PlaceSearch
        # turns plus four TravelTime turns before any arithmetic, and being generous to the
        # baseline is the conservative direction for the claim under test.
        self._agent = initialize_agent(
            self.tools,
            llm=llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=verbose,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
            max_iterations=max_iterations,
        )

    def answer(self, question: str, options: list[str]) -> AgentResult:
        # The tools are pydantic models shared across threads, so the client they read is
        # bound per thread rather than held on them.
        set_client(self.kakao_client)
        self.kakao_client.reset_counters()

        prompt = build_prompt(question, options)
        started = time.time()
        try:
            result = self._agent.invoke({"input": prompt})
        except (APIConnectionError, APITimeoutError) as exc:
            raise LLMUnavailableError(f"{type(exc).__name__}: {exc}") from exc
        except APIStatusError as exc:
            if exc.status_code in REQUEST_STATUS_CODES:
                raise
            raise LLMUnavailableError(f"{type(exc).__name__}: {exc}") from exc
        latency_ms = (time.time() - started) * 1000

        output = str(result.get("output", ""))
        steps = result.get("intermediate_steps", []) or []

        one_based = parse_upstream_answer(output)
        predicted_answer = None
        failure_type = None
        failure_message = None
        if one_based is None:
            failure_type = "answer_parse_failure"
            failure_message = "No ^^N^^ selection in the final answer"
        elif 1 <= one_based <= len(options):
            predicted_answer = one_based - 1
        else:
            # Upstream records this as `verdict: invalid`, including its `^^0^^` refusal,
            # which these benchmarks have no index for.
            failure_type = "answer_parse_failure"
            failure_message = f"Selection ^^{one_based}^^ is outside 1..{len(options)}"

        return AgentResult(
            agent_type=self.agent_type,
            # ReAct has no classification stage, so nothing is predicted here. The intent
            # metric counts only questions an intent was predicted for.
            predicted_intent=None,
            predicted_answer=predicted_answer,
            response=output,
            tool_calls=len(steps),
            api_calls=self.kakao_client.api_call_count,
            cache_hits=self.kakao_client.cache_hit_count,
            cache_misses=self.kakao_client.cache_miss_count,
            reasoning_steps=len(steps) + 1,
            latency_ms=latency_ms,
            failure_type=failure_type,
            failure_message=failure_message,
            trace=_trace(prompt, steps, output),
        )


def _trace(prompt: str, steps: list[Any], output: str) -> list[dict[str, Any]]:
    """Every tool the run reached, with its normalized arguments and its observation."""

    entries: list[dict[str, Any]] = [{"stage": "prompt", "input": prompt}]
    for index, step in enumerate(steps):
        try:
            action, observation = step
        except (TypeError, ValueError):
            entries.append({"stage": "step", "index": index, "raw": str(step)})
            continue
        entries.append(
            {
                "stage": "tool",
                "index": index,
                "tool": getattr(action, "tool", None),
                "arguments": getattr(action, "tool_input", None),
                "observation": str(observation),
            }
        )
    entries.append({"stage": "final", "output": output})
    return entries
