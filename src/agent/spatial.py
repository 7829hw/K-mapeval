"""Spatial-Agent (GeoFlow), driven by this repository's Evaluator.

The agent is upstream's, unmodified: `src/spatial_agent/` is
`ecerybao/Spatial-Agent@6876bba` with the Google Maps client swapped for the Kakao one and
nothing else touched. This adapter only turns its `process_question` result into the
`AgentResult` the Evaluator records, so both architectures are scored by the same code.

Upstream's pipeline is unchanged underneath: `AgentRouting → PlannerAgent (transformation
plan + DAG) → TransformationExecutor → AnswerGenerator`, over `OPERATOR_REGISTRY`.

Two things the harness has to know about it:

- `predicted_option` is already zero-based upstream (`spatial_agent.py` line 35: "predicted_option
  is zero-based"), so unlike the ReAct baseline there is no index conversion here.
- `process_question` catches every exception and reports it as an `error` string rather than
  raising. An LLM outage therefore cannot be told apart from a reasoning failure at this
  boundary, so it is absorbed one level down instead, by the retry budget on the ChatOpenAI
  the agent is built with. A run's `llm_unavailable_count` will under-report on this side;
  `docs/UPSTREAM_MAPPING.md` records it.
"""

from __future__ import annotations

import time
from typing import Any

from src.agent.base import AgentResult, BenchmarkAgent
from src.kakao_maps import KakaoMapsClient
from src.spatial_agent.agent.spatial_agent import SpatialAgent as UpstreamSpatialAgent

# Upstream routes to four intents. This repository's benchmarks classify eight ways, so an
# intent this agent cannot name is not a miss by the router — it is a vocabulary the upstream
# architecture does not have. The report's intent accuracy is read against this set.
UPSTREAM_INTENTS = frozenset({"nearby", "routing", "trip", "poi"})


class SpatialAgent(BenchmarkAgent):
    """Upstream's GeoFlow agent, presented as a `BenchmarkAgent`."""

    agent_type = "spatial"

    def __init__(self, kakao_client: KakaoMapsClient, llm: Any = None) -> None:
        self.kakao_client = kakao_client
        self._agent = UpstreamSpatialAgent(kakao_api_key=kakao_client.api_key)
        # Upstream constructs its own client from the key. Swapping in the caller's is what
        # makes the per-question API and cache counters readable, and — since the Evaluator
        # gives each worker thread its own agent — what keeps two workers from sharing one
        # HTTP client and one counter.
        self._replace_client(kakao_client)
        # The vendored agent likewise builds its own ChatOpenAI from the environment.
        # Replacing it lets one run configure both architectures from the same settings.
        if llm is not None:
            self._replace_llm(llm)

    def _replace_client(self, kakao_client: KakaoMapsClient) -> None:
        constructed = self._agent.kakao_client
        self._agent.kakao_client = kakao_client
        self._agent.transformation_executor.client = kakao_client
        if constructed is not kakao_client:
            constructed.close()

    def _replace_llm(self, llm: Any) -> None:
        self._agent.llm = llm
        self._agent.agent_routing.llm = llm
        self._agent.planner_agent.llm = llm
        self._agent.answer_generator.llm = llm

    def answer(self, question: str, options: list[str]) -> AgentResult:
        self.kakao_client.reset_counters()
        started = time.time()
        # `correct_answer` and the question id stay out: gold and eval-only metadata never
        # reach an agent, and upstream accepts both only for its own logging.
        result = self._agent.process_question(question, options=options)
        latency_ms = (time.time() - started) * 1000

        predicted_option = result.get("predicted_option")
        if not isinstance(predicted_option, int) or not 0 <= predicted_option < len(options):
            predicted_option = None

        error = result.get("error")
        failure_type = None
        failure_message = None
        if error:
            failure_type, failure_message = classify_error(str(error))
        elif predicted_option is None:
            failure_type = "answer_parse_failure"
            failure_message = "The generation stage selected no option"

        concept_flow = result.get("concept_flow") or []
        intent = result.get("intent")

        return AgentResult(
            agent_type=self.agent_type,
            predicted_intent=intent if intent in UPSTREAM_INTENTS else None,
            predicted_answer=predicted_option,
            response=str(result.get("answer", "")),
            tool_calls=len(concept_flow),
            api_calls=self.kakao_client.api_call_count,
            cache_hits=self.kakao_client.cache_hit_count,
            cache_misses=self.kakao_client.cache_miss_count,
            reasoning_steps=len(concept_flow),
            latency_ms=latency_ms,
            failure_type=failure_type,
            failure_message=failure_message,
            trace=_trace(result),
        )


# Provider failures the upstream agent stringified on its way out. `process_question` catches
# every exception and reports `str(e)`, which drops the class name, so the marker has to be
# recovered from the text. Keeping them a distinct `failure_type` is what stops a Kakao blip
# from being recorded as something the architecture did.
PROVIDER_MARKERS = (
    "KakaoTimeoutError",
    "KakaoRateLimitError",
    "KakaoAuthError",
    "KakaoError",
)
# httpx's own wording, for a failure that never reached this module's exception types.
TIMEOUT_MARKERS = ("ConnectTimeout", "ReadTimeout", "TimeoutException", "timed out")


def classify_error(error: str) -> tuple[str, str]:
    """An upstream error string as a `(failure_type, message)` the Evaluator can act on.

    The message is re-prefixed with the marker it matched, because `is_transient_failure`
    identifies a retryable provider failure by the class name the message starts with, and a
    question is only worth asking again when the provider could not answer *right now*.
    """

    for marker in PROVIDER_MARKERS:
        if marker in error:
            return "provider_failure", error if error.startswith(marker) else f"{marker}: {error}"
    if any(marker in error for marker in TIMEOUT_MARKERS):
        return "provider_failure", f"KakaoTimeoutError: {error}"
    return "agent_reasoning_failure", error


def _trace(result: dict[str, Any]) -> list[dict[str, Any]]:
    """The upstream stages a report needs to explain an answer after the fact."""

    entries: list[dict[str, Any]] = [
        {"stage": "intent", "intent": result.get("intent")},
    ]
    for index, step in enumerate(result.get("concept_flow") or []):
        entries.append({"stage": "operator", "index": index, **_as_dict(step)})
    entries.append(
        {
            "stage": "evaluation",
            "evaluation": result.get("evaluation", {}),
            "predicted_option": result.get("predicted_option"),
        }
    )
    entries.append({"stage": "final", "answer": result.get("answer", "")})
    return entries


def _as_dict(step: Any) -> dict[str, Any]:
    if isinstance(step, dict):
        return step
    return {"step": str(step)}
