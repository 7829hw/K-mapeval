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
from src.parsing import parse_answer, parse_json_object
from src.tools import SpatialOperatorRegistry, ToolRegistry

SUPPORTED_INTENTS = frozenset(
    {"nearby", "poi", "routing", "trip", "type", "direction", "distance", "radius"}
)

ROUTER_PROMPT = """Classify this Korean spatial multiple-choice question into exactly one intent:
nearby, poi, routing, trip, type, direction, distance, or radius.
Use distance only for straight-line/geodesic distance, routing for road routes, and radius when
the question gives a search radius. Return JSON only: {"intent":"nearby"}.
Do not answer the question."""

PLANNER_PROMPT = """You are the planning stage of Spatial-Agent. Build an executable evidence plan.
Use only these exact contracts:
- place_search(query, limit) -> [{place_id,name,address,latitude,longitude,category}]
- nearby_places(center, query|category_code, radius_m, limit) -> [Place], nearest first
- directions(origin, destination, mode="driving", priority="RECOMMEND") -> Route
- travel_time has the same arguments and Route output as directions
- haversine_distance(place_a, place_b) -> {distance_m,distance_km}
- bearing_to_direction(place_a, place_b) ->
  {bearing_degrees,direction,direction_ko,cardinal_direction,cardinal_direction_ko}
- filter_by_direction(center, places, direction) -> [Place with distance/direction evidence]
- select_min(items, key) / select_max(items, key); items must be objects
- compare_routes(routes, metric="distance_m")
- sum_route_metrics(routes), where routes are complete Route objects

Return JSON only in this shape:
{"steps":[{"id":"s1","operator":"place_search","arguments":{"query":"경복궁","limit":1}}]}

Reference normalized output fields exactly, for example $s1.0, $s1.0.latitude,
$s1.0.longitude, and $s2.distance_m. Never use Google fields such as geometry, lat, lng,
location, legs, or inputs. Pass a complete Place reference such as $s1.0 to origin,
destination, place_a, and place_b. Gather comparable candidate evidence, use deterministic
calculations locally, and contain no more than {max_steps} steps.
Choose a plan appropriate to the intent:
- type: search the named place and inspect its category field.
- direction: search both places and use bearing_to_direction. For the nearest place in a stated
  direction, retrieve candidates with nearby_places, then use filter_by_direction.
- distance: search both places and use haversine_distance; do not use driving directions for a
  straight-line distance question.
- radius: call nearby_places with the exact radius_m and requested POI query/category, then compare
  the returned place names with every option. A vertical bar in an option separates place names.
Do not select the final answer and do not assume the gold answer."""

EVALUATOR_PROMPT = """You are the evaluation stage of Spatial-Agent. Select exactly one candidate
using the executed spatial evidence. Execution evidence has priority over guesses. Candidate
numbering is 0-based. Return JSON only:
{"predicted_option":1,"confidence":0.8,"reason":"brief evidence-based reason"}
Never return an option outside the supplied candidates."""


class SpatialAgent(BenchmarkAgent):
    """Kakao port preserving Route -> Plan -> Execute -> Evaluate -> Generate."""

    agent_type = "spatial_agent"

    def __init__(self, llm: ChatClient, tools: ToolRegistry, *, max_steps: int = 8) -> None:
        self.llm = llm
        self.tools = tools
        self.operators = SpatialOperatorRegistry()
        self.max_steps = max_steps

    def answer(self, question: str, options: list[str]) -> AgentResult:
        started = time.perf_counter()
        api_before = self.tools.provider.api_call_count
        cache_hits_before = self.tools.provider.cache_hit_count
        cache_misses_before = self.tools.provider.cache_miss_count
        tools_before = self.tools.tool_call_count
        trace: list[dict[str, Any]] = []
        failure_type: str | None = None
        failure_message: str | None = None
        response_text = ""
        predicted: int | None = None
        predicted_intent: str | None = None
        reasoning_steps = 0
        try:
            # Route
            routing = self.llm.chat(
                [
                    {"role": "system", "content": ROUTER_PROMPT},
                    {"role": "user", "content": question},
                ]
            )
            reasoning_steps += 1
            route_json = parse_json_object(routing.content)
            intent = str(route_json.get("intent", "")).lower()
            if intent not in SUPPORTED_INTENTS:
                intent = _heuristic_intent(question)
            predicted_intent = intent
            trace.append({"stage": "route", "intent": intent})

            # Plan
            plan_response = self.llm.chat(
                [
                    {
                        "role": "system",
                        "content": PLANNER_PROMPT.replace("{max_steps}", str(self.max_steps)),
                    },
                    {
                        "role": "user",
                        "content": f"Intent: {intent}\n{format_question(question, options)}",
                    },
                ]
            )
            reasoning_steps += 1
            plan = parse_json_object(plan_response.content)
            raw_steps = plan.get("steps")
            if not isinstance(raw_steps, list):
                raise ValueError("Planner response does not contain a steps list")
            steps = raw_steps[: self.max_steps]
            trace.append({"stage": "plan", "steps": steps})

            # Execute
            results: dict[str, Any] = {}
            execution_log: list[dict[str, Any]] = []
            tool_names = {schema["function"]["name"] for schema in self.tools.schemas()}
            for index, step in enumerate(steps, 1):
                if not isinstance(step, dict):
                    continue
                step_id = str(step.get("id") or f"s{index}")
                operator = str(step.get("operator") or "")
                raw_arguments = step.get("arguments") or {}
                try:
                    arguments = _resolve_references(raw_arguments, results)
                    if operator in tool_names:
                        execution = self.tools.invoke(operator, arguments)
                        if execution.status == "ok":
                            results[step_id] = execution.output
                        else:
                            results[step_id] = {"error": execution.error}
                        entry = {
                            "id": step_id,
                            "operator": operator,
                            "arguments": execution.arguments,
                            **execution.observation(),
                        }
                    else:
                        output = self.operators.invoke(operator, arguments)
                        results[step_id] = output
                        entry = {
                            "id": step_id,
                            "operator": operator,
                            "arguments": arguments,
                            "status": "ok",
                            "result": output,
                        }
                except Exception as exc:
                    results[step_id] = {"error": f"{type(exc).__name__}: {exc}"}
                    entry = {
                        "id": step_id,
                        "operator": operator,
                        "arguments": raw_arguments,
                        "status": "error",
                        "error": results[step_id]["error"],
                    }
                execution_log.append(entry)
            trace.append({"stage": "execute", "steps": execution_log})

            # Evaluate
            evaluation = self.llm.chat(
                [
                    {"role": "system", "content": EVALUATOR_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Intent: {intent}\n{format_question(question, options)}\n\n"
                            "Plan and execution evidence:\n"
                            f"{json.dumps(execution_log, ensure_ascii=False)}"
                        ),
                    },
                ]
            )
            reasoning_steps += 1
            evaluation_json = parse_json_object(evaluation.content)
            predicted = _coerce_option(evaluation_json.get("predicted_option"), len(options))
            if predicted is None:
                predicted = parse_answer(evaluation.content, option_count=len(options))
            trace.append(
                {
                    "stage": "evaluate",
                    "predicted_option": predicted,
                    "confidence": evaluation_json.get("confidence"),
                    "reason": evaluation_json.get("reason", ""),
                }
            )

            # Generate
            if predicted is not None:
                response_text = f"^^{predicted}^^"
            trace.append({"stage": "generate", "response": response_text})
        except Exception as exc:
            failure_type = "agent_reasoning_failure"
            failure_message = f"{type(exc).__name__}: {exc}"
        if predicted is None and failure_type is None:
            provider_failure = find_provider_failure(trace)
            if provider_failure:
                failure_type = "provider_failure"
                failure_message = provider_failure
            else:
                failure_type = "answer_parse_failure"
                failure_message = "No valid 0-based option found during evaluation"
        return AgentResult(
            agent_type=self.agent_type,
            predicted_intent=predicted_intent,
            predicted_answer=predicted,
            response=response_text,
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


def _resolve_references(value: Any, results: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_references(item, results) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_references(item, results) for item in value]
    if not isinstance(value, str) or not value.startswith("$"):
        return value
    parts = value[1:].split(".")
    if parts[0] not in results:
        raise ValueError(f"Unknown plan reference: {value}")
    current = results[parts[0]]
    for part in parts[1:]:
        if isinstance(current, list):
            current = current[int(part)]
            continue
        if not isinstance(current, dict):
            raise ValueError(f"Cannot resolve {part!r} in plan reference: {value}")
        if part == "geometry" and {"latitude", "longitude"} <= current.keys():
            current = {
                "lat": current["latitude"],
                "lng": current["longitude"],
                "location": current,
            }
            continue
        aliases = {"lat": "latitude", "lng": "longitude", "lon": "longitude"}
        key = part if part in current else aliases.get(part, part)
        if key not in current:
            raise ValueError(f"Missing field {part!r} in plan reference: {value}")
        current = current[key]
    return current


def _coerce_option(value: Any, option_count: int) -> int | None:
    try:
        option = int(value)
    except (TypeError, ValueError):
        return None
    return option if 0 <= option < option_count else None


def _heuristic_intent(question: str) -> str:
    lowered = question.lower()
    if any(word in lowered for word in ("반경", "이내", "radius", "within")):
        return "radius"
    if any(word in lowered for word in ("장소 유형", "유형은", "카테고리", "종류", "type")):
        return "type"
    if any(word in lowered for word in ("직선거리", "직선 거리", "직선 거리는", "geodesic")):
        return "distance"
    if any(
        word in lowered
        for word in ("어느 방향", "방향에", "동쪽", "서쪽", "남쪽", "북쪽", "direction")
    ):
        return "direction"
    if any(word in lowered for word in ("가까운", "근처", "nearest", "nearby")):
        return "nearby"
    if any(word in lowered for word in ("경로", "운전", "자동차", "주행", "route", "driving")):
        return "routing"
    if any(word in lowered for word in ("일정", "여행", "순서", "경유", "itinerary")):
        return "trip"
    return "poi"
