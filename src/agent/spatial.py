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
from src.agent.geoflow import (
    OPERATOR_CONTRACTS,
    normalize_analysis,
    normalize_and_validate_graph,
    retrieve_templates,
)
from src.llm import ChatClient
from src.parsing import parse_answer, parse_json_object
from src.tools import SpatialOperatorRegistry, ToolRegistry

SUPPORTED_INTENTS = frozenset(
    {"nearby", "poi", "routing", "trip", "type", "direction", "distance", "radius"}
)

ANALYSIS_PROMPT = """You are Spatial-Agent's Spatial Information Theory Analysis stage.
Extract the spatial entities and assign one scientific core concept and one functional role.
Core concepts: location, object, field, event, network, amount, proportion.
Functional roles: extent, temporal_extent, sub_condition, condition, support, measure.
Classify intent as nearby, poi, routing, trip, type, direction, distance, or radius.
Use distance only for straight-line/geodesic distance, routing for road-network routes, and radius
when an explicit search radius is given. Include all named places and spatial/temporal constraints.
Return JSON only:
{"intent":"direction","concepts":[{"id":"anchor","text":"서울역","concept_type":"location","role":"extent","attributes":{}}],"measure":"direction"}
Do not answer the multiple-choice question and do not invent coordinates."""

GRAPH_PROMPT = """You are Spatial-Agent's Concept Transformation Drafting, GeoFlow Graph
Construction, and Factorization stage. Compose the retrieved pre-validated templates into an
executable operator-concept DAG. Every node must contribute to a Measure node.

The graph must satisfy all five paper constraints:
1. acyclicity; 2. role ordering (sub_condition < condition < support < measure);
3. operator output type compatibility; 4. executable operators and available arguments;
5. connectivity from contextual input through every node to a measure.

Return JSON only:
{"graph":[{"id":"places","operator":"batch_geocode","arguments":{"place_names":["A","B"]},"depends_on":[],"output_type":"object","role":"support"}]}

Exact operator contracts:
- place_search(query, limit=5) -> object (list of Place)
- batch_geocode(place_names, anchor?, radius_m=20000, limit=1) -> object; anchor biases ambiguous
  names toward the question's reference location. Output preserves order and each item has
  {query, place, candidates}; reference the best match as $node.0.place
- geocode(address, limit=5) -> location
- place_details(place_id) -> object
- nearby_places(center, query|category_code, radius_m, limit) -> object, nearest first
- directions(origin, destination, mode="driving", priority) -> field Route
- travel_time(origin, destination, mode="driving", priority) -> field Route
- distance_matrix(origins,destinations OR pairs, mode="driving", priority) -> field;
  pairs is [{origin,destination,label?}], and output routes preserve pair order at $node.routes
- haversine_distance(place_a, place_b) -> amount
- pairwise_distances(pairs=[{place_a,place_b,label?}]) -> field
- bearing_to_direction(place_a, place_b) -> field
- filter_by_direction(center, places, direction) -> object, nearest first
- nearest(anchor, candidates, metric="haversine") -> object
- within_radius(center, candidates, radius_m) -> object
- select_min/select_max(items,key), sort_by(items,key), compare_routes(routes,metric) -> object
- sum_route_metrics(routes) -> amount
- aggregate_route_groups(routes,groups) -> amount; groups contains route indexes per option and
  returns option_totals plus best_distance_option and best_duration_option.

Use normalized fields only: latitude, longitude, distance_m, duration_s. Complete Place objects,
or literal place names for map tools, are valid. Never use Google geometry/lat/lng/legs fields.
Use batch_geocode for anchor/options and distance_matrix for route candidates so the graph remains
within {max_steps} nodes. Supply the question's origin/reference place as batch_geocode.anchor to
disambiguate same-name POIs. For a trip option A→B from S, include route pairs S→A and A→B, in
option order, then aggregate groups. For nearest among explicit options, geocode every option and
compute deterministically. A vertical bar in one option separates grouped place names; preserve its
option index while resolving each name. For a radius question use the exact radius and requested
category/keyword.
Do not select an option and do not use the gold answer."""

REPAIR_PROMPT = """Repair the supplied GeoFlow graph so it passes the listed validation error.
Keep the question semantics and retrieved template, use only the exact operator contracts from the
original system prompt, and stay within {max_steps} nodes. Return only {"graph":[...]} JSON."""

EVALUATOR_PROMPT = """You are Spatial-Agent's grounded response generation stage. Select exactly
one candidate using the final GeoFlow state and the complete topological execution trace. Computed
distance, direction, category, route, and option_totals evidence has priority over prior knowledge.
Respect candidate-index to candidate-text mapping. Numbering is 0-based. Return JSON only:
{"predicted_option":1,"confidence":0.8,"reason":"brief evidence-based reason"}
Never return an option outside the supplied candidates."""


class SpatialAgent(BenchmarkAgent):
    """Paper-aligned concept grounding, GeoFlow composition, execution, and generation."""

    agent_type = "spatial_agent"

    def __init__(self, llm: ChatClient, tools: ToolRegistry, *, max_steps: int = 8) -> None:
        self.llm = llm
        self.tools = tools
        self.operators = SpatialOperatorRegistry()
        self.max_steps = max_steps
        available = {
            *(schema["function"]["name"] for schema in self.tools.schemas()),
            *self.operators.names,
        }
        missing = set(OPERATOR_CONTRACTS) - available
        if missing:
            raise ValueError(f"GeoFlow operators are not executable: {', '.join(sorted(missing))}")

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
            analysis_response = self.llm.chat(
                [
                    {"role": "system", "content": ANALYSIS_PROMPT},
                    {"role": "user", "content": format_question(question, options)},
                ]
            )
            reasoning_steps += 1
            raw_analysis = parse_json_object(analysis_response.content)
            fallback_intent = _heuristic_intent(question)
            analysis = normalize_analysis(raw_analysis, question, fallback_intent)
            intent = _explicit_intent(question) or str(analysis["intent"]).lower()
            if intent not in SUPPORTED_INTENTS:
                intent = fallback_intent
            analysis["intent"] = intent
            predicted_intent = intent
            trace.append({"stage": "analyze", **analysis})

            templates = retrieve_templates(intent, question)
            trace.append(
                {
                    "stage": "retrieve_templates",
                    "templates": [template["name"] for template in templates],
                }
            )

            plan_response = self.llm.chat(
                [
                    {
                        "role": "system",
                        "content": GRAPH_PROMPT.replace("{max_steps}", str(self.max_steps)),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"{format_question(question, options)}\n\n"
                            "Spatial concept analysis:\n"
                            f"{json.dumps(analysis, ensure_ascii=False)}\n\n"
                            "Retrieved pre-validated templates and examples:\n"
                            f"{json.dumps(templates, ensure_ascii=False)}"
                        ),
                    },
                ]
            )
            reasoning_steps += 1
            plan = parse_json_object(plan_response.content)
            trace.append({"stage": "compose", "graph": plan.get("graph") or plan.get("steps")})
            try:
                steps, constraints = normalize_and_validate_graph(plan, max_steps=self.max_steps)
            except ValueError as graph_error:
                trace.append({"stage": "validate", "status": "invalid", "error": str(graph_error)})
                repair_response = self.llm.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                GRAPH_PROMPT.replace("{max_steps}", str(self.max_steps))
                                + "\n\n"
                                + REPAIR_PROMPT.replace("{max_steps}", str(self.max_steps))
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Validation error: {graph_error}\n"
                                f"Question and options:\n{format_question(question, options)}\n"
                                f"Analysis: {json.dumps(analysis, ensure_ascii=False)}\n"
                                f"Templates: {json.dumps(templates, ensure_ascii=False)}\n"
                                f"Invalid graph: {json.dumps(plan, ensure_ascii=False)}"
                            ),
                        },
                    ]
                )
                reasoning_steps += 1
                plan = parse_json_object(repair_response.content)
                trace.append({"stage": "repair", "graph": plan.get("graph") or plan.get("steps")})
                steps, constraints = normalize_and_validate_graph(plan, max_steps=self.max_steps)
            trace.append(
                {
                    "stage": "validate",
                    "status": "valid",
                    "constraints": constraints,
                    "topological_order": [step["id"] for step in steps],
                }
            )

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
                            "role": step["role"],
                            "output_type": step["output_type"],
                            "arguments": execution.arguments,
                            **execution.observation(),
                        }
                    else:
                        output = self.operators.invoke(operator, arguments)
                        results[step_id] = output
                        entry = {
                            "id": step_id,
                            "operator": operator,
                            "role": step["role"],
                            "output_type": step["output_type"],
                            "arguments": arguments,
                            "status": "ok",
                            "result": output,
                        }
                except Exception as exc:
                    results[step_id] = {"error": f"{type(exc).__name__}: {exc}"}
                    entry = {
                        "id": step_id,
                        "operator": operator,
                        "role": step["role"],
                        "output_type": step["output_type"],
                        "arguments": raw_arguments,
                        "status": "error",
                        "error": results[step_id]["error"],
                    }
                entry["state_keys"] = list(results)
                execution_log.append(entry)
            trace.append({"stage": "execute", "steps": execution_log, "final_state": results})

            evaluation = self.llm.chat(
                [
                    {"role": "system", "content": EVALUATOR_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Intent: {intent}\n{format_question(question, options)}\n\n"
                            "Validated GeoFlow topological execution trace:\n"
                            f"{json.dumps(execution_log, ensure_ascii=False)}\n\n"
                            "Final concept state:\n"
                            f"{json.dumps(results, ensure_ascii=False)}"
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
    return _explicit_intent(question) or "poi"


def _explicit_intent(question: str) -> str | None:
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
    if any(word in lowered for word in ("일정", "여행", "순서", "경유", "itinerary")):
        return "trip"
    if any(word in lowered for word in ("경로", "운전", "자동차", "주행", "route", "driving")):
        return "routing"
    if any(word in lowered for word in ("가까운", "근처", "nearest", "nearby")):
        return "nearby"
    return None
