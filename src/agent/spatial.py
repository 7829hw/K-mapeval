from __future__ import annotations

import json
import re
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
    canonical_reference,
    factorize_geoflow,
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
{"intent":"direction","concepts":[{"id":"anchor","text":"서울역","concept_type":"location","role":"extent","attributes":{},"depends_on":[]},{"id":"answer","text":"direction","concept_type":"field","role":"measure","attributes":{},"depends_on":["anchor"]}],"measure":"direction"}
Do not answer the multiple-choice question and do not invent coordinates."""

GRAPH_PROMPT = """You are Spatial-Agent's Concept Transformation Drafting, GeoFlow Graph
Construction, and Factorization stage. Compose the retrieved pre-validated templates into an
executable operator-concept DAG. Every node must contribute to a Measure node.

The graph must satisfy all five paper constraints:
1. acyclicity; 2. role ordering
   (extent/temporal_extent < sub_condition < condition < support < measure);
3. operator output type compatibility; 4. executable operators and available arguments;
5. connectivity from contextual input through every node to a measure.

Return JSON only:
{"graph":[{"id":"places","operator":"batch_geocode","arguments":{"place_names":["A","B"]},"depends_on":[],"output_type":"object","role":"extent","concept_ids":["anchor"]}]}

Exact operator contracts:
- place_search(query, limit=5) -> object (list of Place)
- batch_geocode(place_names, anchor?, radius_m=20000, limit=1) -> object; anchor biases ambiguous
  names toward the question's reference location. Output preserves order and each item has
  {query, place, candidates}; reference the best match as $node.0.place
- geocode(address, limit=5) -> location
- place_details(place_id) -> object
- nearby_places(center, query|category_code, radius_m, limit) -> object, nearest first
  Kakao category codes: MT1 mart, CS2 convenience store, PS3 childcare, SC4 school,
  AC5 academy, PK6 parking, OL7 gas/charging, SW8 subway station, BK9 bank,
  CT1 culture, AG2 real estate, PO3 public institution, AT4 attraction, AD5 lodging,
  FD6 restaurant, CE7 cafe, HP8 hospital, PM9 pharmacy. Use the matching code whenever possible.
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
- merge_places(items) -> object; merges and de-duplicates multiple retrieval branches
- recover_option_places(options,candidates,anchor,radius_m) -> object; conditionally resolves
  options absent from ranked retrieval and merges their POIs into the candidate list
- match_options(options,places,anchor?,mode) -> object; grounds options to retrieved POIs
- match_distance_options(distance,options) -> object; maps a computed distance to numeric options
- match_type_options(place,options) -> object; maps normalized category evidence to options
- events_from_objects(objects,event_type,timestamp_field?) -> event
- filter_events(events,field,operator,value) -> event
- build_route_network(nodes,edges) -> network
- calculate_proportion(numerator,denominator) -> proportion
- open_at_time(schedule,local_time,timezone) -> event
- timezone_convert(local_time,from_timezone,to_timezone) -> event
- calculate_finish_time(start_time,duration_s,timezone) -> event
- tsp_tw(nodes,distance_matrix,time_windows?,start_index=0) -> network
- identity_measure(value) -> object; explicit Measure projection for a single source operator

Use normalized fields only: latitude, longitude, distance_m, duration_s. Complete Place objects,
or literal place names for map tools, are valid. Never use Google geometry/lat/lng/legs fields.
Copy every place name verbatim from the question and candidate options; never shorten, translate,
or remove a store/branch prefix. References must use $node.0.place, never ${node.0.place}.
Use batch_geocode for named endpoints and distance_matrix for route candidates so the graph remains
within {max_steps} nodes. For nearby, direction, and radius questions, geocode only the anchor and
retrieve the requested type with nearby_places. For nearby/direction, use recover_option_places so
only missing option evidence is resolved, then use match_options; do not geocode options upfront.
Supply the question's origin/reference place as batch_geocode.anchor to disambiguate same-name POIs.
For a trip option A→B from S, include route pairs S→A and A→B, in
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
For radius/list questions, compare the resolved in-radius POI names with each option as a set;
never choose an option merely because it contains more items or looks like a list.
When a match_options or match_distance_options result supplies best_option with confidence >= 0.7,
use that grounded option. It is the Measure output of a validated GeoFlow, not a language guess.
Never return an option outside the supplied candidates."""

INTENT_EVALUATION_RULES = {
    "nearby": (
        "For nearest questions, use match_options ranks and distances. Exact or fuzzy "
        "option-to-POI matches outrank unsupported guesses."
    ),
    "direction": (
        "Use only direction-filtered candidates, then choose the nearest supported option from "
        "match_options."
    ),
    "radius": (
        "Treat each option as a set separated by '|'. Compare it with present_option_members and "
        "prefer an exact set match."
    ),
    "distance": (
        "Use the computed Haversine distance and match_distance_options; never substitute route "
        "distance."
    ),
    "type": "Use the normalized POI category returned by place search.",
}


class SpatialAgent(BenchmarkAgent):
    """Concept-graph grounding, constrained factorization, execution, and generation."""

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
            intent = str(analysis["intent"]).lower()
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
                factorized, steps, constraints = _factorize_validate_plan(
                    analysis,
                    plan,
                    question,
                    options,
                    intent,
                    self.max_steps,
                )
            except ValueError as graph_error:
                trace.append(
                    {"stage": "validate", "status": "invalid", "error": str(graph_error)}
                )
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
                                "Question and options:\n"
                                f"{format_question(question, options)}\n"
                                f"Analysis: {json.dumps(analysis, ensure_ascii=False)}\n"
                                f"Templates: {json.dumps(templates, ensure_ascii=False)}\n"
                                f"Invalid graph: {json.dumps(plan, ensure_ascii=False)}"
                            ),
                        },
                    ]
                )
                reasoning_steps += 1
                repaired_plan = parse_json_object(repair_response.content)
                trace.append(
                    {
                        "stage": "repair",
                        "graph": repaired_plan.get("graph") or repaired_plan.get("steps"),
                    }
                )
                try:
                    factorized, steps, constraints = _factorize_validate_plan(
                        analysis,
                        repaired_plan,
                        question,
                        options,
                        intent,
                        self.max_steps,
                    )
                except ValueError:
                    fallback_graph = _bind_prevalidated_template(
                        intent, question, options, _extract_anchor(question, intent)
                    )
                    if not fallback_graph:
                        raise
                    trace.append({"stage": "template_fallback", "graph": fallback_graph})
                    factorized, steps, constraints = _factorize_validate_plan(
                        analysis,
                        {"graph": fallback_graph},
                        question,
                        options,
                        intent,
                        self.max_steps,
                    )
            trace.append({"stage": "factorize", **factorized.as_dict()})
            trace.append(
                {
                    "stage": "validate",
                    "status": "valid",
                    "constraints": constraints,
                    "topological_order": [step["id"] for step in steps],
                }
            )

            results: dict[str, Any] = {}
            concept_state: dict[str, Any] = {}
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
                for binding in step.get("output_bindings") or []:
                    concept_id = str(binding.get("concept_id") or "")
                    if not concept_id:
                        continue
                    concept_state[concept_id] = _resolve_output_binding(
                        results[step_id], str(binding.get("path") or "$")
                    )
                entry["concept_state_keys"] = list(concept_state)
                execution_log.append(entry)
            trace.append(
                {
                    "stage": "execute",
                    "steps": execution_log,
                    "operator_state": results,
                    "final_state": concept_state,
                }
            )

            evaluation = self.llm.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            EVALUATOR_PROMPT
                            + "\n"
                            + INTENT_EVALUATION_RULES.get(intent, "")
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Intent: {intent}\n{format_question(question, options)}\n\n"
                            "Validated GeoFlow topological execution trace:\n"
                            f"{json.dumps(execution_log, ensure_ascii=False)}\n\n"
                            "Final concept state:\n"
                            f"{json.dumps(concept_state, ensure_ascii=False)}"
                        ),
                    },
                ]
            )
            reasoning_steps += 1
            evaluation_json = parse_json_object(evaluation.content)
            predicted = _coerce_option(evaluation_json.get("predicted_option"), len(options))
            if predicted is None:
                predicted = parse_answer(evaluation.content, option_count=len(options))
            grounded = _grounded_prediction(results, len(options))
            if grounded is not None:
                predicted = grounded["predicted_option"]
                evaluation_json = {
                    "predicted_option": predicted,
                    "confidence": grounded["confidence"],
                    "reason": grounded["reason"],
                }
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


def _factorize_validate_plan(
    analysis: dict[str, Any],
    plan: dict[str, Any],
    question: str,
    options: list[str],
    intent: str,
    max_steps: int,
):
    raw_steps = plan.get("graph") if plan.get("graph") is not None else plan.get("steps")
    if not isinstance(raw_steps, list):
        raise ValueError("GeoFlow response does not contain a graph")
    grounded = _ground_graph_literals(raw_steps, question, options, intent)
    factorized = factorize_geoflow(analysis, {"graph": grounded})
    steps, constraints = normalize_and_validate_graph(
        factorized.as_dict(), max_steps=max_steps
    )
    return factorized, steps, constraints


def _resolve_output_binding(output: Any, path: str) -> Any:
    if path in {"", "$"}:
        return output
    reference = path[2:] if path.startswith("$.") else path.lstrip("$")
    current = output
    for part in (value for value in reference.split(".") if value):
        if isinstance(current, list):
            current = current[int(part)]
        elif isinstance(current, dict):
            current = current[part]
        else:
            raise ValueError(f"Cannot resolve output binding path {path!r}")
    return current


def _resolve_references(value: Any, results: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_references(item, results) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_references(item, results) for item in value]
    if not isinstance(value, str):
        return value
    reference = canonical_reference(value)
    if not reference.startswith("$"):
        return value
    parts = reference[1:].split(".")
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


def _ground_graph_literals(
    steps: list[dict[str, Any]], question: str, options: list[str], intent: str
) -> list[dict[str, Any]]:
    """Restore verbatim benchmark entities when an LLM shortens names during binding."""

    anchor = _extract_anchor(question, intent)
    for step in steps:
        if step["operator"] != "batch_geocode":
            continue
        arguments = dict(step["arguments"])
        names = list(arguments.get("place_names") or [])
        if anchor and names and _is_shortened_name(names[0], anchor):
            names[0] = anchor
            if _is_shortened_name(str(arguments.get("anchor") or ""), anchor):
                arguments["anchor"] = anchor
        if (
            intent in {"nearby", "direction", "routing"}
            and len(names) == len(options) + 1
            and all("|" not in option for option in options)
        ):
            names[1:] = options
        arguments["place_names"] = names
        step["arguments"] = arguments
    return steps


def _bind_prevalidated_template(
    intent: str,
    question: str,
    options: list[str],
    anchor: str | None,
) -> list[dict[str, Any]] | None:
    if intent == "distance":
        match = re.search(r"^(.+?)\s+및\s+(.+?)\s+사이의\s+직선거리", question)
        if not match:
            return None
        place_a, place_b = (part.strip() for part in match.groups())
        return [
            _step(
                "places",
                "batch_geocode",
                {"place_names": [place_a, place_b], "anchor": place_a, "limit": 1},
                role="support",
            ),
            _step(
                "distance",
                "haversine_distance",
                {"place_a": "$places.0.place", "place_b": "$places.1.place"},
                depends_on=["places"],
                role="support",
            ),
            _step(
                "option_match",
                "match_distance_options",
                {"distance": "$distance", "options": options},
                depends_on=["distance"],
                role="measure",
            ),
        ]

    if intent not in {"nearby", "direction", "radius"} or not anchor:
        return None
    target = _extract_target_type(question, intent)
    if not target:
        return None
    radius_m = _extract_radius_m(question) if intent == "radius" else 20_000
    retrieval_specs = _nearby_retrieval_specs(target)
    steps = [
        _step(
            "anchor",
            "batch_geocode",
            {"place_names": [anchor], "anchor": anchor, "limit": 1},
            role="support",
        )
    ]
    retrieval_ids: list[str] = []
    for index, spec in enumerate(retrieval_specs):
        step_id = f"nearby_{index + 1}"
        retrieval_ids.append(step_id)
        steps.append(
            _step(
                step_id,
                "nearby_places",
                {
                    "center": "$anchor.0.place",
                    **spec,
                    "radius_m": radius_m,
                    "limit": 45,
                },
                depends_on=["anchor"],
                role="support",
            )
        )
    candidates_ref = f"${retrieval_ids[0]}"
    candidate_dependency = retrieval_ids
    if len(retrieval_ids) > 1:
        steps.append(
            _step(
                "candidates",
                "merge_places",
                {"items": [f"${step_id}" for step_id in retrieval_ids]},
                depends_on=retrieval_ids,
                role="support",
            )
        )
        candidates_ref = "$candidates"
        candidate_dependency = ["candidates"]
    if intent in {"nearby", "direction"}:
        steps.append(
            _step(
                "option_candidates",
                "recover_option_places",
                {
                    "options": options,
                    "candidates": candidates_ref,
                    "anchor": "$anchor.0.place",
                    "radius_m": radius_m,
                },
                depends_on=["anchor", *candidate_dependency],
                role="support",
            )
        )
        candidates_ref = "$option_candidates"
        candidate_dependency = ["option_candidates"]
    if intent == "direction":
        direction = _extract_requested_direction(question)
        if not direction:
            return None
        steps.append(
            _step(
                "directional_candidates",
                "filter_by_direction",
                {
                    "center": "$anchor.0.place",
                    "places": candidates_ref,
                    "direction": direction,
                },
                depends_on=["anchor", *candidate_dependency],
                role="support",
            )
        )
        candidates_ref = "$directional_candidates"
        candidate_dependency = ["directional_candidates"]
    steps.append(
        _step(
            "option_match",
            "match_options",
            {
                "options": options,
                "places": candidates_ref,
                "anchor": "$anchor.0.place",
                "mode": "radius_set" if intent == "radius" else "nearest",
            },
            depends_on=["anchor", *candidate_dependency],
            role="measure",
        )
    )
    return steps


def _step(
    step_id: str,
    operator: str,
    arguments: dict[str, Any],
    *,
    depends_on: list[str] | None = None,
    role: str,
) -> dict[str, Any]:
    from src.agent.geoflow import OPERATOR_CONTRACTS

    return {
        "id": step_id,
        "operator": operator,
        "arguments": arguments,
        "depends_on": depends_on or [],
        "output_type": OPERATOR_CONTRACTS[operator].output_type,
        "role": role,
    }


def _extract_target_type(question: str, intent: str) -> str | None:
    patterns = {
        "nearby": r"가장\s+가까운\s+(.+?)\s+중",
        "direction": r"(?:북쪽|남쪽|동쪽|서쪽)에\s+있는\s+가장\s+가까운\s+(.+?)\s+중",
        "radius": r"안에\s+있는\s+(.+?)\s+목록",
    }
    match = re.search(patterns[intent], question)
    return match.group(1).strip() if match else None


def _nearby_retrieval_specs(target: str) -> list[dict[str, Any]]:
    compact = "".join(target.split())
    official = {
        "대형마트": "MT1",
        "편의점": "CS2",
        "어린이집": "PS3",
        "유치원": "PS3",
        "학교": "SC4",
        "학원": "AC5",
        "주차장": "PK6",
        "주유소": "OL7",
        "충전소": "OL7",
        "역": "SW8",
        "지하철역": "SW8",
        "은행": "BK9",
        "문화시설": "CT1",
        "부동산": "AG2",
        "공공기관": "PO3",
        "관광명소": "AT4",
        "숙박": "AD5",
        "음식점": "FD6",
        "카페": "CE7",
        "병원": "HP8",
        "약국": "PM9",
    }
    if compact in official:
        return [{"category_code": official[compact]}]
    expansions = {
        "패스트푸드점": [{"query": "패스트푸드"}, {"category_code": "FD6"}],
        "빵집": [{"query": "빵집"}, {"query": "베이커리"}],
        "슈퍼마켓": [{"query": "슈퍼마켓"}, {"query": "마트"}],
        "박물관": [{"query": "박물관"}, {"category_code": "CT1"}],
        "갤러리": [{"query": "갤러리"}, {"category_code": "CT1"}],
    }
    return expansions.get(compact, [{"query": target}])


def _extract_radius_m(question: str) -> int:
    match = re.search(r"반경\s*([\d,]+(?:\.\d+)?)\s*(km|m)", question, re.IGNORECASE)
    if not match:
        return 2000
    radius = float(match.group(1).replace(",", ""))
    return round(radius * 1000 if match.group(2).lower() == "km" else radius)


def _extract_requested_direction(question: str) -> str | None:
    return next(
        (
            direction
            for direction in ("북쪽", "남쪽", "동쪽", "서쪽")
            if direction in question
        ),
        None,
    )


def _grounded_prediction(results: dict[str, Any], option_count: int) -> dict[str, Any] | None:
    for step_id in reversed(list(results)):
        result = results[step_id]
        if not isinstance(result, dict) or "best_option" not in result:
            continue
        predicted = _coerce_option(result.get("best_option"), option_count)
        confidence = float(result.get("confidence") or 0.0)
        if predicted is None or confidence < 0.7:
            continue
        return {
            "predicted_option": predicted,
            "confidence": confidence,
            "reason": f"Validated GeoFlow Measure {step_id} selected option {predicted}",
        }
    return None


def _extract_anchor(question: str, intent: str) -> str | None:
    separators = {
        "radius": (" 반경",),
        "trip": ("에서 출발",),
        "routing": ("에서 자동차",),
        "nearby": ("에서 가장 가까운",),
        "direction": ("에서 북쪽", "에서 남쪽", "에서 동쪽", "에서 서쪽"),
    }
    for separator in separators.get(intent, ()):
        if separator in question:
            return question.split(separator, 1)[0].strip()
    return None


def _is_shortened_name(candidate: str, expected: str) -> bool:
    candidate_key = "".join(candidate.split()).casefold()
    expected_key = "".join(expected.split()).casefold()
    return bool(candidate_key and candidate_key != expected_key and candidate_key in expected_key)


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
