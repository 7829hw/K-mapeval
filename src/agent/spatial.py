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
from src.llm import ChatClient, LLMUnavailableError
from src.parsing import parse_answer, parse_json_object
from src.tools import SpatialOperatorRegistry, ToolRegistry

SUPPORTED_INTENTS = frozenset(
    {"nearby", "poi", "routing", "trip", "type", "direction", "distance", "radius"}
)

# Retrieval budget for grounded nearby searches. Nearby results come back nearest-first, so a
# wide radius never hurts ranking, while a narrow one silently truncates the candidate set the
# Measure step compares options against.
RETRIEVAL_LIMIT = 45
RETRIEVAL_RADIUS_M = 20_000

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
   (sub_condition < condition < support < measure; contextual roles are unordered);
3. operator output type compatibility; 4. executable operators and available arguments;
5. connectivity from contextual input through every node to a measure.

Return JSON only:
{"graph":[{"id":"places","operator":"batch_geocode","arguments":{"place_names":["A","B"]},"depends_on":[],"output_type":"object","role":"extent","concept_ids":["anchor"]}]}

Exact operator contracts:
- place_search(query?,center?,category_code?,radius_m?,min_rating?,open_now?,limit=5) -> object
- batch_geocode(place_names, anchor?, radius_m=20000, limit=1) -> object; anchor biases ambiguous
  names toward the question's reference location. Output preserves order and each item has
  {query, place, candidates}; reference the best match as $node.0.place
- geocode(address, limit=5) -> location
- reverse_geocode(latitude,longitude,limit=5) -> location
- place_details(place_id) -> object
- batch_place_details(place_ids) -> object
- nearby_places(center, query|category_code, radius_m, limit) -> object, nearest first
  Kakao category codes: MT1 mart, CS2 convenience store, PS3 childcare, SC4 school,
  AC5 academy, PK6 parking, OL7 gas/charging, SW8 subway station, BK9 bank,
  CT1 culture, AG2 real estate, PO3 public institution, AT4 attraction, AD5 lodging,
  FD6 restaurant, CE7 cafe, HP8 hospital, PM9 pharmacy. Use the matching code whenever possible.
- directions(origin,destination,mode="driving",priority,waypoints?,include_steps=false) -> field
- travel_time(origin, destination, mode="driving", priority) -> field Route
- distance_matrix(origins,destinations OR pairs, mode="driving", priority) -> field;
  pairs is [{origin,destination,label?}], and output routes preserve pair order at $node.routes
- haversine_distance(place_a, place_b) -> amount; output is {distance_m, distance_km}, so
  reference the node itself ($node) or $node.distance_m, never $node.amount
- pairwise_distances(pairs=[{place_a,place_b,label?}]) -> field
- pairwise_extremes(locations) -> amount
- bearing_to_direction(place_a, place_b) -> field
- filter_by_direction(center, places, direction) -> object, nearest first; places is a retrieved
  POI list such as $nearby, not a batch_geocode node
- nearest(anchor,candidates,metric="haversine"|"travel_time",routes?) -> object
- within_radius(center, candidates, radius_m) -> object
- select_min/select_max(items,key), sort_by(items,key), compare_routes(routes,metric) -> object
- filter_routes(routes,keyword,include=true) -> field
- extract_distance(route), extract_duration(route) -> amount
- filter_places(places,min_rating?,price_levels?,required_types?,open_now?) -> object
- steps_analysis(route,landmark?) -> field
- sum_route_metrics(routes) -> amount
- aggregate_route_groups(routes,groups) -> amount; groups contains route indexes per option and
  returns option_totals plus best_distance_option and best_duration_option.
- merge_places(items) -> object; merges and de-duplicates multiple retrieval branches
- recover_option_places(options,candidates,anchor,radius_m) -> object; conditionally resolves
  options absent from ranked retrieval and merges their POIs into the candidate list
- match_options(options,places,anchor?,mode) -> object; grounds options to retrieved POIs
- match_distance_options(distance,options) -> object; maps a computed distance to numeric options.
  Pass the haversine node itself as distance and copy the option texts verbatim
- match_type_options(place,options) -> object; maps normalized category evidence to options
- events_from_objects(objects,event_type,timestamp_field?) -> event
- filter_events(events,field,operator,value) -> event
- build_route_network(nodes,edges) -> network
- calculate_proportion(numerator,denominator) -> proportion
- open_at_time(schedule,local_time,timezone) -> event
- timezone(latitude,longitude,timestamp?) -> event
- timezone_convert(local_time,from_timezone,to_timezone) -> event
- calculate_finish_time(start_time,locations,stay_durations_s?,timezone?,mode?) -> event
- calculate_start_time(arrival_time,duration_s,timezone) -> event
- tsp_tw(nodes,distance_matrix,time_windows?,service_times?,start_index=0,time_budget?) -> network
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
Exact candidate-text matches beat semantic matches, and semantic matches beat inference.
Respect candidate-index to candidate-text mapping. Numbering is 0-based. Return JSON only:
{"predicted_answer":"exact candidate text","predicted_option":1,"confidence":0.8,
"reason":"brief evidence-based reason"}
predicted_answer must be copied verbatim from the candidate list, and predicted_option must be the
0-based index of that same candidate.
For radius/list questions, compare the resolved in-radius POI names with each option as a set;
never choose an option merely because it contains more items or looks like a list.
Treat deterministic best_option fields as evidence, but make the final selection in this generation
step after checking them against the complete final state and trace.
An operator that reports an error contributed no evidence; fall back to the surviving steps rather
than to the failed step's intent.
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
    "poi": (
        "Use the retrieved place attributes and computed coordinates directly. For pairwise "
        "comparisons, compare the metric of every option pair before selecting."
    ),
    "routing": (
        "Read route distance_m and duration_s from the executed routes. Shortest uses "
        "distance_m, fastest uses duration_s, and a named via-route must be isolated first."
    ),
    "trip": (
        "Evaluate each option as an ordered sequence and compare option_totals. Order matters, "
        "so never reorder an option's stops when matching it to the aggregated totals."
    ),
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
                        max(self.max_steps, len(fallback_graph)),
                        expand_retrieval=False,
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
            predicted, selection = _select_option(evaluation_json, options)
            if predicted is None:
                predicted = parse_answer(evaluation.content, option_count=len(options))
                selection = "answer_marker" if predicted is not None else "unresolved"
            trace.append(
                {
                    "stage": "evaluate",
                    "predicted_option": predicted,
                    "predicted_answer": evaluation_json.get("predicted_answer"),
                    "selection_method": selection,
                    "confidence": evaluation_json.get("confidence"),
                    "reason": evaluation_json.get("reason", ""),
                }
            )

            if predicted is not None:
                response_text = f"^^{predicted}^^"
            trace.append({"stage": "generate", "response": response_text})
        except LLMUnavailableError as exc:
            failure_type = "llm_unavailable"
            failure_message = f"{type(exc).__name__}: {exc}"
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
    *,
    expand_retrieval: bool = True,
):
    raw_steps = plan.get("graph") if plan.get("graph") is not None else plan.get("steps")
    if not isinstance(raw_steps, list):
        raise ValueError("GeoFlow response does not contain a graph")
    if len(raw_steps) > max_steps:
        raise ValueError(
            f"GeoFlow graph has {len(raw_steps)} operators, exceeding "
            f"MAX_REASONING_STEPS={max_steps}"
        )
    grounded = _ground_graph_literals(
        raw_steps, question, options, intent, expand_retrieval=expand_retrieval
    )
    factorized = factorize_geoflow(analysis, {"graph": grounded})
    # The planner budget above governs what the planner authored. Retrieval fan-out added
    # during grounding is deterministic and gets its own allowance on top of it.
    steps, constraints = normalize_and_validate_graph(
        factorized.as_dict(), max_steps=max(max_steps, len(grounded))
    )
    return factorized, steps, constraints


def _resolve_output_binding(output: Any, path: str) -> Any:
    if path in {"", "$"}:
        return output
    reference = path[2:] if path.startswith("$.") else path.lstrip("$")
    current = output
    for part in (value for value in reference.split(".") if value):
        resolved = _descend_reference(current, part)
        if resolved is _UNRESOLVED:
            # Concept state materialization must never abort the run; an unusable binding
            # path falls back to the operator output it was meant to project.
            return current
        current = resolved
    return current


_UNRESOLVED = object()

_REFERENCE_ALIASES = {
    "lat": "latitude",
    "lng": "longitude",
    "lon": "longitude",
    "amount": "distance_m",
    "value": "distance_m",
    "meters": "distance_m",
}


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
    root, _, remainder = reference[1:].partition(".")
    if root not in results:
        raise ValueError(f"Unknown plan reference: {value}")
    current = results[root]
    for part in (segment for segment in remainder.split(".") if segment):
        resolved = _descend_reference(current, part)
        if resolved is _UNRESOLVED:
            # An over-specified path degrades to the closest resolvable object rather than
            # failing the operator, mirroring upstream Spatial-Agent's lenient concept
            # reference resolution. Operators normalize the remaining shape themselves.
            return current
        current = resolved
    return current


def _descend_reference(current: Any, part: str) -> Any:
    if isinstance(current, list):
        try:
            return current[int(part)]
        except (ValueError, IndexError):
            return _UNRESOLVED
    if not isinstance(current, dict):
        return _UNRESOLVED
    if part == "geometry" and {"latitude", "longitude"} <= current.keys():
        return {
            "lat": current["latitude"],
            "lng": current["longitude"],
            "location": current,
        }
    key = part if part in current else _REFERENCE_ALIASES.get(part, part)
    if key in current:
        return current[key]
    return _UNRESOLVED


def _ground_graph_literals(
    steps: list[dict[str, Any]],
    question: str,
    options: list[str],
    intent: str,
    *,
    expand_retrieval: bool = True,
) -> list[dict[str, Any]]:
    """Bind verbatim question literals after drafting, before graph validation.

    GeoFlow factors such as a radius, requested direction, and the candidate option texts are
    constants from the question, not values that an LLM should invent or route through a
    synthetic operator output.
    """

    anchor = _extract_anchor(question, intent)
    target = (
        _extract_target_type(question, intent)
        if intent in {"nearby", "direction", "radius"}
        else None
    )
    radius_m = _extract_radius_m(question) if intent == "radius" else None
    specifications = _nearby_retrieval_specs(target) if target else []
    grounded: list[dict[str, Any]] = []
    for step in steps:
        operator = step.get("operator")
        arguments = dict(step.get("arguments") or step.get("params") or {})
        if operator == "nearby_places" and specifications:
            arguments.pop("query", None)
            arguments.pop("category_code", None)
            arguments["radius_m"] = radius_m if radius_m is not None else RETRIEVAL_RADIUS_M
            arguments["limit"] = RETRIEVAL_LIMIT
            grounded.extend(
                _retrieval_steps(step, arguments, specifications, expand=expand_retrieval)
            )
            continue
        if operator == "filter_by_direction" and intent == "direction":
            direction = _extract_requested_direction(question)
            if direction:
                arguments["direction"] = direction
            grounded.append({**step, "arguments": arguments})
            continue
        if operator in {"match_options", "match_distance_options", "match_type_options"}:
            # The Measure step compares against the candidate texts verbatim; a planner that
            # paraphrases or numerically re-types them breaks the comparison.
            arguments["options"] = options
            if operator == "match_options":
                arguments["mode"] = "radius_set" if intent == "radius" else "nearest"
            grounded.append({**step, "arguments": arguments})
            continue
        if operator != "batch_geocode":
            grounded.append(step)
            continue
        names = list(arguments.get("place_names") or [])
        pair = _extract_compared_places(question) if intent == "distance" else None
        if pair and len(names) == 2:
            # These two are POIs the question states precisely, not option shorthand, so a
            # neighbourhood hit still has to match by name: 자양2동문고 must not become 초원책서점.
            arguments["strict_names"] = True
            # The place names are question literals like the option texts are. A planner that
            # "helpfully" completes 만화시장 into 가좌시장만화카페 or 마천1치안센터 into 웅동파출소
            # sends the geocoder after a different POI in a different province, and every operator
            # downstream computes correctly over the wrong evidence.
            names = list(pair)
            arguments["anchor"] = names[0]
        elif anchor:
            if names and names[0] != anchor:
                names[0] = anchor
            arguments["anchor"] = anchor
        if (
            intent in {"nearby", "direction", "routing"}
            and len(names) == len(options) + 1
            and all("|" not in option for option in options)
        ):
            names[1:] = options
        arguments["place_names"] = names
        grounded.append({**step, "arguments": arguments})
    return grounded


def _retrieval_steps(
    step: dict[str, Any],
    arguments: dict[str, Any],
    specifications: list[dict[str, Any]],
    *,
    expand: bool,
) -> list[dict[str, Any]]:
    """Fan a retrieval node out over every Kakao spelling of the requested place type.

    Korean place types map onto several Kakao keywords or category codes (경찰서 also appears
    as 파출소/지구대/치안센터), so a single retrieval silently loses candidates. The branches
    merge back under the planner's original node id, which keeps downstream references valid.
    """

    if not expand or len(specifications) == 1:
        return [{**step, "arguments": {**arguments, **specifications[0]}}]
    step_id = str(step.get("id") or "nearby")
    branches = [
        {
            key: value
            for key, value in step.items()
            if key not in {"concept_ids", "output_bindings", "input_concepts"}
        }
        | {
            "id": f"{step_id}__r{index + 1}",
            "arguments": {**arguments, **specification},
        }
        for index, specification in enumerate(specifications)
    ]
    merged = {
        **step,
        "id": step_id,
        "operator": "merge_places",
        "arguments": {"items": [f"${branch['id']}" for branch in branches]},
        "depends_on": [str(branch["id"]) for branch in branches],
        "output_type": "object",
    }
    return [*branches, merged]


_COMPARED_PLACES = re.compile(r"^(.+?)\s*(?:및|와|과)\s+(.+?)\s+사이의\s+직선거리")


def _extract_compared_places(question: str) -> tuple[str, str] | None:
    """The two POI names a straight-line-distance question compares, verbatim."""

    match = _COMPARED_PLACES.search(question)
    if not match:
        return None
    first, second = (part.strip() for part in match.groups())
    return (first, second) if first and second else None


def _bind_prevalidated_template(
    intent: str,
    question: str,
    options: list[str],
    anchor: str | None,
) -> list[dict[str, Any]] | None:
    if intent == "distance":
        pair = _extract_compared_places(question)
        if not pair:
            return None
        place_a, place_b = pair
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
    radius_m = _extract_radius_m(question) if intent == "radius" else RETRIEVAL_RADIUS_M
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
                    "limit": RETRIEVAL_LIMIT,
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
    """Map a requested Korean place type onto the Kakao searches that actually cover it."""

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
    # Korean place types whose Kakao POI names diverge from the requested word. Each family
    # is generic over the type, never over a question or an option string.
    expansions = {
        "패스트푸드점": [{"query": "패스트푸드"}, {"category_code": "FD6"}],
        "빵집": [{"query": "빵집"}, {"query": "베이커리"}],
        "슈퍼마켓": [{"query": "슈퍼마켓"}, {"query": "마트"}],
        "박물관": [{"query": "박물관"}, {"category_code": "CT1"}],
        "갤러리": [{"query": "갤러리"}, {"category_code": "CT1"}],
        "경찰서": [
            {"query": "경찰서"},
            {"query": "파출소"},
            {"query": "지구대"},
            {"query": "치안센터"},
        ],
        "도서관": [{"query": "도서관"}, {"query": "문고"}, {"query": "도서실"}],
        "우체국": [{"query": "우체국"}, {"query": "우편취급국"}],
        "서점": [{"query": "서점"}, {"query": "책방"}],
        "화장품매장": [{"query": "화장품"}],
        "전자제품매장": [{"query": "전자제품"}, {"query": "가전"}],
        "꽃집": [{"query": "꽃집"}, {"query": "플라워"}],
        "세탁소": [{"query": "세탁소"}, {"query": "빨래방"}],
        "정육점": [{"query": "정육점"}, {"query": "축산"}],
        "문구점": [{"query": "문구"}],
        "안경점": [{"query": "안경"}],
        "관광안내소": [{"query": "관광안내소"}, {"query": "관광안내"}],
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


def _select_option(payload: dict[str, Any], options: list[str]) -> tuple[int | None, str]:
    """Reconcile the generated answer text with the generated index.

    Upstream Spatial-Agent selects on the answer *text* and derives the index from it, because
    a model that names the right candidate can still miscount its position. Exact text wins,
    the declared index is the next authority, and a single containment match is the last
    resort.
    """

    text = payload.get("predicted_answer")
    exact = _match_option_text(text, options, strict=True) if isinstance(text, str) else None
    if exact is not None:
        return exact, "exact_answer_text"
    index = _coerce_option(payload.get("predicted_option"), len(options))
    if index is not None:
        return index, "predicted_option"
    contained = _match_option_text(text, options, strict=False) if isinstance(text, str) else None
    if contained is not None:
        return contained, "answer_text_containment"
    return None, "unresolved"


def _match_option_text(value: str, options: list[str], *, strict: bool) -> int | None:
    key = _option_key(value)
    if not key:
        return None
    matches = [
        index
        for index, option in enumerate(options)
        if (option_key := _option_key(option))
        and (
            option_key == key
            or (not strict and (key in option_key or option_key in key))
        )
    ]
    return matches[0] if len(matches) == 1 else None


def _option_key(value: str) -> str:
    return "".join(str(value).split()).casefold()


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
