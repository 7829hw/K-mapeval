from __future__ import annotations

import json
import re
import time
from datetime import datetime
from difflib import SequenceMatcher
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
    reference_expression,
    reference_roots,
    retrieve_templates,
    split_reference_arithmetic,
)
from src.llm import ChatClient, LLMUnavailableError
from src.parsing import parse_answer, parse_json_object
from src.tools import SpatialOperatorRegistry, ToolRegistry
from src.tools.spatial import (
    parse_clock_text,
    parse_coordinate_literal,
    strip_location_qualifier,
)

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
when an explicit search radius is given. A question that names a cardinal direction (동/서/남/북쪽,
북동/남동/남서/북서쪽) is direction even when it also asks for the nearest one: the direction is
the constraint that decides the answer. A question asking which place is nearest is nearby, not
distance; distance is for a numeric separation.
A question that names a place to start from plus two or more places to visit — with how long to
spend at each, or a total time available, or both — is trip, and asks either for the visiting order
or for how many places fit. Trip wins over routing whenever there is more than one stop to arrange:
routing is one origin to one destination and the properties of that single road route (its
duration, its distance, its turns, its guidance, or which detour through it is fastest).
A question that compares places against each other or relates two named anchors — which pair is
farthest apart, which candidate lies between two places, which one is close to both — is poi. It is
not distance, which reports one numeric separation, and not radius, which searches around a single
anchor with a stated radius.
Also return "target_type": the kind of place that answers the question, as the ordinary Korean
noun for it (편의점, 약국, 주유소, 카페, 은행, 병원, 주차장, 지하철역, 대형마트, 음식점, 학교 …).
When the question only describes a need, infer the kind of place that satisfies it — "우산을 사야
합니다" is 편의점, "두통약을 사야 합니다" is 약국, "기름을 넣어야 합니다" is 주유소, "현금을 찾아야
합니다" is 은행. Use null when the question is not asking for a kind of place at all.
Include all named places and spatial/temporal constraints.
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
  Kakao Mobility routes cars only, so a walking question — 걸어서, 걸어가기에 가장 가까운 —
  is answered with haversine_distance or nearest(metric="haversine"), never by asking these two
  for a walking route: the call fails and the evidence is lost.
- distance_matrix(origins,destinations OR pairs, mode="driving", priority) -> field;
  pairs is [{origin,destination,label?}], and output routes preserve pair order at $node.routes
- haversine_distance(place_a, place_b) -> amount; straight-line distance only. Output is
  {distance_m, distance_km}, so reference the node itself ($node) or $node.distance_m, never
  $node.amount. A question about travelling — 주행 거리, 이동 거리, how far you drive or walk —
  is asking for road distance and must come from directions/distance_matrix, not from this;
  a straight line is roughly four fifths of the road that follows it, which is close enough to
  land on a wrong option and far enough to be wrong
- pairwise_distances(pairs=[{place_a,place_b,label?}]) -> field
- pairwise_extremes(locations) -> amount
- bearing_to_direction(place_a, place_b) -> field
- filter_by_direction(center, places, direction) -> object, nearest first; places is a retrieved
  POI list such as $nearby, not a batch_geocode node
- nearest(anchor,candidates,metric="haversine"|"travel_time",routes?,required_type?) -> object;
  required_type keeps only candidates of that kind before ranking, and is ignored when nothing
  matches
- within_radius(center, candidates, radius_m) -> object
- select_min/select_max(items,key), sort_by(items,key), compare_routes(routes,metric) -> object
- filter_routes(routes,keyword,include=true) -> field
- extract_distance(route), extract_duration(route) -> amount
- filter_places(places,min_rating?,price_levels?,required_types?,open_now?) -> object
- steps_analysis(route,landmark?) -> field; totals are left_turn_count/right_turn_count/
  roundabout_exit_count over the whole drive. With a landmark it also reports landmark_index,
  instruction_after_landmark, and the same counts split as *_before_landmark / *_after_landmark.
  A question about turns "before"/"after" reaching a road must read the split counts, never the
  total — pass the road name as landmark
- sum_route_metrics(routes) -> amount
- aggregate_route_groups(routes,groups) -> amount; groups contains route indexes per option and
  returns option_totals plus best_distance_option and best_duration_option.
- merge_places(items) -> object; merges and de-duplicates multiple retrieval branches
- recover_option_places(options,candidates,anchor,radius_m,category_code?) -> object;
  conditionally resolves options absent from ranked retrieval and merges their POIs into the
  candidate list. Pass the same category_code the retrieval used.
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
- calculate_finish_time(start_time|arrival_time,locations,stay_durations_s?,timezone?,mode?,
  priority?) -> event; routes every leg of `locations` in order and adds the stays. Give
  start_time to ask when an itinerary ends, or arrival_time to ask the latest departure that still
  meets it — the output carries both start_time and finish_time either way. Use this rather than
  summing legs into calculate_start_time by hand
- calculate_start_time(arrival_time,duration_s,timezone) -> event
- tsp_tw(nodes,distance_matrix,time_windows?,service_times?,start_index=0,time_budget?,
  end_index?) -> network; end_index fixes the last stop when the trip must finish
  somewhere (an appointment), leaving only the stops between it and the start to order;
  distance_matrix accepts a distance_matrix node directly ($legs), which carries the square
  duration matrix in seconds; nodes must be the matching place list in the same order, with the
  starting place at start_index. service_times are the stay durations in seconds (0 for the start)
  and time_budget is the available time in seconds. Output is {order, total_cost, feasible}, where
  order indexes nodes.
- identity_measure(value) -> object; explicit Measure projection for a single source operator

Use normalized fields only: latitude, longitude, distance_m, duration_s. Complete Place objects,
or literal place names for map tools, are valid. Never use Google geometry/lat/lng/legs fields.
Copy every place name verbatim from the question and candidate options; never shorten, translate,
or remove a store/branch prefix. References must use $node.0.place, never ${node.0.place}.
Use batch_geocode for named endpoints and distance_matrix for route candidates so the graph remains
within {max_steps} nodes. For nearby, direction, and radius questions, geocode only the anchor and
retrieve the requested type with nearby_places. For nearby/direction, use recover_option_places so
only missing option evidence is resolved, then use match_options; do not geocode options upfront.
This holds even when the options already look like a complete list of places: four named options
are not a candidate set, they are answer texts, and geocoding them and taking the nearest answers
"which of these is closest" instead of the question asked.
When the question describes a *need* rather than naming a kind of place — 우산을 사야 합니다,
두통약을 사야 합니다, 기름을 넣어야 합니다, 현금을 찾아야 합니다, 끼니를 해결해야 합니다 — work out
which kind of place satisfies it and retrieve that kind. The options will include closer places of
other kinds, so a ranking that ignores the kind returns one of those. If you do rank option places
directly, filter_places(required_types=[the Korean noun for that kind]) first.
Supply the question's origin/reference place as batch_geocode.anchor to disambiguate same-name POIs.
For a trip question, geocode the start and every named stop once, then call distance_matrix with
origins = destinations = that full place list so every ordered leg is looked up in one node, and
pass that node to tsp_tw as distance_matrix with the stays as service_times and the stated total
time as time_budget. When the options are visiting orders, still build the same matrix and compare
the orders the options name. When the options are counts of places ("한 곳"/"두 곳"/…), the answer
is how many stops fit the budget, so let tsp_tw decide feasibility — never guess from the number
of places mentioned. Convert hours to seconds (1시간 = 3600).
tsp_tw.total_cost is the whole tour with the stays already in it, and travel_cost/service_cost are
its halves. Feed calculate_start_time the total_cost itself; adding the stays beside it counts
every visit twice. When the trip must finish somewhere — an appointment at a named place, with
errands on the way — give that place as end_index, or the tour will end at an errand instead.
For nearest among explicit options, geocode every option and compute deterministically.
When two anchors bound the question, every option has to be tested against both, so geocode the
two anchors and all the options in one batch_geocode and compute from those places:
- "A에서 B까지 이동하는 경로 위에 있는" asks which option adds least to the A→B trip. Compare
  haversine_distance(A,option) + haversine_distance(option,B) across the options, or route A→B
  through each option as a waypoint and take the smallest. Ranking by the distance to one anchor
  answers a different question.
- "A와 B 양쪽 모두에서 직선거리 R 이내" is an intersection: within_radius(center=A) over the
  option places, then within_radius(center=B) over that result. What survives both is the answer.
Neither shape is a neighbourhood retrieval — nearby_places around one anchor cannot see the other,
so do not answer them with nearby_places plus match_options.
A vertical bar in one option separates grouped place names; preserve its option index while
resolving each name. For a radius question use the exact radius and requested
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
                    if operator not in tool_names:
                        # Local operators spend no API call, so a place they are handed as a
                        # name is only a place if the plan already resolved it. The tools do
                        # their own name resolution through the provider.
                        arguments = _bind_named_places(arguments, results)
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
            predicted, selection = _select_option(evaluation_json, options, results)
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
        raw_steps,
        question,
        options,
        intent,
        expand_retrieval=expand_retrieval,
        inferred_type=analysis.get("target_type"),
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
    canonical = canonical_reference(value)
    expression = reference_expression(canonical)
    if expression is not None and len(expression[0]) > 1:
        # A sum over several nodes: every term must resolve to a number, or the planner meant
        # something we cannot defend and the step fails with its own reference error.
        total = expression[1]
        for name in expression[0]:
            resolved = _resolve_references(name, results)
            if isinstance(resolved, bool) or not isinstance(resolved, int | float):
                raise ValueError(f"Unknown plan reference: {value}")
            total += float(resolved)
        return total
    reference, offset = split_reference_arithmetic(canonical)
    if not reference.startswith("$"):
        return value
    root, _, remainder = reference[1:].partition(".")
    if root not in results:
        raise ValueError(f"Unknown plan reference: {value}")
    current = results[root]
    if isinstance(current, dict) and set(current) == {"error"}:
        # A failed step is recorded as `{"error": ...}` so the run continues, but that marker is
        # not evidence and must not travel on as data. Passed into the next tool it was validated
        # as a Place and produced seven more errors describing the fields an error message does
        # not have, burying the one failure that actually happened.
        raise ValueError(f"Plan reference {value} depends on failed step {root!r}")
    for part in (segment for segment in remainder.split(".") if segment):
        resolved = _descend_reference(current, part)
        if resolved is _UNRESOLVED:
            # An over-specified path degrades to the closest resolvable object rather than
            # failing the operator, mirroring upstream Spatial-Agent's lenient concept
            # reference resolution. Operators normalize the remaining shape themselves.
            return current
        current = resolved
    if offset and isinstance(current, int | float) and not isinstance(current, bool):
        # The offset is a constant the question states; applying it is leniency about where a
        # planner wrote the sum, never about what the sum is. A reference that resolves to
        # anything but a number carries no arithmetic, so it is returned untouched.
        return current + offset
    return current


# Every operator argument that means a place, in the local operator registry's own spelling.
# `batch_geocode`'s `place_names` is deliberately absent: names are what it is asked to resolve.
PLACE_VALUED_ARGUMENTS = frozenset(
    {"anchor", "candidates", "center", "locations", "place_a", "place_b", "places"}
)


def _resolved_place_index(results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every place the plan has already resolved, keyed by each name it is known under."""

    index: dict[str, dict[str, Any]] = {}

    def record(name: Any, place: Any) -> None:
        if not isinstance(name, str) or not isinstance(place, dict):
            return
        if "latitude" not in place or "longitude" not in place:
            return
        key = _name_key_for_match(strip_location_qualifier(name))
        if key:
            index.setdefault(key, place)

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if {"latitude", "longitude"} <= value.keys():
                record(value.get("name"), value)
            if isinstance(value.get("place"), dict):
                record(value.get("query"), value["place"])
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(results)
    return index


def _bind_named_places(arguments: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    """A place written as a name is the place the plan already resolved under that name.

    Planners hand `filter_by_direction` the option texts it just geocoded rather than the
    geocoded places, and the local operators cannot look a name up — they spend no API call by
    design. Dropping the names left an empty sector, which reads as "nothing lies that way"
    while the evidence for all four candidates sat in the previous step's result. A name is
    bound only when the plan itself resolved it, so this grants no evidence the run did not
    already gather; an unknown name is left alone and still fails as a missing place.
    """

    named = {
        key: value
        for key, value in arguments.items()
        if key in PLACE_VALUED_ARGUMENTS and _holds_place_name(value)
    }
    if not named:
        return arguments
    index = _resolved_place_index(results)
    if not index:
        return arguments
    bound = dict(arguments)
    for key, value in named.items():
        if isinstance(value, list):
            bound[key] = [_named_place(item, index) for item in value]
        else:
            bound[key] = _named_place(value, index)
    return bound


def _holds_place_name(value: Any) -> bool:
    if isinstance(value, str):
        return not value.startswith("$") and parse_coordinate_literal(value) is None
    if isinstance(value, list):
        return any(_holds_place_name(item) for item in value)
    return False


def _named_place(value: Any, index: dict[str, dict[str, Any]]) -> Any:
    if not _holds_place_name(value):
        return value
    return index.get(_name_key_for_match(strip_location_qualifier(str(value))), value)


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


_INDEXED_REFERENCE = re.compile(r"^(\$[A-Za-z_][\w-]*)\.\d+(?:\.\w+)*$")


def _whole_list_reference(value: Any) -> Any:
    """An itinerary is the whole geocoded list, so an index into it is a planner slip.

    `calculate_finish_time` has no legs to time when it is handed one stop, and a plan that
    wrote `$places.0` where it meant `$places` failed as a validation error before the clock
    ran. The node it indexed into is the itinerary it geocoded, so drop the index.
    """

    if not isinstance(value, str):
        return value
    match = _INDEXED_REFERENCE.match(canonical_reference(value))
    return match.group(1) if match else value


# Every `trip_latest_departure` question opens the same way: the appointment's place, its clock
# time, and only then the errands on the way. The deadline is what makes that place the end of
# the trip, so the clock has to be part of the pattern — a bare "X에서" is any of the stops.
_DESTINATION_PATTERNS = (
    r"^\s*(.+?)에서\s*(?:오전|오후|아침|저녁|밤)?\s*\d{1,2}\s*시(?:\s*\d{1,2}\s*분)?에?\s*"
    r"(?:약속이|미팅이|모임이)",
    r"(.+?)(?:에|까지)\s*(?:오전|오후|아침|저녁|밤)?\s*\d{1,2}\s*시(?:\s*\d{1,2}\s*분)?(?:까지)?\s*"
    r"(?:도착|가야)",
)


def _extract_trip_destination(question: str) -> str | None:
    for pattern in _DESTINATION_PATTERNS:
        match = re.search(pattern, question)
        if match:
            return match.group(1).strip()
    return None


def _index_of_name(names: list[str], wanted: str) -> int | None:
    """Position of a question's place among the names the plan geocoded."""

    key = _name_key_for_match(wanted)
    for index, name in enumerate(names):
        if _name_key_for_match(name) == key:
            return index
    for index, name in enumerate(names):
        candidate = _name_key_for_match(name)
        if candidate and key and (candidate in key or key in candidate):
            return index
    return None


def _name_key_for_match(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


# "…를 차례로 둘러본 뒤 가예로 돌아옵니다" — the return is a leg the question states, not an
# optional flourish, and a plan that stops at the last sight computes an arrival one drive short.
_RETURN_PATTERNS = (
    r"(?:으로|로)\s*돌아\s*(?:옵니다|온다|와야|가|옵)",
    r"다시\s*\S+(?:으로|로)\s*(?:돌아|와)",
)


def _closed_itinerary(
    itinerary: list[Any], question: str, trip_node_names: list[str]
) -> list[Any] | None:
    """The round trip the question states: out from the base, through the stops, back to it."""

    base = _trip_origin(question, [_location_name(stop) for stop in itinerary])
    if base is None:
        base = _trip_origin(question, trip_node_names)
    if base is None:
        # Without a stated departure place there is nothing to close the loop on; leave the
        # plan's own itinerary alone rather than guess which end is the base.
        return None
    key = _name_key_for_match(base)
    middle = [stop for stop in itinerary if _name_key_for_match(_location_name(stop)) != key]
    closed = [base, *middle, base]
    return closed if closed != itinerary else None


def _trip_origin(question: str, names: list[str]) -> str | None:
    """Which of the plan's own names the question departs from.

    Matched against the names in hand rather than read out of the sentence: a free-form parse of
    "오전 10시 00분에 키이토에서 자동차로 출발해" has to decide for itself where the clock ends
    and the place begins, and the plan already knows what the places are called.
    """

    for name in names:
        if name and re.search(
            rf"{re.escape(name)}\s*(?:에서|에)\s*(?:자동차로\s*)?출발", question
        ):
            return name
    return None


def _named_stop(value: Any, trip_node_names: list[str]) -> Any:
    """An itinerary entry as the question names it, resolving a reference into the node list."""

    if not isinstance(value, str) or not value.strip().startswith("$"):
        return value
    reference = canonical_reference(value)
    parts = [part for part in reference.lstrip("$").split(".") if part]
    index = next((int(part) for part in parts[1:] if part.isdigit()), None)
    if index is not None and 0 <= index < len(trip_node_names):
        return trip_node_names[index]
    return value


def _returns_to_start(question: str) -> bool:
    return any(re.search(pattern, question) for pattern in _RETURN_PATTERNS)


_TOUR_TOTAL_FIELDS = frozenset({"total_cost", ""})


def _counts_its_own_stays(value: str, tour_totals: set[str]) -> bool:
    """Does this reference a `tsp_tw` whole-tour cost, which already includes the stays?"""

    parsed = reference_expression(canonical_reference(value))
    if parsed is None:
        return False
    for name in parsed[0]:
        root, _, path = name[1:].partition(".")
        if root in tour_totals and path in _TOUR_TOTAL_FIELDS:
            return True
    return False


def _sole_reference(value: str) -> str | None:
    parsed = reference_expression(canonical_reference(value))
    return parsed[0][0] if parsed and len(parsed[0]) == 1 else None


def _carries_written_sum(value: str) -> bool:
    """Did the planner already add something to this reference?"""

    parsed = reference_expression(canonical_reference(value))
    return parsed is not None and (len(parsed[0]) > 1 or parsed[1] != 0.0)


def _step_sources(step: dict[str, Any]) -> set[str]:
    """Every node id a step reads from, whether declared as a dependency or only referenced."""

    sources = {str(name) for name in step.get("depends_on") or []}
    sources.update(reference_roots(step.get("arguments") or step.get("params") or {}))
    return sources


def _ground_graph_literals(
    steps: list[dict[str, Any]],
    question: str,
    options: list[str],
    intent: str,
    *,
    expand_retrieval: bool = True,
    inferred_type: str | None = None,
) -> list[dict[str, Any]]:
    """Bind verbatim question literals after drafting, before graph validation.

    GeoFlow factors such as a radius, requested direction, and the candidate option texts are
    constants from the question, not values that an LLM should invent or route through a
    synthetic operator output.
    """

    anchor = _extract_anchor(question, intent)
    target = None
    if intent in {"nearby", "direction", "radius"}:
        # The question's own words first; the Analysis stage's inference when it did not name a
        # type. A question that describes a need never states one, and without this the
        # retrieval loses its category and the ranking answers "nearest of anything".
        target = _extract_target_type(question, intent) or inferred_type
    radius_m = _extract_radius_m(question) if intent == "radius" else None
    specifications = _nearby_retrieval_specs(target) if target else []
    # tsp_tw's service_times are positional, so the stays can only be bound once the node list the
    # plan geocoded is known — it is the place order every downstream index refers to.
    route_priority = _extract_route_priority(question)
    trip_node_names: list[str] = []
    if intent == "trip":
        trip_node_names = next(
            (
                [str(name) for name in (step.get("arguments") or {}).get("place_names") or []]
                for step in steps
                if step.get("operator") == "batch_geocode"
                and len((step.get("arguments") or {}).get("place_names") or []) > 2
            ),
            [],
        )
    # `steps_analysis` has nothing to count when the route it reads was fetched without its
    # turn-by-turn guidance, and `directions` omits them by default. The operator then reported
    # zero turns for every question and the generation stage answered from prose instead, which
    # is a confident wrong number rather than a failure. A route a step analysis consumes is a
    # route whose steps are needed, so bind it here rather than hope the prompt lands.
    # Which node ids produce a tour whose cost already carries the stays.
    tour_totals = {
        str(step.get("id"))
        for step in steps
        if step.get("operator") == "tsp_tw"
    }
    stepwise_sources = {
        source
        for step in steps
        if step.get("operator") == "steps_analysis"
        for source in _step_sources(step)
    }
    # The place names the question states outright, for repairing one a plan copied short: a plan
    # that geocoded `문래` where the question says `빈칸 문래` routed from another place entirely
    # and counted another route's turns, with every stage reporting success.
    question_places = [
        name
        for name in (
            anchor,
            _extract_trip_destination(question),
            *(_extract_compared_places(question) or ()),
        )
        if name
    ]
    grounded: list[dict[str, Any]] = []
    for step in steps:
        operator = step.get("operator")
        arguments = _verbatim_place_names(
            dict(step.get("arguments") or step.get("params") or {}),
            question,
            options,
            question_places,
        )
        if operator == "nearby_places" and specifications:
            arguments.pop("query", None)
            arguments.pop("category_code", None)
            arguments["radius_m"] = radius_m if radius_m is not None else RETRIEVAL_RADIUS_M
            arguments["limit"] = RETRIEVAL_LIMIT
            grounded.extend(
                _retrieval_steps(step, arguments, specifications, expand=expand_retrieval)
            )
            continue
        if route_priority and operator in _PRIORITY_OPERATORS:
            arguments["priority"] = route_priority
        if operator == "calculate_start_time" and intent == "trip":
            stays, _ = _extract_trip_schedule(question)
            # Only when the travel total is computed by another node: a reference to a route sum
            # carries travel and nothing else, so the stays are certainly missing. A literal may
            # already include them, and binding on top of that would count them twice.
            duration = arguments.get("duration_s")
            if isinstance(duration, str) and duration.strip().startswith("$"):
                if _counts_its_own_stays(duration, tour_totals):
                    # `tsp_tw.total_cost` is the whole tour, stays included, so anything added
                    # beside it counts every visit twice — a whole stay, wider than the gap
                    # between two options. That is true of a written `+ 4500` and equally of the
                    # stays bound here, so strip the one and withhold the other. The operator's
                    # contract says so regardless of what any constant happens to equal.
                    arguments["duration_s"] = _sole_reference(duration) or duration
                    arguments.pop("stay_durations_s", None)
                elif _carries_written_sum(duration):
                    # Some other written sum: whatever the planner added, adding the stays on top
                    # would count them twice.
                    arguments.pop("stay_durations_s", None)
                elif stays:
                    arguments["stay_durations_s"] = list(stays.values())
            grounded.append({**step, "arguments": arguments})
            continue
        if operator == "identity_measure" and not arguments.get("value"):
            # A Measure with nothing to measure is a planner leftover; what it meant is the node
            # it depends on. Failing here threw away a graph whose evidence was already gathered.
            source = next(iter(step.get("depends_on") or []), None)
            if source:
                arguments["value"] = f"${source}"
            grounded.append({**step, "arguments": arguments})
            continue
        if operator == "directions" and step.get("id") in stepwise_sources:
            arguments["include_steps"] = True
            if route_priority and operator in _PRIORITY_OPERATORS:
                arguments["priority"] = route_priority
            grounded.append({**step, "arguments": arguments})
            continue
        if operator == "calculate_finish_time" and intent == "trip":
            stays, _ = _extract_trip_schedule(question)
            locations = _whole_list_reference(arguments.get("locations"))
            arguments["locations"] = locations
            # One stay per location, in the order the itinerary visits them. The stays are stated
            # in the question exactly; a plan that drops the last one or invents one for the
            # return lands a whole visit away, which is wider than the gap between two options.
            # When `locations` is a reference the names are not in hand here — but the itinerary
            # is exactly what the trip's `batch_geocode` node lists, and the operator resolves the
            # reference to that same list. Without this the planner's own stays were left to
            # mismatch the resolved length, and the args model rejected the call outright.
            itinerary: list[Any] = []
            if isinstance(locations, list) and len(locations) > 1:
                # A stop written as `$geo.1.place` is a name the geocode node already holds, and
                # looking a stay up by the reference text finds nothing — which bound every stay
                # to zero and lost four hours off a finish time without failing anything.
                itinerary = [_named_stop(item, trip_node_names) for item in locations]
            elif isinstance(locations, str) and len(trip_node_names) > 1:
                itinerary = list(trip_node_names)
            if itinerary and _returns_to_start(question):
                # "X에서 출발해 …를 둘러본 뒤 X로 돌아옵니다" states both endpoints; only the order
                # of the stops between them is the plan's business. A plan that drops the return
                # arrives one drive early, and one that drops the departure loses its first leg
                # *and* shifts every stay onto the wrong stop — neither fails, both answer an
                # option away.
                closed = _closed_itinerary(itinerary, question, trip_node_names)
                if closed is not None:
                    itinerary = closed
                    arguments["locations"] = closed
            if stays and itinerary:
                arguments["stay_durations_s"] = [
                    _stay_stated_for(question, _location_name(item)) for item in itinerary
                ]
            grounded.append({**step, "arguments": arguments})
            continue
        if operator == "tsp_tw" and intent == "trip":
            stays, budget = _extract_trip_schedule(question)
            if budget is not None:
                arguments["time_budget"] = budget
            names = trip_node_names
            destination = _extract_trip_destination(question)
            if destination and names:
                # The place the deadline names is where the trip ends, positionally against the
                # same node list the stays line up with. Left free, the search reordered the
                # itinerary so the tour finished at an errand — cheaper, and an answer to a
                # different question.
                index = _index_of_name(names, destination)
                if index is not None and index != int(arguments.get("start_index") or 0):
                    arguments["end_index"] = index
            if names and stays:
                # service_times must line up with the node list, and the start is not a visit.
                arguments["service_times"] = [
                    0.0 if index == 0 else _stay_for(stays, name)
                    for index, name in enumerate(names)
                ]
            grounded.append({**step, "arguments": arguments})
            continue
        if operator == "nearest" and target:
            # Bound here rather than asked for in the prompt: a planner that ranks the option
            # texts directly produces a graph with no retrieval to carry the category, and told
            # only in prose it keeps doing it. The kind asked for is a question literal like the
            # radius and the direction, so it is bound like one.
            arguments["required_type"] = target
            grounded.append({**step, "arguments": arguments})
            continue
        if operator == "filter_by_direction" and intent == "direction":
            direction = _extract_requested_direction(question)
            if direction:
                arguments["direction"] = direction
            grounded.append({**step, "arguments": arguments})
            continue
        if operator == "recover_option_places":
            # Recovery has to look for the same kind of place the retrieval did, or an option is
            # satisfied by any namesake: "목동" in a station question matched 교보문고 목동점.
            arguments["options"] = options
            if radius_m is not None:
                arguments["radius_m"] = radius_m
            category = next(
                (
                    specification["category_code"]
                    for specification in specifications
                    if specification.get("category_code")
                ),
                None,
            )
            if category and len(specifications) == 1:
                arguments["category_code"] = category
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
            # `arguments` is the copy every branch above edits; appending the original step here
            # threw those edits away, which is how a bound routing priority never reached the
            # `directions` call it was bound for.
            grounded.append({**step, "arguments": arguments})
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


def _stay_stated_for(question: str, name: str) -> float:
    """How long the question says to spend at this place, in seconds; 0 when it says nothing.

    Looked up by the name the plan already holds rather than parsed out of the prose: reading
    names out of the sentence swallowed the clause in front of the first one, so the starting
    point inherited a visit it never makes.
    """

    key = name.strip()
    if not key:
        return 0.0
    match = re.search(
        rf"{re.escape(key)}\s*(?:을|를|에서|에)?\s*(?:약\s*)?([\d.]+)\s*(시간|분)", question
    )
    if not match:
        return 0.0
    amount = float(match.group(1))
    return amount * 3600 if match.group(2) == "시간" else amount * 60


def _location_name(value: Any) -> str:
    """The name a planner used for an itinerary stop, whatever shape it wrote it in."""

    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("name", "query", "place_name", "address"):
            found = value.get(key)
            if isinstance(found, str) and found:
                return found
            if isinstance(found, dict):
                nested = found.get("name")
                if isinstance(nested, str) and nested:
                    return nested
    return ""


def _stay_for(stays: dict[str, float], name: str) -> float:
    """Look a node's stay up by the name the question used, tolerating a decorated node label."""

    if name in stays:
        return stays[name]
    for stated, seconds in stays.items():
        if stated in name or name in stated:
            return seconds
    return 0.0


def _extract_trip_schedule(question: str) -> tuple[dict[str, float], float | None]:
    """Read the stays and the total time a trip question states, in seconds.

    These are question literals exactly as a radius or a direction is: the plan may choose the
    order, but how long each visit takes and how much time there is are given, not inferred. A
    planner that rounds 1.5시간 to an hour, or reads the budget off the wrong number, produces a
    feasible-looking plan for a trip nobody asked about.
    """

    stays: dict[str, float] = {}
    # A stop is stated as "X를 2시간" or as "X에서 30분"; reading only the first shape returned
    # nothing for a question full of errands and left the departure time short by all of them.
    for match in re.finditer(
        r"([^,.]+?)(?:을|를|에서|에)\s*(?:약\s*)?([\d.]+)\s*(시간|분)", question
    ):
        name = match.group(1).strip()
        # The sentence that introduces the stay list ends in "…있습니다. " — keep only the name.
        name = re.split(r"[.!?]\s*", name)[-1].strip()
        if not name:
            continue
        amount = float(match.group(2))
        stays[name] = amount * 3600 if match.group(3) == "시간" else amount * 60
    budget_match = re.search(r"총\s*([\d.]+)\s*시간", question)
    budget = float(budget_match.group(1)) * 3600 if budget_match else None
    return stays, budget


# A radius is stated in ordinary Korean, not in one keyword. "반경 600m", "직선거리 600m 이내"
# and a bare "600m 안에" all name the same constraint, and recognizing only the first silently
# substituted the 2000 m default for the number the question actually asked about.
_RADIUS_PATTERNS = (
    r"(?:반경|직선거리|거리)\s*([\d,]+(?:\.\d+)?)\s*(km|m)\b",
    r"([\d,]+(?:\.\d+)?)\s*(km|m)\s*(?:이내|안|이하|미만)",
)


# Which route a question means. Kakao's RECOMMEND re-optimizes against live traffic, so the road
# it picks — and therefore its distance, its turns and the roads it names — changes between calls.
# A question that has to be gradable says which route it means, and the choice is bound here like
# the radius and the direction rather than left to whichever default a tool happens to carry.
_PRIORITY_PHRASES = (
    ("DISTANCE", ("최단 경로", "가장 짧은 경로", "최단거리 경로", "거리가 가장 짧은")),
    ("TIME", ("가장 빠른 경로", "최단 시간", "가장 빨리", "제일 빠른")),
)

_PRIORITY_OPERATORS = frozenset(
    {"directions", "travel_time", "distance_matrix", "calculate_finish_time"}
)


def _extract_route_priority(question: str) -> str | None:
    for priority, phrases in _PRIORITY_PHRASES:
        if any(phrase in question for phrase in phrases):
            return priority
    return None


def _extract_radius_m(question: str) -> int:
    for pattern in _RADIUS_PATTERNS:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            radius = float(match.group(1).replace(",", ""))
            return round(radius * 1000 if match.group(2).lower() == "km" else radius)
    return 2000


def _extract_requested_direction(question: str) -> str | None:
    return next(
        (
            direction
            for direction in ("북쪽", "남쪽", "동쪽", "서쪽")
            if direction in question
        ),
        None,
    )


# "I am at X" is said several ways, and an anchor phrasing the splitter does not know reads as
# no anchor at all — the geocoder then loses its disambiguation and `recover_option_places` its
# centre. Tried before the intent-specific separators because it names the anchor outright.
_ANCHOR_PATTERNS = (
    r"지금\s+(.+?)에\s+(?:있|와\s*있|머물)",
    r"현재\s+(.+?)에\s+(?:있|머물)",
    r"^(.+?)에\s+있는데",
)


# "A와 B 양쪽 모두에서 …" reads to the splitters as one long place name. It is two, and there is
# no single anchor to bind — the question asks for the intersection of two neighbourhoods.
_TWO_ANCHOR_MARKERS = ("양쪽", "둘 다", "모두에서")


def _extract_anchor(question: str, intent: str) -> str | None:
    for pattern in _ANCHOR_PATTERNS:
        match = re.search(pattern, question)
        if match and match.group(1).strip():
            return _single_anchor(match.group(1).strip())
    separators = {
        "radius": (" 반경", "에서 직선거리"),
        "trip": ("에서 출발",),
        # "A에서 B까지 자동차로" is the other half of the routing phrasing, and the first "에서"
        # is where the drive starts. Without it a plan that geocoded `문래` for the question's
        # `빈칸 문래` kept the shortened name, resolved a different place, and counted the turns
        # of a route nobody asked about.
        "routing": ("에서 자동차", "에서"),
        "nearby": ("에서 가장 가까운",),
        "direction": ("에서 북쪽", "에서 남쪽", "에서 동쪽", "에서 서쪽"),
    }
    for separator in separators.get(intent, ()):
        if separator in question:
            return _single_anchor(question.split(separator, 1)[0].strip())
    return None


# Arguments that carry a place *name* the question or the options wrote down. `query` is absent
# on purpose: a retrieval's query is a kind of place, and the question need not contain the word.
_NAME_ARGUMENTS = ("place_names", "options", "anchor", "origin", "destination")
# How close a planner's spelling has to be to a literal before it is treated as that literal.
_TRANSCRIPTION_SIMILARITY = 0.85


def _verbatim_place_names(
    arguments: dict[str, Any],
    question: str,
    options: list[str],
    question_places: list[str] | None = None,
) -> dict[str, Any]:
    """Restore a place name the planner copied out wrong.

    The prompt says to copy every name verbatim, and mostly they are. When they are not the
    lookup fails outright — `잠원한강공원 눈쌨매장` for the question's `눈썰매장` matched nothing,
    and the step that needed it, plus everything downstream, was lost. A name that is nearly a
    literal the question wrote is that literal; a name that resembles nothing in the question is
    left exactly as the planner wrote it, so this can only ever restore evidence.
    """

    literals = [option for option in options if option.strip()]
    stated = [name for name in (question_places or []) if name.strip()]
    corrected = dict(arguments)
    for key in _NAME_ARGUMENTS:
        value = corrected.get(key)
        if isinstance(value, str):
            corrected[key] = _verbatim_name(value, question, literals, stated)
        elif isinstance(value, list):
            corrected[key] = [
                _verbatim_name(item, question, literals, stated) if isinstance(item, str) else item
                for item in value
            ]
    return corrected


def _verbatim_name(
    name: str, question: str, options: list[str], stated: list[str] | None = None
) -> str:
    candidate = name.strip()
    if candidate in options:
        return name
    for literal in stated or []:
        # A name the question states, of which the planner wrote only a part. `빈칸 문래` came
        # through as `문래`, which resolves — to 문래동창작촌, a different place, so the route
        # measured was a different route and every stage reported success.
        if _is_shortened_name(candidate, literal):
            return literal
    if len(candidate) < 4 or candidate in question:
        return name
    best = candidate
    best_ratio = _TRANSCRIPTION_SIMILARITY
    for literal in [*options, *_question_spans(question, len(candidate))]:
        ratio = SequenceMatcher(None, candidate, literal).ratio()
        if ratio > best_ratio:
            best, best_ratio = literal, ratio
    return best


def _question_spans(question: str, length: int) -> list[str]:
    """Every span of the question about as long as the name, as written."""

    text = question.strip()
    spans: list[str] = []
    for width in (length - 1, length, length + 1):
        if width < 4:
            continue
        spans.extend(text[start : start + width] for start in range(0, len(text) - width + 1))
    return spans


def _single_anchor(candidate: str) -> str | None:
    """The anchor, unless the text in hand names two of them.

    `'가좌동 마을극장과 증산역 6호선 양쪽 모두'` was bound as one place name and searched as one:
    no place matched, and the retrieval it anchored never happened. Two anchors are not an anchor,
    and the plan's own two-place composition answers the question.
    """

    if any(marker in candidate for marker in _TWO_ANCHOR_MARKERS):
        return None
    return candidate or None


def _is_shortened_name(candidate: str, expected: str) -> bool:
    candidate_key = "".join(candidate.split()).casefold()
    expected_key = "".join(expected.split()).casefold()
    return bool(candidate_key and candidate_key != expected_key and candidate_key in expected_key)


# The operators default to it and every question in these families is stated in it.
_CLOCK_TIMEZONE = "Asia/Seoul"


def _computed_clock_option(results: dict[str, Any] | None, options: list[str]) -> int | None:
    """The option nearest the wall clock the graph computed, when there is exactly one."""

    if not results or len(options) < 2:
        return None
    parsed_options = [parse_clock_text(option, _CLOCK_TIMEZONE) for option in options]
    if any(option is None for option in parsed_options):
        return None
    # A clock operator reports both ends and computes one of them: run forwards and the start is
    # the question's, run backwards and the finish is. Which is which is not visible in the field
    # names — preferring `finish_time` answered "when must I leave" with the deadline the question
    # had just handed over — so the operator names the field it derived and only that one counts.
    derived = {
        parsed
        for value in results.values()
        if isinstance(value, dict)
        and isinstance(value.get("derived_clock"), str)
        and isinstance(value.get(str(value["derived_clock"])), str)
        and (parsed := _clock_moment(str(value[str(value["derived_clock"])]))) is not None
    }
    if len(derived) != 1:
        # No computed clock, or two of them and no way to know which the question asked for.
        return None
    computed = next(iter(derived))
    minutes = [
        option.hour * 60 + option.minute for option in parsed_options if option is not None
    ]
    distances = sorted(
        (abs(value - (computed.hour * 60 + computed.minute)), index)
        for index, value in enumerate(minutes)
    )
    (nearest_gap, nearest), (runner_up_gap, _) = distances[0], distances[1]
    if nearest_gap * 2 >= runner_up_gap:
        # The nearest option is always *some* option, so the clock counts only when it picks one
        # decisively — twice as close as the next. A plan that lost its stays computed 12:30
        # against options at 14:23 and 15:23, was 113 minutes from one and 173 from the other,
        # and took the first; the generation stage had added the four stated hours itself and
        # written the right answer. A clock that does land on its option still outranks that
        # prose, because it is computed evidence and the prose is recalled.
        return None
    return nearest


def _clock_moment(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return parse_clock_text(value, _CLOCK_TIMEZONE)


def _select_option(
    payload: dict[str, Any],
    options: list[str],
    results: dict[str, Any] | None = None,
) -> tuple[int | None, str]:
    """Reconcile the generated answer text with the generated index.

    Upstream Spatial-Agent selects on the answer *text* and derives the index from it, because
    a model that names the right candidate can still miscount its position. Exact text wins,
    the declared index is the next authority, and a single containment match is the last
    resort.

    A clock the operators computed outranks all three. When every option is a wall-clock time and
    the graph produced exactly one, the generation stage's job is to report that time, not to
    revise it — and revising is what it did: a trace reading 14:40 was answered as 16:33
    "accounting for real-world traffic, parking, and navigation variations", and one reading
    13:36 as 15:46 for an "unrecorded return trip". Both adjustments are invented evidence, and
    both moved the answer exactly one option.
    """

    computed = _computed_clock_option(results, options)
    if computed is not None:
        return computed, "computed_clock"
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
