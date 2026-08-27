from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
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
    ExampleEmbedder,
    OperatorContract,
    canonical_reference,
    factorize_geoflow,
    normalize_analysis,
    normalize_and_validate_graph,
    reference_expression,
    retrieve_examples,
    retrieve_templates,
    split_reference_arithmetic,
)
from src.agent.semantics import (
    factorize_semantic_graph,
    transform_catalogue,
)
from src.llm import (
    ChatClient,
    LLMContextOverflowError,
    LLMOutputTruncatedError,
    LLMUnavailableError,
    TokenUsage,
)
from src.parsing import parse_answer, parse_json_object
from src.tools import SpatialOperatorRegistry, ToolRegistry
from src.tools.map import canonical_retrieval_specs
from src.tools.spatial import (
    parse_coordinate_literal,
    split_place_type,
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
When the question only describes a need rather than naming a kind, infer the kind of place that
satisfies it — "우산을 사야 합니다" is 편의점. Name the kind that actually satisfies the need, at
the granularity the need implies: a neighbouring kind will usually be closer, and naming it
answers a different question. Use null when the question is not asking for a kind of place at
all.
Include all named places and spatial/temporal constraints.
The intent label is evaluation metadata only. Later stages use the concepts, roles, attributes,
measure, and question text, so make those fields complete without relying on the label.
When the question states a search radius or a compass sector, put the value on the concept that
carries it, as "attributes": {"radius_m": 600} in metres and {"direction": "북동쪽"}. Grounding
reads these only when it cannot find the literal in the question itself, so a phrasing you had to
interpret is exactly the case worth recording; do not restate one the sentence spells out and do
not invent one the question does not give.
Return JSON only:
{"intent":"direction","concepts":[{"id":"anchor","text":"서울역","concept_type":"location","role":"extent","attributes":{},"depends_on":[]},{"id":"sector","text":"북동쪽","concept_type":"field","role":"sub_condition","attributes":{"direction":"북동쪽"},"depends_on":["anchor"]},{"id":"answer","text":"direction","concept_type":"field","role":"measure","attributes":{},"depends_on":["anchor"]}],"measure":"direction"}
Do not answer the multiple-choice question and do not invent coordinates."""

GRAPH_PROMPT = """You are Spatial-Agent's Concept Transformation stage. Read the spatial concept
analysis and say which transformations the question needs, in what order. You are not choosing
tools: which operator performs a transformation is decided afterwards, deterministically, from the
concept types and the transformation you name. Name the relation, not the implementation.

Transformations:
{transform_catalogue}

Every node is {"id", "transform", "inputs", "role"} and may carry "concept_ids", "factors" and
"via".
- inputs are the ids of the nodes this one consumes, in order. The first input of a search, a
  filter, a sort or a route is the place it is measured from.
- via, on a ROUTE_MEASURE, lists what the route passes through on the way -- concept ids, in the
  order the route reaches them. A route stated as "A에서 B를 들러 C까지" is one route from A to C
  with B in via, never two routes. Leave it out when the route passes through nothing.
- concept_ids are the analysis concepts this node is about. A RESOLVE_PLACES node lists the
  concepts whose places it resolves, and the names come from those concepts -- do not retype a
  place name, and do not translate or shorten one.
- role is one of extent, temporal_extent, sub_condition, condition, support, measure.
- factors are semantic parameters only: ordinal (1-based: second nearest is 2), measure
  ("distance" or "duration"), key, extreme ("min"/"max"), aggregate ("sum"/"difference"/
  "proportion"/"count"), scope ("options" to resolve the candidate texts, "listed" to resolve the
  ones the question itself lists, "attribute" to narrow by the stated kind's modifier, "one" for
  a single place).
  Never put a radius, a compass sector, a stay, a time budget or a place name in factors: the
  analysis already extracted those and they are bound for you.

The graph must satisfy all five paper constraints:
1. acyclicity; 2. role ordering
   (sub_condition < condition < support < measure; contextual roles are unordered);
3. output type compatibility; 4. every transformation reachable; 5. connectivity from contextual
   input through every node to a measure. Keep it within {max_steps} nodes.

Return JSON only:
{"graph":[{"id":"anchor","transform":"RESOLVE_PLACES","inputs":[],"concept_ids":["anchor"],"role":"extent"},{"id":"found","transform":"PLACE_SEARCH","inputs":["anchor"],"concept_ids":["target"],"role":"support"},{"id":"ranked","transform":"SORT","inputs":["anchor","found"],"role":"support"},{"id":"kth","transform":"ORDINAL_SELECT","inputs":["ranked"],"factors":{"ordinal":2},"role":"support"},{"id":"answer","transform":"MATCH_OPTIONS","inputs":["kth"],"role":"measure"}]}

What the question is about, not what the tools are:
- The candidate options are answer texts, not a candidate set, and the analysis lists them as
  concepts only so you can see them. Whenever the question asks for a *kind* of place -- a bank,
  a pharmacy, a cafe, the nearest or k-th nearest of something -- the graph is
  RESOLVE_PLACES (the anchor only) -> PLACE_SEARCH -> SORT -> ORDINAL_SELECT -> MATCH_OPTIONS.
  Do not resolve the option texts as the candidates. The gold answer is the k-th of the whole
  neighbourhood, and the options are a handful of places drawn from various ranks, so a graph
  that resolves the four options and ranks them answers "which of these four is closest" -- a
  different question, with a different answer. Resolve the options only when the question names
  them as the things to compare against each other.
- When the question describes a *need* rather than naming a kind of place, the analysis has
  already worked out which kind satisfies it. PLACE_SEARCH will use it.
- A question that ranks by 주행거리/이동거리 is decided in metres and one that ranks by time in
  seconds. Say which with factors.measure; the orders a trip question offers sit about 2% apart,
  so the wrong measure answers a different question.
- The whole distance or duration of an itinerary the question already orders is ROUTE_MATRIX ->
  SELECT_LEGS -> AGGREGATE. The matrix holds every pair of stops; the trip drives only the
  consecutive ones, and SELECT_LEGS is how the graph says which those are.
- When the options are counts of places, the answer is how many stops fit the stated time, so let
  ROUTE_OPTIMIZE decide feasibility rather than counting the places the question mentions.
- Two anchors bound some questions -- "A와 B 양쪽 모두에서", "A에서 B까지 이동하는 경로 위에" --
  and every option has to be tested against both. Resolve both anchors and compose two
  transformations over them; a single neighbourhood search around one anchor cannot see the other.
- An ordinal is a factor, not a shape: "두 번째로 가까운" is SORT then ORDINAL_SELECT with
  ordinal=2, over the same graph the nearest question uses with ordinal=1.
- A question that lists its own candidates -- "… 아래 목록 중 몇 곳인가요? (A, B, C, D)" -- is
  asking about those and not about the neighbourhood. Resolve them with RESOLVE_PLACES and
  scope="listed", measure each from the anchor, FILTER, then AGGREGATE with aggregate="count".
  A retrieval answers "how many are there", which is a different number.
- A kind of place written as a modifier and a broad kind -- 중식 음식점, 양식 음식점 -- needs the
  narrowing between the retrieval and the ranking: PLACE_SEARCH the broad kind, FILTER with
  scope="attribute", then rank. The analysis extracted both halves and the filter is bound for
  you. Ranking straight off the retrieval answers with the nearest place of the broad kind, and
  the other options are exactly that.
- A measure between two places has to say which two. Give DISTANCE_MEASURE the two nodes that
  resolved them, or name the two concepts in concept_ids. Two measures over one resolved list
  are the same measure twice, and their difference is zero.

Do not select an option and do not use the gold answer."""

REPAIR_PROMPT = """Repair the supplied transformation graph so it passes the listed validation
error. Keep the question's semantics and the retrieved template, answer in the same
transformation vocabulary as before -- the error names operators because that is what the
transformations were mapped onto, but you choose transformations, not operators -- and stay within
{max_steps} nodes. Return only {"graph":[...]} JSON."""

EVALUATOR_PROMPT = """You are Spatial-Agent's grounded response generation stage. Select exactly
one candidate using the final GeoFlow state and topological execution evidence. Large collections
are losslessly counted but represented by bounded samples; `_omitted_items` states how many
records were left out of a sample. Computed
distance, direction, category, route, and option_totals evidence has priority over prior knowledge.
Exact candidate-text matches beat semantic matches, and semantic matches beat inference.
Respect candidate-index to candidate-text mapping. Numbering is 0-based. Return JSON only:
{"predicted_answer":"exact candidate text","predicted_option":1,"confidence":0.8,
"reason":"brief evidence-based reason"}
predicted_answer must be copied verbatim from the candidate list, and predicted_option must be the
0-based index of that same candidate.
Treat deterministic operator output as the primary evidence: the graph was built to compute this
answer, and its Measure node is where the answer is read from. Fields such as best_option, ranked,
option_totals and present_option_members are computed, not guessed, so prefer them over anything
you would otherwise infer -- but make the selection here, after checking them against the complete
final state and the trace.
An operator that reports an error contributed no evidence; fall back to the surviving steps rather
than to the failed step's intended purpose.
Read every quantity in the unit the question asks for and the operator reports it in. Do not
substitute one measure for another -- a road distance is not a straight-line distance, seconds are
not metres, and an option that lists more items is not thereby a larger set.
Never return an option outside the supplied candidates."""

# Execution traces are retained in full for auditability, but copying every resolved Place through
# every downstream argument and concept binding into the final LLM prompt grows quadratically with
# a retrieval.  Forty-five nearby results pushed that prompt past a 65,536-token context window.
# These limits apply only to the evaluation prompt's evidence projection, never to execution or to
# the trace stored in the report.
EVALUATION_ARGUMENT_LIST_LIMIT = 4
EVALUATION_RESULT_LIST_LIMIT = 10
EVALUATION_STATE_LIST_LIMIT = 6
EVALUATION_MAX_DEPTH = 6
EVALUATION_STRING_LIMIT = 512

class SpatialAgent(BenchmarkAgent):
    """Concept-graph grounding, constrained factorization, execution, and generation."""

    agent_type = "spatial_agent"

    def __init__(
        self,
        llm: ChatClient,
        tools: ToolRegistry,
        *,
        max_steps: int = 15,
        example_embedder: ExampleEmbedder | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        #: Optional embedding backend for few-shot example retrieval. `None` uses the
        #: deterministic concept-overlap scorer, which is what runs here: the deployment these
        #: benchmarks use serves one chat model and answers `/v1/embeddings` with 404.
        self.example_embedder = example_embedder
        self.operators = SpatialOperatorRegistry()
        self.max_steps = max_steps
        available = {
            *(schema["function"]["name"] for schema in self.tools.schemas()),
            *self.operators.names,
        }
        missing = set(OPERATOR_CONTRACTS) - available
        if missing:
            raise ValueError(f"GeoFlow operators are not executable: {', '.join(sorted(missing))}")
        #: What factorization may map a transformation onto. Taken from the registries rather
        #: than from the contract table so a deployment missing a tool cannot have a graph built
        #: against it.
        self.executable_operators = frozenset(available)

    def _graph_prompt(self) -> str:
        """The planner prompt, with the vocabulary rendered from the transform table itself.

        Rendered rather than duplicated so a transformation cannot exist in one and not the
        other: adding one to `TRANSFORMS` is what puts it in front of the planner.
        """

        return GRAPH_PROMPT.replace(
            "{transform_catalogue}", transform_catalogue()
        ).replace("{max_steps}", str(self.max_steps))

    def answer(self, question: str, options: list[str]) -> AgentResult:
        started = time.perf_counter()
        api_before = self.tools.provider.api_call_count
        cache_hits_before = self.tools.provider.cache_hit_count
        cache_misses_before = self.tools.provider.cache_miss_count
        tools_before = self.tools.tool_call_count
        trace = self.new_trace()
        failure_type: str | None = None
        failure_message: str | None = None
        response_text = ""
        predicted: int | None = None
        predicted_intent: str | None = None
        reasoning_steps = 0
        usage = TokenUsage()
        execution_errors: list[dict[str, str]] = []
        semantic_nodes = 0
        concrete_nodes = 0
        try:
            analysis_response = self.llm.chat(
                [
                    {"role": "system", "content": ANALYSIS_PROMPT},
                    {"role": "user", "content": format_question(question, options)},
                ]
            )
            reasoning_steps += 1
            usage += analysis_response.usage
            raw_analysis = parse_json_object(analysis_response.content)
            fallback_intent = _heuristic_intent(question)
            # Read the question's stated factors from the raw reply first, so Concept Analysis
            # can be completed from them when the stage came up short. Nothing here is new
            # evidence -- it is what the deterministic extractors already found in the same
            # question, which the fallback used to discard in favour of the question text.
            stated = extract_facts(raw_analysis, question)
            analysis = normalize_analysis(raw_analysis, question, fallback_intent, facts=stated)
            predicted_intent = str(analysis.pop("intent")).lower()
            if predicted_intent not in SUPPORTED_INTENTS:
                predicted_intent = fallback_intent
            trace.append({"stage": "analyze", "intent": predicted_intent, **analysis})
            # The label remains in the trace/report, but every executable stage receives a view
            # that cannot inspect it. Popping it makes the label reporting metadata rather than a
            # hidden template, grounding, validation, repair, or evaluation router.
            runtime_analysis = analysis
            facts = extract_facts(runtime_analysis, question)

            # Two retrievals, two questions. Which macro-template the concept graph calls for is
            # structural -- a measure over a network, a field narrowed by a sub-condition -- and
            # is read off the concepts and roles. Which worked example resembles this question is
            # a similarity judgement, and that is what an embedding is for.
            templates = retrieve_templates(runtime_analysis, question)
            examples = retrieve_examples(
                runtime_analysis, question, embed=self.example_embedder
            )
            trace.append(
                {
                    "stage": "retrieve_templates",
                    "templates": [template["name"] for template in templates],
                    "examples": [example["name"] for example in examples],
                }
            )

            plan_response = self.llm.chat(
                [
                    {
                        "role": "system",
                        "content": self._graph_prompt(),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"{format_question(question, options)}\n\n"
                            "Spatial concept analysis:\n"
                            f"{json.dumps(runtime_analysis, ensure_ascii=False)}\n\n"
                            "Retrieved pre-validated templates and examples:\n"
                            f"{json.dumps(templates, ensure_ascii=False)}"
                        ),
                    },
                ]
            )
            reasoning_steps += 1
            usage += plan_response.usage
            plan = parse_json_object(plan_response.content)
            trace.append({"stage": "compose", "graph": plan.get("graph") or plan.get("steps")})
            try:
                factorized, steps, constraints, semantic = _factorize_validate_plan(
                    runtime_analysis,
                    plan,
                    question,
                    options,
                    facts,
                    self.max_steps,
                    retrieval_specs=self.tools.retrieval_specs,
                    available_operators=self.executable_operators,
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
                                self._graph_prompt()
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
                                f"Analysis: {json.dumps(runtime_analysis, ensure_ascii=False)}\n"
                                f"Templates: {json.dumps(templates, ensure_ascii=False)}\n"
                                f"Invalid graph: {json.dumps(plan, ensure_ascii=False)}"
                            ),
                        },
                    ]
                )
                reasoning_steps += 1
                usage += repair_response.usage
                repaired_plan = parse_json_object(repair_response.content)
                trace.append(
                    {
                        "stage": "repair",
                        "graph": repaired_plan.get("graph") or repaired_plan.get("steps"),
                    }
                )
                try:
                    factorized, steps, constraints, semantic = _factorize_validate_plan(
                        runtime_analysis,
                        repaired_plan,
                        question,
                        options,
                        facts,
                        self.max_steps,
                        retrieval_specs=self.tools.retrieval_specs,
                        available_operators=self.executable_operators,
                    )
                except ValueError as repair_error:
                    # Upstream has no type or role check to fail in the first place. The repair
                    # may already have fixed the structural error and be rejected only by one of
                    # those local checks, so relax the *repaired* graph first. Going straight back
                    # to the original discarded a valid repair and retried the very structural
                    # error the repair had removed. The original remains the final fallback when
                    # the repair is structurally invalid too; this order is identical for every
                    # intent and introduces no family-specific solver.
                    try:
                        factorized, steps, constraints, semantic = _factorize_validate_plan(
                            runtime_analysis,
                            repaired_plan,
                            question,
                            options,
                            facts,
                            self.max_steps,
                            retrieval_specs=self.tools.retrieval_specs,
                            available_operators=self.executable_operators,
                            strict_types=False,
                        )
                    except ValueError as repaired_lenient_error:
                        trace.append(
                            {
                                "stage": "validate",
                                "status": "invalid",
                                "mode": "lenient",
                                "plan_source": "repaired",
                                "error": str(repaired_lenient_error),
                            }
                        )
                        try:
                            factorized, steps, constraints, semantic = _factorize_validate_plan(
                                runtime_analysis,
                                plan,
                                question,
                                options,
                                facts,
                                self.max_steps,
                                retrieval_specs=self.tools.retrieval_specs,
                                available_operators=self.executable_operators,
                                strict_types=False,
                            )
                        except ValueError as original_lenient_error:
                            trace.append(
                                {
                                    "stage": "validate",
                                    "status": "invalid",
                                    "mode": "lenient",
                                    "plan_source": "original",
                                    "error": str(original_lenient_error),
                                }
                            )
                            raise GraphValidationError(
                                str(original_lenient_error)
                            ) from original_lenient_error
                        trace.append(
                            {
                                "stage": "validate",
                                "status": "lenient",
                                "plan_source": "original",
                                "error": str(graph_error),
                            }
                        )
                    else:
                        trace.append(
                            {
                                "stage": "validate",
                                "status": "lenient",
                                "plan_source": "repaired",
                                "error": str(repair_error),
                            }
                        )
            # G -> G' as this run performed it: which transformation each node asked for, which
            # operator answered, and which precedence rule decided. `concrete_nodes` counts the
            # nodes the planner named an operator for anyway, which is the measure of whether
            # the semantic vocabulary took.
            semantic_nodes = len(semantic.graph)
            concrete_nodes = len(semantic.concrete_nodes)
            trace.append({"stage": "transform", **semantic.as_dict()})
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
                    # A place written as a name is the place this plan already resolved. The
                    # local operators spend no API call, so a name they are handed is only a
                    # place if the plan resolved it — and since the five baseline tools stopped
                    # resolving names behind the tool call, the same now holds for a `directions`
                    # or `nearby_places` node. Binding grants no evidence the run had not already
                    # gathered; a name the plan never resolved is left alone and still fails.
                    arguments = _bind_named_pairs(
                        _bind_named_places(arguments, results), results
                    )
                    if operator not in tool_names:
                        arguments = _bind_step_references(arguments, results)
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
                if entry.get("status") == "error":
                    execution_errors.append(
                        {
                            "step_id": step_id,
                            "operator": operator,
                            "error": str(entry.get("error") or "Unknown operator execution error"),
                        }
                    )
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

            evaluation_evidence = _compact_evaluation_evidence(
                execution_log, concept_state
            )
            evaluation = self.llm.chat(
                [
                    {"role": "system", "content": EVALUATOR_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"{format_question(question, options)}\n\n"
                            "Compacted GeoFlow topological execution evidence:\n"
                            f"{json.dumps(evaluation_evidence, ensure_ascii=False)}"
                        ),
                    },
                ]
            )
            reasoning_steps += 1
            usage += evaluation.usage
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
        except LLMOutputTruncatedError as exc:
            failure_type = "llm_output_truncated"
            failure_message = f"{type(exc).__name__}: {exc}"
            usage += exc.usage
        except LLMContextOverflowError as exc:
            failure_type = "llm_context_overflow"
            failure_message = f"{type(exc).__name__}: {exc}"
        except GraphValidationError as exc:
            # Neither the draft nor the repair produced a graph that passes the paper's
            # constraints. Nothing was executed; say so, rather than filing it beside a crash.
            failure_type = "graph_validation_failure"
            failure_message = str(exc)
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
            llm_calls=reasoning_steps,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            reasoning_chars=usage.reasoning_chars,
            execution_errors=execution_errors,
            graph_nodes=semantic_nodes,
            planner_named_operator_nodes=concrete_nodes,
            latency_ms=(time.perf_counter() - started) * 1000,
            failure_type=failure_type,
            failure_message=failure_message,
            trace=trace,
        )


class GraphValidationError(RuntimeError):
    """Every validation attempt was spent and no graph passed.

    Reported as its own failure type rather than as `agent_reasoning_failure`. The distinction is
    what a report needs to separate "the planner could not draft a valid graph for this question"
    from "something in this port raised while running one" -- the first is a result about the
    architecture, the second is a defect. They were pooled, and the pooled count was read as the
    first.
    """


#: How a canonical place type is searched for. The provider answers it -- a category code is
#: Kakao's vocabulary, not the operator graph's -- and the default keeps this module runnable
#: with no provider at all, which is what every grounding test and the offline replay use.
RetrievalSpecs = Callable[[str], list[dict[str, Any]]]


def _factorize_validate_plan(
    analysis: dict[str, Any],
    plan: dict[str, Any],
    question: str,
    options: list[str],
    facts: GroundingFacts,
    max_steps: int,
    *,
    strict_types: bool = True,
    retrieval_specs: RetrievalSpecs = canonical_retrieval_specs,
    available_operators: frozenset[str] = frozenset(OPERATOR_CONTRACTS),
):
    raw_steps = plan.get("graph") if plan.get("graph") is not None else plan.get("steps")
    if not isinstance(raw_steps, list):
        raise ValueError("GeoFlow response does not contain a graph")
    if len(raw_steps) > max_steps:
        raise ValueError(
            f"GeoFlow graph has {len(raw_steps)} operators, exceeding "
            f"MAX_REASONING_STEPS={max_steps}"
        )
    # G -> G'. The planner answered in transformations; which operator performs each is decided
    # here, deterministically, from the concept types and the transformation alone. A planner
    # that named operators anyway is carried through as already-factorized and counted, because
    # losing a graph that would have executed to a vocabulary preference is a worse trade than
    # knowing how often it happens.
    semantic = factorize_semantic_graph(
        raw_steps,
        concepts=analysis.get("concepts") or [],
        options=options,
        facts=facts,
        available=available_operators,
    )
    grounded = _ground_graph_literals(
        semantic.graph, question, options, facts, retrieval_specs=retrieval_specs
    )
    factorized = factorize_geoflow(analysis, {"graph": grounded}, strict_types=strict_types)
    # The planner budget above governs what the planner authored. Retrieval fan-out added
    # during grounding is deterministic and gets its own allowance on top of it.
    steps, constraints = normalize_and_validate_graph(
        factorized.as_dict(),
        max_steps=max(max_steps, len(grounded)),
        strict_types=strict_types,
    )
    return factorized, steps, constraints, semantic


def _compact_evaluation_evidence(
    execution_log: list[dict[str, Any]], concept_state: dict[str, Any]
) -> dict[str, Any]:
    """Project executed evidence into a bounded prompt without changing the stored trace.

    Operator arguments contain resolved upstream outputs, so a 45-place retrieval is repeated in
    the retrieval result, ranking arguments, ranking result, later arguments, and concept state.
    The evaluator needs the operation, status, constants, measurements, ranks and candidate names;
    it does not need every duplicate provider record.  Collection samples retain their exact size
    through an ``_omitted_items`` marker, while downstream Measure results (normally already small)
    remain intact.
    """

    steps: list[dict[str, Any]] = []
    for entry in execution_log:
        projected = {
            key: entry[key]
            for key in ("id", "operator", "role", "output_type", "status", "error")
            if key in entry
        }
        if "arguments" in entry:
            projected["arguments"] = _compact_evaluation_value(
                entry["arguments"], list_limit=EVALUATION_ARGUMENT_LIST_LIMIT
            )
        if "result" in entry:
            projected["result"] = _compact_evaluation_value(
                entry["result"], list_limit=EVALUATION_RESULT_LIST_LIMIT
            )
        steps.append(projected)
    return {
        "steps": steps,
        "final_state": _compact_evaluation_value(
            concept_state, list_limit=EVALUATION_STATE_LIST_LIMIT
        ),
    }


def _compact_evaluation_value(
    value: Any, *, list_limit: int, depth: int = 0
) -> Any:
    """Recursively bound repeated collections while preserving answer-bearing scalar fields."""

    if value is None:
        return None
    if isinstance(value, str):
        if len(value) <= EVALUATION_STRING_LIMIT:
            return value
        omitted = len(value) - EVALUATION_STRING_LIMIT
        return f"{value[:EVALUATION_STRING_LIMIT]}… <{omitted} chars omitted>"
    if isinstance(value, dict):
        if depth >= EVALUATION_MAX_DEPTH:
            return {"_omitted_fields": len(value)}
        return {
            str(key): _compact_evaluation_value(
                item, list_limit=list_limit, depth=depth + 1
            )
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list | tuple):
        if depth >= EVALUATION_MAX_DEPTH:
            return {"_collection_size": len(value)}
        sample = [
            _compact_evaluation_value(item, list_limit=list_limit, depth=depth + 1)
            for item in value[:list_limit]
        ]
        if len(value) > list_limit:
            sample.append({"_omitted_items": len(value) - list_limit})
        return sample
    return value


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
    {
        "anchor",
        "candidates",
        "center",
        "destination",
        "destinations",
        "locations",
        "origin",
        "origins",
        "place_a",
        "place_b",
        "places",
        "waypoints",
    }
)
# The endpoint keys inside a `pairs` entry, which is where the pairwise operators keep theirs.
PLACE_VALUED_PAIR_KEYS = frozenset({"place_a", "place_b", "origin", "destination"})


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


def _bind_step_references(arguments: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    """A list of bare node ids where their results belong.

    `select_max(items=["d0","d1","d2","d3"], key="distance_m")` names four steps of the plan and
    forgets the `$`. Nothing resolved them, so the ranking had no comparable item and the
    comparison the whole plan was built for was lost. Only when *every* entry names a step the
    run has already executed — one that does not is a string the planner meant literally.
    """

    items = arguments.get("items")
    if not isinstance(items, list) or not items:
        return arguments
    if not all(isinstance(item, str) and item in results for item in items):
        return arguments
    return {**arguments, "items": [results[item] for item in items]}


def _bind_named_pairs(arguments: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    """The same binding one level down, where `pairs` keeps its endpoints.

    `pairwise_distances` takes `[{place_a, place_b}]`, and a planner fills those with the option
    texts it geocoded a step earlier. Unbound they are names the operator cannot look up, and the
    pair is reported as an unresolved endpoint — 64 of them in one run of the farthest-pair
    family, which is every comparison the question asks for.
    """

    pairs = arguments.get("pairs")
    if not isinstance(pairs, list) or not any(isinstance(pair, dict) for pair in pairs):
        return arguments
    index = _resolved_place_index(results)
    if not index:
        return arguments
    return {
        **arguments,
        "pairs": [
            {
                key: (
                    _named_place(value, index) if key in PLACE_VALUED_PAIR_KEYS else value
                )
                for key, value in pair.items()
            }
            if isinstance(pair, dict)
            else pair
            for pair in pairs
        ],
    }


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



def _closed_itinerary(
    itinerary: list[Any], facts: GroundingFacts, trip_node_names: list[str]
) -> list[Any] | None:
    """The round trip the question states: out from the base, through the stops, back to it.

    The base is the departure `extract_facts` already read out of the question; matching it here
    against the plan's own names is binding, not a second reading of the sentence.
    """

    base = _match_stated_name(facts.trip_origin, [_location_name(stop) for stop in itinerary])
    if base is None:
        base = _match_stated_name(facts.trip_origin, trip_node_names)
    if base is None:
        # Without a stated departure place there is nothing to close the loop on; leave the
        # plan's own itinerary alone rather than guess which end is the base.
        return None
    key = _name_key_for_match(base)
    middle = [stop for stop in itinerary if _name_key_for_match(_location_name(stop)) != key]
    closed = [base, *middle, base]
    return closed if closed != itinerary else None


def _match_stated_name(stated: str | None, names: list[str]) -> str | None:
    """Which of the plan's own names is the one the question stated.

    Matched rather than re-parsed: the plan already knows what its places are called, and the
    stated name may carry a particle or a clause the plan's copy dropped.
    """

    if not stated:
        return None
    key = _name_key_for_match(stated)
    for name in names:
        if name and _name_key_for_match(name) == key:
            return name
    for name in names:
        if name and (key in _name_key_for_match(name) or _name_key_for_match(name) in key):
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


@dataclass(frozen=True)
class GroundingFacts:
    """What the question states, read once, before any node of the graph is looked at.

    Each field is a spatial factor the paper's concept graph is supposed to carry: the extent to
    search from, the kind of thing sought, the sub-conditions that narrow it, and the objective.
    Grounding used to reach for these one at a time and only when `analysis["intent"]` matched a
    hard-coded set -- a radius was read only from a question the Analysis stage had labelled
    `radius`, a compared pair only from one it had labelled `distance`. Whether a question
    *states* a radius is a property of the question; whether a classifier said "radius" is a
    property of the classifier, and on the recorded runs the two disagree often: 21 of 90
    `nearby_subtype_kth` graphs were labelled `poi`, so their retrieval never received the kind
    of place the question names.

    Presence is the gate now. A fact that is not in the question is `None`, and the branch that
    needs it does not run.
    """

    anchor: str | None = None
    target_type: str | None = None
    #: The half of a stated kind that narrows it: 중식 of "중식 음식점". The broad half decides
    #: what to retrieve and this decides which of it qualifies, and they are different questions.
    target_subtype: str | None = None
    #: The candidates a question offers outright, as a parenthesised list. A question that names
    #: its candidates is asking about those and not about the neighbourhood: "반경 300m 이내에
    #: 있는 은행은 아래 목록 중 몇 곳인가요? (A, B, C, D)" counts A..D, and a retrieval counts
    #: every bank Kakao knows within the radius, which is a different number.
    listed_places: tuple[str, ...] = ()
    radius_m: int | None = None
    direction: str | None = None
    compared_pair: tuple[str, str] | None = None
    route_priority: str | None = None
    returns_to_start: bool = False
    stated_order: bool = False
    # The temporal sub-conditions a trip states. `stays` is a tuple of pairs rather than a mapping
    # so the whole record stays frozen and hashable, which is what lets it be attached to the
    # concept graph and compared between revisions.
    stays: tuple[tuple[str, float], ...] = ()
    time_budget_s: float | None = None
    trip_destination: str | None = None
    trip_origin: str | None = None
    # Which measure the question ranks by. `None` is "the question did not say", not "seconds".
    route_objective: str | None = None

    def stated_stay(self, name: str) -> float:
        """The stay stated for exactly this place, or zero when the question states none."""

        key = name.strip()
        return next((seconds for stated, seconds in self.stays if stated == key), 0.0)

    def stay_for(self, name: str) -> float:
        """The same, tolerating a node label the plan decorated around the stated name."""

        exact = self.stated_stay(name)
        if exact:
            return exact
        for stated, seconds in self.stays:
            if stated and (stated in name or name in stated):
                return seconds
        return 0.0

    def stated_places(self) -> tuple[str, ...]:
        """Every place the question names outright, for repairing one a plan copied short."""

        named = (
            self.anchor,
            self.trip_destination,
            *(self.compared_pair or ()),
            *self.listed_places,
            *(name for name, _ in self.stays),
        )
        return tuple(name for name in named if name)


def extract_facts(analysis: dict[str, Any], question: str) -> GroundingFacts:
    """Read the question's stated factors once, from the concept graph and the question text.

    Which of the two leads is decided per fact rather than globally, and the rule is what kind of
    fact it is.

    A radius, a compass sector, a pair of compared names, a routing preference, a closing leg and
    a fixed order are all written in the sentence verbatim. For those the question is the record
    and the scan over it is exact, so it goes first; an LLM re-transcription of a literal can only
    introduce error, which is the same reason the option texts and the place names are bound from
    the question rather than accepted from the planner. The concept graph is consulted *after*,
    to recover a literal the scan did not find -- a phrasing the patterns do not know is exactly
    the case the analysis can still describe.

    `target_type` is the other kind. "우산을 사야 합니다" states no kind of place at all, and the
    answer is 편의점 only by inference, which is the Analysis stage's job and not a regex's. The
    question's own words still win when it names one; `analysis["target_type"]` fills the rest.
    """

    radius_m = _extract_radius_m(question)
    if radius_m is None:
        radius_m = _stated_number(analysis, "radius_m", "radius")
    direction = _extract_requested_direction(question) or _stated_text(
        analysis, "direction", "bearing"
    )
    stays, budget = _extract_trip_schedule(question)
    if not stays:
        stays = _stated_stays(analysis)
    if budget is None:
        budget = _stated_seconds(analysis, "time_budget_s", "time_budget")
    anchor = _extract_anchor(question)
    stated_type = _extract_target_type(question) or analysis.get("target_type")
    broad_type, subtype = split_place_type(str(stated_type)) if stated_type else (None, None)
    return GroundingFacts(
        anchor=anchor,
        target_type=broad_type,
        target_subtype=subtype,
        listed_places=_extract_listed_places(question),
        radius_m=radius_m,
        direction=direction,
        compared_pair=_extract_compared_places(question),
        route_priority=_extract_route_priority(question),
        returns_to_start=_returns_to_start(question),
        stated_order=_states_visiting_order(question),
        stays=tuple(stays.items()),
        time_budget_s=budget,
        trip_destination=_extract_trip_destination(question),
        trip_origin=_stated_departure(question, anchor, stays),
        route_objective="distance" if _asks_for_distance(question) else None,
    )


def _concept_attributes(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        concept["attributes"]
        for concept in (analysis.get("concepts") or [])
        if isinstance(concept, dict) and isinstance(concept.get("attributes"), dict)
    ]


def _stated_number(analysis: dict[str, Any], *keys: str) -> int | None:
    """A metre count the Analysis stage attached to a concept, under any of these keys.

    Written leniently on purpose: the stage returns free-form attributes today, so "600",
    "600m" and 600 all have to read as 600. It is a recovery path -- the question scan has
    already failed by the time this runs.
    """

    for attributes in _concept_attributes(analysis):
        for key in keys:
            if key not in attributes:
                continue
            found = re.search(r"([\d,]+(?:\.\d+)?)\s*(km)?", str(attributes[key]), re.IGNORECASE)
            if not found:
                continue
            value = float(found.group(1).replace(",", ""))
            return round(value * 1000 if found.group(2) else value)
    return None


def _stated_stays(analysis: dict[str, Any]) -> dict[str, float]:
    """Visit durations the Analysis stage attached to concepts, when the question scan found none.

    Same standing as the radius and the sector recovery below it: the sentence is the record and
    the scan over it is exact, so this only runs when the scan came back empty. A concept carries
    its own name in `text`, so the stay is keyed by that.
    """

    stays: dict[str, float] = {}
    for concept in analysis.get("concepts") or []:
        if not isinstance(concept, dict):
            continue
        attributes = concept.get("attributes")
        name = str(concept.get("text") or "").strip()
        if not isinstance(attributes, dict) or not name:
            continue
        for key in ("visit_duration_s", "visit_duration", "stay_duration_s", "stay_duration"):
            seconds = _seconds_from(attributes.get(key)) if key in attributes else None
            if seconds is not None:
                stays[name] = seconds
                break
    return stays


def _stated_seconds(analysis: dict[str, Any], *keys: str) -> float | None:
    for attributes in _concept_attributes(analysis):
        for key in keys:
            if key in attributes:
                seconds = _seconds_from(attributes[key])
                if seconds is not None:
                    return seconds
    return None


def _seconds_from(value: Any) -> float | None:
    """A duration an LLM may have written as a number, as "3시간", or as "180분"."""

    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str):
        return None
    written = _duration_seconds(value.strip())
    if written is not None:
        return written
    try:
        return float(value.strip())
    except ValueError:
        return None


# "오전 10시 00분에 키이토에서 자동차로 출발해" — a free parse has to decide for itself where the
# clock ends and the place begins, so the departure is matched against the places the question has
# already been read to name rather than segmented out of the sentence a second time.
_DEPARTS = "(?:에서|에)\\s*(?:자동차로\\s*)?출발"


# "오전 10시 00분에 가예에서 출발해" — the anchor splitter keeps the clock, because it splits at
# the first "에서" and the clock sits in front of it. The place is what is left after it.
_CLOCK_PREFIX = re.compile(
    r"^\s*(?:오전|오후|아침|저녁|밤)?\s*\d{1,2}\s*시(?:\s*\d{1,2}\s*분)?\s*에?\s*"
)


def _stated_departure(question: str, anchor: str | None, stays: dict[str, float]) -> str | None:
    trimmed = _CLOCK_PREFIX.sub("", anchor).strip() if anchor else None
    for name in (trimmed, anchor, *stays):
        if name and re.search(rf"{re.escape(name)}\s*{_DEPARTS}", question):
            return name
    return None


def _stated_text(analysis: dict[str, Any], *keys: str) -> str | None:
    for attributes in _concept_attributes(analysis):
        for key in keys:
            value = attributes.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _ground_graph_literals(
    steps: list[dict[str, Any]],
    question: str,
    options: list[str],
    facts: GroundingFacts,
    *,
    retrieval_specs: RetrievalSpecs = canonical_retrieval_specs,
) -> list[dict[str, Any]]:
    """Bind verbatim question literals after drafting, before graph validation.

    GeoFlow factors such as a radius, requested direction, and the candidate option texts are
    constants from the question, not values that an LLM should invent or route through a
    synthetic operator output. `facts` is that reading of the question; every branch below asks
    whether a fact is present, never what a classifier called the question.
    """

    anchor = facts.anchor
    target = facts.target_type
    # Which half of a stated kind discriminates. "중식 음식점" retrieves 음식점 -- Kakao files no
    # 중식 category -- and qualifies on 중식, which is the half that appears in the category path
    # of the places that came back.
    qualifier = facts.target_subtype or facts.target_type
    radius_m = facts.radius_m
    specifications = list(retrieval_specs(target)) if target else []
    # tsp_tw's service_times are positional, so the stays can only be bound once the node list the
    # plan geocoded is known — it is the place order every downstream index refers to.
    route_priority = facts.route_priority
    # The itinerary is whichever `batch_geocode` node lists more than two places. That structural
    # test is the real guard; the old trip-label conjunct that sat in front of it did
    # nothing a trip plan can notice, and in a plan the Analysis stage labelled something else it
    # withheld the stays from an operator that still ran -- binding every stay to zero rather
    # than refusing.
    trip_node_names: list[str] = next(
        (
            [str(name) for name in (step.get("arguments") or {}).get("place_names") or []]
            for step in steps
            if step.get("operator") == "batch_geocode"
            and len((step.get("arguments") or {}).get("place_names") or []) > 2
        ),
        [],
    )
    # Which node ids produce a tour whose cost already carries the stays.
    tour_totals = {
        str(step.get("id"))
        for step in steps
        if step.get("operator") == "tsp_tw"
    }
    # The place names the question states outright, for repairing one a plan copied short: a plan
    # that geocoded `문래` where the question says `빈칸 문래` routed from another place entirely
    # and counted another route's turns, with every stage reporting success.
    # Any place the question gives a duration to is stated exactly as its anchor is, and
    # `_extract_trip_schedule` already reads each one to bind its stay. Not gated on intent: this
    # list is what the *question* says, so it is gathered the same way whatever the question is
    # about, and the regex simply finds nothing in a question that states no stay. Without these a
    # planner that mis-segments a Korean particle -- `백련산꿈마을숲정이를` copied as
    # `백련산꿈마을숲정` -- geocoded nothing, and the loss surfaced three nodes later as
    # `tsp_tw distance_matrix must be square`, which names neither the place nor the problem.
    question_places = list(facts.stated_places())
    indexed_references = _indexed_references_by_root(steps)
    grounded: list[dict[str, Any]] = []
    for step in steps:
        operator = step.get("operator")
        if not isinstance(operator, str):
            # A planner that writes `"operator": ["directions", "travel_time"]` is a planner that
            # failed to plan, and saying so is the whole fix. Left alone it reached a set lookup as
            # an unhashable list and came back as `TypeError: unhashable type: 'list'` -- a crash
            # in this file, recorded against the agent as if it had reasoned its way there.
            raise ValueError(
                f"GeoFlow node {step.get('id')!r} names no operator: {operator!r}"
            )
        raw_arguments = step.get("arguments")
        if raw_arguments is None:
            raw_arguments = step.get("params")
        if raw_arguments and not isinstance(raw_arguments, dict):
            # A planner writes `"arguments": "$nearest_result"` when a node just forwards what it
            # depends on -- the same intent the auto-generated closing Measure step expresses as
            # `{"value": "$source"}`. `dict("$nearest_result")` does not raise that name; Python
            # iterates the string and reports "dictionary update sequence element #0 has length 1,
            # 2 is required", which told the repair round nothing about what was actually wrong.
            # Wrapped under the operator's one required slot when it has exactly one -- the same
            # binding `normalize_and_validate_graph` makes from a lone dependency -- or left as
            # written otherwise, so the graph's own "arguments must be an object" is what a
            # multi-argument operator is refused with.
            contract = OPERATOR_CONTRACTS.get(operator, OperatorContract("object"))
            required = contract.required_arguments
            raw_arguments = {required[0]: raw_arguments} if len(required) == 1 else raw_arguments
        arguments = (
            _verbatim_place_names(dict(raw_arguments), question, options, question_places)
            if isinstance(raw_arguments, dict)
            else raw_arguments
        )
        if not isinstance(arguments, dict):
            grounded.append({**step, "arguments": arguments})
            continue
        if operator == "nearby_places" and specifications:
            arguments.pop("query", None)
            arguments.pop("category_code", None)
            arguments["radius_m"] = radius_m if radius_m is not None else RETRIEVAL_RADIUS_M
            arguments["limit"] = RETRIEVAL_LIMIT
            grounded.extend(
                _retrieval_steps(step, arguments, specifications)
            )
            continue
        if route_priority and operator in _PRIORITY_OPERATORS:
            arguments["priority"] = route_priority
        if operator == "calculate_start_time":
            stays = facts.stays
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
                    arguments["stay_durations_s"] = [seconds for _, seconds in stays]
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
        if operator == "calculate_finish_time":
            stays = facts.stays
            locations = _whole_list_reference(arguments.get("locations"))
            arguments["locations"] = locations
            # One stay per location, in the order the itinerary visits them. The stays are stated
            # in the question exactly; a plan that drops the last one or invents one for the
            # return lands a whole visit away, which is wider than the gap between two options.
            # When `locations` is a reference the names are not in hand here — but the itinerary
            # is exactly what the trip's `batch_geocode` node lists, and the operator resolves the
            # reference to that same list. Without this the planner's own stays were left to
            # mismatch the resolved length, and the args model rejected the call outright.
            if isinstance(locations, list) and len(locations) == 1 and isinstance(
                locations[0], list
            ):
                # `locations: ["$places"]` resolves to one list holding the whole itinerary. The
                # tool flattens it; the stays are bound here, so they have to be counted against
                # the same list or the args model rejects the call for a length mismatch.
                locations = list(locations[0])
                arguments["locations"] = locations
            itinerary: list[Any] = []
            if isinstance(locations, list) and len(locations) > 1:
                # A stop written as `$geo.1.place` is a name the geocode node already holds, and
                # looking a stay up by the reference text finds nothing — which bound every stay
                # to zero and lost four hours off a finish time without failing anything.
                itinerary = [_named_stop(item, trip_node_names) for item in locations]
            elif isinstance(locations, str) and len(trip_node_names) > 1:
                itinerary = list(trip_node_names)
            if itinerary and facts.returns_to_start:
                # "X에서 출발해 …를 둘러본 뒤 X로 돌아옵니다" states both endpoints; only the order
                # of the stops between them is the plan's business. A plan that drops the return
                # arrives one drive early, and one that drops the departure loses its first leg
                # *and* shifts every stay onto the wrong stop — neither fails, both answer an
                # option away.
                closed = _closed_itinerary(itinerary, facts, trip_node_names)
                if closed is not None:
                    itinerary = closed
                    arguments["locations"] = closed
            if stays and itinerary:
                arguments["stay_durations_s"] = [
                    facts.stated_stay(_location_name(item)) for item in itinerary
                ]
            grounded.append({**step, "arguments": arguments})
            continue
        if operator == "tsp_tw":
            stays, budget = facts.stays, facts.time_budget_s
            if facts.route_objective == "distance":
                # "총 주행거리가 가장 짧은 방문 순서" is a question about metres. The tours it
                # chooses between are ~2% apart, so ranking them by seconds is not an
                # approximation of ranking them by metres — it is a different answer. The stays
                # and the budget are decoys in a distance question and the operator refuses them
                # beside a metre matrix, so they go.
                arguments["metric"] = "distance"
                stays, budget = (), None
                for key in ("service_times", "time_budget", "time_windows"):
                    arguments.pop(key, None)
            elif stays or budget is not None or arguments.get("time_windows") is not None:
                # A stay, a time window or a time budget can only be combined with seconds. The
                # question supplied the clock constraint, so binding `duration` is the same kind
                # of literal grounding as binding the budget itself; retaining a planner's
                # `distance` here creates metre-plus-second arithmetic the operator must refuse.
                arguments["metric"] = "duration"
            if budget is not None:
                arguments["time_budget"] = budget
            names = trip_node_names
            if facts.returns_to_start:
                arguments["return_to_start"] = True
                # `return_to_start` is how the operator represents a closed tour. `end_index=0`
                # is a redundant spelling of the same fact that the runtime deliberately rejects,
                # while another end contradicts the question. The question literal wins either
                # way, so no fixed endpoint remains.
                arguments.pop("end_index", None)
            if facts.stated_order:
                # The order is a question literal exactly as the stays and the budget are. Left to
                # the planner it was set in one graph out of fifty-nine, and the search reordered
                # an itinerary the question had already ordered.
                arguments["fixed_order"] = True
                # Under a stated order the sequence names its own last stop, so a fixed end is at
                # best redundant and at worst contradicts it.
                arguments.pop("end_index", None)
            destination = facts.trip_destination
            if (
                destination
                and names
                and not arguments.get("fixed_order")
                and not arguments.get("return_to_start")
            ):
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
                    0.0 if index == 0 else facts.stay_for(name)
                    for index, name in enumerate(names)
                ]
            grounded.append({**step, "arguments": arguments})
            continue
        if operator == "nearest" and target:
            # Bound here rather than asked for in the prompt: a planner that ranks the option
            # texts directly produces a graph with no retrieval to carry the category, and told
            # only in prose it keeps doing it. The kind asked for is a question literal like the
            # radius and the direction, so it is bound like one.
            arguments["required_type"] = qualifier
            grounded.append({**step, "arguments": arguments})
            continue
        if operator == "filter_by_direction":
            if facts.direction:
                arguments["direction"] = facts.direction
            grounded.append({**step, "arguments": arguments})
            continue
        if operator == "filter_places" and facts.target_subtype:
            # The narrowing half of the stated kind, bound as a literal exactly like the radius
            # and the sector. A planner asked to transcribe it writes the whole phrase, and
            # "중식 음식점" matches no category path at all.
            arguments["required_types"] = [facts.target_subtype]
            grounded.append({**step, "arguments": arguments})
            continue
        if operator == "filter_by_distance" and radius_m is not None:
            # The same stated literal `within_radius` gets, on the shape that measured first.
            arguments["max_distance_m"] = radius_m
            grounded.append({**step, "arguments": arguments})
            continue
        if operator == "within_radius" and radius_m is not None:
            # The radius is the whole content of this filter, and it is a question literal like
            # any other. Grounding bound it onto the retrieval and onto option recovery but never
            # onto the filter itself, because no planner had written one -- the semantic layer
            # composes FILTER over a stated radius, so now one exists on every radius question.
            arguments["radius_m"] = radius_m
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
            if facts.direction:
                # Recovery runs after the candidates were filtered and adds the option texts the
                # retrieval missed — regardless of where they are, until it is told. A recovered
                # mart 271 m *south* of the anchor out-ranked the northern one the filter had
                # correctly found at 961 m, and the direction the question asks about was gone
                # from the answer. The sector is the recovered option's constraint like the
                # radius already is. A question that names no sector has none to bind.
                arguments["direction"] = facts.direction
            grounded.append({**step, "arguments": arguments})
            continue
        if operator in {"match_options", "match_distance_options", "match_type_options"}:
            # The Measure step compares against the candidate texts verbatim; a planner that
            # paraphrases or numerically re-types them breaks the comparison.
            arguments["options"] = options
            if operator == "match_options":
                # A stated radius is what makes the options a set to be matched whole
                # rather than candidates to be ranked; the classifier's label was standing in
                # for the same fact.
                arguments["mode"] = "radius_set" if radius_m is not None else "nearest"
            grounded.append({**step, "arguments": arguments})
            continue
        if operator != "batch_geocode":
            # `arguments` is the copy every branch above edits; appending the original step here
            # threw those edits away, which is how a bound routing priority never reached the
            # `directions` call it was bound for.
            grounded.append({**step, "arguments": arguments})
            continue
        names = list(arguments.get("place_names") or [])
        pair = facts.compared_pair
        written_anchor = arguments.get("anchor")
        batch_anchor = anchor or (
            written_anchor if isinstance(written_anchor, str) and written_anchor.strip() else None
        )
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
        elif batch_anchor:
            # Only the node that *has* an anchor slot gets the anchor written into it. A plan may
            # geocode the anchor in one node and the four option texts in another, and replacing
            # the head of the second deleted an option — the gold one, in a radius question whose
            # every other stage then worked: the anchor stood 0 m from itself, was the only place
            # inside 600 m, and the generation stage picked from the leftovers.
            if names and (
                len(names) == len(options) + 1
                or names[0] == batch_anchor
                or _is_shortened_name(names[0], batch_anchor)
            ):
                names[0] = batch_anchor
            arguments["anchor"] = batch_anchor
            # `anchor` normally only biases ambiguous name resolution and is not another output.
            # A plan that references exactly one record beyond its listed names, however, states
            # through its own dataflow that it expects [anchor, *names].  Prepend only under that
            # structural proof; doing it for every anchor-bearing batch would shift correct option
            # indices and turn a disambiguation hint into data the planner never requested.
            highest_index = indexed_references.get(str(step.get("id") or ""), -1)
            if (
                names
                and highest_index >= len(names)
                and all(_option_key(name) != _option_key(batch_anchor) for name in names)
            ):
                names.insert(0, batch_anchor)
        if (
            _ranks_the_options(steps, str(step.get("id") or ""))
            and len(names) == len(options) + 1
            and all("|" not in option for option in options)
        ):
            names[1:] = options
        arguments["place_names"] = names
        grounded.append({**step, "arguments": arguments})
    return grounded


# What a batch of geocoded names is *for*, read off the graph rather than off a classifier's
# label. A node whose places end up in an option match or a nearest-of ranking is a candidate
# search, and its list after the anchor is the option texts. A node whose places end up in a tour
# or a schedule is an itinerary, and overwriting its stops with the option texts would answer a
# different trip.
_OPTION_RANKING_OPERATORS = frozenset(
    {
        "match_options",
        "match_distance_options",
        "match_type_options",
        "nearest",
        "filter_by_direction",
        "recover_option_places",
    }
)
_ITINERARY_OPERATORS = frozenset(
    {"tsp_tw", "calculate_finish_time", "calculate_start_time", "aggregate_route_groups"}
)


def _ranks_the_options(steps: list[dict[str, Any]], node_id: str) -> bool:
    """Does everything downstream of this node treat its places as the candidate options?

    This replaces the old three-label allowlist, whose real content was "not a trip": the excluded
    label that mattered was `trip`, because splicing the option texts into an
    itinerary rewrites the stops. Asking the graph is both narrower and safer -- a routing plan
    the Analysis stage happened to call `trip` used to lose the splice, and a trip plan it called
    `routing` used to get one.
    """

    if not node_id:
        return False
    reachable = _downstream_operators(steps, node_id)
    return bool(reachable & _OPTION_RANKING_OPERATORS) and not (reachable & _ITINERARY_OPERATORS)


def _downstream_operators(steps: list[dict[str, Any]], node_id: str) -> set[str]:
    """Every operator reachable from this node, following `depends_on` and `$node` references."""

    consumers: dict[str, set[str]] = {}
    operators: dict[str, str] = {}
    for step in steps:
        current = str(step.get("id") or "")
        if not current:
            continue
        operators[current] = str(step.get("operator") or "")
        for source in _referenced_nodes(step):
            consumers.setdefault(source, set()).add(current)
    seen: set[str] = set()
    frontier = [node_id]
    while frontier:
        current = frontier.pop()
        for consumer in consumers.get(current, ()):
            if consumer not in seen:
                seen.add(consumer)
                frontier.append(consumer)
    return {operators[node] for node in seen if node in operators}


def _referenced_nodes(step: dict[str, Any]) -> set[str]:
    """The node ids this step reads: its declared dependencies plus any `$node` it writes."""

    sources = {str(value) for value in (step.get("depends_on") or [])}

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str):
            for match in re.finditer(r"\$([A-Za-z_][\w-]*)", canonical_reference(value)):
                sources.add(match.group(1))

    walk(step.get("arguments") if step.get("arguments") is not None else step.get("params"))
    return sources


def _indexed_references_by_root(steps: list[dict[str, Any]]) -> dict[str, int]:
    """Largest literal list index each planner node is referenced with anywhere downstream."""

    highest: dict[str, int] = {}

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
            return
        if isinstance(value, list):
            for item in value:
                walk(item)
            return
        if not isinstance(value, str):
            return
        for match in re.finditer(
            r"\$([A-Za-z_][\w-]*)\.(\d+)(?:\.|\b)", canonical_reference(value)
        ):
            root, index = match.group(1), int(match.group(2))
            highest[root] = max(highest.get(root, -1), index)

    walk(steps)
    return highest


def _retrieval_steps(
    step: dict[str, Any],
    arguments: dict[str, Any],
    specifications: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fan a retrieval node out over every Kakao spelling of the requested place type.

    Korean place types map onto several Kakao keywords or category codes (경찰서 also appears
    as 파출소/지구대/치안센터), so a single retrieval silently loses candidates. The branches
    merge back under the planner's original node id, which keeps downstream references valid.
    """

    if len(specifications) == 1:
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


_COMPARED_PLACE_PATTERNS = (
    re.compile(
        r"^(.+?)\s*(?:및|와|과)\s+(.+?)\s+(?:사이|간)(?:의)?\s+직선\s*거리"
    ),
    re.compile(r"^(.+?)에서\s+(.+?)까지(?:의)?\s+직선\s*거리"),
)

# How many separations the question actually states. Two of them is a difference question over
# three places -- "A에서 B까지의 직선거리와 A에서 C까지의 직선거리는 얼마나 차이가 나나요?" --
# and the patterns above read only the first pair out of it.
_SEPARATION_CLAUSE = re.compile(r"(?:까지|사이|간)(?:의)?\s*직선\s*거리")


def _extract_compared_places(question: str) -> tuple[str, str] | None:
    """The two POI names a straight-line-distance question compares, verbatim.

    Only when the question compares exactly two. "A에서 B까지의 직선거리와 A에서 C까지의
    직선거리는 얼마나 차이가 나나요?" names three places and states two separations, and reading
    the first pair off it and binding `place_names` to it deletes C from the plan -- the question
    then measures one of the two distances it asked to compare. That was already happening on the
    `poi_distance_difference` rows the Analysis stage happened to label `distance`; it is refused
    outright now rather than more widely.
    """

    if len(_SEPARATION_CLAUSE.findall(question)) != 1:
        return None
    for pattern in _COMPARED_PLACE_PATTERNS:
        match = pattern.search(question)
        if match:
            first, second = (part.strip() for part in match.groups())
            if first and second:
                return first, second
    return None


# The kind of place a question asks for sits between a semantic lead-in and a grammatical tail.
# Keep those pieces independent: enumerating complete observed sentences makes the extractor a
# function of one generator, while the same relation can be written with different particles,
# endings and ordinary synonyms.
# `아래`/`위` sits between the particle and the tail noun in "은행은 아래 목록 중": without it
# the non-greedy body grew past the particle and read the kind of place as "은행은 아래".
_TARGET_TYPE_TAIL = r"\s*(?:은|는|이|가|을|를)?\s*(?:아래|위)?\s*(?:다음|어디|무엇|어느|중|목록)"

_TARGET_TYPE_LEADS: dict[str, tuple[str, ...]] = {
    "nearby": (r"(?:가장\s*)?(?:가까운|인접한)\s+",),
    "direction": (
        r"(?:북동|남동|남서|북서|북|남|동|서)쪽\s*(?:방향)?(?:에|으로)\s*있는\s*"
        r"(?:가장\s*가까운\s*)?",
        r"(?:북동|남동|남서|북서|북|남|동|서)쪽\s*(?:방향)?에서\s*"
        r"(?:가장\s*가까운\s*)?",
    ),
    "radius": (r"(?:이내|안|내)에\s*(?:있는|위치한)\s+",),
}

# Flattened deliberately. The leads were keyed by intent and only the matching key was tried,
# which meant a question the Analysis stage mislabelled had its stated kind of place read as "no
# kind at all". The leads do not compete: each names a different relation, and over every question
# in `dataset/` trying all three finds a type in exactly the `nearby`, `radius`, `direction` and
# legacy `poi` rows and in no `trip` or `routing` row.
_TARGET_TYPE_PATTERNS: tuple[str, ...] = tuple(
    lead + r"(.+?)" + _TARGET_TYPE_TAIL
    for leads in _TARGET_TYPE_LEADS.values()
    for lead in leads
)


# What a question says when it is *not* naming a kind of place. "다음 중 걸어가기에 가장 가까운
# 곳" is the inferred-category family, whose whole point is that the kind is never stated; reading
# 곳 as the type would retrieve "places" and hand the ranking every kind there is.
_PLACEHOLDER_TYPES = frozenset({"곳", "장소", "것", "데", "지점", "위치"})


#: A question that offers its candidates writes them as a parenthesised, comma-separated list at
#: the end. Greedy from the first bracket to the last, because the names carry brackets of their
#: own and not always balanced ones: "기업은행 구)용산2가 무인 ATM" closes one that never opened,
#: and a nesting-aware pattern found no list at all in that question. The comma split below is
#: depth-aware, and two or more entries are required, which is what keeps an ordinary
#: parenthetical aside from being read as a candidate list.
_LISTED_PLACES = re.compile(r"\((.+)\)\s*$", re.DOTALL)


def _extract_listed_places(question: str) -> tuple[str, ...]:
    """The candidates a question names outright, in the order it names them.

    A question that lists its candidates is asking about those. Counting a retrieval instead
    answers "how many banks are within 300m", which is a different number from "how many of these
    four are", and it is the one every `nearby_within_radius_count` row was returning.

    Two or more entries are required: a single parenthesised phrase at the end of a sentence is
    an aside, not a candidate list.
    """

    match = _LISTED_PLACES.search(question.strip())
    if not match:
        return ()
    names: list[str] = []
    depth = 0
    current: list[str] = []
    for character in match.group(1):
        if character == "(":
            depth += 1
        elif character == ")":
            depth = max(depth - 1, 0)
        if character == "," and depth == 0:
            names.append("".join(current).strip())
            current = []
            continue
        current.append(character)
    names.append("".join(current).strip())
    listed = tuple(name for name in names if name)
    return listed if len(listed) >= 2 else ()


def _extract_target_type(question: str) -> str | None:
    for pattern in _TARGET_TYPE_PATTERNS:
        match = re.search(pattern, question)
        found = match.group(1).strip() if match else ""
        if found and "".join(found.split()) not in _PLACEHOLDER_TYPES:
            return found
    return None


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


# What the question is asking to minimise, in its own words. `거리`/`주행거리`/`km` is metres;
# `시간`/`분` is seconds. A trip question states one or the other outright, so this is a literal
# to read, not a preference to infer.
_ASKS_DISTANCE = re.compile(r"(주행\s*거리|이동\s*거리|총\s*거리|거리가\s*가장\s*짧|distance)")
_ASKS_DURATION = re.compile(
    r"(소요\s*시간|이동\s*시간|시간이\s*가장\s*짧|가장\s*빠(?:른|르)|duration)"
)


def _asks_for_distance(question: str) -> bool:
    """Does the trip question rank its options by metres rather than by seconds?"""

    return bool(_ASKS_DISTANCE.search(question)) and not _ASKS_DURATION.search(question)


# "…둘러본 뒤 다시 제일모텔로 돌아옵니다" closes the tour. The cheapest open path is not the
# cheapest loop, so whether the drive home counts is a question literal like the stays are.
# "…를 차례로 둘러본 뒤 가예로 돌아옵니다" — the return is a leg the question states, not an
# optional flourish, and a plan that stops at the last sight computes an arrival one drive short.
# There were two definitions of this predicate for a while, this one shadowing a `_RETURN_PATTERNS`
# tuple defined 600 lines earlier that nothing could reach. They agreed on every question in
# `dataset/`, so deleting the unreachable one changed no answer — but nothing in the suite said so,
# which is why the round-trip cases below now pin it. `ruff`'s F811 is on so a second definition is
# a lint error rather than a silent shadow.
_RETURNS_TO_START = re.compile(r"(다시\s*\S+(?:으)?로\s*돌아|돌아옵니다|돌아온다|돌아와|출발지로"
                               r"|return(?:ing)?\s+to\s+(?:the\s+)?start)")


def _returns_to_start(question: str) -> bool:
    """Does the trip come back to where it started?"""

    return bool(_RETURNS_TO_START.search(question))


# A trip question that fixes its own sequence says so in the sentence that lists the stops.
# These are the phrasings the Korean generators produce and the ones a person writes: "적힌 순서
# 대로", "순서대로", "차례대로", "이 순서로", and the English a mixed-language question may use.
_STATED_ORDER = re.compile(
    r"(적힌\s*순서|나열된\s*순서|이\s*순서|위\s*순서|순서대로|차례대로|순서로"
    r"|in\s+(?:the\s+)?(?:listed|written|given|stated|this)\s+order|in\s+order)"
)


def _states_visiting_order(question: str) -> bool:
    """Does the question fix the order of the stops, or only the set of them?"""

    return bool(_STATED_ORDER.search(question))


_DURATION_TEXT = r"(?:[\d.]+\s*시간(?:\s*[\d.]+\s*분)?|[\d.]+\s*분)"


def _duration_seconds(value: str) -> float | None:
    match = re.fullmatch(
        r"\s*(?:(?P<hours>[\d.]+)\s*시간)?\s*(?:(?P<minutes>[\d.]+)\s*분)?\s*",
        value,
    )
    if not match or not any(match.group(name) for name in ("hours", "minutes")):
        return None
    hours = float(match.group("hours") or 0)
    minutes = float(match.group("minutes") or 0)
    return hours * 3600 + minutes * 60


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
        rf"([^,.!?]+?)(?:을|를|에서|에)\s*(?:약\s*)?({_DURATION_TEXT})", question
    ):
        name = match.group(1).strip()
        # If the first visit shares a clause with the departure, retain the name after the
        # departure verb: "A에서 출발해 B를 30분" spends time at B, not at a place named by the
        # whole clause. Punctuation and commas are already clause boundaries in the regex.
        name = re.split(r"출발(?:해|해서|하여|하고|한\s*뒤)?\s*", name)[-1].strip()
        if not name:
            continue
        seconds = _duration_seconds(match.group(2))
        if seconds is not None:
            stays[name] = seconds
    budget_match = re.search(rf"(?:총|전체)\s*({_DURATION_TEXT})", question)
    budget = _duration_seconds(budget_match.group(1)) if budget_match else None
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


def _extract_radius_m(question: str) -> int | None:
    """The radius the question states, in metres, or nothing when it states none.

    It used to answer 2,000 for a question with no radius in it. That was harmless only because
    nothing called it unless a classifier had already said "radius"; with presence as the gate,
    a made-up default would be a stated constraint the question never stated.
    """

    for pattern in _RADIUS_PATTERNS:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            radius = float(match.group(1).replace(",", ""))
            return round(radius * 1000 if match.group(2).lower() == "km" else radius)
    return None


def _extract_requested_direction(question: str) -> str | None:
    return next(
        (
            direction
            # Longer diagonal names must precede their cardinal substrings: 북동쪽 contains 동쪽.
            for direction in (
                "북동쪽",
                "남동쪽",
                "남서쪽",
                "북서쪽",
                "북쪽",
                "남쪽",
                "동쪽",
                "서쪽",
            )
            if direction in question
        ),
        None,
    )


# "I am at X" is said several ways, and an anchor phrasing the splitter does not know reads as
# no anchor at all — the geocoder then loses its disambiguation and `recover_option_places` its
# centre. Tried before the relational patterns because it names the anchor outright.
_ANCHOR_PATTERNS = (
    r"지금\s+(.+?)에\s+(?:있|와\s*있|머물)",
    r"현재\s+(.+?)에\s+(?:있|머물)",
    r"^(.+?)에\s+있는데",
)

# Then the relation the question states between the anchor and what it asks about: nearest-of,
# within-a-radius, in-a-sector. These were three per-intent tables and only the entry matching the
# Analysis stage's guess was ever tried, so a question it mislabelled lost its anchor -- which
# costs the geocoder its disambiguation and `recover_option_places` its centre. They are one
# ordered list now, most specific first, and a question that states none of these relations falls
# through to the separators exactly as before.
# Ordered most-constrained first, which is what the per-intent keying used to do implicitly. A
# sector question also says "가장 가까운" -- "서울역 기준 북쪽에서 가장 가까운 편의점" -- so the
# nearest-of pattern would swallow "서울역 기준 북쪽" as the anchor if it ran first. Likewise a
# radius question says "에서"; its metre count is what distinguishes it.
_ANCHOR_RELATIONS = (
    r"^(.+?)(?:에서|을\s*기준으로|를\s*기준으로|\s기준(?:으로)?)\s*"
    r"(?:볼\s*때\s*)?(?:북동|남동|남서|북서|북|남|동|서)쪽\s*(?:방향)?(?:에|에서|으로)",
    r"^(.+?)(?:에서|으로부터)\s*(?:반경|직선\s*거리|거리)?\s*[\d,.]+\s*(?:km|m)",
    r"^(.+?)(?:에서|으로부터|와|과)\s*(?:가장\s*)?(?:가까운|인접한)",
)

# Last, the plain phrase splits, ordered longest-and-most-specific first so the bare "에서" that
# ends the list only ever runs when nothing more definite matched. "A에서 B까지 자동차로" is the
# routing phrasing and the first "에서" is where the drive starts: without it a plan that geocoded
# `문래` for the question's `빈칸 문래` kept the shortened name, resolved a different place, and
# counted the turns of a route nobody asked about.
_ANCHOR_SEPARATORS = (
    "에서 가장 가까운",
    "에서 직선거리",
    "에서 북쪽",
    "에서 남쪽",
    "에서 동쪽",
    "에서 서쪽",
    "에서 출발",
    "에서 자동차",
    " 반경",
    "에서",
)


# "A와 B 양쪽 모두에서 …" reads to the splitters as one long place name. It is two, and there is
# no single anchor to bind — the question asks for the intersection of two neighbourhoods.
_TWO_ANCHOR_MARKERS = ("양쪽", "둘 다", "모두에서")


def _extract_anchor(question: str) -> str | None:
    """The place the question measures from, read from the question and nothing else."""

    for pattern in (*_ANCHOR_PATTERNS, *_ANCHOR_RELATIONS):
        match = re.search(pattern, question)
        if match and match.group(1).strip():
            return _single_anchor(match.group(1).strip())
    for separator in _ANCHOR_SEPARATORS:
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


# "헤이갤러리 근처에서 분위기가 가장 좋은 카페" names 헤이갤러리 and then says "near it". The
# vicinity word is not part of the name, and binding it produced `batch_geocode("헤이갤러리 근처")`
# -- a place that does not exist, written over the option the plan had geocoded. Only ever
# stripped from the *tail*, and only when something is left.
_VICINITY_TAIL = re.compile(r"\s*(?:근처|인근|주변|부근|일대)$")


def _single_anchor(candidate: str) -> str | None:
    """The anchor, unless the text in hand names two of them.

    `'가좌동 마을극장과 증산역 6호선 양쪽 모두'` was bound as one place name and searched as one:
    no place matched, and the retrieval it anchored never happened. Two anchors are not an anchor,
    and the plan's own two-place composition answers the question.
    """

    if any(marker in candidate for marker in _TWO_ANCHOR_MARKERS):
        return None
    return _VICINITY_TAIL.sub("", candidate).strip() or None


def _is_shortened_name(candidate: Any, expected: Any) -> bool:
    # A planner may fill `place_names` with objects rather than names -- one graph passed
    # `{"cost": "$cost_0", "index": 0}` per entry, using `batch_geocode` to build a record list.
    # The registry refuses that cleanly, but this predicate ran first and `.split()` on a dict
    # threw the whole question away with a Python internals leak. Whatever is not a name is not
    # a shortened one, so answer the question that was asked and let the operator do the refusing.
    if not isinstance(candidate, str) or not isinstance(expected, str):
        return False
    candidate_key = "".join(candidate.split()).casefold()
    expected_key = "".join(expected.split()).casefold()
    return bool(candidate_key and candidate_key != expected_key and candidate_key in expected_key)


def _select_option(
    payload: dict[str, Any],
    options: list[str],
) -> tuple[int | None, str]:
    """Reconcile the generated answer text with the generated index.

    Upstream Spatial-Agent selects on the answer *text* and derives the index from it, because
    a model that names the right candidate can still miscount its position. Exact text wins,
    the declared index is the next authority, and a single containment match is the last
    resort.

    Operator state is evidence supplied to the generation stage, not a second answer channel in
    the harness. Keeping selection here limited to the model's declared answer makes every intent
    follow the same rule; deterministic clock values no longer receive a family-specific override.
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
    if any(word in lowered for word in ("반경", "이내", "radius", "within")) or re.search(
        r"[\d,.]+\s*(?:km|m)\s*(?:내|안)(?:에|의)", lowered
    ):
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
    if any(word in lowered for word in ("일정", "여행", "순서", "경유", "둘러", "itinerary")):
        return "trip"
    if any(word in lowered for word in ("경로", "운전", "자동차", "주행", "route", "driving")):
        return "routing"
    if any(word in lowered for word in ("가까운", "인접한", "근처", "nearest", "nearby")):
        return "nearby"
    return None
