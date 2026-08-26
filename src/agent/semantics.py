"""Semantic transformations, and the deterministic map from them to executable operators.

The paper's pipeline puts a *concept transformation* stage between the concept graph and the
executable operator graph. This port had no such stage: `GRAPH_PROMPT` handed the LLM the
operator contracts and asked it to write `nearby_places(center, category_code="PM9", ...)`
directly, so choosing which tool computes a spatial relation -- a mechanical question with one
right answer given the concept types -- was being answered by a language model, per question,
from a prompt listing Kakao category codes.

Here the LLM says *what* transformation the question needs and this module decides *which*
operator performs it. Two properties matter and both are tested:

* It never sees the question. Its inputs are the semantic graph, the concept analysis, the facts
  the analysis extracted, and the names the tool registry can execute.
* It is deterministic. The same semantic graph and the same registry always produce the same
  executable graph; where several operators could serve, `PRECEDENCE` below decides, and the
  rule that fired is recorded on the node so a report can count them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

#: What a semantic node may carry besides its inputs. These are *semantic* parameters -- an
#: ordinal position, which measure to rank by, which way to aggregate -- never a question
#: literal. Radii, sectors, stays and budgets come from the analysis facts and are bound after
#: factorization, so a planner cannot mistranscribe one.
SEMANTIC_FACTORS = frozenset(
    {"ordinal", "measure", "aggregate", "extreme", "key", "target", "scope"}
)


@dataclass(frozen=True)
class Transform:
    """One semantic transformation and the operators that can perform it."""

    name: str
    summary: str
    #: Candidate operators, most specific first. The first whose guard accepts wins.
    candidates: tuple[tuple[str, Callable[[Resolution], bool]], ...]
    output_type: str

    def choose(self, resolution: Resolution) -> tuple[str, str] | None:
        for operator, guard in self.candidates:
            if guard(resolution):
                return operator, guard.__name__
        return None


@dataclass(frozen=True)
class Resolution:
    """Everything the choice may depend on. Deliberately does not include the question."""

    factors: dict[str, Any]
    #: How many upstream nodes feed this one.
    arity: int
    #: Output types of those upstream nodes, in order.
    input_types: tuple[str, ...]
    #: Facts the Concept Analysis stage extracted from the question, already typed.
    facts: Any
    #: Operator names the tool registry and the operator registry can actually run.
    available: frozenset[str]

    def factor(self, name: str, default: Any = None) -> Any:
        return self.factors.get(name, default)

    def has(self, name: str) -> bool:
        value = self.factors.get(name)
        return value is not None and value != ""


def _always(_: Resolution) -> bool:
    return True


def _pairwise(resolution: Resolution) -> bool:
    """Two distinct places to separate, rather than a list to cross-join."""

    return resolution.arity >= 2


def _over_a_collection(resolution: Resolution) -> bool:
    return resolution.arity <= 1


def _ranks_by_distance_from_an_anchor(resolution: Resolution) -> bool:
    """A sort with an anchor is a nearest-of, which is the operator that knows about distance."""

    return resolution.arity >= 2 or resolution.factor("key") in {None, "", "distance"}


def _by_stated_key(_: Resolution) -> bool:
    return True


def _minimum(resolution: Resolution) -> bool:
    return str(resolution.factor("extreme", "min")).lower() == "min"


def _maximum(resolution: Resolution) -> bool:
    return str(resolution.factor("extreme", "min")).lower() == "max"


def _duration(resolution: Resolution) -> bool:
    return str(resolution.factor("measure", "distance")).lower() in {"duration", "time", "seconds"}


def _distance(_: Resolution) -> bool:
    return True


def _within_a_stated_radius(resolution: Resolution) -> bool:
    return resolution.facts is not None and getattr(resolution.facts, "radius_m", None) is not None


def _within_a_stated_sector(resolution: Resolution) -> bool:
    return resolution.facts is not None and bool(getattr(resolution.facts, "direction", None))


def _by_place_attribute(_: Resolution) -> bool:
    return True


def _around_an_extent(resolution: Resolution) -> bool:
    """A neighbourhood search needs somewhere to be the centre of."""

    return resolution.arity >= 1


def _without_an_extent(_: Resolution) -> bool:
    return True


def _one_place(resolution: Resolution) -> bool:
    return str(resolution.factor("scope", "")).lower() in {"one", "single"}


def _a_collection(_: Resolution) -> bool:
    return True


def _difference(resolution: Resolution) -> bool:
    return str(resolution.factor("aggregate", "sum")).lower() in {"difference", "subtract", "minus"}


def _proportion(resolution: Resolution) -> bool:
    return str(resolution.factor("aggregate", "sum")).lower() in {"proportion", "share", "ratio"}


def _grouped(resolution: Resolution) -> bool:
    return str(resolution.factor("scope", "")).lower() in {"groups", "grouped", "legs"}


def _sum(_: Resolution) -> bool:
    return True


def _against_a_numeric_measure(resolution: Resolution) -> bool:
    return "amount" in resolution.input_types


def _against_a_category(resolution: Resolution) -> bool:
    return str(resolution.factor("key", "")).lower() in {"type", "category", "kind"}


def _against_place_names(_: Resolution) -> bool:
    return True


def _arrival(resolution: Resolution) -> bool:
    return str(resolution.factor("measure", "finish")).lower() in {"start", "departure", "latest"}


def _finish(_: Resolution) -> bool:
    return True


#: The semantic vocabulary. Every transformation names a spatial relation the question asks
#: about; none names a tool. Ordered as a reader would meet them: gather, measure, narrow, rank,
#: aggregate, answer.
TRANSFORMS: dict[str, Transform] = {
    "RESOLVE_PLACES": Transform(
        "RESOLVE_PLACES",
        "Turn the place names the question states into located places.",
        (("batch_geocode", _always),),
        "object",
    ),
    "PLACE_SEARCH": Transform(
        "PLACE_SEARCH",
        "Find places of the requested kind, around an extent when one is given.",
        (("nearby_places", _around_an_extent), ("place_search", _without_an_extent)),
        "object",
    ),
    "PLACE_DETAILS": Transform(
        "PLACE_DETAILS",
        "Read the stored attributes of places already located.",
        (("place_details", _one_place), ("batch_place_details", _a_collection)),
        "object",
    ),
    "DISTANCE_MEASURE": Transform(
        "DISTANCE_MEASURE",
        "Straight-line separation between places.",
        (("haversine_distance", _pairwise), ("pairwise_distances", _over_a_collection)),
        "amount",
    ),
    "ROUTE_MEASURE": Transform(
        "ROUTE_MEASURE",
        "The road route from one place to another.",
        (("travel_time", _duration), ("directions", _distance)),
        "field",
    ),
    "ROUTE_MATRIX": Transform(
        "ROUTE_MATRIX",
        "Road cost between every pair of an itinerary's stops.",
        (("distance_matrix", _always),),
        "field",
    ),
    "ROUTE_EXTRACT": Transform(
        "ROUTE_EXTRACT",
        "One number out of a computed route.",
        (("extract_duration", _duration), ("extract_distance", _distance)),
        "amount",
    ),
    "ROUTE_STEPS": Transform(
        "ROUTE_STEPS",
        "Turn-by-turn guidance and its counts.",
        (("steps_analysis", _always),),
        "field",
    ),
    "ROUTE_COMPARE": Transform(
        "ROUTE_COMPARE",
        "Choose between road routes already computed.",
        (("compare_routes", _always),),
        "object",
    ),
    "ROUTE_OPTIMIZE": Transform(
        "ROUTE_OPTIMIZE",
        "Order an itinerary's stops under the question's stated constraints.",
        (("tsp_tw", _always),),
        "network",
    ),
    "SCHEDULE": Transform(
        "SCHEDULE",
        "Clock arithmetic over an itinerary: when it finishes, or when it must start.",
        (("calculate_start_time", _arrival), ("calculate_finish_time", _finish)),
        "event",
    ),
    "FILTER": Transform(
        "FILTER",
        "Narrow a set of places by a sub-condition the question states.",
        (
            ("within_radius", _within_a_stated_radius),
            ("filter_by_direction", _within_a_stated_sector),
            ("filter_places", _by_place_attribute),
        ),
        "object",
    ),
    "SORT": Transform(
        "SORT",
        "Put candidates in the order the question ranks them by.",
        (("nearest", _ranks_by_distance_from_an_anchor), ("sort_by", _by_stated_key)),
        "object",
    ),
    "ORDINAL_SELECT": Transform(
        "ORDINAL_SELECT",
        "Take the k-th member of an ordering. `ordinal` is 1-based, as the question counts.",
        (("select_by_index", _always),),
        "object",
    ),
    "EXTREME_SELECT": Transform(
        "EXTREME_SELECT",
        "Take the smallest or largest by the stated key.",
        (
            ("pairwise_extremes", _over_a_collection),
            ("select_min", _minimum),
            ("select_max", _maximum),
        ),
        "object",
    ),
    "AGGREGATE": Transform(
        "AGGREGATE",
        "Combine numbers: a total, a difference between two, or a share of a whole.",
        (
            ("difference", _difference),
            ("calculate_proportion", _proportion),
            ("aggregate_route_groups", _grouped),
            ("sum_amounts", _sum),
        ),
        "amount",
    ),
    "MATCH_OPTIONS": Transform(
        "MATCH_OPTIONS",
        "Map the computed evidence onto the candidate answer texts.",
        (
            ("match_distance_options", _against_a_numeric_measure),
            ("match_type_options", _against_a_category),
            ("match_options", _against_place_names),
        ),
        "object",
    ),
    "MEASURE": Transform(
        "MEASURE",
        "The node the answer is read from.",
        (("identity_measure", _always),),
        "object",
    ),
}


def transform_catalogue() -> str:
    """The vocabulary as the planner prompt states it -- names and meanings, never operators."""

    return "\n".join(f"- {t.name}: {t.summary}" for t in TRANSFORMS.values())


def resolve_operator(
    transform_name: str,
    factors: dict[str, Any],
    *,
    input_types: Sequence[str],
    facts: Any,
    available: frozenset[str],
) -> tuple[str, str]:
    """Which operator performs this transformation here, and which rule chose it.

    Raises when the vocabulary does not contain the transformation, or when every operator that
    could serve it is missing from the registry -- both are graph-construction failures and are
    reported as such rather than executed around.
    """

    transform = TRANSFORMS.get(transform_name.upper())
    if transform is None:
        raise ValueError(
            f"Unknown semantic transformation {transform_name!r}; "
            f"expected one of {', '.join(sorted(TRANSFORMS))}"
        )
    resolution = Resolution(
        factors={k: v for k, v in factors.items() if k in SEMANTIC_FACTORS},
        arity=len(input_types),
        input_types=tuple(input_types),
        facts=facts,
        available=available,
    )
    chosen = transform.choose(resolution)
    if chosen is None or chosen[0] not in available:
        runnable = [name for name, _ in transform.candidates if name in available]
        if not runnable:
            raise ValueError(
                f"No executable operator for {transform.name}: "
                f"{', '.join(name for name, _ in transform.candidates)} are not in the registry"
            )
        # Every guard declined, or the preferred operator is not executable here. Falling through
        # to the first runnable candidate keeps the precedence explicit and the result
        # reproducible; it is recorded so a report can count how often it happens.
        return runnable[0], "fallback_first_runnable"
    return chosen


# ---------------------------------------------------------------------------------------------
# Argument wiring
#
# Once the operator is chosen, its arguments follow from the graph: which node feeds it, what
# that node produced, and which semantic factors the transformation carries. None of it depends
# on the question, so all of it is decided here rather than asked of a model.
#
# Two reference shapes recur. A `batch_geocode` node returns one record per requested name, and
# the located place is `.N.place` of it; a `nearest` node returns its ordering under `.ranked`.
# Getting either wrong is silent -- the operator receives a record where it wanted a place -- so
# the producing operator decides the suffix rather than the planner remembering to write it.
# ---------------------------------------------------------------------------------------------

#: Operators whose output is a per-name record list rather than a place.
_RECORD_LIST_OPERATORS = frozenset({"batch_geocode"})
#: Operators that publish their ordering under a named field.
_ORDERED_FIELD = {"nearest": "ranked"}


def _place_ref(node_id: str, producer: str | None, index: int = 0) -> str:
    if producer in _RECORD_LIST_OPERATORS:
        return f"${node_id}.{index}.place"
    return f"${node_id}"


def _items_ref(node_id: str, producer: str | None) -> str:
    field = _ORDERED_FIELD.get(producer or "")
    return f"${node_id}.{field}" if field else f"${node_id}"


@dataclass(frozen=True)
class _Wiring:
    """The producing node of each input, so a reference can carry the right suffix."""

    inputs: tuple[str, ...]
    producers: tuple[str | None, ...]
    factors: dict[str, Any]

    def place(self, position: int, index: int = 0) -> str:
        node = self.inputs[position]
        return _place_ref(node, self.producers[position], index)

    def whole(self, position: int) -> str:
        return f"${self.inputs[position]}"

    def items(self, position: int) -> str:
        return _items_ref(self.inputs[position], self.producers[position])

    def two_places(self) -> tuple[str, str]:
        """The two places to relate: one node holding both, or one node each."""

        if len(self.inputs) >= 2:
            return self.place(0), self.place(1)
        return self.place(0, 0), self.place(0, 1)

    def by_type(self, wanted: str, produced_by: dict[str, str]) -> str | None:
        for node in self.inputs:
            if produced_by.get(node) == wanted:
                return f"${node}"
        return None


def _ordinal_index(factors: dict[str, Any]) -> int:
    """`ordinal` counts the way the question does -- second nearest is 2 -- and `index` is 0-based.

    This is the whole of what `Retrieve-Rank-Ordinal` used to be a template for. An ordinal is a
    factor on a selection, not a family of question.
    """

    raw = factors.get("ordinal", 1)
    try:
        position = int(raw)
    except (TypeError, ValueError):
        position = 1
    return max(position, 1) - 1


def _rank_key(factors: dict[str, Any]) -> str:
    key = str(factors.get("key") or factors.get("measure") or "distance").lower()
    return {"time": "duration_s", "duration": "duration_s", "distance": "distance_m"}.get(key, key)


def wire_arguments(
    operator: str, wiring: _Wiring, *, output_types: dict[str, str]
) -> dict[str, Any]:
    """Arguments for a chosen operator, derived from its inputs and semantic factors."""

    factors = wiring.factors
    arity = len(wiring.inputs)
    if operator == "batch_geocode":
        return {}                                        # names are bound from the concept graph
    if operator == "nearby_places":
        return {"center": wiring.place(0)} if arity else {}
    if operator == "batch_place_details":
        return {"place_ids": wiring.whole(0)} if arity else {}
    if operator == "place_details":
        # The id lives on the located place, so the reference is that place's `place_id` field.
        return {"place_id": f"{wiring.place(0)}_id"} if arity else {}
    if operator == "place_search":
        return {}
    if operator == "haversine_distance":
        first, second = wiring.two_places()
        return {"place_a": first, "place_b": second}
    if operator == "pairwise_distances":
        return {"pairs": wiring.whole(0)} if arity else {}
    if operator == "pairwise_extremes":
        return {"locations": wiring.whole(0)} if arity else {}
    if operator in {"directions", "travel_time"}:
        origin, destination = wiring.two_places()
        return {"origin": origin, "destination": destination}
    if operator == "distance_matrix":
        return {"origins": wiring.whole(0), "destinations": wiring.whole(0)} if arity else {}
    if operator in {"extract_distance", "extract_duration", "steps_analysis"}:
        return {"route": wiring.whole(0)} if arity else {}
    if operator == "compare_routes":
        return {"routes": [wiring.whole(i) for i in range(arity)]}
    if operator == "tsp_tw":
        nodes = wiring.by_type("object", output_types) or (wiring.whole(0) if arity else None)
        matrix = wiring.by_type("field", output_types)
        arguments: dict[str, Any] = {}
        if nodes:
            arguments["nodes"] = nodes
        if matrix:
            arguments["distance_matrix"] = matrix
        return arguments
    if operator == "calculate_finish_time":
        return {"locations": wiring.whole(0)} if arity else {}
    if operator == "calculate_start_time":
        return {"duration_s": wiring.whole(0)} if arity else {}
    if operator == "within_radius":
        return (
            {"center": wiring.place(0), "candidates": wiring.whole(1)}
            if arity >= 2
            else {"candidates": wiring.whole(0)}
        )
    if operator == "filter_by_direction":
        return (
            {"center": wiring.place(0), "places": wiring.whole(1)}
            if arity >= 2
            else {"places": wiring.whole(0)}
        )
    if operator == "filter_places":
        return {"places": wiring.whole(0)} if arity else {}
    if operator == "nearest":
        return (
            {"anchor": wiring.place(0), "candidates": wiring.whole(1)}
            if arity >= 2
            else {"candidates": wiring.whole(0)}
        )
    if operator == "sort_by":
        return {"items": wiring.items(0), "key": _rank_key(factors)} if arity else {}
    if operator == "select_by_index":
        return {"items": wiring.items(0), "index": _ordinal_index(factors)} if arity else {}
    if operator in {"select_min", "select_max"}:
        return {"items": wiring.items(0), "key": _rank_key(factors)} if arity else {}
    if operator == "sum_amounts":
        return {"amounts": [wiring.whole(i) for i in range(arity)]}
    if operator == "difference":
        return (
            {"minuend": wiring.whole(0), "subtrahend": wiring.whole(1)}
            if arity >= 2
            else {"minuend": wiring.whole(0)}
        )
    if operator == "aggregate_route_groups":
        return (
            {"routes": wiring.whole(0), "groups": wiring.whole(1)}
            if arity >= 2
            else {"routes": wiring.whole(0)}
        )
    if operator == "calculate_proportion":
        return (
            {"numerator": wiring.whole(0), "denominator": wiring.whole(1)}
            if arity >= 2
            else {"numerator": wiring.whole(0)}
        )
    if operator == "match_options":
        return {"places": wiring.whole(0)} if arity else {}
    if operator == "match_distance_options":
        return {"distance": wiring.whole(0)} if arity else {}
    if operator == "match_type_options":
        return {"place": wiring.place(0)} if arity else {}
    if operator == "identity_measure":
        return {"value": wiring.whole(0)} if arity else {}
    # An operator the vocabulary can reach but this table has no wiring for. Forwarding the first
    # input under its single required slot is what `normalize_and_validate_graph` does for a lone
    # dependency, so it is the same convention rather than a new one.
    return {"value": wiring.whole(0)} if arity else {}


# ---------------------------------------------------------------------------------------------
# Factorization G -> G'
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticFactorization:
    """The executable graph, and a record of how each node was chosen."""

    graph: list[dict[str, Any]]
    #: One row per node: the transformation asked for, the operator picked, and the rule.
    decisions: tuple[dict[str, str], ...]
    #: Nodes the planner named an operator for instead of a transformation.
    concrete_nodes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "graph": self.graph,
            "decisions": [dict(row) for row in self.decisions],
            "concrete_nodes": list(self.concrete_nodes),
        }


def is_semantic_graph(steps: Sequence[Any]) -> bool:
    """Did the planner answer in transformations rather than in operators?"""

    return any(
        isinstance(step, dict) and str(step.get("transform") or "").strip() for step in steps
    )


def factorize_semantic_graph(
    steps: Sequence[Any],
    *,
    concepts: Sequence[dict[str, Any]],
    options: Sequence[str],
    facts: Any,
    available: frozenset[str],
) -> SemanticFactorization:
    """Map a semantic transformation graph onto executable operators.

    The question is not a parameter, and that is the point: choosing the operator for a spatial
    relation is mechanical once the concept types and the transformation are known, so it is done
    here, once, the same way every time -- not asked of a language model per question from a
    prompt that had to list Kakao's category codes to make the answer possible.
    """

    concept_text = {
        str(concept.get("id")): str(concept.get("text") or "")
        for concept in concepts
        if isinstance(concept, dict)
    }
    graph: list[dict[str, Any]] = []
    decisions: list[dict[str, str]] = []
    concrete: list[str] = []
    produced_by: dict[str, str] = {}
    output_types: dict[str, str] = {}

    for position, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"GeoFlow node {position} is not an object: {step!r}")
        node_id = str(step.get("id") or f"n{position + 1}")
        inputs = [
            str(value)
            for value in (step.get("inputs") or step.get("depends_on") or [])
            if str(value) in produced_by or str(value)
        ]
        inputs = [value for value in inputs if value in produced_by]
        factors = step.get("factors")
        factors = dict(factors) if isinstance(factors, dict) else {}
        transform_name = str(step.get("transform") or "").strip()

        if not transform_name:
            # The planner named an operator. Kept rather than refused -- a graph that would have
            # executed is not worth losing to a vocabulary preference -- but counted, because
            # "how often does the planner still reach for a tool" is what says whether the
            # semantic layer took.
            operator = step.get("operator")
            if operator is None or operator == "":
                raise ValueError(
                    f"GeoFlow node {node_id!r} names neither a transformation nor an operator"
                )
            concrete.append(node_id)
            # Passed through exactly as written, including an operator that is not a string and
            # arguments that are not an object. Both are planner faults with their own diagnoses
            # downstream, and normalising them here would replace a message that names the
            # problem with one that does not.
            arguments = step.get("arguments")
            if arguments is None:
                arguments = step.get("params")
            if arguments is None:
                arguments = {}
            rule = "planner_named_the_operator"
            output_type = str(step.get("output_type") or "object")
        else:
            operator, rule = resolve_operator(
                transform_name,
                factors,
                input_types=[output_types.get(name, "object") for name in inputs],
                facts=facts,
                available=available,
            )
            wiring = _Wiring(
                inputs=tuple(inputs),
                producers=tuple(produced_by.get(name) for name in inputs),
                factors=factors,
            )
            arguments = wire_arguments(operator, wiring, output_types=output_types)
            arguments = _bind_named_entities(
                operator, arguments, step, factors, concept_text, options
            )
            output_type = TRANSFORMS[transform_name.upper()].output_type

        produced_by[node_id] = operator if isinstance(operator, str) else ""
        output_types[node_id] = output_type
        decisions.append(
            {
                "id": node_id,
                "transform": transform_name or "(operator)",
                "operator": operator if isinstance(operator, str) else repr(operator),
                "rule": rule,
            }
        )
        graph.append(
            {
                "id": node_id,
                "operator": operator,
                "arguments": arguments,
                "depends_on": list(inputs),
                "output_type": output_type,
                "role": str(step.get("role") or "support"),
                "concept_ids": [str(value) for value in (step.get("concept_ids") or [])],
            }
        )
    return SemanticFactorization(graph, tuple(decisions), tuple(concrete))


def _bind_named_entities(
    operator: str,
    arguments: dict[str, Any],
    step: dict[str, Any],
    factors: dict[str, Any],
    concept_text: dict[str, str],
    options: Sequence[str],
) -> dict[str, Any]:
    """Fill the place names a geocode needs from the concept graph, never from the planner.

    The entities are already in the concept graph -- that is what the Analysis stage extracted
    them for -- so a `RESOLVE_PLACES` node names the concepts it resolves and the text comes from
    there. Letting the planner retype them is how `백련산꿈마을숲정이` became
    `백련산꿈마을숲정` and geocoded nothing, three nodes before the failure surfaced.
    """

    if operator != "batch_geocode":
        return arguments
    if str(factors.get("scope") or "").lower() in {"options", "candidates"}:
        arguments["place_names"] = list(options)
        return arguments
    named = [
        concept_text[str(value)]
        for value in (step.get("concept_ids") or [])
        if str(value) in concept_text and concept_text[str(value)].strip()
    ]
    if named:
        arguments["place_names"] = named
    elif isinstance(step.get("place_names"), list):
        arguments["place_names"] = [str(value) for value in step["place_names"]]
    return arguments


# ---------------------------------------------------------------------------------------------
# Lifting G' back to G
#
# The inverse map exists for one reason: to check this module against the behaviour it replaces.
# Every macro-template's worked example was written as concrete operators, and lifting one to
# transformations and factorizing it back has to return the operators it started from. That is
# how `Retrieve-Rank-Ordinal` earns its deletion -- not by argument, but by the composition
# reproducing it.
# ---------------------------------------------------------------------------------------------

_LIFT: dict[str, str] = {
    "batch_geocode": "RESOLVE_PLACES",
    "geocode": "RESOLVE_PLACES",
    "nearby_places": "PLACE_SEARCH",
    "place_search": "PLACE_SEARCH",
    "batch_place_details": "PLACE_DETAILS",
    "place_details": "PLACE_DETAILS",
    "haversine_distance": "DISTANCE_MEASURE",
    "pairwise_distances": "DISTANCE_MEASURE",
    "directions": "ROUTE_MEASURE",
    "travel_time": "ROUTE_MEASURE",
    "distance_matrix": "ROUTE_MATRIX",
    "extract_distance": "ROUTE_EXTRACT",
    "extract_duration": "ROUTE_EXTRACT",
    "steps_analysis": "ROUTE_STEPS",
    "compare_routes": "ROUTE_COMPARE",
    "tsp_tw": "ROUTE_OPTIMIZE",
    "calculate_finish_time": "SCHEDULE",
    "calculate_start_time": "SCHEDULE",
    "within_radius": "FILTER",
    "filter_by_direction": "FILTER",
    "filter_places": "FILTER",
    "nearest": "SORT",
    "sort_by": "SORT",
    "select_by_index": "ORDINAL_SELECT",
    "select_min": "EXTREME_SELECT",
    "select_max": "EXTREME_SELECT",
    "pairwise_extremes": "EXTREME_SELECT",
    "sum_amounts": "AGGREGATE",
    "difference": "AGGREGATE",
    "calculate_proportion": "AGGREGATE",
    "aggregate_route_groups": "AGGREGATE",
    "match_options": "MATCH_OPTIONS",
    "match_distance_options": "MATCH_OPTIONS",
    "match_type_options": "MATCH_OPTIONS",
    "identity_measure": "MEASURE",
}

_LIFT_FACTORS: dict[str, dict[str, Any]] = {
    "travel_time": {"measure": "duration"},
    "extract_duration": {"measure": "duration"},
    "select_min": {"extreme": "min"},
    "select_max": {"extreme": "max"},
    "difference": {"aggregate": "difference"},
    "calculate_proportion": {"aggregate": "proportion"},
    "aggregate_route_groups": {"scope": "groups"},
    "calculate_start_time": {"measure": "start"},
    "sort_by": {"key": "distance"},
    "place_details": {"scope": "one"},
}


def lift_to_semantic(graph: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Express a concrete operator graph in the semantic vocabulary."""

    lifted: list[dict[str, Any]] = []
    for step in graph:
        operator = str(step.get("operator") or "")
        transform = _LIFT.get(operator)
        node: dict[str, Any] = {
            "id": str(step.get("id") or ""),
            "inputs": [str(value) for value in (step.get("depends_on") or [])],
            "role": str(step.get("role") or "support"),
        }
        if step.get("concept_ids"):
            node["concept_ids"] = [str(value) for value in step["concept_ids"]]
        if transform is None:
            # Not in the vocabulary. Left concrete, which the factorizer accepts and counts.
            node["operator"] = operator
            node["arguments"] = step.get("arguments") or {}
            lifted.append(node)
            continue
        node["transform"] = transform
        factors = dict(_LIFT_FACTORS.get(operator, {}))
        arguments = step.get("arguments") or {}
        if operator == "select_by_index" and isinstance(arguments, dict):
            index = arguments.get("index")
            factors["ordinal"] = (int(index) + 1) if isinstance(index, int) else 1
        if operator in {"sort_by", "select_min", "select_max"} and isinstance(arguments, dict):
            if arguments.get("key"):
                factors["key"] = str(arguments["key"])
        if operator == "batch_geocode" and isinstance(arguments, dict):
            names = arguments.get("place_names")
            if isinstance(names, list):
                node["place_names"] = [str(value) for value in names]
        if factors:
            node["factors"] = factors
        lifted.append(node)
    return lifted
