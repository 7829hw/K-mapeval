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
from dataclasses import dataclass, field
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
    #: Which upstream nodes emit a set of candidates rather than a single place.
    input_is_collection: tuple[bool, ...] = ()
    #: The operators those upstream nodes were factorized to.
    input_operators: tuple[str, ...] = ()
    #: How many places the node declared as waypoints. A route through somewhere is a different
    #: route from the one between its ends, and only the graph can say which is asked for.
    via_count: int = 0

    @property
    def fans_out(self) -> bool:
        """One place measured against many, which is a ranking rather than a pair."""

        return len(self.input_is_collection) >= 2 and (
            not self.input_is_collection[0] and any(self.input_is_collection[1:])
        )

    def factor(self, name: str, default: Any = None) -> Any:
        return self.factors.get(name, default)

    def has(self, name: str) -> bool:
        value = self.factors.get(name)
        return value is not None and value != ""


def _always(_: Resolution) -> bool:
    return True


def _one_place_against_many(resolution: Resolution) -> bool:
    """"How far is each of these from that" is a ranking, and `nearest` is what computes it.

    Wiring it as `haversine_distance(place_a=anchor, place_b=$candidates.0.place)` measures the
    first candidate and throws the rest away -- which is what it did, on 190 recorded graphs.
    """

    return resolution.fans_out


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


def _over_a_matrix(resolution: Resolution) -> bool:
    """Many routes are not a route; asking a grid of them for `distance_m` gets a dict and an
    error. A selection of legs is the same shape, already narrowed to the ones travelled."""

    return bool({"distance_matrix", "select_legs"} & set(resolution.input_operators))


def _through_waypoints(resolution: Resolution) -> bool:
    """A route the question sends through somewhere is one route with waypoints.

    `travel_time` carries an origin and a destination and nothing between them, so a route
    stated as "A에서 B를 들러 C까지" has to be measured by the operator that takes waypoints --
    otherwise the drive that gets measured is a different drive.
    """

    return resolution.via_count > 0


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


#: How a node says its aggregate covers the legs an itinerary drives rather than every pair a
#: matrix holds. `groups` is the older spelling and means the same thing when no explicit group
#: list is supplied.
_LEG_SCOPES = frozenset({"groups", "grouped", "legs", "consecutive", "consecutive_legs",
                         "itinerary", "route_legs"})


def _grouped(resolution: Resolution) -> bool:
    """Per-option totals: one group of route indexes per candidate order."""

    return str(resolution.factor("scope", "")).lower() in _LEG_SCOPES


def _over_route_legs(resolution: Resolution) -> bool:
    """A total over routes that have already been narrowed to the ones the trip drives."""

    return "select_legs" in resolution.input_operators


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
        "Straight-line separation between places: a pair, or each of many from one.",
        (
            ("nearest", _one_place_against_many),
            ("haversine_distance", _pairwise),
            ("pairwise_distances", _over_a_collection),
        ),
        "object",
    ),
    "ROUTE_MEASURE": Transform(
        "ROUTE_MEASURE",
        "The road route from one place to another, optionally through stated waypoints.",
        (("directions", _through_waypoints), ("travel_time", _duration),
         ("directions", _distance)),
        "field",
    ),
    "ROUTE_MATRIX": Transform(
        "ROUTE_MATRIX",
        "Road cost between every pair of an itinerary's stops.",
        (("distance_matrix", _always),),
        "field",
    ),
    "SELECT_LEGS": Transform(
        "SELECT_LEGS",
        "Keep only the legs an ordered itinerary drives, out of a matrix of every pair.",
        (("select_legs", _always),),
        "field",
    ),
    "ROUTE_EXTRACT": Transform(
        "ROUTE_EXTRACT",
        "One number out of a computed route, or the total over a matrix of them.",
        (
            ("sum_route_metrics", _over_a_matrix),
            ("extract_duration", _duration),
            ("extract_distance", _distance),
        ),
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
            # Ahead of the grouped total: once the legs the trip drives have been selected there
            # is one group and it is all of them, and `aggregate_route_groups` would be asked for
            # a group list nothing in the graph carries.
            ("sum_route_metrics", _over_route_legs),
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
    input_is_collection: Sequence[bool] | None = None,
    input_operators: Sequence[str] | None = None,
    via_count: int = 0,
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
        input_is_collection=tuple(input_is_collection or ()),
        input_operators=tuple(input_operators or ()),
        via_count=via_count,
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
#: Operators whose output is a set of candidates rather than one place. Measuring *from* a place
#: *to* one of these is a fan-out, not a pair, and wiring it as a pair measures the first
#: candidate and silently discards the rest.
_COLLECTION_OPERATORS = frozenset(
    {
        "nearby_places",
        "place_search",
        "within_radius",
        "filter_by_direction",
        "filter_places",
        "nearest",
        "sort_by",
        "merge_places",
        "recover_option_places",
    }
)
#: Operators that already return an ordering, so a SORT over one of them has nothing to do.
_ORDERING_OPERATORS = frozenset({"nearest", "sort_by"})
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
    #: The node a route matrix was built at, if the graph built one. `tsp_tw` needs it and a
    #: planner that lists only the places forgets to feed it.
    matrix_source: str | None = None
    #: References to the places the route passes through, in the order it passes them. Named by
    #: the graph's own `via` relation, never inferred from where an input happens to sit.
    via: tuple[str, ...] = ()
    #: Positions inside a single record-list input that `via` claimed, so the ends of the route
    #: are read from the positions it did not.
    via_positions: frozenset[int] = frozenset()
    #: How many places the single record-list input resolved, so the far end can be found.
    resolved_count: int = 0
    #: How many places each input resolved, so a stop list gathers all of them.
    resolved_sizes: tuple[int, ...] = ()
    #: The stop references every matrix node was built over, keyed by node id. A tour indexes
    #: into its cost matrix, so its node list has to be that matrix's own stops in that order.
    matrix_stops: dict[str, list[str]] = field(default_factory=dict)

    def place(self, position: int, index: int = 0) -> str:
        node = self.inputs[position]
        return _place_ref(node, self.producers[position], index)

    def whole(self, position: int) -> str:
        return f"${self.inputs[position]}"

    def items(self, position: int) -> str:
        return _items_ref(self.inputs[position], self.producers[position])

    def two_places(self) -> tuple[str, str]:
        """The two places to relate: one node holding both, or one node each.

        When the graph declared waypoints inside a single resolved list, the ends of the route
        are the first and *last* positions the waypoints did not claim. Reading the second
        position as the destination is what measured 인디스타 → 소설호텔 for a question about
        인디스타 → 소설호텔 → 공작지.
        """

        if not self.inputs:
            return "", ""
        if len(self.inputs) >= 2:
            return self.place(0), self.place(1)
        if self.via_positions:
            free = [
                index
                for index in range(self.resolved_count)
                if index not in self.via_positions
            ]
            if len(free) >= 2:
                return self.place(0, free[0]), self.place(0, free[-1])
        return self.place(0, 0), self.place(0, 1)

    def every_place(self) -> list[str]:
        """One reference per place across every input, in the order the inputs were given.

        An input that resolved several names contributes each of them: an itinerary written as
        one anchor node and one node holding the rest is the common shape, and taking `.0.place`
        of the second lost every stop after the first.
        """

        references: list[str] = []
        for position in range(len(self.inputs)):
            size = self.resolved_sizes[position] if position < len(self.resolved_sizes) else 0
            if self.producers[position] in _RECORD_LIST_OPERATORS and size > 1:
                references.extend(self.place(position, index) for index in range(size))
            else:
                references.append(self.place(position))
        return references

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
        return {"place_a": first, "place_b": second} if first else {}
    if operator == "pairwise_distances":
        return {"pairs": wiring.whole(0)} if arity else {}
    if operator == "pairwise_extremes":
        return {"locations": wiring.whole(0)} if arity else {}
    if operator in {"directions", "travel_time"}:
        origin, destination = wiring.two_places()
        if not origin:
            return {}
        routed: dict[str, Any] = {"origin": origin, "destination": destination}
        if wiring.via and operator == "directions":
            # In the order the graph listed them: Kakao drives the waypoints as given, so
            # reversing two of them measures a different drive that still routes.
            routed["waypoints"] = list(wiring.via)
        return routed
    if operator == "distance_matrix":
        if not arity:
            return {}
        # Several resolved nodes are several stops, not a matrix over the first one. A planner
        # that resolves each stop separately -- which it does about half the time -- produced a
        # 1x1 grid, and the leg selection then had a single route and no leg to take.
        stops = wiring.every_place() if arity >= 2 else wiring.whole(0)
        return {"origins": stops, "destinations": stops}
    if operator in {"extract_distance", "extract_duration", "steps_analysis"}:
        return {"route": wiring.whole(0)} if arity else {}
    if operator == "select_legs":
        return {"routes": wiring.whole(0)} if arity else {}
    if operator == "sum_route_metrics":
        return {"routes": wiring.whole(0)} if arity else {}
    if operator == "compare_routes":
        return {"routes": [wiring.whole(i) for i in range(arity)]}
    if operator == "tsp_tw":
        matrix = wiring.by_type("field", output_types) or wiring.matrix_source
        # The tour's order indexes into its cost matrix, so the nodes are that matrix's own
        # stops, in the order it was built over. Taking them from whichever input happened to be
        # object-typed gave a five-stop trip a one-place node list beside a six-place matrix.
        stops = wiring.matrix_stops.get(str(matrix or "").lstrip("$")) if matrix else None
        nodes: Any = stops or wiring.by_type("object", output_types)
        if not nodes and arity:
            nodes = wiring.whole(0)
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
        # The radius is a stated fact and grounding binds it; the centre is whichever input is a
        # single place. A FILTER given only the candidate set has no centre to offer, and asking
        # for one it does not have is a refusal rather than a graph.
        if arity >= 2:
            return {"center": wiring.place(0), "candidates": wiring.whole(1)}
        if arity == 1:
            return {"center": wiring.place(0), "candidates": wiring.whole(0)}
        return {}
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


#: Transformations that read a computed route. Given places instead, the route between them is
#: what they meant, and composing it is the same macro expansion `tsp_tw`'s cost matrix gets.
_ROUTE_READING_TRANSFORMS = frozenset({"ROUTE_EXTRACT", "ROUTE_STEPS"})


#: Operators whose output is unambiguously a located place. A route reader fed one of these was
#: given an endpoint, never a route.
_GEOCODERS = frozenset({"batch_geocode", "geocode", "reverse_geocode"})


def _needs_a_route_composed(
    transform_name: str, inputs: Sequence[str], producers: Sequence[str]
) -> bool:
    """A route reader handed the endpoints of a route rather than the route.

    `ROUTE_EXTRACT` over two `batch_geocode` nodes is how a planner writes "the distance of the
    leg from A to B", and it wired `extract_distance(route=$A)` -- a place where a route belongs.
    Every graph that decomposed a drive into legs wrote it this way: two of the recorded
    `routing_turn_count_via` graphs and three of `trip_total_distance`, each one an errored step
    or a repair round.

    Only a geocoder counts. `compare_routes` is typed `object` too and its output is a route that
    was chosen; composing a drive from that end to itself is what the first version of this rule
    did to two `routing_detour_cost` graphs, and the replay caught it before the benchmark did.
    """

    if transform_name.upper() not in _ROUTE_READING_TRANSFORMS:
        return False
    if len(inputs) < 2:
        return False
    return all(producer in _GEOCODERS for producer in producers)


#: Transformations that would otherwise total every pair of a matrix rather than the legs a
#: trip drives. `ROUTE_EXTRACT` is here because `sum_route_metrics` over a matrix node is the
#: same sum under a different name.
_MATRIX_TOTALLING_TRANSFORMS = frozenset({"AGGREGATE", "ROUTE_EXTRACT"})


def _totals_a_bare_matrix(
    transform_name: str,
    factors: dict[str, Any],
    inputs: Sequence[str],
    produced_by: dict[str, str],
    square_matrices: set[str],
) -> bool:
    """Is this node about to add up a whole origins x destinations grid?"""

    if transform_name.upper() not in _MATRIX_TOTALLING_TRANSFORMS:
        return False
    if len(inputs) != 1 or inputs[0] not in square_matrices:
        return False
    if produced_by.get(inputs[0]) != "distance_matrix":
        return False
    aggregate = str(factors.get("aggregate", "sum")).lower()
    return aggregate not in {"difference", "subtract", "minus", "proportion", "share", "ratio"}


def _resolve_via(
    step: dict[str, Any],
    inputs: Sequence[str],
    follow: Callable[[str], str],
    produced_by: dict[str, str],
    resolved_concepts: dict[str, list[str]],
) -> tuple[tuple[str, ...], frozenset[int], tuple[str, ...]]:
    """The places a route passes through, as the graph named them.

    `via` is an explicit relation on the node -- a list of ids, in the order the route reaches
    them. Each id is either an upstream node that resolved one place, or a concept an upstream
    `RESOLVE_PLACES` bound; both become a reference to that place. Nothing here reads which
    input sat where: a waypoint that the graph did not declare is not a waypoint, because the
    alternative is deciding that the middle of any three places is one, and "A와 B 중 C에 더
    가까운 곳" has a middle too.

    Returns the references, which positions of a single resolved list they claimed, and which
    upstream nodes they depend on.
    """

    declared = step.get("via")
    if declared is None:
        declared = step.get("waypoints")
    if isinstance(declared, str):
        declared = [declared]
    if not isinstance(declared, list) or not declared:
        return (), frozenset(), ()

    references: list[str] = []
    positions: set[int] = set()
    depends: list[str] = []
    for raw in declared:
        name = str(raw).lstrip("$").strip()
        if not name:
            continue
        node = follow(name.split(".", 1)[0])
        if node in produced_by:
            references.append(_place_ref(node, produced_by[node]))
            depends.append(node)
            continue
        for source in inputs:
            aligned = resolved_concepts.get(source) or []
            if name in aligned:
                index = aligned.index(name)
                references.append(_place_ref(source, produced_by.get(source), index))
                positions.add(index)
                break
    return tuple(references), frozenset(positions), tuple(dict.fromkeys(depends))


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

    # Synthetic concepts are excluded outright. The role completion inserts one, carrying the
    # whole question as its text, whenever an analysis produced no extent -- it exists so the
    # graph has a contextual root, and it names no place. A planner that lists it in
    # `concept_ids` would otherwise have a `batch_geocode` sent after a sentence.
    concept_text = {
        str(concept.get("id")): str(concept.get("text") or "")
        for concept in concepts
        if isinstance(concept, dict) and not (concept.get("attributes") or {}).get("synthetic")
    }
    graph: list[dict[str, Any]] = []
    decisions: list[dict[str, str]] = []
    concrete: list[str] = []
    produced_by: dict[str, str] = {}
    output_types: dict[str, str] = {}
    # A node the factorization folded into its input: the id still resolves, to the node that
    # already did the work. `SORT` over a `nearest` is the case that matters -- the ordering
    # exists, so emitting a second node to re-sort it is at best redundant and at worst asks an
    # operator for a field its input does not carry.
    fused: dict[str, str] = {}
    # Whether a node emits a set of candidates. An operator can say so by itself, but a
    # `batch_geocode` depends on how many names it was given: four option texts are a candidate
    # set and one anchor is not, and measuring from an anchor *to* a set is a ranking.
    emits_collection: dict[str, bool] = {}
    # Which concept every earlier RESOLVE_PLACES already claimed, so a later one that names no
    # resolvable concept does not take the same places again.
    claimed: set[str] = set()
    # Which concept sits at which position of a resolved list, for the nodes where the binding
    # came from named concepts. `via` names a concept and the reference it becomes is that
    # concept's position, so a route through the second of three stops is written as such
    # rather than assumed from where the argument fell.
    resolved_concepts: dict[str, list[str]] = {}
    # How many places each node resolved, so the far end of a route can be found.
    resolved_counts: dict[str, int] = {}
    # Matrix nodes whose origins and destinations are the same list, so their routes form a
    # square grid whose consecutive legs are the ones an itinerary drives.
    square_matrices: set[str] = set()
    # The stop references every matrix node was built over, so a tour that indexes into one can
    # be given the same list in the same order.
    matrix_stops: dict[str, list[str]] = {}

    def follow(name: str) -> str:
        seen: set[str] = set()
        while name in fused and name not in seen:
            seen.add(name)
            name = fused[name]
        return name

    for position, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"GeoFlow node {position} is not an object: {step!r}")
        node_id = str(step.get("id") or f"n{position + 1}")
        inputs = [
            str(value)
            for value in (step.get("inputs") or step.get("depends_on") or [])
            if str(value) in produced_by or str(value)
        ]
        # A planner sometimes writes an input as a path -- "R1.start" -- meaning one part of a
        # node's output. The node is what the graph depends on; the part is a wiring detail this
        # module decides. Dropping the whole input instead left the node with no inputs at all.
        inputs = [follow(str(value).split(".", 1)[0]) for value in inputs]
        inputs = [value for value in inputs if value in produced_by]
        inputs = list(dict.fromkeys(inputs))
        factors = step.get("factors")
        factors = dict(factors) if isinstance(factors, dict) else {}
        transform_name = str(step.get("transform") or "").strip()
        via_refs, via_positions, via_nodes = _resolve_via(
            step, inputs, follow, produced_by, resolved_concepts
        )
        # A waypoint is something the route depends on, so it joins the dependencies even when
        # the node did not also list it as an input.
        inputs = list(dict.fromkeys([*inputs, *via_nodes]))

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
            if _needs_a_route_composed(
                transform_name, inputs, [produced_by.get(name) or "" for name in inputs]
            ) and (
                "directions" in available
            ):
                route_id = f"{node_id}_route"
                route_wiring = _Wiring(
                    inputs=tuple(inputs),
                    producers=tuple(produced_by.get(name) for name in inputs),
                    factors=factors,
                    via=via_refs,
                    via_positions=via_positions,
                    resolved_count=resolved_counts.get(inputs[0], 0) if inputs else 0,
                    resolved_sizes=tuple(resolved_counts.get(name, 0) for name in inputs),
                )
                graph.append(
                    {
                        "id": route_id,
                        "operator": "directions",
                        "arguments": wire_arguments(
                            "directions", route_wiring, output_types=output_types
                        ),
                        "depends_on": list(inputs),
                        "output_type": "field",
                        "role": "support",
                        "concept_ids": [],
                    }
                )
                produced_by[route_id] = "directions"
                emits_collection[route_id] = False
                output_types[route_id] = "field"
                decisions.append(
                    {
                        "id": route_id,
                        "transform": "ROUTE_MEASURE",
                        "operator": "directions",
                        "rule": "composed_route_for_a_route_reader",
                    }
                )
                inputs = [route_id]
                via_refs, via_positions = (), frozenset()
            if (
                transform_name.upper() == "SELECT_LEGS"
                and len(inputs) >= 2
                and all(produced_by.get(name) in _GEOCODERS for name in inputs)
                and "distance_matrix" in available
            ):
                # The legs of an itinerary come out of the costs between its stops, so a
                # SELECT_LEGS handed the stops themselves is asking for the matrix first. Same
                # expansion as the route a ROUTE_EXTRACT needs, one shape up.
                grid_id = f"{node_id}_matrix"
                stops = _Wiring(
                    inputs=tuple(inputs),
                    producers=tuple(produced_by.get(name) for name in inputs),
                    factors=factors,
                    resolved_sizes=tuple(resolved_counts.get(name, 0) for name in inputs),
                ).every_place()
                graph.append(
                    {
                        "id": grid_id,
                        "operator": "distance_matrix",
                        "arguments": {"origins": stops, "destinations": stops},
                        "depends_on": list(inputs),
                        "output_type": "field",
                        "role": "support",
                        "concept_ids": [],
                    }
                )
                produced_by[grid_id] = "distance_matrix"
                square_matrices.add(grid_id)
                matrix_stops[grid_id] = list(stops)
                emits_collection[grid_id] = False
                output_types[grid_id] = "field"
                decisions.append(
                    {
                        "id": grid_id,
                        "transform": "ROUTE_MATRIX",
                        "operator": "distance_matrix",
                        "rule": "composed_matrix_for_leg_selection",
                    }
                )
                inputs = [grid_id]
            if _totals_a_bare_matrix(
                transform_name, factors, inputs, produced_by, square_matrices
            ) and "select_legs" in available:
                # A square matrix holds every pair; an itinerary drives the consecutive ones.
                # Totalling the matrix answers a question about n^2 legs when the trip has
                # n-1 of them, which is how `trip_total_distance` returned a confident number
                # roughly four times too large with no step reporting an error. The selection
                # is a node of its own so the grouping stays visible in the graph.
                legs_id = f"{node_id}_legs"
                source = inputs[0]
                graph.append(
                    {
                        "id": legs_id,
                        "operator": "select_legs",
                        "arguments": {"routes": f"${source}"},
                        "depends_on": [source],
                        "output_type": "field",
                        "role": "support",
                        "concept_ids": [],
                    }
                )
                produced_by[legs_id] = "select_legs"
                emits_collection[legs_id] = False
                output_types[legs_id] = "field"
                decisions.append(
                    {
                        "id": legs_id,
                        "transform": "SELECT_LEGS",
                        "operator": "select_legs",
                        "rule": "composed_consecutive_legs",
                    }
                )
                inputs = [legs_id]
            operator, rule = resolve_operator(
                transform_name,
                factors,
                input_types=[output_types.get(name, "object") for name in inputs],
                facts=facts,
                available=available,
                input_is_collection=[emits_collection.get(name, False) for name in inputs],
                input_operators=[produced_by.get(name) or "" for name in inputs],
                via_count=len(via_refs),
            )
            if (
                transform_name.upper() == "SORT"
                and len(inputs) == 1
                and produced_by.get(inputs[0]) in _ORDERING_OPERATORS
            ):
                # The ordering already exists. Fold this node into the one that produced it and
                # let every downstream reference resolve there.
                fused[node_id] = inputs[0]
                decisions.append(
                    {
                        "id": node_id,
                        "transform": transform_name,
                        "operator": produced_by[inputs[0]],
                        "rule": "fused_into_existing_ordering",
                    }
                )
                continue
            wiring = _Wiring(
                inputs=tuple(inputs),
                producers=tuple(produced_by.get(name) for name in inputs),
                factors=factors,
                matrix_source=next(
                    (f"${name}" for name, op in produced_by.items() if op == "distance_matrix"),
                    None,
                ),
                via=via_refs,
                via_positions=via_positions,
                resolved_count=resolved_counts.get(inputs[0], 0) if inputs else 0,
                resolved_sizes=tuple(resolved_counts.get(name, 0) for name in inputs),
                matrix_stops=matrix_stops,
            )
            arguments = wire_arguments(operator, wiring, output_types=output_types)
            arguments, aligned = _bind_named_entities(
                operator, arguments, step, factors, concept_text, options, concepts, claimed
            )
            names = arguments.get("place_names") or []
            for name in names:
                claimed.add(str(name))
            if operator == "batch_geocode":
                resolved_counts[node_id] = len(names)
                if len(aligned) == len(names):
                    resolved_concepts[node_id] = aligned
            output_type = TRANSFORMS[transform_name.upper()].output_type

        if (
            operator == "tsp_tw"
            and isinstance(arguments, dict)
            and not arguments.get("distance_matrix")
            and inputs
        ):
            # Ordering an itinerary needs the cost between its stops, and a planner that names
            # ROUTE_OPTIMIZE over the places alone has not asked for one. The matrix is implied
            # by the transformation rather than chosen by it, so it is composed in here -- this
            # is the macro-template expansion the deterministic stage is for. Nine of 319
            # recorded graphs needed it.
            matrix_id = f"{node_id}_matrix"
            places = f"${inputs[0]}"
            graph.append(
                {
                    "id": matrix_id,
                    "operator": "distance_matrix",
                    "arguments": {"origins": places, "destinations": places},
                    "depends_on": [inputs[0]],
                    "output_type": "field",
                    "role": "support",
                    "concept_ids": [],
                }
            )
            produced_by[matrix_id] = "distance_matrix"
            square_matrices.add(matrix_id)
            matrix_stops[matrix_id] = [places]
            emits_collection[matrix_id] = False
            output_types[matrix_id] = "field"
            decisions.append(
                {
                    "id": matrix_id,
                    "transform": "ROUTE_MATRIX",
                    "operator": "distance_matrix",
                    "rule": "composed_for_route_optimize",
                }
            )
            arguments["distance_matrix"] = f"${matrix_id}"
            inputs = [*inputs, matrix_id]
        if operator == "distance_matrix" and isinstance(arguments, dict):
            origins, destinations = arguments.get("origins"), arguments.get("destinations")
            if origins is not None and origins == destinations:
                square_matrices.add(node_id)
                if isinstance(origins, list):
                    matrix_stops[node_id] = list(origins)
        produced_by[node_id] = operator if isinstance(operator, str) else ""
        # `operator` may not be a string here: a planner that writes
        # `"operator": ["extract_distance", "extract_distance"]` is carried through so the graph
        # validator can name that fault, and a set lookup on a list raises `unhashable type`
        # instead -- a crash in this file, recorded against the agent as if it had reasoned its
        # way there.
        emits_collection[node_id] = isinstance(operator, str) and (
            operator in _COLLECTION_OPERATORS
            or (
                operator == "batch_geocode"
                and len((arguments or {}).get("place_names") or []) > 1
            )
        )
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
    concepts: Sequence[dict[str, Any]],
    claimed: set[str],
) -> tuple[dict[str, Any], list[str]]:
    """Fill the place names a geocode needs from the concept graph, never from the planner.

    The entities are already in the concept graph -- that is what the Analysis stage extracted
    them for -- so a `RESOLVE_PLACES` node names the concepts it resolves and the text comes from
    there. Letting the planner retype them is how `백련산꿈마을숲정이` became
    `백련산꿈마을숲정` and geocoded nothing, three nodes before the failure surfaced.

    The planner does not always name a concept that exists. Measured over 642 recorded nodes it
    named one 520 times and invented an id 122 times -- `Option 0`, `candidate_options`, a slug
    of the anchor's name -- usually when it meant the candidate texts, or when the Analysis stage
    had returned no usable concepts at all. So the lookup falls through an explicit chain rather
    than leaving `place_names` empty, which is a `batch_geocode` that geocodes nothing and a
    graph that fails four nodes later.
    """

    if operator != "batch_geocode":
        return arguments, []
    requested = [str(value) for value in (step.get("concept_ids") or [])]

    # 1. The node said outright that it resolves the candidate texts.
    if str(factors.get("scope") or "").lower() in {"options", "candidates"}:
        arguments["place_names"] = list(options)
        return arguments, []

    # 2. Concepts the analysis actually has.
    aligned = [
        value
        for value in requested
        if value in concept_text and concept_text[value].strip()
    ]
    named = [concept_text[value] for value in aligned]
    if named:
        arguments["place_names"] = named
        return arguments, aligned

    # 3. Every id it named is invented. When they read as the options -- `Option 0`,
    #    `candidate_options`, `option_places` -- that is what it meant.
    if requested and all(
        "option" in value.lower() or "candidate" in value.lower() for value in requested
    ):
        arguments["place_names"] = list(options)
        return arguments, []

    # 4. Otherwise take the located concepts no earlier node has claimed. This is the case where
    #    the Analysis stage returned only `question_context` and the planner had nothing to name.
    unclaimed = [
        text
        for text in (
            str(concept.get("text") or "").strip()
            for concept in concepts
            if isinstance(concept, dict)
            and concept.get("concept_type") in {"location", "object"}
            and concept.get("role") != "measure"
            # A synthetic concept is a structural placeholder the role completion inserted so
            # the graph has an extent, and its text is the whole question. It is not an entity,
            # and geocoding it geocodes a sentence.
            and not (concept.get("attributes") or {}).get("synthetic")
        )
        if text and text not in claimed
    ]
    if unclaimed:
        arguments["place_names"] = unclaimed
        return arguments, []

    # 5. Nothing in the concept graph to resolve. The options are the only places left that the
    #    question certainly names, and an empty geocode is a certain failure.
    arguments["place_names"] = list(options)
    return arguments, []


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
    "select_legs": "SELECT_LEGS",
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
