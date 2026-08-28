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

from src.agent.geoflow import OPERATOR_CONTRACTS
from src.tools.spatial import options_state_counts

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
    #: How many places the node named as the endpoints of its measure. A pairwise measure that
    #: says which pair it is between is a pair, whatever its inputs look like.
    endpoint_count: int = 0
    #: Do the candidate texts answer "how many"? Read off the options, which factorization
    #: already receives; a count matched to "세 곳" is a different relation from a distance
    #: matched to "약 2.4km", and the options are what say which.
    options_are_counts: bool = False

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
    """ "How far is each of these from that" is a ranking, and `nearest` is what computes it.

    Wiring it as `haversine_distance(place_a=anchor, place_b=$candidates.0.place)` measures the
    first candidate and throws the rest away -- which is what it did, on 190 recorded graphs.

    Three or more inputs is the same relation written a different way: a pair has two ends, so a
    measure over an anchor and three candidate nodes is a ranking whatever their types say.
    """

    return resolution.fans_out or (resolution.arity >= 3 and resolution.endpoint_count != 2)


def _between_places_only(resolution: Resolution) -> bool:
    """A comparison handed places rather than routes is a route to measure, not a choice.

    "A에서 B까지 거리가 가장 짧은 경로로 운전합니다" is written `ROUTE_COMPARE` by a planner
    reading `compare_routes` as "pick the best route", and the graph that says it has computed no
    routes at all -- its two inputs are geocoded places. Resolved as a comparison it was expanded
    into a cost matrix and a tour, and a one-node tour is not a route: `steps_analysis` returned a
    single record and `select_by_index(3)` refused it as "outside a collection of 1", which is how
    `routing_nth_turn` and `routing_turn_count_via` read 0 of 14 in every run of this stack.

    Only when *every* input is a geocoder, which is the same guard `_needs_a_route_composed`
    uses and for the same reason: `compare_routes` output is a route too, and composing a drive
    from one of those to itself is a defect this file has already had once.
    """

    # Only the inputs something in the graph produced. A planner names `mode` and `constraint`
    # beside the two places, and those are leaves the question supplied -- reading them as
    # "not a geocoder" made this predicate false on every graph it exists for.
    produced = [operator for operator in resolution.input_operators if operator]
    return bool(produced) and all(operator in _GEOCODERS for operator in produced)


def _pairwise(resolution: Resolution) -> bool:
    """Two distinct places to separate, rather than a list to cross-join."""

    return resolution.endpoint_count == 2 or resolution.arity == 2


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


#: Operators whose output carries a per-item distance, so a radius filter over one of them has
#: the measurement already and needs no centre of its own.
_MEASURED_OPERATORS = frozenset(
    {"nearest", "within_radius", "pairwise_distances", "sort_by", "haversine_distance"}
)


def _over_measured_candidates(resolution: Resolution) -> bool:
    """The graph measured first and is narrowing what it measured.

    This is the shape the question states -- "how far is each of these, keep the ones inside R" --
    and it is the one a planner writes. `within_radius` measures and filters together, so a FILTER
    that follows a DISTANCE_MEASURE has no centre to give it and wired the candidate list itself
    as `center`, which is exactly what "center has no resolved coordinates" was.
    """

    return (
        _stated_radius(resolution)
        and resolution.arity == 1
        and bool(set(resolution.input_operators) & _MEASURED_OPERATORS)
    )


def _stated_radius(resolution: Resolution) -> bool:
    return resolution.facts is not None and getattr(resolution.facts, "radius_m", None) is not None


def _within_a_stated_radius(resolution: Resolution) -> bool:
    return _stated_radius(resolution)


def _within_a_stated_sector(resolution: Resolution) -> bool:
    return resolution.facts is not None and bool(getattr(resolution.facts, "direction", None))


def _by_a_stated_attribute(resolution: Resolution) -> bool:
    """A kind narrowed by a modifier -- 중식 of "중식 음식점" -- is an attribute of the places."""

    return resolution.facts is not None and bool(getattr(resolution.facts, "target_subtype", None))


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
_LEG_SCOPES = frozenset(
    {"groups", "grouped", "legs", "consecutive", "consecutive_legs", "itinerary", "route_legs"}
)


def _grouped(resolution: Resolution) -> bool:
    """Per-option totals: one group of route indexes per candidate order."""

    return str(resolution.factor("scope", "")).lower() in _LEG_SCOPES


def _over_route_legs(resolution: Resolution) -> bool:
    """A total over routes that have already been narrowed to the ones the trip drives."""

    return "select_legs" in resolution.input_operators


def _counts_a_set(resolution: Resolution) -> bool:
    """ "몇 곳" is how many, not how much. A set of places carries nothing to add up."""

    if str(resolution.factor("aggregate", "sum")).lower() in {"count", "how_many", "cardinality"}:
        return True
    # A total over a collection of *places* can only be a count: `sum_amounts` reads a
    # measurement off each item and a place carries none, which raised "received nothing to add".
    return resolution.arity == 1 and any(resolution.input_is_collection)


def _sum(_: Resolution) -> bool:
    return True


def _against_a_count(resolution: Resolution) -> bool:
    """A count is a whole number of things, and the option that states it is the answer.

    `match_distance_options` reads a magnitude out of an option and takes the nearest, and
    "한 곳"/"두 곳" carries no digit for it to read -- so every counting question's matcher
    reported no best option and the response stage answered from the words itself.
    """

    return resolution.options_are_counts and bool(
        {"count_items", "tsp_tw"} & set(resolution.input_operators)
    )


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


#: Transformations whose declared output type is one of several a graph may legitimately name.
#: Read through `accepted_output_types` so canonicalization and validation cannot disagree about
#: what a transformation is allowed to produce.
_POLYMORPHIC_OUTPUTS = {
    "RESOLVE_PLACES": frozenset({"location", "object"}),
    "DISTANCE_MEASURE": frozenset({"amount", "field", "object"}),
    "FILTER": frozenset({"field", "object"}),
    "SORT": frozenset({"field", "object"}),
    "ORDINAL_SELECT": frozenset({"location", "object"}),
    "EXTREME_SELECT": frozenset({"amount", "location", "object"}),
    "ROUTE_COMPARE": frozenset({"field", "object"}),
    "ROUTE_OPTIMIZE": frozenset({"network", "object"}),
    "AGGREGATE": frozenset({"amount", "proportion"}),
    "MEASURE": frozenset({"amount", "event", "field", "network", "object", "proportion"}),
}
_TRANSFORM_INPUTS = {
    "RESOLVE_PLACES": frozenset({"location", "object"}),
    "PLACE_SEARCH": frozenset({"location", "object"}),
    "PLACE_DETAILS": frozenset({"location", "object"}),
    "DISTANCE_MEASURE": frozenset({"field", "location", "object"}),
    "ROUTE_MEASURE": frozenset({"location", "network", "object"}),
    "ROUTE_MATRIX": frozenset({"location", "network", "object"}),
    "SELECT_LEGS": frozenset({"field", "network"}),
    "ROUTE_EXTRACT": frozenset({"field"}),
    # A route arrives typed by whichever transformation produced it: a FIELD from ROUTE_MEASURE,
    # an OBJECT from ROUTE_COMPARE's polymorphic output, a NETWORK from ROUTE_OPTIMIZE. Reading
    # its turns is the same operator in all three cases, and refusing two of them refuses a plan
    # `steps_analysis` would have run.
    "ROUTE_STEPS": frozenset({"field", "network", "object"}),
    "ROUTE_COMPARE": frozenset({"field", "object"}),
    # `tsp_tw` orders places. A place typed LOCATION -- which is what RESOLVE_PLACES,
    # ORDINAL_SELECT and EXTREME_SELECT all emit -- is one.
    "ROUTE_OPTIMIZE": frozenset({"field", "location", "network", "object"}),
    "SCHEDULE": frozenset({"amount", "event", "field", "network"}),
    # DISTANCE_MEASURE emits AMOUNT among its polymorphic outputs, and narrowing candidates by a
    # measured separation is what `filter_by_distance` is for.
    "FILTER": frozenset({"amount", "field", "location", "object"}),
    "SORT": frozenset({"amount", "field", "object"}),
    "ORDINAL_SELECT": frozenset({"field", "object"}),
    "EXTREME_SELECT": frozenset({"amount", "field", "object"}),
    # Counting places is an aggregation, and a place may be typed LOCATION.
    "AGGREGATE": frozenset({"amount", "field", "location", "object", "proportion"}),
    "MATCH_OPTIONS": frozenset({"amount", "event", "field", "network", "object", "proportion"}),
    "MEASURE": frozenset(
        {"amount", "event", "field", "location", "network", "object", "proportion"}
    ),
}

def accepted_output_types(transformation: str) -> frozenset[str]:
    """The core-concept types a transformation may name as its output.

    The single source of truth for G3's output half. Canonicalization coerces a produced concept
    into this set before validation reads it, so the two stages can never hold different opinions
    about what `ROUTE_MEASURE` produces.
    """

    name = transformation.upper()
    transform = TRANSFORMS.get(name)
    if transform is None:
        return frozenset()
    return _POLYMORPHIC_OUTPUTS.get(name, frozenset({transform.output_type}))


def accepted_input_types(transformation: str) -> frozenset[str] | None:
    """The core-concept types a transformation may consume, or None where anything may be.

    Read rather than duplicated because two stages complete graphs against it: the implicit
    concept completion must not inject a concept this table then refuses, which is exactly what
    `implicit_route` did to `ROUTE_MEASURE` on 36 questions in one run.
    """

    return _TRANSFORM_INPUTS.get(transformation.upper())


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
        (("directions", _through_waypoints), ("travel_time", _duration), ("directions", _distance)),
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
        (("directions", _between_places_only), ("compare_routes", _always)),
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
            ("filter_by_distance", _over_measured_candidates),
            ("within_radius", _within_a_stated_radius),
            ("filter_by_direction", _within_a_stated_sector),
            ("filter_places", _by_a_stated_attribute),
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
            ("count_items", _counts_a_set),
            ("sum_amounts", _sum),
        ),
        "amount",
    ),
    "MATCH_OPTIONS": Transform(
        "MATCH_OPTIONS",
        "Map the computed evidence onto the candidate answer texts.",
        (
            ("match_count_options", _against_a_count),
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


def transform_catalogue(*, include_mcq: bool = True) -> str:
    """The vocabulary as the planner prompt states it -- names and meanings, never operators."""

    return "\n".join(
        f"- {transform.name}: {transform.summary}"
        for transform in TRANSFORMS.values()
        if include_mcq or transform.name != "MATCH_OPTIONS"
    )


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
    endpoint_count: int = 0,
    options_are_counts: bool = False,
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
        endpoint_count=endpoint_count,
        options_are_counts=options_are_counts,
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
    #: Input nodes that `via` claimed, for the graphs that resolve each place separately.
    via_inputs: frozenset[str] = frozenset()
    #: The two places the graph named as the ends of this measure, in that order.
    endpoints: tuple[str, ...] = ()
    #: How many places the single record-list input resolved, so the far end can be found.
    resolved_count: int = 0
    #: How many places each input resolved, so a stop list gathers all of them.
    resolved_sizes: tuple[int, ...] = ()
    #: The stop references every matrix node was built over, keyed by node id. A tour indexes
    #: into its cost matrix, so its node list has to be that matrix's own stops in that order.
    matrix_stops: dict[str, list[str]] = field(default_factory=dict)
    #: The arguments each input node was wired with. A retrieval records the place it searched
    #: around, so a filter over that retrieval can measure from the same centre instead of being
    #: handed the candidate list as one.
    producer_arguments: tuple[dict[str, Any], ...] = ()

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

        if len(self.endpoints) == 2:
            return self.endpoints[0], self.endpoints[1]
        if not self.inputs:
            return "", ""
        if len(self.inputs) >= 2:
            ends = [
                position for position, node in enumerate(self.inputs) if node not in self.via_inputs
            ]
            if len(ends) >= 2:
                return self.place(ends[0]), self.place(ends[-1])
            return self.place(0), self.place(1)
        if self.via_positions:
            free = [
                index for index in range(self.resolved_count) if index not in self.via_positions
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

    def gathered(self, start: int = 0) -> Any:
        """Every input from this position on, as one slot's worth of evidence.

        A slot that takes a collection takes *all* of it. A planner that measured three
        distances in three nodes and asked for the smallest, or retrieved in two branches and
        filtered the union, wrote several inputs into one slot; reading the first of them
        returned a minimum over one candidate and a filter over half the evidence.
        """

        positions = range(start, len(self.inputs))
        references = [self.items(position) for position in positions]
        if not references:
            return None
        return references[0] if len(references) == 1 else references

    def anchor_place(self) -> tuple[str | None, int]:
        """The single place to measure *from*, and which input position it came from.

        Position 0 is where it sits in a graph that resolves the anchor first, and reading it
        positionally is right for those -- so that case is answered first and unchanged. It is
        wrong for the graph that ranks a retrieval and then ranks the ranking: both of that
        node's inputs are collections, `anchor` was bound to one of them, and the operator
        refused with `anchor has no resolved coordinates`, taking every step that referenced it
        down as well. What that graph measures from is the place its retrieval searched around,
        which the retrieval recorded, so ask it rather than the input list.

        Returns `-1` for the position when the anchor came from outside the inputs, so no input
        is dropped from the candidates.
        """

        for position, producer in enumerate(self.producers):
            if producer not in _COLLECTION_OPERATORS:
                return self.place(position), position
        return self.upstream_center(), -1

    def gathered_except(self, skip: int) -> Any:
        """Every input but the one already spoken for, as one slot's worth of evidence."""

        references = [
            self.items(position) for position in range(len(self.inputs)) if position != skip
        ]
        if not references:
            return None
        return references[0] if len(references) == 1 else references

    def upstream_center(self) -> str | None:
        """The place an input node searched around, as that node recorded it."""

        for arguments in self.producer_arguments:
            center = arguments.get("center") if isinstance(arguments, dict) else None
            if isinstance(center, str) and center.startswith("$"):
                return center
        return None

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
        return {}  # names are bound from the concept graph
    if operator == "nearby_places":
        return {"center": wiring.place(0)} if arity else {}
    if operator == "batch_place_details":
        return {"place_ids": wiring.whole(0)} if arity else {}
    if operator == "place_details":
        # The id lives on the located place, so the reference is that place's `place_id` field.
        # A record list ends in `.place` and the id is the sibling field `.place_id`; anything
        # else is the place itself and the id is a field of it. Appending `_id` to a bare node
        # reference produced `$place_id`, which names a node called `place_id` and not a field of
        # anything -- the multi-input check found it, because the input then went unread.
        if not arity:
            return {}
        reference = wiring.place(0)
        suffix = "_id" if reference.endswith(".place") else ".place_id"
        return {"place_id": f"{reference}{suffix}"}
    if operator == "place_search":
        return {}
    if operator == "haversine_distance":
        first, second = wiring.two_places()
        return {"place_a": first, "place_b": second} if first else {}
    if operator == "pairwise_distances":
        return {"pairs": wiring.gathered()} if arity else {}
    if operator == "pairwise_extremes":
        return {"locations": wiring.gathered()} if arity else {}
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
        # The legs come out of the matrix, whichever input that is; the stop nodes beside it are
        # what the matrix was built over and are already inside it.
        source = wiring.by_type("field", output_types) or (wiring.whole(0) if arity else None)
        return {"routes": source} if source else {}
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
        return {"locations": wiring.gathered()} if arity else {}
    if operator == "calculate_start_time":
        return {"duration_s": wiring.whole(0)} if arity else {}
    if operator == "within_radius":
        # The radius is a stated fact and grounding binds it; the centre is whichever input is a
        # single place. Given only the candidate set, the centre is the one the retrieval that
        # produced it already searched around -- read off that node's own arguments, not guessed.
        # Wiring the candidate list itself as `center` is what "center has no resolved
        # coordinates" was, on seven rows of one pass.
        if arity >= 2:
            return {"center": wiring.place(0), "candidates": wiring.whole(1)}
        if arity == 1:
            center = wiring.upstream_center()
            if center is None:
                raise ValueError(
                    "FILTER by radius has no place to measure from: give it the anchor as its "
                    "first input, or measure the candidates before filtering them"
                )
            return {"center": center, "candidates": wiring.items(0)}
        return {}
    if operator == "filter_by_direction":
        return (
            {"center": wiring.place(0), "places": wiring.whole(1)}
            if arity >= 2
            else {"places": wiring.whole(0)}
        )
    if operator == "filter_places":
        return {"places": wiring.gathered()} if arity else {}
    if operator == "filter_by_distance":
        # The radius is a stated fact and grounding binds it, exactly as it does for
        # `within_radius`; the measured set is whatever the measure steps produced.
        return {"items": wiring.gathered()} if arity else {}
    if operator == "count_items":
        return {"items": wiring.gathered()} if arity else {}
    if operator == "nearest":
        # The comment that used to sit here said this binding was safe because the shape guard
        # only reaches `nearest` when the first input is already a single place. It does not:
        # `_one_place_against_many` asks whether the measure fans out, never what input 0 holds,
        # and the same comment admits `SET_MEASURE` removed the precondition. So the anchor is
        # found rather than assumed. A graph whose input 0 *is* the place binds exactly as
        # before; only the graphs that were failing take the new branch.
        if arity < 2:
            return {"candidates": wiring.gathered()} if arity else {}
        anchor, position = wiring.anchor_place()
        if anchor is None:
            return {"candidates": wiring.gathered()}
        if position == 0:
            return {"anchor": anchor, "candidates": wiring.gathered(1)}
        candidates = wiring.gathered_except(position)
        return (
            {"anchor": anchor, "candidates": candidates}
            if candidates is not None
            else {"candidates": wiring.gathered()}
        )
    if operator == "sort_by":
        return {"items": wiring.gathered(), "key": _rank_key(factors)} if arity else {}
    if operator == "select_by_index":
        return {"items": wiring.gathered(), "index": _ordinal_index(factors)} if arity else {}
    if operator in {"select_min", "select_max"}:
        return {"items": wiring.gathered(), "key": _rank_key(factors)} if arity else {}
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
    if operator == "match_count_options":
        return {"count": wiring.whole(0)} if arity else {}
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


def consumed_inputs(arguments: Any) -> set[str]:
    """Every node id a wired argument set actually reads, however deeply it is nested."""

    found: set[str] = set()
    stack: list[Any] = [arguments]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, list | tuple):
            stack.extend(value)
        elif isinstance(value, str) and value.startswith("$"):
            found.add(value[1:].split(".", 1)[0])
    return found


def unconsumed_inputs(inputs: Sequence[str], arguments: Any) -> list[str]:
    """Inputs the graph declared that the wiring never reads.

    Every defect this module has shipped had this shape. `distance_matrix` over four resolved
    stops read `inputs[0]` and built a 1x1 grid; `tsp_tw` took its node list from whichever input
    was object-typed and got one place beside a six-place matrix; a via-route read the first two
    inputs and drove to the waypoint; two `DISTANCE_MEASURE` nodes over one list measured the same
    thing twice. None of them failed loudly -- each returned a confident number computed over less
    evidence than the graph had gathered.

    So it is checked rather than remembered: a node that declares an input and does not read it is
    a wiring bug in this file, and the check names it at the node where it happens.
    """

    read = consumed_inputs(arguments)
    return [name for name in inputs if name not in read]


#: Operators that deliberately read fewer nodes than the graph makes them depend on. Each is a
#: real relation, not an oversight, and each is listed with why.
_PARTIAL_CONSUMERS: dict[str, str] = {
    # A tour reads its stops through the matrix they were priced in, so the stop nodes reach it
    # only as the matrix's own list.
    "tsp_tw": "reads its stops through the cost matrix",
    # The legs are in the matrix; the stop nodes beside it are what the matrix was built over.
    "select_legs": "reads its stops through the cost matrix",
    # The names come from the concept graph, never from an upstream node: a planner that makes
    # one geocode depend on another means "after that one", and the dependency is real even
    # though nothing of its output is read.
    "batch_geocode": "resolves concepts, not upstream output",
}
# `within_radius` and `place_search` were listed here too and neither was reachable: a radius
# filter reads its candidate input and takes the centre from the retrieval's own arguments, and
# a `place_search` given any input resolves to `nearby_places`, which reads it as the centre.
# Both were removed rather than kept as insurance -- an exemption that never fires is a claim
# nothing tests. Verified by replaying all 283 recorded graphs with each one removed.


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
    #: Departures worth counting that are not worth refusing over. See `AgentResult`.
    diagnostics: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "graph": self.graph,
            "decisions": [dict(row) for row in self.decisions],
            "concrete_nodes": list(self.concrete_nodes),
            "diagnostics": [dict(row) for row in self.diagnostics],
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

    found = _place_references(declared, inputs, follow, produced_by, resolved_concepts)

    def shares_its_node(node: str) -> bool:
        """Whether the waypoint sits in a batch beside other places, or has the node to itself.

        A position only tells the ends apart from the waypoint when they were resolved together.
        When the graph geocoded the waypoint in its own node, that node *is* the waypoint, and
        reporting a position instead left it in the running for the far end -- five of eight
        detour graphs routed to the stop and passed through the destination.
        """

        return len(resolved_concepts.get(node, ())) > 1

    return (
        tuple(reference for reference, _, _ in found),
        frozenset(
            position
            for _, node, position in found
            if position is not None and shares_its_node(node)
        ),
        tuple(
            dict.fromkeys(
                node
                for _, node, position in found
                if position is None or not shares_its_node(node)
            )
        ),
    )


def _place_references(
    names: Sequence[Any],
    inputs: Sequence[str],
    follow: Callable[[str], str],
    produced_by: dict[str, str],
    resolved_concepts: dict[str, list[str]],
) -> list[tuple[str, str, int | None]]:
    """Resolve ids the graph wrote into references to located places, in the order written.

    An id names either an upstream node that resolved a place, or a concept an upstream
    `RESOLVE_PLACES` bound. Both are things the *graph* said; neither is read out of the question
    and neither is inferred from which argument slot an input happened to fall into.

    Each result carries the node it resolves through and, for a concept, its position inside that
    node's resolved list -- which is what lets the ends of a route be the positions a waypoint did
    not claim.
    """

    found: list[tuple[str, str, int | None]] = []
    for raw in names:
        name = str(raw).lstrip("$").strip()
        if not name:
            continue
        node = follow(name.split(".", 1)[0])
        if node in produced_by:
            found.append((_place_ref(node, produced_by[node]), node, None))
            continue
        for source in inputs:
            aligned = resolved_concepts.get(source) or []
            if name in aligned:
                index = aligned.index(name)
                found.append((_place_ref(source, produced_by.get(source), index), source, index))
                break
    return found


#: Transformations that relate exactly two places, and so have a pair to name.
_PAIRWISE_TRANSFORMS = frozenset({"DISTANCE_MEASURE", "ROUTE_MEASURE"})


#: Concept roles that name a *restriction* rather than a thing to measure. A question's radius,
#: compass sector and kind of place arrive as concepts in these roles.
_CONSTRAINT_ROLES = frozenset({"sub_condition", "condition"})


def constraint_concepts(concepts: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The concepts that restrict a measurement rather than supply one of its operands."""

    return {
        str(concept.get("id")): concept
        for concept in concepts
        if isinstance(concept, dict)
        and str(concept.get("role") or "") in _CONSTRAINT_ROLES
        and concept.get("concept_type") not in {"location"}
    }


def _constraint_inputs(
    declared_inputs: Sequence[Any], constraints: dict[str, dict[str, Any]]
) -> list[str]:
    """Constraint concepts this node names as inputs.

    A planner writes `FILTER(inputs=["distance_measure", "constraint"])`, and it is right to:
    the restriction is a thing the question stated and the node is where it applies. Read as a
    data dependency it is a reference to nothing, and the whole node was refused for it. Read as
    what it is, it is the semantic graph saying which node carries the constraint -- which is
    what makes constraint preservation checkable rather than inferred from operator choice.
    """

    return [
        str(value).lstrip("$").strip()
        for value in declared_inputs
        if str(value).lstrip("$").strip() in constraints
    ]


def _dangling_references(
    declared_inputs: Sequence[Any],
    follow: Callable[[str], str],
    produced_by: dict[str, str],
    resolved_concepts: dict[str, list[str]],
    constraints: dict[str, dict[str, Any]],
) -> list[str]:
    """Inputs that name nothing this graph produced, no concept it resolved, no constraint."""

    bound = {name for names in resolved_concepts.values() for name in names}
    missing: list[str] = []
    for value in declared_inputs:
        name = str(value).lstrip("$").strip()
        if not name:
            continue
        node = follow(name.split(".", 1)[0])
        if node in produced_by or name in bound or name in constraints:
            continue
        missing.append(repr(name))
    return list(dict.fromkeys(missing))


def _resolvable_sources(
    inputs: Sequence[str],
    produced_by: dict[str, str],
    resolved_concepts: dict[str, list[str]],
) -> list[str]:
    """The nodes a concept id could be found in: this node's inputs, then every resolved list.

    A node that named its endpoints as concepts sometimes lists no input at all -- the places it
    means were resolved elsewhere in the graph and it simply did not say so. Widening the search
    past its own inputs is what lets that graph resolve; the ids it names are still the graph's
    own words.
    """

    return list(dict.fromkeys([*inputs, *resolved_concepts]))


def _resolve_endpoints(
    step: dict[str, Any],
    inputs: Sequence[str],
    declared_inputs: Sequence[Any],
    follow: Callable[[str], str],
    produced_by: dict[str, str],
    resolved_concepts: dict[str, list[str]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The two places a measure is between, when the graph named them as concepts.

    A pairwise measure needs to say *which* pair. The planner says it two ways and neither was
    read: it writes the concept ids in `inputs` -- "inputs": ["anchor", "target1"] -- where node
    ids belong, or it writes them in `concept_ids` beside a single resolved node. Both were
    dropped, so two `DISTANCE_MEASURE` nodes over the same list factorized identically and their
    difference was zero, or measured nothing at all. That is 13 of 33 `poi_distance_difference`
    rows in one pass, every one of them reported as a clean run.

    Only consulted when the node's own inputs cannot already supply a pair, so a graph that wires
    two nodes keeps wiring two nodes.
    """

    sources = [name for name in inputs if produced_by.get(name)]
    if len(sources) >= 2:
        return (), ()
    unresolved = [
        value
        for value in declared_inputs
        if str(value).lstrip("$").split(".", 1)[0] not in produced_by
    ]
    for candidates in (unresolved, step.get("concept_ids") or []):
        found = _place_references(candidates, inputs, follow, produced_by, resolved_concepts)
        if len(found) == 2:
            return (
                tuple(reference for reference, _, _ in found),
                tuple(dict.fromkeys(node for _, node, _ in found)),
            )
    return (), ()


def factorize_semantic_graph(
    steps: Sequence[Any],
    *,
    concepts: Sequence[dict[str, Any]],
    options: Sequence[str],
    facts: Any,
    available: frozenset[str],
    strict_types: bool = True,
) -> SemanticFactorization:
    """Map a semantic transformation graph onto executable operators.

    The question is not a parameter, and that is the point: choosing the operator for a spatial
    relation is mechanical once the concept types and the transformation are known, so it is done
    here, once, the same way every time -- not asked of a language model per question from a
    prompt that had to list Kakao's category codes to make the answer possible.

    `strict_types=False` is the last attempt before a question is given up on, and it steps aside
    from this port's own heuristics -- upstream has none of them -- while every rule about what
    the graph *is* still refuses. What it relaxes here is the unconsumed-input check: that rule
    predicts one step would read less evidence than the graph handed it, which is worth refusing
    a draft for so the repair round is told, but not worth refusing the question for. It was the
    single largest cause of `graph_validation_failure` in a run at 28 of 100, and the step it
    refuses is one the executor would have run.
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
    counting_question = options_state_counts(options)
    constraint_by_id = constraint_concepts(concepts)
    graph: list[dict[str, Any]] = []
    decisions: list[dict[str, str]] = []
    concrete: list[str] = []
    diagnostics: list[dict[str, Any]] = []
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
    # The arguments each node was wired with, so a filter can read the centre its retrieval used.
    wired_arguments: dict[str, dict[str, Any]] = {}
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
        declared_inputs = list(step.get("inputs") or step.get("depends_on") or [])
        factors = step.get("factors")
        factors = dict(factors) if isinstance(factors, dict) else {}
        transform_name = str(step.get("transform") or "").strip()
        via_refs, via_positions, via_nodes = _resolve_via(
            step, inputs, follow, produced_by, resolved_concepts
        )
        # A waypoint is something the route depends on, so it joins the dependencies even when
        # the node did not also list it as an input.
        inputs = list(dict.fromkeys([*inputs, *via_nodes]))
        # An input that names neither an upstream node nor a concept an upstream node resolved
        # is a reference to nothing. Silently dropping it left a `FILTER` over three named
        # candidates with no inputs at all, so the node was wired empty and the narrowing it
        # carried never applied -- one of the eight graphs that came out with no narrowing in
        # them. A reference that resolves to nothing is a planner fault with a name.
        carried = _constraint_inputs(declared_inputs, constraint_by_id)
        for reference in _dangling_references(
            declared_inputs, follow, produced_by, resolved_concepts, constraint_by_id
        ):
            # Recorded, not refused. As a refusal this cost `nearby_kth_nearest` 27.8 points and
            # took validation below 90%: the graphs it names are malformed, and repair does not
            # fix them, so enforcing it traded a wrong answer for no answer. The reference is
            # still a defect and still worth counting.
            diagnostics.append(
                {
                    "kind": "invalid_semantic_reference",
                    "node": node_id,
                    "transform": transform_name or "(operator)",
                    "reference": reference.strip("'"),
                }
            )
        endpoints: tuple[str, ...] = ()
        if transform_name.upper() in _PAIRWISE_TRANSFORMS:
            endpoints, endpoint_nodes = _resolve_endpoints(
                step,
                _resolvable_sources(inputs, produced_by, resolved_concepts),
                declared_inputs,
                follow,
                produced_by,
                resolved_concepts,
            )
            inputs = list(dict.fromkeys([*inputs, *endpoint_nodes]))

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
            ) and ("directions" in available):
                route_id = f"{node_id}_route"
                route_wiring = _Wiring(
                    inputs=tuple(inputs),
                    producers=tuple(produced_by.get(name) for name in inputs),
                    factors=factors,
                    via=via_refs,
                    via_positions=via_positions,
                    via_inputs=frozenset(via_nodes),
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
            if (
                _totals_a_bare_matrix(transform_name, factors, inputs, produced_by, square_matrices)
                and "select_legs" in available
            ):
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
                endpoint_count=len(endpoints),
                options_are_counts=counting_question,
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
                via_inputs=frozenset(via_nodes),
                endpoints=endpoints,
                resolved_count=resolved_counts.get(inputs[0], 0) if inputs else 0,
                resolved_sizes=tuple(resolved_counts.get(name, 0) for name in inputs),
                matrix_stops=matrix_stops,
                producer_arguments=tuple(wired_arguments.get(name, {}) for name in inputs),
            )
            arguments = wire_arguments(operator, wiring, output_types=output_types)
            arguments, aligned = _bind_named_entities(
                operator,
                arguments,
                step,
                factors,
                concept_text,
                options,
                concepts,
                claimed,
                facts,
            )
            missed = [] if operator in _PARTIAL_CONSUMERS else unconsumed_inputs(inputs, arguments)
            if missed:
                complaint = (
                    f"GeoFlow node {node_id!r} ({transform_name} -> {operator}) declares "
                    f"{len(inputs)} inputs and its wiring reads {len(inputs) - len(missed)}: "
                    f"{', '.join(missed)} would be gathered and never used"
                )
                if strict_types:
                    raise ValueError(complaint)
                diagnostics.append(
                    {"node": node_id, "rule": "unconsumed_inputs", "detail": complaint}
                )
            names = arguments.get("place_names") or []
            for name in names:
                claimed.add(str(name))
            if operator == "batch_geocode":
                resolved_counts[node_id] = len(names)
                if len(aligned) == len(names):
                    resolved_concepts[node_id] = aligned
            # What the node produces is what the operator that performs it produces, whenever
            # that is one of the types the transformation may produce. The transformation's own
            # declared type is a default for the operator that usually performs it, and a
            # polymorphic transformation has more than one: `ROUTE_COMPARE` is typed `object`
            # for `compare_routes`, and resolving it to `directions` -- which is what a
            # comparison handed two places actually needs -- makes the node a route FIELD.
            # Reading the default there refused the plan the executor could have run, which is
            # exactly what `AGENTS.md` says a declared-type table must never do.
            declared = TRANSFORMS[transform_name.upper()].output_type
            produced = OPERATOR_CONTRACTS.get(operator)
            output_type = (
                produced.output_type
                if produced is not None
                and produced.output_type in accepted_output_types(transform_name)
                else declared
            )

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
        if isinstance(arguments, dict):
            wired_arguments[node_id] = arguments
        produced_by[node_id] = operator if isinstance(operator, str) else ""
        # `operator` may not be a string here: a planner that writes
        # `"operator": ["extract_distance", "extract_distance"]` is carried through so the graph
        # validator can name that fault, and a set lookup on a list raises `unhashable type`
        # instead -- a crash in this file, recorded against the agent as if it had reasoned its
        # way there.
        emits_collection[node_id] = isinstance(operator, str) and (
            operator in _COLLECTION_OPERATORS
            or (operator == "batch_geocode" and len((arguments or {}).get("place_names") or []) > 1)
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
                # Which stated restrictions this node was told to apply. Grounding binds their
                # literals, and constraint preservation is checked against them.
                **({"constraint_ids": carried} if carried else {}),
            }
        )
    return SemanticFactorization(graph, tuple(decisions), tuple(concrete), tuple(diagnostics))


def _bind_named_entities(
    operator: str,
    arguments: dict[str, Any],
    step: dict[str, Any],
    factors: dict[str, Any],
    concept_text: dict[str, str],
    options: Sequence[str],
    concepts: Sequence[dict[str, Any]],
    claimed: set[str],
    facts: Any = None,
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
    scope = str(factors.get("scope") or "").lower()

    # 1. The node said it resolves the candidates the *question* listed. Those are a question
    #    literal like the radius, so they come from the facts the analysis extracted and not from
    #    a planner retyping four place names it read once.
    listed = tuple(getattr(facts, "listed_places", ()) or ()) if facts is not None else ()
    if scope in {"listed", "named", "offered"} and listed:
        arguments["place_names"] = list(listed)
        return arguments, []

    # 2. The node said outright that it resolves the candidate texts.
    if scope in {"options", "candidates"}:
        arguments["place_names"] = list(options)
        return arguments, []

    # 3. Concepts the analysis actually has.
    aligned = [
        value for value in requested if value in concept_text and concept_text[value].strip()
    ]
    named = [concept_text[value] for value in aligned]
    if named:
        arguments["place_names"] = named
        return arguments, aligned

    # 4. Every id it named is invented. When they read as the options -- `Option 0`,
    #    `candidate_options`, `option_places` -- that is what it meant.
    if requested and all(
        "option" in value.lower() or "candidate" in value.lower() for value in requested
    ):
        arguments["place_names"] = list(options)
        return arguments, []

    # 5. Otherwise take the located concepts no earlier node has claimed. This is the case where
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

    # 6. Nothing in the concept graph to resolve. The options are the only places left that the
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
    "match_count_options": "MATCH_OPTIONS",
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
