from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

CORE_CONCEPTS = frozenset(
    {"location", "object", "field", "event", "network", "amount", "proportion"}
)
FUNCTIONAL_ROLES = frozenset(
    {"extent", "temporal_extent", "sub_condition", "condition", "support", "measure"}
)
CONTEXTUAL_ROLES = frozenset({"extent", "temporal_extent"})
ROLE_PRIORITY = {
    "sub_condition": 0,
    "condition": 1,
    "support": 2,
    "measure": 3,
}


@dataclass(frozen=True)
class OperatorContract:
    output_type: str
    required_arguments: tuple[str, ...] = ()
    optional_arguments: tuple[str, ...] = ()
    # Arguments whose values may be outputs of earlier graph nodes.  This does not make every
    # bare string a reference: normalization only promotes a value when it exactly names one of
    # the node's declared dependencies.  Keeping the information on the contract prevents a
    # literal such as metric="distance" or mode="driving" from being mistaken for a node id.
    reference_arguments: tuple[str, ...] = ()

    @property
    def allowed_arguments(self) -> frozenset[str]:
        return frozenset((*self.required_arguments, *self.optional_arguments))


@dataclass(frozen=True)
class ConceptNode:
    id: str
    text: str
    concept_type: str
    role: str
    attributes: dict[str, Any]
    derived: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "concept_type": self.concept_type,
            "role": self.role,
            "attributes": self.attributes,
            "derived": self.derived,
        }


@dataclass(frozen=True)
class ConceptGraph:
    nodes: tuple[ConceptNode, ...]
    edges: tuple[tuple[str, str], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.as_dict() for node in self.nodes],
            "edges": [{"source": source, "target": target} for source, target in self.edges],
        }


@dataclass(frozen=True)
class OperatorHyperedge:
    operator_id: str
    input_concepts: tuple[str, ...]
    output_bindings: tuple[dict[str, str], ...]
    parameters: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "input_concepts": list(self.input_concepts),
            "output_bindings": list(self.output_bindings),
            "parameters": self.parameters,
        }


@dataclass(frozen=True)
class FactorizedGeoFlow:
    concept_graph: ConceptGraph
    graph: tuple[dict[str, Any], ...]
    hyperedges: tuple[OperatorHyperedge, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "concept_graph": self.concept_graph.as_dict(),
            "graph": list(self.graph),
            "hyperedges": [edge.as_dict() for edge in self.hyperedges],
        }


OPERATOR_CONTRACTS: dict[str, OperatorContract] = {
    "identity_measure": OperatorContract(
        "object", ("value",), reference_arguments=("value",)
    ),
    "place_search": OperatorContract(
        "object",
        optional_arguments=(
            "query",
            "center",
            "category_code",
            "radius_m",
            "min_rating",
            "open_now",
            "limit",
        ),
        reference_arguments=("center",),
    ),
    "batch_geocode": OperatorContract(
        "object",
        ("place_names",),
        ("anchor", "radius_m", "limit", "strict_names"),
        ("anchor",),
    ),
    "geocode": OperatorContract("location", ("address",), ("limit",)),
    "reverse_geocode": OperatorContract(
        "location", ("latitude", "longitude"), ("limit",), ("latitude", "longitude")
    ),
    "place_details": OperatorContract(
        "object", ("place_id",), reference_arguments=("place_id",)
    ),
    "batch_place_details": OperatorContract(
        "object", ("place_ids",), reference_arguments=("place_ids",)
    ),
    "nearby_places": OperatorContract(
        "object",
        ("center",),
        ("query", "category_code", "radius_m", "limit"),
        ("center",),
    ),
    # `candidates` defaults to an empty list on the tool: recovering options with nothing
    # retrieved yet is a legitimate plan, and requiring it here refused one.
    "recover_option_places": OperatorContract(
        "object",
        ("options", "anchor"),
        ("candidates", "category_code", "radius_m", "direction"),
        ("anchor", "candidates"),
    ),
    "directions": OperatorContract(
        "field",
        ("origin", "destination"),
        ("mode", "priority", "waypoints", "include_steps"),
        ("origin", "destination", "waypoints"),
    ),
    "travel_time": OperatorContract(
        "field",
        ("origin", "destination"),
        ("mode", "priority", "waypoints", "include_steps"),
        ("origin", "destination", "waypoints"),
    ),
    "distance_matrix": OperatorContract(
        "field",
        optional_arguments=("origins", "destinations", "pairs", "mode", "priority"),
        reference_arguments=("origins", "destinations", "pairs"),
    ),
    "haversine_distance": OperatorContract(
        "amount",
        optional_arguments=(
            "place_a",
            "place_b",
            "lat1",
            "lon1",
            "lng1",
            "lat2",
            "lon2",
            "lng2",
            "start_lat",
            "start_lng",
            "start_lon",
            "end_lat",
            "end_lng",
            "end_lon",
        ),
        reference_arguments=("place_a", "place_b"),
    ),
    "pairwise_distances": OperatorContract(
        "field", ("pairs",), reference_arguments=("pairs",)
    ),
    "pairwise_extremes": OperatorContract(
        "amount", ("locations",), reference_arguments=("locations",)
    ),
    "bearing_to_direction": OperatorContract(
        "field",
        optional_arguments=(
            "place_a",
            "place_b",
            "lat1",
            "lon1",
            "lng1",
            "lat2",
            "lon2",
            "lng2",
            "start_lat",
            "start_lng",
            "start_lon",
            "end_lat",
            "end_lng",
            "end_lon",
        ),
        reference_arguments=("place_a", "place_b"),
    ),
    "filter_by_direction": OperatorContract(
        "object",
        ("center", "places", "direction"),
        reference_arguments=("center", "places"),
    ),
    "nearest": OperatorContract(
        "object",
        ("anchor", "candidates"),
        ("metric", "routes", "required_type"),
        ("anchor", "candidates", "routes"),
    ),
    "within_radius": OperatorContract(
        "object",
        ("center", "candidates", "radius_m"),
        reference_arguments=("center", "candidates"),
    ),
    "select_min": OperatorContract(
        "object", ("items",), ("key",), ("items",)
    ),
    "select_max": OperatorContract(
        "object", ("items",), ("key",), ("items",)
    ),
    "sort_by": OperatorContract(
        "object", ("items", "key"), ("descending",), ("items",)
    ),
    "select_by_index": OperatorContract(
        "object", ("items", "index"), ("key", "descending"), ("items", "index")
    ),
    "compare_routes": OperatorContract(
        "object", ("routes",), ("metric",), ("routes",)
    ),
    "filter_routes": OperatorContract(
        "field", ("routes", "keyword"), ("include",), ("routes",)
    ),
    "extract_distance": OperatorContract(
        "amount", ("route",), reference_arguments=("route",)
    ),
    "extract_duration": OperatorContract(
        "amount", ("route",), reference_arguments=("route",)
    ),
    "filter_places": OperatorContract(
        "object",
        ("places",),
        ("min_rating", "price_levels", "required_types", "open_now"),
        ("places",),
    ),
    "steps_analysis": OperatorContract(
        "field", ("route",), ("landmark",), ("route",)
    ),
    "sum_route_metrics": OperatorContract(
        "amount", ("routes",), ("metric",), ("routes",)
    ),
    "sum_amounts": OperatorContract(
        "amount", ("amounts",), ("key",), ("amounts",)
    ),
    "difference": OperatorContract(
        "amount", ("minuend", "subtrahend"), ("key",), ("minuend", "subtrahend")
    ),
    "aggregate_route_groups": OperatorContract(
        "amount", ("routes", "groups"), reference_arguments=("routes", "groups")
    ),
    "merge_places": OperatorContract(
        "object", ("items",), reference_arguments=("items",)
    ),
    "match_options": OperatorContract(
        "object",
        ("options", "places"),
        ("anchor", "mode", "minimum_similarity"),
        ("places", "anchor"),
    ),
    "match_distance_options": OperatorContract(
        "object", ("distance", "options"), reference_arguments=("distance",)
    ),
    "match_type_options": OperatorContract(
        "object", ("place", "options"), reference_arguments=("place",)
    ),
    "events_from_objects": OperatorContract(
        "event", ("objects",), ("event_type", "timestamp_field"), ("objects",)
    ),
    "filter_events": OperatorContract(
        "event",
        ("events", "field", "operator", "value"),
        reference_arguments=("events", "value"),
    ),
    "build_route_network": OperatorContract(
        "network", ("nodes", "edges"), reference_arguments=("nodes", "edges")
    ),
    "calculate_proportion": OperatorContract(
        "proportion",
        ("numerator", "denominator"),
        reference_arguments=("numerator", "denominator"),
    ),
    "open_at_time": OperatorContract(
        "event",
        ("schedule", "local_time", "timezone"),
        reference_arguments=("schedule", "local_time"),
    ),
    "timezone": OperatorContract(
        "event",
        ("latitude", "longitude"),
        ("timestamp",),
        ("latitude", "longitude", "timestamp"),
    ),
    "timezone_convert": OperatorContract(
        "event",
        ("local_time", "from_timezone", "to_timezone"),
        reference_arguments=("local_time",),
    ),
    # Only `locations` is unconditionally required: the itinerary is anchored by *either*
    # start_time or arrival_time, and `CalculateFinishTimeArgs` enforces exactly-one with a
    # message G4 cannot express. Leaving start_time required here refused every plan that used
    # the reverse mode the prompt had just told planners to use.
    "calculate_finish_time": OperatorContract(
        "event",
        ("locations",),
        ("start_time", "arrival_time", "stay_durations_s", "timezone", "mode", "priority"),
        ("locations", "start_time", "arrival_time", "stay_durations_s"),
    ),
    "calculate_start_time": OperatorContract(
        "event",
        ("arrival_time", "duration_s", "timezone"),
        ("stay_durations_s",),
        ("arrival_time", "duration_s", "stay_durations_s"),
    ),
    "tsp_tw": OperatorContract(
        "network",
        ("nodes", "distance_matrix"),
        (
            "time_windows",
            "service_times",
            "start_index",
            "time_budget",
            "end_index",
            "fixed_order",
            "metric",
            "return_to_start",
        ),
        (
            "nodes",
            "distance_matrix",
            "time_windows",
            "service_times",
            "start_index",
            "time_budget",
            "end_index",
        ),
    ),
}

# Names planners wrote for an operation that now exists, mapped onto the operator that does it.
# Only exact synonyms belong here -- same operation, arguments `_normalize_arguments` already
# accepts. `select_second_closest` and its kin are deliberately absent: turning that name into
# `select_by_index(index=1)` means reading an ordinal out of an identifier, and a question
# answered one rung off is indistinguishable from one answered wrongly.
OPERATOR_SYNONYMS: dict[str, str] = {
    "subtraction": "difference",
    "subtract": "difference",
    "calculate_difference": "difference",
    "sum_distances": "sum_amounts",
}


# The declared table must never be stricter than the implementation. `_normalize_arguments` in
# `src.tools.spatial` accepts several spellings for the same slot, but the required-argument check
# only ever looked for the canonical one -- so a plan the executor would have run was refused
# before it ran. `sum_amounts(items=[$leg1, $leg2])` was written exactly that way and lost its
# question to "missing arguments: amounts". These are the same spellings the normalizer accepts.
ARGUMENT_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "select_min": {"items": ("values", "inputs", "list", "candidates")},
    "select_max": {"items": ("values", "inputs", "list", "candidates")},
    "select_by_index": {
        "items": ("list", "candidates", "values", "places", "routes"),
        "index": ("i",),
    },
    "sum_amounts": {
        "amounts": ("items", "values", "inputs", "distances", "legs", "routes", "numbers"),
    },
    # `center` is `nearby_places`'s name for the same point, and a planner that retrieves with one
    # node and ranks with the next writes one vocabulary across both. `_normalize_arguments`
    # accepts these; the required-argument check has to accept the same spellings or it refuses a
    # plan the executor would have run.
    "nearest": {"anchor": ("center", "origin", "from_place", "reference")},
    "extract_distance": {"route": ("routes", "legs", "route_list")},
    "extract_duration": {"route": ("routes", "legs", "route_list")},
    "difference": {
        "minuend": (
            "a",
            "first",
            "left",
            "value_a",
            "amount1",
            "x",
            "values",
            "amounts",
            "items",
            "inputs",
        ),
        "subtrahend": (
            "b",
            "second",
            "right",
            "value_b",
            "amount2",
            "y",
            "values",
            "amounts",
            "items",
            "inputs",
        ),
    },
    "sum_route_metrics": {"routes": ("inputs", "legs")},
    # These names state the same boolean.  The executor already canonicalizes them, so the graph
    # contract must not reject a plan that the implementation accepts.
    "tsp_tw": {
        "fixed_order": (
            "preserve_order",
            "keep_order",
            "in_order",
            "sequential",
            "ordered",
        )
    },
}


OPERATOR_INPUT_TYPES: dict[str, dict[str, frozenset[str]]] = {
    "identity_measure": {
        "value": frozenset(CORE_CONCEPTS),
    },
    "batch_geocode": {"anchor": frozenset({"location", "object"})},
    "batch_place_details": {"place_ids": frozenset({"object"})},
    "place_details": {"place_id": frozenset({"object"})},
    "nearby_places": {"center": frozenset({"location", "object"})},
    "recover_option_places": {
        "candidates": frozenset({"object"}),
        "anchor": frozenset({"location", "object"}),
    },
    "directions": {
        "origin": frozenset({"location", "object"}),
        "destination": frozenset({"location", "object"}),
        "waypoints": frozenset({"location", "object"}),
    },
    "travel_time": {
        "origin": frozenset({"location", "object"}),
        "destination": frozenset({"location", "object"}),
    },
    "distance_matrix": {
        "origins": frozenset({"location", "object"}),
        "destinations": frozenset({"location", "object"}),
        "pairs": frozenset({"location", "object"}),
    },
    "haversine_distance": {
        "place_a": frozenset({"location", "object"}),
        "place_b": frozenset({"location", "object"}),
        "lat1": frozenset({"location", "object"}),
        "lon1": frozenset({"location", "object"}),
        "lng1": frozenset({"location", "object"}),
        "lat2": frozenset({"location", "object"}),
        "lon2": frozenset({"location", "object"}),
        "lng2": frozenset({"location", "object"}),
    },
    "pairwise_distances": {"pairs": frozenset({"location", "object"})},
    "pairwise_extremes": {"locations": frozenset({"location", "object"})},
    "bearing_to_direction": {
        "place_a": frozenset({"location", "object"}),
        "place_b": frozenset({"location", "object"}),
    },
    "filter_by_direction": {
        "center": frozenset({"location", "object"}),
        "places": frozenset({"object"}),
    },
    "nearest": {
        "anchor": frozenset({"location", "object"}),
        "candidates": frozenset({"object"}),
        "routes": frozenset({"field"}),
    },
    "within_radius": {
        "center": frozenset({"location", "object"}),
        "candidates": frozenset({"object"}),
    },
    # Picking the smallest, the largest or a sort order is a property of the collection, not of
    # what the collection holds: these read a named key off whatever they are given. Restricting
    # them refused a plan that ranked itineraries, which are events.
    "select_min": {"items": frozenset(CORE_CONCEPTS)},
    "select_max": {"items": frozenset(CORE_CONCEPTS)},
    "sort_by": {"items": frozenset(CORE_CONCEPTS)},
    "select_by_index": {"items": frozenset(CORE_CONCEPTS)},
    # Arithmetic reads a number off whatever carries one -- a route field, a measured
    # amount, a place record with a distance on it -- so nothing is refused here that the
    # implementation would have run.
    "sum_amounts": {"amounts": frozenset(CORE_CONCEPTS)},
    "difference": {
        "minuend": frozenset(CORE_CONCEPTS),
        "subtrahend": frozenset(CORE_CONCEPTS),
    },
    "compare_routes": {"routes": frozenset({"field"})},
    "filter_routes": {"routes": frozenset({"field"})},
    "extract_distance": {"route": frozenset({"field"})},
    "extract_duration": {"route": frozenset({"field"})},
    "filter_places": {"places": frozenset({"object"})},
    "steps_analysis": {"route": frozenset({"field"})},
    "sum_route_metrics": {"routes": frozenset({"field"})},
    "aggregate_route_groups": {"routes": frozenset({"field"})},
    "merge_places": {"items": frozenset({"object"})},
    "match_options": {
        "places": frozenset({"object"}),
        "anchor": frozenset({"object", "location"}),
    },
    # Anything that can carry a measurement, because that is what the operator reads:
    # `_distance_value` takes a number or pulls `distance_m`/`distance`/`value`/`amount`/`meters`/
    # `distance_km` off a record, and says so plainly when the record holds none. A share is a
    # number like a distance is, and `select_max` over three haversine results returns the winning
    # *record*, not its metres -- a correct plan that this line refused on four of v6's rows
    # before it was widened, and a brand-share plan before that.
    "match_distance_options": {
        "distance": frozenset({"amount", "proportion", "object", "field"})
    },
    "match_type_options": {"place": frozenset({"object", "location"})},
    "events_from_objects": {"objects": frozenset({"object"})},
    "filter_events": {"events": frozenset({"event"})},
    "build_route_network": {
        "nodes": frozenset({"object", "location"}),
        "edges": frozenset({"field", "amount"}),
    },
    "calculate_proportion": {
        "numerator": frozenset({"amount", "event", "object"}),
        "denominator": frozenset({"amount", "event", "object"}),
    },
    "open_at_time": {"schedule": frozenset({"object", "field", "event"})},
    "timezone_convert": {"local_time": frozenset({"event", "field"})},
    "calculate_finish_time": {
        "locations": frozenset({"object", "location"}),
    },
    "calculate_start_time": {
        "arrival_time": frozenset({"event", "field"}),
        "duration_s": frozenset({"amount", "field"}),
    },
    "tsp_tw": {
        "nodes": frozenset({"object", "location"}),
        "distance_matrix": frozenset({"field", "network"}),
    },
}


TEMPLATES = {
    "object_field_measure": {
        "name": "Object-Field-Measure",
        # A share or a count over a neighbourhood is this family too, and a neighbourhood question
        # routes on its radius. Without these the pattern was unreachable for the very questions
        # that need it: "반경 600m 안의 편의점 중 이 브랜드가 몇 퍼센트인가".
        "intents": {"poi", "type", "radius", "nearby"},
        "keywords": ("속성", "분포", "비율", "field", "proportion"),
        "pattern": "objects -> event/field restriction -> amount or proportion Measure",
        "example": {
            "graph": [
                {
                    "id": "objects",
                    "operator": "place_search",
                    "arguments": {"query": "대상 장소", "limit": 15},
                    "role": "extent",
                },
                {
                    "id": "events",
                    "operator": "events_from_objects",
                    "arguments": {"objects": "$objects", "event_type": "observation"},
                    "depends_on": ["objects"],
                    "role": "support",
                },
                {
                    "id": "proportion",
                    "operator": "calculate_proportion",
                    "arguments": {"numerator": "$events", "denominator": "$objects"},
                    "depends_on": ["objects", "events"],
                    "role": "measure",
                },
            ]
        },
    },
    "place_attribute": {
        "name": "Place-Attribute-Query",
        "intents": {"poi", "type"},
        "keywords": ("유형", "종류", "카테고리", "주소", "정보"),
        "pattern": "place_search(name) -> inspect normalized place attribute -> Measure",
        "example": {
            "graph": [
                {
                    "id": "place",
                    "operator": "place_search",
                    "arguments": {"query": "농협은행 불암지점", "limit": 1},
                    "depends_on": [],
                    "output_type": "object",
                    "role": "extent",
                },
                {
                    "id": "details",
                    "operator": "place_details",
                    "arguments": {"place_id": "$place.0.place_id"},
                    "depends_on": ["place"],
                    "role": "support",
                },
                {
                    "id": "attribute",
                    "operator": "identity_measure",
                    "arguments": {"value": "$details"},
                    "depends_on": ["details"],
                    "role": "measure",
                },
            ]
        },
    },
    "bearing": {
        "name": "Location-Bearing-Classify",
        "intents": {"poi", "direction"},
        "keywords": ("방향", "동쪽", "서쪽", "남쪽", "북쪽"),
        "pattern": "batch_geocode(locations) -> bearing/filter direction -> Measure",
        "example": {
            "graph": [
                {
                    "id": "locations",
                    "operator": "batch_geocode",
                    "arguments": {
                        "place_names": ["기준 장소", "대상 장소"],
                        "anchor": "기준 장소",
                        "limit": 1,
                    },
                    "depends_on": [],
                    "output_type": "object",
                    "role": "extent",
                },
                {
                    "id": "direction",
                    "operator": "bearing_to_direction",
                    "arguments": {
                        "place_a": "$locations.0.place",
                        "place_b": "$locations.1.place",
                    },
                    "depends_on": ["locations"],
                    "output_type": "field",
                    "role": "measure",
                },
            ]
        },
    },
    "geocode_compare": {
        "name": "Geocode-Batch-Compare",
        "intents": {"nearby", "poi", "distance", "direction"},
        "keywords": ("가까운", "거리", "짧은", "nearest", "distance"),
        "pattern": "batch_geocode(anchor and candidates) -> deterministic distance comparison",
        "example": {
            "graph": [
                {
                    "id": "places",
                    "operator": "batch_geocode",
                    "arguments": {
                        "place_names": ["기준", "선택지 0", "선택지 1"],
                        "anchor": "기준",
                    },
                    "depends_on": [],
                    "output_type": "object",
                    "role": "extent",
                },
                {
                    "id": "nearest",
                    "operator": "nearest",
                    "arguments": {
                        "anchor": "$places.0.place",
                        "candidates": ["$places.1.place", "$places.2.place"],
                    },
                    "depends_on": ["places"],
                    "output_type": "object",
                    "role": "measure",
                },
            ]
        },
    },
    # "두 번째로 가까운 편의점" is not "which of these is nearest". The k-th nearest place is the
    # k-th of the *neighbourhood*, and the options are four names drawn from the top six of it --
    # so ranking the options against each other answers a different question and lands on the gold
    # only by luck. Retrieval was the missing step, not the ordinal: across the v7 runs 130 plans
    # for these two families composed a graph and 4 of them retrieved anything, because
    # `Geocode-Batch-Compare` outranked everything else on "가까운" and its example does exactly
    # what the planner then did.
    "ordinal_nearby": {
        "name": "Retrieve-Rank-Ordinal",
        # The Analysis stage labels these questions `nearby` about three times in four and `poi`
        # the rest of the time, and the fourteen it called `poi` were offered
        # `Geocode-Batch-Compare` instead -- which spans four intents and shows the option-ranking
        # shape. A template gated on one intent label is gated on that stage guessing right.
        "intents": {"nearby", "poi"},
        "keywords": ("번째", "가까운", "nearest", "second", "third"),
        # An example that answers a different question is worse than no second example: the
        # planner copies whichever shape it recognises, and 27 of the 40 plans that were offered
        # this template still built `Geocode-Batch-Compare`'s. Where one pattern supersedes
        # another for a question shape, the loser is dropped rather than offered beside it.
        "supersedes": ("geocode_compare",),
        "pattern": (
            "nearby_places(center, category/keyword, limit≈15) -> nearest -> select_by_index(k-1) "
            "-> match_options"
        ),
        "example": {
            "graph": [
                {
                    "id": "anchor",
                    "operator": "batch_geocode",
                    "arguments": {"place_names": ["기준 장소"], "limit": 1},
                    "depends_on": [],
                    "output_type": "object",
                    "role": "extent",
                },
                {
                    "id": "neighbourhood",
                    "operator": "nearby_places",
                    "arguments": {
                        "center": "$anchor.0.place",
                        "query": "편의점",
                        # Retrieve as deep as the ordinal needs, not as deep as the tool allows.
                        # The example asked for 45 and the planner then wrote 100 (clamped to 45)
                        # in fourteen plans: 45 place records travel through `nearest` into every
                        # later prompt, which pushed the median question to 27k tokens and the
                        # worst to 68k against a 65,536 window -- five of fifty-four questions
                        # died of `llm_context_overflow` rather than of anything they got wrong.
                        # A k-th nearest question needs k plus margin.
                        "limit": 15,
                    },
                    "depends_on": ["anchor"],
                    "output_type": "object",
                    "role": "support",
                },
                {
                    "id": "ranking",
                    "operator": "nearest",
                    "arguments": {
                        "anchor": "$anchor.0.place",
                        "candidates": "$neighbourhood",
                    },
                    "depends_on": ["neighbourhood"],
                    "output_type": "object",
                    "role": "support",
                },
                {
                    "id": "kth",
                    "operator": "select_by_index",
                    "arguments": {"items": "$ranking.ranked", "index": 1},
                    "depends_on": ["ranking"],
                    "output_type": "object",
                    "role": "support",
                },
                {
                    "id": "answer",
                    "operator": "match_options",
                    "arguments": {"options": ["선택지 0", "선택지 1"], "places": ["$kth"]},
                    "depends_on": ["kth"],
                    "output_type": "object",
                    "role": "measure",
                },
            ]
        },
    },
    "radius": {
        "name": "Filter-Aggregate-Measure",
        "intents": {"nearby", "radius"},
        "keywords": ("반경", "이내", "within", "radius"),
        "pattern": "nearby_places(center, exact radius and category/keyword) -> Measure",
        "example": {
            "graph": [
                {
                    "id": "center",
                    "operator": "batch_geocode",
                    "arguments": {"place_names": ["서울역"], "limit": 1},
                    "role": "extent",
                },
                {
                    "id": "nearby",
                    "operator": "nearby_places",
                    "arguments": {
                        "center": "$center.0.place",
                        "query": "편의점",
                        "radius_m": 500,
                        "limit": 45,
                    },
                    "depends_on": ["center"],
                    "output_type": "object",
                    "role": "measure",
                }
            ]
        },
    },
    "routes": {
        "name": "Multi-Route-Compare",
        "intents": {"routing"},
        "keywords": ("경로", "자동차", "주행", "route", "driving"),
        "pattern": "distance_matrix(one origin, all option destinations) -> Measure",
        "example": {
            "graph": [
                {
                    "id": "routes",
                    "operator": "distance_matrix",
                    "arguments": {
                        "origins": ["기준 장소"],
                        "destinations": ["선택지 0", "선택지 1"],
                        "priority": "DISTANCE",
                    },
                    "depends_on": [],
                    "output_type": "field",
                    "role": "extent",
                },
                {
                    "id": "best_route",
                    "operator": "compare_routes",
                    "arguments": {"routes": "$routes.routes", "metric": "distance_m"},
                    "depends_on": ["routes"],
                    "role": "measure",
                },
            ]
        },
    },
    "trip": {
        "name": "Multi-Segment-Aggregate",
        "intents": {"trip"},
        "keywords": ("일정", "차례", "경유", "여행", "trip", "itinerary"),
        "pattern": (
            "distance_matrix(explicit ordered segment pairs) -> aggregate_route_groups -> Measure"
        ),
        "example": {
            "graph": [
                {
                    "id": "segments",
                    "operator": "distance_matrix",
                    "arguments": {
                        "pairs": [
                            {"origin": "출발지", "destination": "선택지 0 첫 장소"},
                            {"origin": "선택지 0 첫 장소", "destination": "선택지 0 둘째 장소"},
                            {"origin": "출발지", "destination": "선택지 1 첫 장소"},
                            {"origin": "선택지 1 첫 장소", "destination": "선택지 1 둘째 장소"},
                        ],
                        "priority": "DISTANCE",
                    },
                    "depends_on": [],
                    "output_type": "field",
                    "role": "extent",
                },
                {
                    "id": "totals",
                    "operator": "aggregate_route_groups",
                    "arguments": {"routes": "$segments.routes", "groups": [[0, 1], [2, 3]]},
                    "depends_on": ["segments"],
                    "output_type": "amount",
                    "role": "measure",
                },
            ]
        },
    },
    "route_optimize": {
        "name": "Route-Optimize",
        "intents": {"trip"},
        "keywords": ("최적 순서", "시간창", "방문 순서", "tsp"),
        "pattern": (
            "locations -> distance_matrix(all ordered pairs) -> tsp_tw(service times, budget)"
            " -> Measure"
        ),
        "example": {
            "graph": [
                {
                    "id": "locations",
                    "operator": "batch_geocode",
                    "arguments": {"place_names": ["출발지", "방문지 1", "방문지 2"]},
                    "role": "extent",
                },
                {
                    "id": "legs",
                    "operator": "distance_matrix",
                    "arguments": {
                        "origins": ["출발지", "방문지 1", "방문지 2"],
                        "destinations": ["출발지", "방문지 1", "방문지 2"],
                    },
                    "depends_on": ["locations"],
                    "role": "support",
                },
                {
                    "id": "optimized",
                    "operator": "tsp_tw",
                    "arguments": {
                        "nodes": "$locations",
                        # The matrix is looked up, never written down: $legs carries the square
                        # duration matrix built from every ordered pair above.
                        "distance_matrix": "$legs",
                        "service_times": [0, 7200, 5400],
                        "time_budget": 28800,
                        "start_index": 0,
                    },
                    "depends_on": ["locations", "legs"],
                    "role": "measure",
                },
            ]
        },
    },
    "route_step_extract": {
        "name": "Route-Step-Extract",
        "intents": {"routing"},
        "keywords": ("경로 단계", "도로", "회전", "step"),
        "pattern": "directions -> route-step field extraction -> Measure",
        "example": {
            "graph": [
                {
                    "id": "endpoints",
                    "operator": "batch_geocode",
                    "arguments": {"place_names": ["출발지", "도착지"]},
                    "role": "extent",
                },
                {
                    "id": "route",
                    "operator": "directions",
                    "arguments": {
                        "origin": "$endpoints.0.place",
                        "destination": "$endpoints.1.place",
                        "include_steps": True,
                    },
                    "depends_on": ["endpoints"],
                    "role": "support",
                },
                {
                    "id": "step_analysis",
                    "operator": "steps_analysis",
                    "arguments": {"route": "$route"},
                    "depends_on": ["route"],
                    "role": "measure",
                },
            ]
        },
    },
    "time_window_reverse": {
        "name": "Time-Window-Reverse",
        "intents": {"trip", "routing"},
        "keywords": ("도착 시간", "출발 시간", "영업시간", "time window"),
        "pattern": "TEXTENT -> route duration -> reverse/finish-time calculation -> Measure",
        "example": {
            "graph": [
                {
                    "id": "time_context",
                    "operator": "timezone_convert",
                    "arguments": {
                        "local_time": "2026-01-01T09:00:00",
                        "from_timezone": "Asia/Seoul",
                        "to_timezone": "Asia/Seoul",
                    },
                    "role": "temporal_extent",
                },
                {
                    "id": "departure",
                    "operator": "calculate_start_time",
                    "arguments": {
                        "arrival_time": "$time_context.converted_time",
                        "duration_s": 3600,
                        "timezone": "Asia/Seoul",
                    },
                    "depends_on": ["time_context"],
                    "role": "measure",
                },
            ]
        },
    },
}


def retrieve_templates(intent: str, question: str, *, limit: int = 2) -> list[dict[str, Any]]:
    """Retrieve pre-validated macro templates with deterministic semantic hints."""

    lowered = question.casefold()
    ranked: list[tuple[int, str, dict[str, Any]]] = []
    for key, template in TEMPLATES.items():
        score = 4 if intent in template["intents"] else 0
        score += sum(1 for keyword in template["keywords"] if keyword.casefold() in lowered)
        if score:
            ranked.append((score, key, template))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    chosen: list[dict[str, Any]] = []
    blocked: set[str] = set()
    for _, key, template in ranked:
        if key in blocked:
            continue
        chosen.append(template)
        # Only a template that already outranked it can supersede one, since `ranked` is sorted.
        blocked |= set(template.get("supersedes", ()))
        if len(chosen) >= limit:
            break
    return [
        {"name": template["name"], "pattern": template["pattern"], "example": template["example"]}
        for template in chosen
    ]


def normalize_analysis(
    payload: dict[str, Any], question: str, fallback_intent: str
) -> dict[str, Any]:
    intent = str(payload.get("intent", "")).strip().lower() or fallback_intent
    raw_concepts = payload.get("concepts") or payload.get("concept_entities") or []
    concepts: list[dict[str, Any]] = []
    if isinstance(raw_concepts, list):
        for index, item in enumerate(raw_concepts):
            if not isinstance(item, dict):
                continue
            concept_type = str(item.get("concept_type") or item.get("type") or "object").lower()
            role = str(item.get("role") or item.get("functional_role") or "support").lower()
            concepts.append(
                {
                    "id": str(item.get("id") or f"c{index + 1}"),
                    "text": str(item.get("text") or item.get("name") or ""),
                    "concept_type": concept_type if concept_type in CORE_CONCEPTS else "object",
                    "role": role if role in FUNCTIONAL_ROLES else "support",
                    "attributes": (
                        item.get("attributes") if isinstance(item.get("attributes"), dict) else {}
                    ),
                    "depends_on": [str(value) for value in (item.get("depends_on") or [])],
                }
            )
    if not concepts:
        concepts = [
            {
                "id": "question_context",
                "text": question,
                "concept_type": "object",
                "role": "extent",
                "attributes": {},
                "depends_on": [],
            },
            {
                "id": "requested_answer",
                "text": str(payload.get("measure") or "answer choice"),
                "concept_type": "amount",
                "role": "measure",
                "attributes": {},
                "depends_on": ["question_context"],
            },
        ]
    concepts = _complete_analysis_roles(concepts, question, str(payload.get("measure") or intent))
    # The kind of place the question is asking for, which the Analysis stage may have had to
    # infer ("우산을 사야 합니다" -> 편의점). Grounding binds it when the question text does not
    # name a type outright; dropping it here is what made a need-shaped question unanswerable.
    target = payload.get("target_type") or payload.get("place_type")
    target_type = str(target).strip() if isinstance(target, str) and target.strip() else None
    return {
        "intent": intent,
        "concepts": concepts,
        "measure": payload.get("measure", intent),
        "target_type": target_type,
    }


def build_concept_graph(analysis: dict[str, Any]) -> ConceptGraph:
    """Build the paper-level G=(V,E,lambda,rho) from normalized analysis concepts."""

    nodes = tuple(
        ConceptNode(
            id=str(concept["id"]),
            text=str(concept.get("text") or ""),
            concept_type=str(concept["concept_type"]),
            role=str(concept["role"]),
            attributes=dict(concept.get("attributes") or {}),
        )
        for concept in analysis.get("concepts", [])
    )
    node_ids = {node.id for node in nodes}
    edges: list[tuple[str, str]] = []
    for concept in analysis.get("concepts", []):
        target = str(concept["id"])
        for source in concept.get("depends_on") or []:
            if str(source) in node_ids and str(source) != target:
                edges.append((str(source), target))
    return ConceptGraph(nodes=nodes, edges=tuple(dict.fromkeys(edges)))


def factorize_geoflow(
    analysis: dict[str, Any], payload: dict[str, Any], *, strict_types: bool = True
) -> FactorizedGeoFlow:
    """Map concept graph G to an executable operator-concept hypergraph G'.

    Each operator hyperedge records input concepts, literal factor parameters, and one or more
    output bindings.  Analysis concepts that supply a radius, direction, category, or other
    literal argument are factor inputs; they must not be fabricated as operator outputs merely
    to satisfy connectivity.

    `strict_types=False` skips the concept-level role-ordering rule, the same way
    `normalize_and_validate_graph` skips the node-level one. They are one rule applied to two
    graphs, and only half of it was skippable when the lenient pass was added -- so three of the
    seven Spatial-Agent failures in the first run on the new model were still this port refusing a
    plan for a reason upstream does not have.
    """

    raw_steps = payload.get("graph") if payload.get("graph") is not None else payload.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("GeoFlow response does not contain a non-empty graph")
    concept_graph = build_concept_graph(analysis)
    concepts = {node.id: node for node in concept_graph.nodes}
    step_ids = [
        str(raw.get("id") or f"s{index + 1}") if isinstance(raw, dict) else f"s{index + 1}"
        for index, raw in enumerate(raw_steps)
    ]
    dependency_map: dict[str, list[str]] = {}
    for index, raw in enumerate(raw_steps):
        if not isinstance(raw, dict):
            continue
        step_id = step_ids[index]
        declared = raw.get("depends_on") or raw.get("before") or []
        dependencies = [_normalize_dependency(value, step_ids) for value in declared]
        dependencies.extend(reference_roots(raw.get("arguments") or raw.get("params") or {}))
        dependency_map[step_id] = list(
            dict.fromkeys(value for value in dependencies if value in step_ids and value != step_id)
        )
    outgoing = {step_id: set() for step_id in step_ids}
    for step_id, dependencies in dependency_map.items():
        for dependency in dependencies:
            outgoing[dependency].add(step_id)

    source_ids = {step_id for step_id in step_ids if not dependency_map.get(step_id)}
    sink_ids = {step_id for step_id in step_ids if not outgoing[step_id]}
    extent_concepts = [
        node.id for node in concept_graph.nodes if node.role in {"extent", "temporal_extent"}
    ]
    measure_concepts = [node.id for node in concept_graph.nodes if node.role == "measure"]
    role_buckets: dict[str, list[str]] = {role: [] for role in FUNCTIONAL_ROLES}
    for node in concept_graph.nodes:
        role_buckets[node.role].append(node.id)

    roles: dict[str, str] = {}
    for index, raw in enumerate(raw_steps):
        if not isinstance(raw, dict):
            continue
        step_id = step_ids[index]
        roles[step_id] = _factorized_role(
            step_id,
            str(raw.get("operator") or ""),
            str(raw.get("role") or ""),
            source_ids,
            sink_ids,
        )
    # LLM-authored labels are advisory. A procedural child cannot run at an earlier role than
    # its parent, so raise the child role to the strongest role required by real dependencies.
    for _ in step_ids:
        changed = False
        for step_id in step_ids:
            if step_id in source_ids or step_id in sink_ids:
                continue
            required = max(
                (
                    ROLE_PRIORITY[roles[parent]]
                    for parent in dependency_map.get(step_id, [])
                    if roles[parent] in ROLE_PRIORITY
                ),
                default=-1,
            )
            current = ROLE_PRIORITY.get(roles[step_id], -1)
            if required > current:
                roles[step_id] = _role_for_priority(required)
                changed = True
        if not changed:
            break

    graph: list[dict[str, Any]] = []
    bound_concepts: set[str] = set()
    step_concepts: dict[str, list[str]] = {}
    for index, raw in enumerate(raw_steps):
        if not isinstance(raw, dict):
            raise ValueError(f"GeoFlow node {index} is not an object")
        step = dict(raw)
        step_id = step_ids[index]
        step["id"] = step_id
        step["depends_on"] = dependency_map.get(step_id, [])
        operator = str(step.get("operator") or "")
        role = roles[step_id]
        step["role"] = role
        explicit_ids = step.get("concept_ids") or []
        concept_ids = [
            str(value)
            for value in explicit_ids
            if str(value) in concepts and str(value) not in bound_concepts
        ]
        if not concept_ids:
            available = [value for value in role_buckets[role] if value not in bound_concepts]
            concept_ids = available
        if source_ids and step_id in source_ids and not concept_ids:
            concept_ids = extent_concepts[:1]
        if sink_ids and step_id in sink_ids and not concept_ids:
            concept_ids = measure_concepts[:1]
        if not concept_ids:
            derived_id = f"derived_{step_id}"
            derived = ConceptNode(
                id=derived_id,
                text=f"{operator} result",
                concept_type=OPERATOR_CONTRACTS.get(
                    operator, OperatorContract("object")
                ).output_type,
                role=role,
                attributes={"operator_id": step_id},
                derived=True,
            )
            concepts[derived_id] = derived
            role_buckets[role].append(derived_id)
            concept_ids = [derived_id]
        step["concept_ids"] = list(dict.fromkeys(concept_ids))
        for concept_id in step["concept_ids"]:
            if concepts[concept_id].role != role:
                concepts[concept_id] = replace(concepts[concept_id], role=role)
        step_concepts[step_id] = step["concept_ids"]
        bound_concepts.update(step["concept_ids"])
        graph.append(step)

    if not any(step["role"] == "measure" for step in graph):
        source_step = graph[-1]
        measure_id = measure_concepts[0] if measure_concepts else "derived_measure"
        if measure_id not in concepts:
            concepts[measure_id] = ConceptNode(
                id=measure_id,
                text="requested answer",
                concept_type="object",
                role="measure",
                attributes={},
                derived=True,
            )
        measure_step_id = _unique_id("measure", {step["id"] for step in graph})
        measure_step = {
            "id": measure_step_id,
            "operator": "identity_measure",
            "arguments": {"value": f"${source_step['id']}"},
            "depends_on": [source_step["id"]],
            "output_type": "object",
            "role": "measure",
            "concept_ids": [measure_id],
        }
        graph.append(measure_step)
        step_concepts[measure_step_id] = [measure_id]
        bound_concepts.add(measure_id)

    # A measure is an operator output, never a supplementary factor. Explicit LLM bindings can
    # occasionally omit it, so bind any remaining measure to the final sink.
    remaining_measures = [
        concept_id
        for concept_id, concept in concepts.items()
        if concept_id not in bound_concepts and concept.role == "measure"
    ]
    if remaining_measures:
        sink = next(step for step in reversed(graph) if step["role"] == "measure")
        sink["concept_ids"].extend(remaining_measures)
        step_concepts[sink["id"]].extend(remaining_measures)
        bound_concepts.update(remaining_measures)

    analysis_dependencies = {
        str(concept.get("id")): [str(value) for value in concept.get("depends_on") or []]
        for concept in analysis.get("concepts", [])
        if isinstance(concept, dict) and concept.get("id")
    }
    factor_inputs: dict[str, list[str]] = {step["id"]: [] for step in graph}
    for concept_id, concept in list(concepts.items()):
        if concept_id in bound_concepts:
            continue
        consumer = _select_factor_step(concept, graph, analysis_dependencies.get(concept_id, []))
        factor_inputs[consumer["id"]].append(concept_id)
        consumer_role = consumer["role"]
        if (
            concept.role in ROLE_PRIORITY
            and consumer_role in ROLE_PRIORITY
            and ROLE_PRIORITY[concept.role] > ROLE_PRIORITY[consumer_role]
        ):
            concepts[concept_id] = replace(concept, role=consumer_role)

    hyperedges: list[OperatorHyperedge] = []
    concept_edges: list[tuple[str, str]] = []
    # Keep only explicit semantic dependencies that agree with normalized functional roles.
    for source, target in concept_graph.edges:
        if source == target:
            continue
        if not _violates_procedural_order(concepts[source].role, concepts[target].role):
            concept_edges.append((source, target))
    for step in graph:
        input_concepts = list(
            dict.fromkeys(
                [
                    concept_id
                    for dependency in step["depends_on"]
                    for concept_id in step_concepts.get(dependency, [])
                ]
                + factor_inputs[step["id"]]
            )
        )
        bindings = step.get("output_bindings")
        valid_bindings = {
            str(binding.get("concept_id")): dict(binding)
            for binding in bindings or []
            if isinstance(binding, dict)
            and str(binding.get("concept_id")) in step["concept_ids"]
        }
        bindings = [
            valid_bindings.get(concept_id, {"concept_id": concept_id, "path": "$"})
            for concept_id in step["concept_ids"]
        ]
        step["input_concepts"] = input_concepts
        step["output_bindings"] = bindings
        step["factor_parameters"] = _literal_arguments(step.get("arguments") or {})
        hyperedges.append(
            OperatorHyperedge(
                operator_id=step["id"],
                input_concepts=tuple(input_concepts),
                output_bindings=tuple(dict(binding) for binding in bindings),
                parameters=step["factor_parameters"],
            )
        )
        for source in input_concepts:
            concept_edges.extend(
                (source, target) for target in step["concept_ids"] if source != target
            )

    # A factor without an explicit analysis dependency is still contextualized by the query's
    # EXTENT. Add a single, local anchoring edge rather than the old role-adjacency complete
    # bipartite fallback.
    contextual = [
        concept_id
        for concept_id, concept in concepts.items()
        if concept.role in CONTEXTUAL_ROLES
    ]
    incoming = {concept_id: set() for concept_id in concepts}
    for source, target in concept_edges:
        incoming[target].add(source)
    if contextual:
        anchor = contextual[0]
        for concept_id, concept in concepts.items():
            if concept.role not in CONTEXTUAL_ROLES and not incoming[concept_id]:
                concept_edges.append((anchor, concept_id))

    complete_graph = ConceptGraph(
        nodes=tuple(concepts.values()),
        edges=tuple(dict.fromkeys(concept_edges)),
    )
    return FactorizedGeoFlow(complete_graph, tuple(graph), tuple(hyperedges))


def _role_for_priority(priority: int) -> str:
    return next(role for role, value in ROLE_PRIORITY.items() if value == priority)


def _select_factor_step(
    concept: ConceptNode,
    graph: list[dict[str, Any]],
    dependencies: list[str],
) -> dict[str, Any]:
    """Select the operator that consumes an unmaterialized concept as a factor."""

    normalized_text = re.sub(r"\s+", "", concept.text).lower()
    direction_words = {"동쪽", "서쪽", "남쪽", "북쪽", "북동", "북서", "남동", "남서"}
    category_codes = {
        "카페": "ce7",
        "편의점": "cs2",
        "음식점": "fd6",
        "병원": "hp8",
        "약국": "pm9",
        "주유소": "ol7",
        "은행": "bk9",
    }

    def score(step: dict[str, Any]) -> tuple[int, int]:
        arguments = step.get("arguments") or step.get("params") or {}
        rendered = re.sub(r"\s+", "", repr(arguments)).lower()
        operator = str(step.get("operator") or "")
        value = 0
        if normalized_text and normalized_text in rendered:
            value += 20
        if concept.concept_type == "amount" and any(
            key in arguments for key in ("radius_m", "distance", "limit", "threshold")
        ):
            value += 10
        if any(word in normalized_text for word in direction_words) and operator in {
            "bearing_to_direction",
            "filter_by_direction",
        }:
            value += 12
        expected_code = next(
            (code for word, code in category_codes.items() if word in normalized_text), None
        )
        if expected_code and str(arguments.get("category_code") or "").lower() == expected_code:
            value += 15
        if concept.concept_type in {"location", "object"} and operator in {
            "place_search",
            "nearby_places",
            "filter_places",
            "batch_geocode",
            "batch_place_details",
        }:
            value += 4
        if dependencies and step.get("depends_on"):
            value += 2
        if step["role"] == "measure":
            value += 1
        return value, -graph.index(step)

    return max(graph, key=score)


def _literal_arguments(value: Any) -> Any:
    """Return supplementary literal parameters, excluding operator-state references."""

    if isinstance(value, dict):
        return {
            key: literal
            for key, item in value.items()
            if (literal := _literal_arguments(item)) is not None
        }
    if isinstance(value, list):
        literals = [literal for item in value if (literal := _literal_arguments(item)) is not None]
        return literals or None
    if isinstance(value, str) and canonical_reference(value).startswith("$"):
        return None
    return value


def _complete_analysis_roles(
    concepts: list[dict[str, Any]], question: str, measure: str
) -> list[dict[str, Any]]:
    completed = [dict(concept) for concept in concepts]
    known_ids = {str(concept["id"]) for concept in completed}
    extents = [
        concept
        for concept in completed
        if concept["role"] in {"extent", "temporal_extent"}
    ]
    if not extents:
        extent_id = _unique_id("question_context", known_ids)
        completed.insert(
            0,
            {
                "id": extent_id,
                "text": question,
                "concept_type": "object",
                "role": "extent",
                "attributes": {"synthetic": True},
                "depends_on": [],
            },
        )
        known_ids.add(extent_id)
        extents = [completed[0]]
    if not any(concept["role"] == "measure" for concept in completed):
        measure_id = _unique_id("requested_answer", known_ids)
        completed.append(
            {
                "id": measure_id,
                "text": measure,
                "concept_type": "amount" if "거리" in question else "object",
                "role": "measure",
                "attributes": {"synthetic": True},
                "depends_on": [str(extents[0]["id"])],
            }
        )
    return completed


def _factorized_role(
    step_id: str,
    operator: str,
    declared_role: str,
    source_ids: set[str],
    sink_ids: set[str],
) -> str:
    if step_id in source_ids:
        if operator in {
            "open_at_time",
            "calculate_finish_time",
            "calculate_start_time",
            "timezone_convert",
            "timezone",
        }:
            return "temporal_extent"
        return "extent"
    if step_id in sink_ids:
        return "measure"
    semantic_roles = {
        "nearby_places": "sub_condition",
        "within_radius": "sub_condition",
        "filter_events": "sub_condition",
        "merge_places": "condition",
        "recover_option_places": "support",
        "filter_by_direction": "support",
        "open_at_time": "condition",
        "build_route_network": "support",
        "distance_matrix": "support",
        "filter_routes": "condition",
        "filter_places": "condition",
        "steps_analysis": "support",
        "tsp_tw": "support",
    }
    if operator in semantic_roles:
        return semantic_roles[operator]
    if declared_role in {"sub_condition", "condition", "support"}:
        return declared_role
    return "support"


def _unique_id(prefix: str, known_ids: set[str]) -> str:
    if prefix not in known_ids:
        return prefix
    index = 2
    while f"{prefix}_{index}" in known_ids:
        index += 1
    return f"{prefix}_{index}"


def _missing_arguments(
    arguments: dict[str, Any],
    required: tuple[str, ...],
    aliases: dict[str, tuple[str, ...]],
) -> list[str]:
    """The required slots the planner filled under no spelling this port accepts."""

    return [
        name
        for name in required
        if name not in arguments and not any(alias in arguments for alias in aliases.get(name, ()))
    ]


def _accepted_argument_names(
    contract: OperatorContract, aliases: dict[str, tuple[str, ...]]
) -> frozenset[str]:
    """Every canonical spelling and exact alias the implementation accepts."""

    return contract.allowed_arguments | frozenset(
        alias for names in aliases.values() for alias in names
    )


def _reference_argument_names(
    contract: OperatorContract, aliases: dict[str, tuple[str, ...]]
) -> frozenset[str]:
    """Canonical and aliased data slots in which a declared dependency may be written bare."""

    names = set(contract.reference_arguments)
    for canonical in contract.reference_arguments:
        names.update(aliases.get(canonical, ()))
    return frozenset(names)


def _validate_statically_known_reference_shapes(
    steps: list[dict[str, Any]], by_id: dict[str, dict[str, Any]]
) -> None:
    """Refuse projections that cannot exist on an output whose shape is already known.

    Most operator output shapes are only known after provider execution, so validation must stay
    lenient about their paths.  A ``batch_geocode`` node is different: it returns one ordered
    record per literal ``place_names`` entry.  Its top-level cardinality and list shape are known
    before execution.  Letting ``$places.3`` through when the node was given three names degraded
    to the whole list in the executor and made the next spatial operator compute over the wrong
    evidence.  Likewise, ``$places.anchor_place`` can never name a field on a list.

    This is a structural data-availability rule, so it applies to the final lenient pass as well
    as strict validation.  Unknown paths below a valid record (for example Google's
    ``$places.0.geometry.location`` spelling) remain deliberately lenient.
    """

    for step in steps:
        for reference in _reference_strings(step["arguments"]):
            parts = [part for part in reference.lstrip("$").split(".") if part]
            if len(parts) < 2:
                continue
            producer = by_id.get(parts[0])
            if not producer or producer["operator"] != "batch_geocode":
                continue
            projection = parts[1]
            if not projection.isdigit():
                raise ValueError(
                    "Data availability violation: "
                    f"{reference} projects field {projection!r} from batch_geocode list "
                    f"{producer['id']!r}; use a numeric record index first"
                )
            names = producer["arguments"].get("place_names")
            if isinstance(names, list) and int(projection) >= len(names):
                raise ValueError(
                    "Data availability violation: "
                    f"{reference} indexes batch_geocode node {producer['id']!r} at "
                    f"{projection}, but it has {len(names)} place_names"
                )


def normalize_and_validate_graph(
    payload: dict[str, Any], *, max_steps: int, strict_types: bool = True
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    """Normalize a planner graph and refuse the ones that cannot run.

    `strict_types=False` keeps every structural rule -- an unknown operator, a dependency that is
    not a node, a cycle, a graph with no Measure -- and skips the two that are this port's own
    invention: declared output-type compatibility and functional-role ordering. Upstream has
    neither (there is no type check anywhere in `spatial-agent`), so a graph they reject is a
    graph upstream would have executed, and refusing it measures our validator rather than the
    architecture. They stay on by default because their message is what the repair round is given
    to work with; the lenient pass is the last thing tried before a question is given up on.
    """

    raw_steps = payload.get("graph")
    if raw_steps is None:
        raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("GeoFlow response does not contain a non-empty graph")
    if len(raw_steps) > max_steps:
        raise ValueError(
            f"GeoFlow graph has {len(raw_steps)} operators, exceeding "
            f"MAX_REASONING_STEPS={max_steps}"
        )

    raw_ids = [
        str(raw.get("id") or f"s{index + 1}") if isinstance(raw, dict) else f"s{index + 1}"
        for index, raw in enumerate(raw_steps)
    ]
    known_ids = set(raw_ids)
    steps: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_steps):
        if not isinstance(raw, dict):
            raise ValueError(f"GeoFlow node {index} is not an object")
        step_id = str(raw.get("id") or f"s{index + 1}")
        if step_id in seen:
            raise ValueError(f"Duplicate GeoFlow node id: {step_id}")
        seen.add(step_id)
        operator = str(raw.get("operator") or "")
        operator = OPERATOR_SYNONYMS.get(operator, operator)
        contract = OPERATOR_CONTRACTS.get(operator)
        if contract is None:
            raise ValueError(f"Unknown GeoFlow operator: {operator}")
        arguments = raw.get("arguments")
        if arguments is None:
            arguments = raw.get("params")
        if not isinstance(arguments, dict):
            raise ValueError(f"GeoFlow node {step_id} arguments must be an object")
        # `select_min(items, index=1)` is an ordinal, not a minimum, and both questions that
        # wrote it meant the second nearest place. The index is explicit, so honouring it is not
        # reading intent out of a name -- and dropping it would answer with the *nearest* place,
        # which is a confident wrong answer where the old contract at least failed and went to
        # the repair round.
        if operator in {"select_min", "select_max"} and "index" in arguments:
            arguments = dict(arguments)
            if operator == "select_max":
                arguments.setdefault("descending", True)
            operator = "select_by_index"
            contract = OPERATOR_CONTRACTS[operator]
        declared = raw.get("depends_on") or raw.get("before") or []
        if not isinstance(declared, list):
            raise ValueError(f"GeoFlow node {step_id} depends_on must be a list")
        declared_dependencies = [_normalize_dependency(value, raw_ids) for value in declared]
        aliases = ARGUMENT_ALIASES.get(operator, {})
        unexpected = sorted(set(arguments) - _accepted_argument_names(contract, aliases))
        if unexpected:
            raise ValueError(
                f"GeoFlow node {step_id} gives {operator} unsupported arguments: "
                f"{', '.join(unexpected)}"
            )
        missing = _missing_arguments(arguments, contract.required_arguments, aliases)
        # A node whose whole input is its one dependency, written as `arguments: {}` with
        # `depends_on: [previous]`. The planner said where the value comes from and the operator
        # has exactly one slot to put it in, so there is one binding consistent with the plan and
        # nothing to guess -- `extract_distance` after a `distance_matrix`, `sum_amounts` after
        # the extraction. Refused, it cost `trip_total_distance` questions on both 300-row draws:
        # seven of the ten `agent_reasoning_failure` rows across them were a missing argument, and
        # the repair round routinely filled one such node and left the next one empty.
        #
        # Deliberately narrow. Two missing arguments, or two dependencies, and which value belongs
        # in which slot is a guess; this fills nothing then and the plan is refused as before.
        if len(missing) == 1 and len(declared_dependencies) == 1:
            arguments = {**arguments, missing[0]: f"${declared_dependencies[0]}"}
            missing = _missing_arguments(arguments, contract.required_arguments, aliases)
        if missing:
            raise ValueError(f"GeoFlow node {step_id} is missing arguments: {', '.join(missing)}")
        if operator == "place_search" and not (
            "query" in arguments
            or (
                "center" in arguments
                and ({"query", "category_code"} & arguments.keys())
            )
        ):
            raise ValueError("place_search requires query, or center with query/category_code")
        if operator == "nearby_places" and not ({"query", "category_code"} & arguments.keys()):
            raise ValueError("nearby_places requires query or category_code")
        if operator == "distance_matrix" and not (
            "pairs" in arguments or {"origins", "destinations"} <= arguments.keys()
        ):
            raise ValueError("distance_matrix requires pairs or origins and destinations")
        if operator in {"haversine_distance", "bearing_to_direction"} and not (
            {"place_a", "place_b"} <= arguments.keys() or _has_coordinate_pairs(arguments)
        ):
            raise ValueError(f"{operator} requires two places or two coordinate pairs")
        arguments = _rewrite_placeholder_references(
            arguments,
            known_ids=known_ids,
            declared_dependencies=declared_dependencies,
            bare_reference_arguments=_reference_argument_names(contract, aliases),
        )
        inferred = reference_roots(arguments)
        dependencies = list(dict.fromkeys([*declared_dependencies, *inferred]))
        # The operator's contract is authoritative about what it returns; the planner's declared
        # output_type is a guess it has no authority over. Failing the whole graph over that
        # disagreement cost a correct plan its answer — `tsp_tw` declared `object` and was thrown
        # away even though every leg had been looked up. Correct it and carry on.
        output_type = contract.output_type
        default_role = "measure" if index == len(raw_steps) - 1 else "support"
        role = str(raw.get("role") or default_role).lower()
        # Demoted below once the edges are known: a Measure is what the answer is read from, so a
        # node another node consumes is not one however the planner labelled it. Left as declared
        # here because the consumers are not known until every step has been read.
        if role not in FUNCTIONAL_ROLES:
            raise ValueError(f"Unknown functional role on {step_id}: {role}")
        steps.append(
            {
                "id": step_id,
                "operator": operator,
                "arguments": arguments,
                "depends_on": dependencies,
                "output_type": output_type,
                "role": role,
                "concept_ids": [str(value) for value in (raw.get("concept_ids") or [])],
                "input_concepts": [str(value) for value in (raw.get("input_concepts") or [])],
                "output_bindings": list(raw.get("output_bindings") or []),
            }
        )

    by_id = {step["id"]: step for step in steps}
    _validate_statically_known_reference_shapes(steps, by_id)
    consumed = {
        dependency
        for step in steps
        for dependency in step["depends_on"]
        if dependency in by_id
    }
    for step in steps:
        if step["role"] == "measure" and step["id"] in consumed:
            # G2 orders sub_condition < condition < support < measure, so a Measure feeding
            # another node breaks the ordering. The plan is fine; the label is not.
            step["role"] = "support"
    if not any(step["role"] == "measure" for step in steps):
        # A Measure is what the answer is read from, which is what nothing consumes. Promoting
        # the terminals completes the demotion above rather than leaving a graph with no Measure.
        for step in steps:
            if step["id"] not in consumed and step["role"] not in CONTEXTUAL_ROLES:
                step["role"] = "measure"
    for step in steps:
        # `depends_on` is the planner's declaration; the `$` references in the arguments are what
        # execution actually follows, and they were merged in already. A declared name that
        # resolves to nothing — a typo, or arithmetic the planner meant to perform later — adds
        # no edge, so drop it and keep the graph. Only a step left with nothing at all is a
        # genuine break.
        resolvable = [name for name in step["depends_on"] if name in by_id]
        if len(resolvable) != len(step["depends_on"]):
            if not resolvable and step["role"] not in CONTEXTUAL_ROLES:
                unknown = [name for name in step["depends_on"] if name not in by_id]
                raise ValueError(
                    f"Unknown dependency {unknown[0]!r} on GeoFlow node {step['id']}"
                )
            step["depends_on"] = resolvable
        for dependency in step["depends_on"]:
            if strict_types and _violates_procedural_order(
                by_id[dependency]["role"], step["role"]
            ):
                raise ValueError(
                    f"Role ordering violation: {dependency} ({by_id[dependency]['role']}) -> "
                    f"{step['id']} ({step['role']})"
                )
        accepted_inputs = OPERATOR_INPUT_TYPES.get(step["operator"], {}) if strict_types else {}
        for argument_name, references in _references_by_argument(step["arguments"]).items():
            accepted = accepted_inputs.get(argument_name)
            if not accepted:
                continue
            paths = {
                reference.lstrip("$").split(".")[0]: reference
                for reference in _reference_strings(step["arguments"].get(argument_name))
            }
            for dependency in references:
                if dependency not in by_id:
                    # An unresolvable reference root is not a node, so it has no declared type to
                    # check. Indexing `by_id` with it raised a `KeyError` from inside validation,
                    # which is outside the per-step isolation and lost the whole question before a
                    # tool ran. Execution reports it as that step's own failure instead.
                    continue
                node_type = by_id[dependency]["output_type"]
                dependency_type = _reference_type(paths.get(dependency, dependency), node_type)
                if dependency_type is not None and dependency_type not in accepted:
                    raise ValueError(
                        f"Type compatibility violation: {dependency} outputs {dependency_type}, "
                        f"but {step['operator']}.{argument_name} accepts {sorted(accepted)}"
                    )

    ordered = _topological_sort(steps)
    extents = {
        step["id"]
        for step in steps
        if step["role"] in {"extent", "temporal_extent"}
    }
    if not extents:
        raise ValueError("GeoFlow graph has no EXTENT or TEXTENT contextual node")
    measures = {step["id"] for step in steps if step["role"] == "measure"}
    if not measures:
        raise ValueError("GeoFlow graph has no Measure node")
    outgoing: dict[str, set[str]] = {step["id"]: set() for step in steps}
    for step in steps:
        for dependency in step["depends_on"]:
            outgoing[dependency].add(step["id"])

    # A node nothing consumes is a planner leftover, not a broken plan: the rest of the graph
    # answers the question perfectly well without it. Dropping it normalizes the draft into one
    # that satisfies G5, where refusing the whole graph cost the question instead.
    while True:
        unused = [
            step["id"]
            for step in steps
            if step["role"] != "measure" and not _reaches_measure(step["id"], outgoing, measures)
        ]
        if not unused:
            break
        dropped = set(unused)
        steps = [step for step in steps if step["id"] not in dropped]
        for step in steps:
            step["depends_on"] = [
                dependency for dependency in step["depends_on"] if dependency not in dropped
            ]
        by_id = {step["id"]: step for step in steps}
        extents = [step["id"] for step in steps if step["role"] in CONTEXTUAL_ROLES]
        outgoing = {step["id"]: set() for step in steps}
        for step in steps:
            for dependency in step["depends_on"]:
                outgoing[dependency].add(step["id"])
    if not extents:
        raise ValueError("GeoFlow graph has no EXTENT or TEXTENT contextual node")
    # The order was computed before pruning, so it still lists nodes that are gone.
    ordered = _topological_sort(steps)

    for step in steps:
        if not _reachable_from_context(step["id"], by_id, extents):
            raise ValueError(
                f"Disconnected GeoFlow node is not reachable from EXTENT/TEXTENT: {step['id']}"
            )
        if not _reaches_measure(step["id"], outgoing, measures):
            raise ValueError(
                f"Disconnected GeoFlow node does not contribute to a Measure: {step['id']}"
            )

    concept_graph_payload = payload.get("concept_graph")
    if isinstance(concept_graph_payload, dict):
        concept_nodes = concept_graph_payload.get("nodes") or []
        concept_ids = {
            str(node.get("id"))
            for node in concept_nodes
            if isinstance(node, dict) and node.get("id")
        }
        bound_ids = {
            concept_id
            for step in steps
            for concept_id in [*step["concept_ids"], *step["input_concepts"]]
        }
        missing_bindings = concept_ids - bound_ids
        if missing_bindings:
            raise ValueError(
                "Concept-to-operator factorization is incomplete: "
                + ", ".join(sorted(missing_bindings))
            )
        for step in steps:
            if not step["concept_ids"]:
                raise ValueError(f"GeoFlow operator has no concept binding: {step['id']}")
            binding_ids = {
                str(binding.get("concept_id"))
                for binding in step["output_bindings"]
                if isinstance(binding, dict)
            }
            if not set(step["concept_ids"]) <= binding_ids:
                raise ValueError(
                    f"GeoFlow operator output bindings are incomplete: {step['id']}"
                )
        concept_by_id = {
            str(node["id"]): node
            for node in concept_nodes
            if isinstance(node, dict) and node.get("id")
        }
        concept_outgoing: dict[str, set[str]] = {
            concept_id: set() for concept_id in concept_by_id
        }
        concept_incoming: dict[str, set[str]] = {
            concept_id: set() for concept_id in concept_by_id
        }
        concept_edges: list[tuple[str, str]] = []
        for edge in concept_graph_payload.get("edges") or []:
            if not isinstance(edge, dict):
                continue
            source, target = str(edge.get("source") or ""), str(edge.get("target") or "")
            if source not in concept_by_id or target not in concept_by_id:
                raise ValueError(
                    f"Concept graph edge references an unknown node: {source}->{target}"
                )
            concept_edges.append((source, target))
        # A concept something else is built from is not the Measure, whatever the Analysis stage
        # labelled it — the same demotion the operator graph already applies, applied to the
        # concepts. A radius is a condition on a search, and one plan calling it a measure was
        # refused outright with its retrieval already specified.
        concept_consumed = {source for source, _ in concept_edges}
        demoted = [
            node
            for concept_id, node in concept_by_id.items()
            if node.get("role") == "measure" and concept_id in concept_consumed
        ]
        for node in demoted:
            node["role"] = "support"
        if not any(node.get("role") == "measure" for node in concept_by_id.values()):
            # A Measure is what the answer is read from, which is what nothing is built from.
            # Promoting the terminals completes the demotion rather than leaving a concept graph
            # with none — exactly as the operator graph does.
            terminals = [
                (concept_id, node)
                for concept_id, node in concept_by_id.items()
                if concept_id not in concept_consumed
            ]
            promoted = [
                node for _, node in terminals if node.get("role") not in CONTEXTUAL_ROLES
            ]
            if not promoted:
                # Every terminal is labelled with a contextual role. The Analysis stage does that
                # for an answer it thinks of as a place — "the nearest bank" is a location.
                promoted = [node for _, node in terminals]
            if not promoted:
                # No terminal at all: an Analysis stage that made its concepts depend on each
                # other in a ring leaves nothing that nothing is built from. Demoting then takes
                # the last Measure away and refuses a plan whose operator graph is correct — a
                # direction question lost its answer that way with the retrieval already
                # specified. The Analysis stage's own labelling stands when there is nothing
                # better to promote.
                promoted = demoted
            for node in promoted:
                node["role"] = "measure"
        for source, target in concept_edges:
            source_role = str(concept_by_id[source].get("role") or "support")
            target_role = str(concept_by_id[target].get("role") or "support")
            if strict_types and _violates_procedural_order(source_role, target_role):
                raise ValueError(
                    f"Concept role ordering violation: {source} ({source_role}) -> "
                    f"{target} ({target_role})"
                )
            concept_outgoing[source].add(target)
            concept_incoming[target].add(source)
        concept_extents = {
            concept_id
            for concept_id, node in concept_by_id.items()
            if node.get("role") in {"extent", "temporal_extent"}
        }
        concept_measures = {
            concept_id
            for concept_id, node in concept_by_id.items()
            if node.get("role") == "measure"
        }
        if not concept_extents or not concept_measures:
            raise ValueError("Concept graph requires contextual and Measure concepts")
        for concept_id in concept_by_id:
            if not _reachable_in_graph(concept_id, concept_incoming, concept_extents):
                raise ValueError(
                    f"Concept is not reachable from EXTENT/TEXTENT: {concept_id}"
                )
            if not _reachable_in_graph(concept_id, concept_outgoing, concept_measures):
                raise ValueError(f"Concept does not contribute to a Measure: {concept_id}")

    constraints = {
        "acyclicity": True,
        "role_ordering": True,
        "type_compatibility": True,
        "data_availability": True,
        "connectivity": True,
        "contextual_connectivity": True,
        "concept_factorization": True,
        "concept_connectivity": True,
    }
    return ordered, constraints


def reference_roots(value: Any) -> list[str]:
    roots: list[str] = []
    if isinstance(value, dict):
        for item in value.values():
            roots.extend(reference_roots(item))
    elif isinstance(value, list):
        for item in value:
            roots.extend(reference_roots(item))
    elif isinstance(value, str):
        canonical = canonical_reference(value)
        parsed = reference_expression(canonical)
        names = parsed[0] if parsed else [canonical]
        for name in names:
            if name.startswith("$"):
                roots.append(name[1:].split(".", 1)[0])
    return list(dict.fromkeys(roots))


# A reference that names a field is a projection, not the operator's whole output, so the node's
# declared type does not describe it: `tsp_tw` outputs a network, but `$tsp.total_cost` is the
# tour's duration and belongs wherever an amount does. Rejecting those cost eleven questions in
# one run to plans that were composed correctly. Types for the projections planners actually
# take; any other path is left unconstrained rather than refused.
OUTPUT_FIELD_TYPES: dict[str, str] = {
    "total_cost": "amount",
    "travel_cost": "amount",
    "service_cost": "amount",
    "duration_s": "amount",
    "travel_duration_s": "amount",
    "stay_duration_s": "amount",
    "distance_m": "amount",
    "distance_km": "amount",
    "percentage": "amount",
    "proportion": "proportion",
    "routes": "field",
    "steps": "field",
    "order": "object",
    "nearest": "object",
    "ranked": "object",
    "place": "object",
    "candidates": "object",
    "finish_time": "event",
    "start_time": "event",
}


def _reference_type(reference: str, node_type: str) -> str | None:
    """The type a `$node.path` reference actually carries, or None when nothing is claimed."""

    parts = [part for part in reference.lstrip("$").split(".") if part]
    if len(parts) < 2:
        return node_type
    for part in reversed(parts[1:]):
        if part in OUTPUT_FIELD_TYPES:
            return OUTPUT_FIELD_TYPES[part]
        if part.isdigit():
            continue
        return None
    return node_type


def _references_by_argument(arguments: dict[str, Any]) -> dict[str, list[str]]:
    return {name: reference_roots(value) for name, value in arguments.items()}


def _reference_strings(value: Any) -> list[str]:
    """Every `$...` reference in an argument tree, with its path kept."""

    found: list[str] = []
    if isinstance(value, str) and value.strip().startswith("$"):
        found.append(split_reference_arithmetic(canonical_reference(value))[0])
    elif isinstance(value, dict):
        for item in value.values():
            found.extend(_reference_strings(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_reference_strings(item))
    return found


def canonical_reference(value: str) -> str:
    """Normalize common LLM reference spellings into the GeoFlow `$node.path` form."""

    reference = value.strip()
    if reference.startswith("${") and reference.endswith("}"):
        reference = "$" + reference[2:-1]
    return re.sub(r"\[(\d+)]", r".\1", reference)


# A planner that has a scalar in one node and a stated constant in the question writes the sum
# into the reference itself ("$travel_s + 2700" for the 45 minutes of errands the question
# states). The node it names is a real node and the constant is a question literal, so the
# reference is a reference plus an offset — not a broken id. Read undecorated it made
# `reference_roots` hand `by_id` a key that does not exist, and the `KeyError` escaped the
# per-step isolation and lost the whole question before any tool ran.
_EXPRESSION_TERM = re.compile(r"(?P<sign>^|[+-])\s*(?P<term>\$[\w.-]+|\d+(?:\.\d+)?)\s*")


def reference_expression(value: str) -> tuple[list[str], float] | None:
    """Read `$a.b + 2700 + $c` as the references it sums and the constant it adds.

    A planner with a scalar in one node and a stated constant in the question writes the sum
    into the reference itself, and a three-leg errand run writes several: `$dur1 + 2700 + $dur2
    + 900 + $dur3`. Every `$` name in it is a real node and every number is a question literal,
    so the whole string is an expression over the graph — not a broken id. Returns None when the
    text is not a `+`/`-` chain of references and numbers, which leaves anything unrecognized to
    fail exactly as before.
    """

    text = value.strip()
    if not text.startswith("$"):
        return None
    references: list[str] = []
    constant = 0.0
    position = 0
    while position < len(text):
        match = _EXPRESSION_TERM.match(text, position)
        if not match or (position and not match.group("sign")):
            return None
        sign = -1.0 if match.group("sign") == "-" else 1.0
        term = match.group("term")
        if term.startswith("$"):
            if sign < 0:
                # A subtracted reference has no meaning we can defend; leave it unrecognized.
                return None
            references.append(term)
        else:
            constant += sign * float(term)
        position = match.end()
    return (references, constant) if references else None


def split_reference_arithmetic(value: str) -> tuple[str, float]:
    """Split `$node.path + 2700` into its reference and the offset applied to it.

    The offset is `0.0` when the string is a plain reference, so callers that only want the
    reference can ignore it. An expression naming several references has no single reference to
    return, so it is handed back unchanged for `reference_expression` to deal with.
    """

    parsed = reference_expression(value)
    if parsed is None or len(parsed[0]) != 1:
        return value, 0.0
    return parsed[0][0], parsed[1]


def _normalize_dependency(value: Any, raw_ids: list[str]) -> str:
    if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
        index = int(value)
        if 0 <= index < len(raw_ids):
            return raw_ids[index]
    text = str(value)
    if text in raw_ids:
        return text
    # Planners sometimes write the arithmetic they intend into the dependency itself
    # ("drive_time + 3600"). The dependency it names is still the node it depends on, and the
    # sum belongs in the argument, so read the node out rather than failing the whole graph.
    named = [candidate for candidate in raw_ids if re.search(rf"\b{re.escape(candidate)}\b", text)]
    if len(named) == 1:
        return named[0]
    return text


_NESTED_REFERENCE_KEYS = frozenset(
    {
        "anchor",
        "candidates",
        "center",
        "destination",
        "edges",
        "events",
        "items",
        "locations",
        "nodes",
        "objects",
        "origin",
        "place",
        "place_a",
        "place_b",
        "places",
        "route",
        "routes",
        "value",
    }
)


def _rewrite_placeholder_references(
    value: Any,
    *,
    known_ids: set[str],
    declared_dependencies: list[str],
    bare_reference_arguments: frozenset[str] = frozenset(),
    allow_bare_reference: bool = False,
) -> Any:
    if isinstance(value, dict):
        return {
            key: _rewrite_placeholder_references(
                item,
                known_ids=known_ids,
                declared_dependencies=declared_dependencies,
                bare_reference_arguments=bare_reference_arguments,
                allow_bare_reference=(
                    key in _NESTED_REFERENCE_KEYS
                    if allow_bare_reference
                    else key in bare_reference_arguments
                ),
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _rewrite_placeholder_references(
                item,
                known_ids=known_ids,
                declared_dependencies=declared_dependencies,
                bare_reference_arguments=bare_reference_arguments,
                allow_bare_reference=allow_bare_reference,
            )
            for item in value
        ]
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if (
        allow_bare_reference
        and stripped in known_ids
        and stripped in declared_dependencies
    ):
        return f"${stripped}"
    reference = canonical_reference(value)
    if not reference.startswith("$"):
        return value
    root, separator, remainder = reference[1:].partition(".")
    if (
        root not in known_ids
        and root in {"node", "step", "result", "input"}
        and len(declared_dependencies) == 1
        and declared_dependencies[0] in known_ids
    ):
        suffix = f".{remainder}" if separator else ""
        return f"${declared_dependencies[0]}{suffix}"
    return reference


def _has_coordinate_pairs(arguments: dict[str, Any]) -> bool:
    coordinate_sets = (
        {"lat1", "lon1", "lat2", "lon2"},
        {"lat1", "lng1", "lat2", "lng2"},
        {"start_lat", "start_lng", "end_lat", "end_lng"},
        {"start_lat", "start_lon", "end_lat", "end_lon"},
    )
    return any(keys <= arguments.keys() for keys in coordinate_sets)


def _topological_sort(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {step["id"]: step for step in steps}
    indegree = {step["id"]: len(step["depends_on"]) for step in steps}
    outgoing: dict[str, list[str]] = {step["id"]: [] for step in steps}
    source_order = {step["id"]: index for index, step in enumerate(steps)}
    for step in steps:
        for dependency in step["depends_on"]:
            outgoing[dependency].append(step["id"])
    ready = [step_id for step_id, degree in indegree.items() if degree == 0]
    ordered: list[dict[str, Any]] = []
    while ready:
        ready.sort(
            key=lambda step_id: (
                _execution_priority(by_id[step_id]["role"]),
                source_order[step_id],
            )
        )
        step_id = ready.pop(0)
        ordered.append(by_id[step_id])
        for target in outgoing[step_id]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if len(ordered) != len(steps):
        raise ValueError("GeoFlow graph violates acyclicity")
    return ordered


def _violates_procedural_order(source_role: str, target_role: str) -> bool:
    if source_role not in ROLE_PRIORITY or target_role not in ROLE_PRIORITY:
        return False
    return ROLE_PRIORITY[source_role] > ROLE_PRIORITY[target_role]


def _execution_priority(role: str) -> int:
    return -1 if role in CONTEXTUAL_ROLES else ROLE_PRIORITY[role]


def _reaches_measure(node: str, outgoing: dict[str, set[str]], measures: set[str]) -> bool:
    pending = [node]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current in measures:
            return True
        if current in visited:
            continue
        visited.add(current)
        pending.extend(outgoing[current])
    return False


def _reachable_from_context(
    node: str,
    by_id: dict[str, dict[str, Any]],
    extents: set[str],
) -> bool:
    pending = [node]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current in extents:
            return True
        if current in visited:
            continue
        visited.add(current)
        pending.extend(by_id[current]["depends_on"])
    return False


def _reachable_in_graph(
    node: str,
    adjacency: dict[str, set[str]],
    targets: set[str],
) -> bool:
    pending = [node]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current in targets:
            return True
        if current in visited:
            continue
        visited.add(current)
        pending.extend(adjacency[current])
    return False
