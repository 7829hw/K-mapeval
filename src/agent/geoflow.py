from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from src.agent.semantics import lift_to_semantic
from src.spatial_contracts import MATRIX_METRICS, normalize_tsp_metric
from src.tools.spatial import split_place_type

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
    "select_legs": OperatorContract(
        "field", ("routes",), ("order",), ("routes",)
    ),
    "filter_by_distance": OperatorContract(
        "object", ("items", "max_distance_m"), ("key",), ("items",)
    ),
    "count_items": OperatorContract(
        "amount", ("items",), reference_arguments=("items",)
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
    "select_legs": {"routes": frozenset({"field"})},
    "filter_by_distance": {"items": frozenset({"object", "field"})},
    "count_items": {"items": frozenset({"object", "field"})},
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
        "affinity": {"proportion", "percentage", "distribution"},
        "keywords": ("속성", "분포", "비율", "field", "proportion"),
        "pattern": "PLACE_SEARCH -> FILTER -> AGGREGATE -> MEASURE",
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
        "affinity": {"type", "category", "address", "attribute", "information"},
        "keywords": ("유형", "종류", "카테고리", "주소", "정보"),
        "pattern": "PLACE_SEARCH -> PLACE_DETAILS -> MEASURE",
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
        "affinity": {"direction", "bearing", "sector"},
        "keywords": ("방향", "동쪽", "서쪽", "남쪽", "북쪽"),
        "pattern": "RESOLVE_PLACES -> FILTER (the stated sector) -> MEASURE",
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
        "affinity": {"nearest", "closest", "distance", "proximity", "comparison"},
        "keywords": ("가까운", "거리", "짧은", "먼", "멀리", "nearest", "farthest", "distance"),
        # Narrowed deliberately. It used to read "anchor and candidates", which a question asking
        # for a *kind* of place copied -- resolving the four answer texts and ranking those.
        "pattern": (
            "RESOLVE_PLACES (every place the question names) -> DISTANCE_MEASURE -> "
            "EXTREME_SELECT -> MATCH_OPTIONS; for questions whose candidates are themselves the "
            "places being measured, never for one that asks for a kind of place"
        ),
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
    "search_rank_ordinal": {
        "name": "Search-Rank-Ordinal",
        # A shape, not a task type. What made the deleted `Retrieve-Rank-Ordinal` a benchmark
        # family wearing a template's clothes was that it carried its own operator recipe and was
        # retrieved on "번째"; this carries a transformation structure, the factorizer still picks
        # every operator, and `k` is a factor. Without it a "네 번째로 가까운 은행" question
        # retrieves `Geocode-Batch-Compare` and ranks the four answer texts.
        "affinity": {"nearest", "closest", "ordinal", "rank", "proximity", "kind"},
        "keywords": ("가까운", "번째", "인접한", "nearest", "closest"),
        "target_literal": True,
        "supersedes": ("geocode_compare",),
        "pattern": (
            "RESOLVE_PLACES (the anchor only) -> PLACE_SEARCH (the kind asked for) -> "
            "DISTANCE_MEASURE -> ORDINAL_SELECT (ordinal=k) -> MATCH_OPTIONS"
        ),
        "example": {"graph": []},
    },
    "distance_difference": {
        "name": "Pairwise-Difference",
        "affinity": {"difference", "comparison", "distance", "gap"},
        "keywords": ("차이", "차이가", "difference"),
        "pattern": (
            "RESOLVE_PLACES -> DISTANCE_MEASURE x2 -> AGGREGATE (aggregate=difference) -> "
            "MATCH_OPTIONS"
        ),
        "example": {"graph": []},
    },
    "radius": {
        "name": "Filter-Aggregate-Measure",
        "affinity": {"radius", "within", "count"},
        "radius_literal": True,
        "keywords": ("반경", "이내", "within", "radius"),
        "pattern": (
            "RESOLVE_PLACES -> PLACE_SEARCH -> FILTER (the stated radius) -> AGGREGATE -> MEASURE"
        ),
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
    "listed_candidates_count": {
        "name": "Listed-Measure-Filter-Count",
        "affinity": {"radius", "within", "count", "candidates"},
        "listed_literal": True,
        "keywords": ("반경", "이내", "목록", "몇 곳"),
        "supersedes": ("radius",),
        "pattern": (
            "RESOLVE_PLACES (anchor) + RESOLVE_PLACES (scope=listed) -> DISTANCE_MEASURE -> "
            "FILTER (the stated radius) -> AGGREGATE (count) -> MEASURE"
        ),
        "example": {"graph": []},
    },
    "search_narrow_rank": {
        "name": "Search-Narrow-Rank",
        "affinity": {"nearest", "closest", "kind", "subtype", "cuisine"},
        "subtype_literal": True,
        "keywords": ("가장 가까운", "가까운", "nearest"),
        "supersedes": ("search_rank_ordinal",),
        "pattern": (
            "RESOLVE_PLACES (the anchor only) -> PLACE_SEARCH (the broad kind) -> "
            "FILTER (scope=attribute) -> DISTANCE_MEASURE -> ORDINAL_SELECT -> MATCH_OPTIONS"
        ),
        "example": {"graph": []},
    },
    "routes": {
        "name": "Multi-Route-Compare",
        "affinity": {"route", "routing", "travel", "duration", "driving"},
        "network_literal": True,
        "keywords": ("경로", "자동차", "주행", "route", "driving"),
        "pattern": "ROUTE_MATRIX -> ROUTE_COMPARE -> MEASURE",
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
        "affinity": {"trip", "itinerary", "total_distance", "total_duration"},
        "trip_literal": True,
        "keywords": ("일정", "차례", "경유", "여행", "trip", "itinerary"),
        "pattern": (
            "ROUTE_MATRIX -> SELECT_LEGS (the consecutive legs) -> AGGREGATE -> MEASURE"
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
        "affinity": {
            "trip",
            "order",
            "sequence",
            "feasibility",
            "visited_count",
            "time_budget",
        },
        "trip_literal": True,
        "keywords": ("최적 순서", "시간창", "방문 순서", "몇 곳", "tsp"),
"pattern": "RESOLVE_PLACES -> ROUTE_MATRIX -> ROUTE_OPTIMIZE -> MEASURE",
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
        "affinity": {"turn", "step", "instruction", "guidance"},
        "network_literal": True,
        "guidance_literal": True,
        "keywords": ("경로 단계", "도로", "회전", "step"),
        "pattern": (
            "RESOLVE_PLACES -> ROUTE_MEASURE (via = anything the route passes through) -> "
            "ROUTE_STEPS -> MEASURE"
        ),
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
        "affinity": {"departure", "arrival", "finish_time", "start_time", "time_window"},
        "keywords": ("도착 시간", "출발 시간", "영업시간", "time window"),
        "pattern": "ROUTE_MEASURE (measure=duration) -> SCHEDULE -> MEASURE",
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


# ---------------------------------------------------------------------------------------------
# Semantic skeletons
#
# What shape of graph a question needs is not operator knowledge and does not belong in the
# factorizer, but it is knowledge, and deleting the 163-line planner prompt threw it away: the
# semantic rewrite scored 50.9% against 82.1%, and the single clearest cause was that a
# "네 번째로 가까운 은행" question retrieved `Geocode-Batch-Compare` -- whose pattern reads
# "resolve the anchor and the candidates, then rank them" -- and copied it. That answers "which
# of these four is closest", which is a different question with a different answer.
#
# These are that knowledge, rewritten as transformation structure. They name no operator, no
# argument and no provider category; the factorizer still decides all three. Each is one worked
# composition with the factors that make it the shape it is.
# ---------------------------------------------------------------------------------------------

SKELETONS: dict[str, list[dict[str, Any]]] = {
    # A kind of place around one anchor, ranked, and the k-th taken. `ordinal` is the whole
    # difference between "가장 가까운" (1) and "네 번째로 가까운" (4) -- one factor, not a family.
    # Only the anchor is resolved: the candidate texts are answers, and resolving them makes the
    # ranking a ranking of the answers.
    "search_rank_ordinal": [
        {"id": "anchor", "transform": "RESOLVE_PLACES", "inputs": [],
         "concept_ids": ["<the anchor concept>"], "role": "extent"},
        {"id": "found", "transform": "PLACE_SEARCH", "inputs": ["anchor"], "role": "support"},
        {"id": "ranked", "transform": "DISTANCE_MEASURE", "inputs": ["anchor", "found"],
         "role": "support"},
        {"id": "kth", "transform": "ORDINAL_SELECT", "inputs": ["ranked"],
         "factors": {"ordinal": 4}, "role": "support"},
        {"id": "answer", "transform": "MATCH_OPTIONS", "inputs": ["kth"], "role": "measure"},
    ],
    # A count or a set inside a stated radius. The radius is a fact the analysis extracted; the
    # skeleton only says where it applies.
    "radius": [
        {"id": "anchor", "transform": "RESOLVE_PLACES", "inputs": [],
         "concept_ids": ["<the anchor concept>"], "role": "extent"},
        # The candidates: the ones the question lists if it lists any, otherwise a retrieval of
        # the kind it asks for. A question that names its candidates is asking about those.
        {"id": "candidates", "transform": "PLACE_SEARCH", "inputs": ["anchor"],
         "role": "support"},
        {"id": "measured", "transform": "DISTANCE_MEASURE", "inputs": ["anchor", "candidates"],
         "role": "support"},
        {"id": "inside", "transform": "FILTER", "inputs": ["measured"], "role": "support"},
        {"id": "count", "transform": "AGGREGATE", "inputs": ["inside"],
         "factors": {"aggregate": "count"}, "role": "support"},
        {"id": "answer", "transform": "MEASURE", "inputs": ["count"], "role": "measure"},
    ],
    # A question that lists its own candidates: resolve those, measure each from the anchor,
    # keep the ones the stated radius admits, and count them. The radius is a fact the analysis
    # extracted; the skeleton only says where it applies.
    "listed_candidates_count": [
        {"id": "anchor", "transform": "RESOLVE_PLACES", "inputs": [],
         "concept_ids": ["<the anchor concept>"], "role": "extent"},
        {"id": "listed", "transform": "RESOLVE_PLACES", "inputs": [],
         "factors": {"scope": "listed"}, "role": "extent"},
        {"id": "measured", "transform": "DISTANCE_MEASURE", "inputs": ["anchor", "listed"],
         "role": "support"},
        {"id": "inside", "transform": "FILTER", "inputs": ["measured"], "role": "support"},
        {"id": "count", "transform": "AGGREGATE", "inputs": ["inside"],
         "factors": {"aggregate": "count"}, "role": "support"},
        {"id": "answer", "transform": "MEASURE", "inputs": ["count"], "role": "measure"},
    ],
    # A kind of place narrowed by an attribute of it -- 중식 of "중식 음식점". Retrieve the broad
    # kind, narrow to the attribute, then rank: a ranking that skips the narrowing answers with
    # the nearest place of any kind, and the closer options are exactly the other kinds.
    "search_narrow_rank": [
        {"id": "anchor", "transform": "RESOLVE_PLACES", "inputs": [],
         "concept_ids": ["<the anchor concept>"], "role": "extent"},
        {"id": "found", "transform": "PLACE_SEARCH", "inputs": ["anchor"], "role": "support"},
        {"id": "narrowed", "transform": "FILTER", "inputs": ["found"],
         "factors": {"scope": "attribute"}, "role": "support"},
        {"id": "ranked", "transform": "DISTANCE_MEASURE", "inputs": ["anchor", "narrowed"],
         "role": "support"},
        {"id": "kth", "transform": "ORDINAL_SELECT", "inputs": ["ranked"],
         "factors": {"ordinal": 1}, "role": "support"},
        {"id": "answer", "transform": "MATCH_OPTIONS", "inputs": ["kth"], "role": "measure"},
    ],
    # A compass sector narrows the candidates before the ranking does. Filtering after the rank
    # answers with whichever place was nearest regardless of where it lies.
    "bearing": [
        {"id": "anchor", "transform": "RESOLVE_PLACES", "inputs": [],
         "concept_ids": ["<the anchor concept>"], "role": "extent"},
        {"id": "found", "transform": "PLACE_SEARCH", "inputs": ["anchor"], "role": "support"},
        {"id": "sector", "transform": "FILTER", "inputs": ["anchor", "found"],
         "role": "support"},
        {"id": "ranked", "transform": "DISTANCE_MEASURE", "inputs": ["anchor", "sector"],
         "role": "support"},
        {"id": "answer", "transform": "MATCH_OPTIONS", "inputs": ["ranked"], "role": "measure"},
    ],
    # Places the question itself names, compared against each other. This is the shape for
    # "which pair is farthest", "which of A and B is nearer to C" -- questions whose candidates
    # *are* the things being measured. It is not the shape for a question that asks for a kind.
    "geocode_compare": [
        {"id": "places", "transform": "RESOLVE_PLACES", "inputs": [],
         "concept_ids": ["<every place the question names>"], "role": "extent"},
        {"id": "spans", "transform": "DISTANCE_MEASURE", "inputs": ["places"], "role": "support"},
        {"id": "pick", "transform": "EXTREME_SELECT", "inputs": ["spans"],
         "factors": {"extreme": "max"}, "role": "support"},
        {"id": "answer", "transform": "MATCH_OPTIONS", "inputs": ["pick"], "role": "measure"},
    ],
    # Two separations and the gap between them: three places, two measures, one difference.
    "distance_difference": [
        {"id": "places", "transform": "RESOLVE_PLACES", "inputs": [],
         "concept_ids": ["<the anchor and both destinations>"], "role": "extent"},
        {"id": "first", "transform": "DISTANCE_MEASURE", "inputs": ["places"], "role": "support"},
        {"id": "second", "transform": "DISTANCE_MEASURE", "inputs": ["places"],
         "role": "support"},
        {"id": "gap", "transform": "AGGREGATE", "inputs": ["first", "second"],
         "factors": {"aggregate": "difference"}, "role": "support"},
        {"id": "answer", "transform": "MATCH_OPTIONS", "inputs": ["gap"], "role": "measure"},
    ],
    # One road route and something read off its guidance: a turn count, the n-th turn, the road
    # a turn happens on. A via-point is part of the route, not a second route.
    "route_step_extract": [
        {"id": "ends", "transform": "RESOLVE_PLACES", "inputs": [],
         "concept_ids": ["<origin, any via point, destination, in that order>"],
         "role": "extent"},
        {"id": "route", "transform": "ROUTE_MEASURE", "inputs": ["ends"],
         "via": ["<the concept the route passes through, omit when it passes through nothing>"],
         "role": "support"},
        {"id": "steps", "transform": "ROUTE_STEPS", "inputs": ["route"], "role": "support"},
        {"id": "answer", "transform": "MATCH_OPTIONS", "inputs": ["steps"], "role": "measure"},
    ],
    # Road routes compared against each other -- which detour is cheapest, which option lies on
    # the way. Every candidate route is measured before any of them is chosen.
    "routes": [
        {"id": "places", "transform": "RESOLVE_PLACES", "inputs": [],
         "concept_ids": ["<the endpoints and every candidate>"], "role": "extent"},
        {"id": "legs", "transform": "ROUTE_MATRIX", "inputs": ["places"], "role": "support"},
        {"id": "pick", "transform": "ROUTE_COMPARE", "inputs": ["legs"], "role": "support"},
        {"id": "answer", "transform": "MATCH_OPTIONS", "inputs": ["pick"], "role": "measure"},
    ],
    # The whole distance or duration of an itinerary the question already orders. Every leg is
    # looked up in one matrix and totalled; the measure the question names decides the unit.
    "trip": [
        {"id": "stops", "transform": "RESOLVE_PLACES", "inputs": [],
         "concept_ids": ["<the start and every stop, in the order stated>"], "role": "extent"},
        {"id": "legs", "transform": "ROUTE_MATRIX", "inputs": ["stops"], "role": "support"},
        {"id": "path", "transform": "SELECT_LEGS", "inputs": ["legs"],
         "factors": {"scope": "consecutive"}, "role": "support"},
        {"id": "total", "transform": "AGGREGATE", "inputs": ["path"],
         "factors": {"aggregate": "sum", "measure": "distance"},
         "role": "support"},
        {"id": "answer", "transform": "MATCH_OPTIONS", "inputs": ["total"], "role": "measure"},
    ],
    # An itinerary the question does *not* order: which sequence is shortest, or how many stops
    # fit the time. The stays, the budget, the closure and the stated order are facts the
    # analysis extracted and the factorizer binds -- the skeleton only asks for the ordering.
    "route_optimize": [
        {"id": "stops", "transform": "RESOLVE_PLACES", "inputs": [],
         "concept_ids": ["<the start and every stop>"], "role": "extent"},
        {"id": "legs", "transform": "ROUTE_MATRIX", "inputs": ["stops"], "role": "support"},
        {"id": "tour", "transform": "ROUTE_OPTIMIZE", "inputs": ["stops", "legs"],
         "factors": {"measure": "distance"}, "role": "support"},
        {"id": "answer", "transform": "MATCH_OPTIONS", "inputs": ["tour"], "role": "measure"},
    ],
    # When a clock is involved: how long the drive takes, then what time that makes it.
    "time_window_reverse": [
        {"id": "ends", "transform": "RESOLVE_PLACES", "inputs": [],
         "concept_ids": ["<origin and destination>"], "role": "extent"},
        {"id": "drive", "transform": "ROUTE_MEASURE", "inputs": ["ends"],
         "factors": {"measure": "duration"}, "role": "support"},
        {"id": "clock", "transform": "SCHEDULE", "inputs": ["drive"], "role": "support"},
        {"id": "answer", "transform": "MATCH_OPTIONS", "inputs": ["clock"], "role": "measure"},
    ],
    # A property of one named place, read off the record rather than computed.
    "place_attribute": [
        {"id": "place", "transform": "RESOLVE_PLACES", "inputs": [],
         "concept_ids": ["<the named place>"], "role": "extent"},
        {"id": "detail", "transform": "PLACE_DETAILS", "inputs": ["place"], "role": "support"},
        {"id": "answer", "transform": "MATCH_OPTIONS", "inputs": ["detail"], "role": "measure"},
    ],
    # A share or a count over a neighbourhood.
    "object_field_measure": [
        {"id": "anchor", "transform": "RESOLVE_PLACES", "inputs": [],
         "concept_ids": ["<the anchor concept>"], "role": "extent"},
        {"id": "found", "transform": "PLACE_SEARCH", "inputs": ["anchor"], "role": "support"},
        {"id": "matching", "transform": "FILTER", "inputs": ["anchor", "found"],
         "role": "support"},
        {"id": "share", "transform": "AGGREGATE", "inputs": ["matching", "found"],
         "factors": {"aggregate": "proportion"}, "role": "support"},
        {"id": "answer", "transform": "MATCH_OPTIONS", "inputs": ["share"], "role": "measure"},
    ],
}


def retrieve_templates(
    analysis: dict[str, Any], question: str, *, limit: int = 2
) -> list[dict[str, Any]]:
    """Retrieve templates from concept/role content and the question's literal wording.

    The Analysis stage still predicts an ``intent`` for reporting, but that label is deliberately
    excluded here. A template is relevant because the concept graph names its measure,
    constraints, attributes, or target object, not because a classifier put the question in one
    bucket. Question keywords remain the exact-literal fallback for a graph that omitted a hint.
    """

    return [
        {"name": template["name"], "pattern": template["pattern"]}
        for _key, template in _rank_templates(analysis, question, limit=limit)
    ]


def _rank_templates(
    analysis: dict[str, Any], question: str, *, limit: int
) -> list[tuple[str, dict[str, Any]]]:
    """Score every template on concept/role structure, then on the question's literal wording."""

    lowered = question.casefold()
    concept_hints = _analysis_retrieval_hints(analysis)
    ranked: list[tuple[int, str, dict[str, Any]]] = []
    for key, template in TEMPLATES.items():
        score = 4 * sum(
            1 for hint in template.get("affinity", ()) if hint.casefold() in concept_hints
        )
        score += _template_shape_score(template, analysis, lowered)
        score += sum(1 for keyword in template["keywords"] if keyword.casefold() in lowered)
        if score:
            ranked.append((score, key, template))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    chosen: list[tuple[str, dict[str, Any]]] = []
    blocked: set[str] = set()
    for _, key, template in ranked:
        if key in blocked:
            continue
        chosen.append((key, template))
        # Only a template that already outranked it can supersede one, since `ranked` is sorted.
        # `Search-Rank-Ordinal` supersedes `Geocode-Batch-Compare` for exactly the reason the
        # measurement gave: they are rival shapes for one question, and the loser's worked
        # example is the wrong answer sitting beside the right one.
        blocked |= set(template.get("supersedes", ()))
        if len(chosen) >= limit:
            break
    return chosen


#: Turn texts into vectors. Supplied by the caller so the retrieval policy is a configuration
#: choice rather than a hard dependency: the deployment these runs use serves one chat model and
#: answers `/v1/embeddings` with 404, so `None` -- the deterministic scorer below -- is what
#: actually runs here.
ExampleEmbedder = Callable[[list[str]], list[list[float]]]


def retrieve_examples(
    analysis: dict[str, Any],
    question: str,
    *,
    limit: int = 2,
    embed: ExampleEmbedder | None = None,
) -> list[dict[str, Any]]:
    """Top-k few-shot example graphs, retrieved separately from the macro-templates.

    The two retrievals answer different questions and had been answering them with one score.
    *Which macro-template* a question needs is a structural fact -- the concept graph names a
    measure over a network, or a field restricted by a sub-condition -- and `retrieve_templates`
    reads it off the concepts and roles. *Which worked example* helps most is a similarity
    question, and similarity over prose is what an embedding is for.

    `embed` is the seam. With one, examples are ranked by cosine similarity between the question
    and each example's description. Without one, they are ranked by the same deterministic
    concept overlap as before, so behaviour is unchanged where no embedding service exists.
    """

    if embed is not None:
        ranked = _embedding_ranking(_example_bank(), question, embed)
        return [{"name": e["name"], "example": e["example"]} for e in ranked[:limit]]
    # Structural retrieval, and the *same* structural retrieval the templates got. Ranking the
    # two independently let them disagree -- a trip question was shown `Route-Optimize`'s pattern
    # beside `Geocode-Batch-Compare`'s worked graph, and a planner copies the graph.
    bank = {entry["key"]: entry for entry in _example_bank()}
    return [
        {"name": bank[key]["name"], "example": bank[key]["example"]}
        for key, _template in _rank_templates(analysis, question, limit=limit)
        if key in bank
    ]


def _example_bank() -> list[dict[str, Any]]:
    """The examples available for few-shot retrieval.

    Today this is one worked graph per macro-template, which is where they already lived. Keeping
    the bank behind a function is what lets it grow into recorded successful graphs without the
    retrieval policy or its callers changing.
    """

    return [
        {
            "key": key,
            "name": template["name"],
            # The authored skeleton when there is one -- that is where the question-shape
            # knowledge lives. Otherwise the worked operator graph lifted into the vocabulary,
            # which is also what makes the round-trip test possible.
            "example": {
                "graph": SKELETONS.get(key)
                or lift_to_semantic(template["example"]["graph"])
            },
            "pattern": template["pattern"],
            "affinity": tuple(template.get("affinity", ())),
            "keywords": tuple(template.get("keywords", ())),
        }
        for key, template in TEMPLATES.items()
    ]


def _embedding_ranking(
    bank: list[dict[str, Any]], question: str, embed: ExampleEmbedder
) -> list[dict[str, Any]]:
    """Cosine similarity between the question and each example's own description."""

    descriptions = [f"{entry['name']}: {entry['pattern']}" for entry in bank]
    vectors = embed([question, *descriptions])
    if len(vectors) != len(descriptions) + 1:
        raise ValueError("embedder returned one vector per input, and did not")
    asked, rest = vectors[0], vectors[1:]
    scored = sorted(
        zip(rest, bank, strict=True),
        key=lambda pair: (-_cosine(asked, pair[0]), pair[1]["name"]),
    )
    return [entry for _, entry in scored]


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    scale = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / scale if scale else 0.0


def _analysis_retrieval_hints(analysis: dict[str, Any]) -> str:
    """Flatten only concept-graph evidence used for deterministic template affinity.

    Top-level ``intent`` is intentionally not traversed. Keeping this projection explicit makes
    it impossible for arbitrary analysis metadata to become another hidden router.
    """

    values: list[str] = []
    for key in ("measure", "target_type"):
        value = analysis.get(key)
        if value not in (None, ""):
            values.append(str(value))
    for concept in analysis.get("concepts") or []:
        if not isinstance(concept, dict):
            continue
        # Location names and generic types such as ``object``/``amount`` say nothing about which
        # macro should run. Restrict free text to the roles that state a requested measure or a
        # narrowing condition; extent names remain available through the question fallback.
        if concept.get("role") in {"measure", "condition", "sub_condition"}:
            values.append(str(concept.get("text") or ""))
        attributes = concept.get("attributes")
        if isinstance(attributes, dict):
            for key, value in attributes.items():
                values.extend((str(key), str(value)))
    return " ".join(values).casefold()


def _template_shape_score(
    template: dict[str, Any], analysis: dict[str, Any], lowered_question: str
) -> int:
    """Score declared graph shapes before falling back to literal keyword overlap."""

    concepts = [value for value in (analysis.get("concepts") or []) if isinstance(value, dict)]
    spatial_count = sum(
        concept.get("concept_type") in {"location", "object"}
        and concept.get("role") != "measure"
        for concept in concepts
    )
    score = 0
    if (
        template.get("trip_literal")
        and spatial_count >= 3
        and re.search(
            r"(→.*→|(?:\d+(?:\.\d+)?\s*(?:시간|분)).*(?:\d+(?:\.\d+)?\s*(?:시간|분))"
            r"|방문\s*순서|몇\s*곳|일정|차례|순서로|순서대로|둘러|itinerary|trip)",
            lowered_question,
        )
    ):
        score += 16
    if template.get("target_literal") and (
        analysis.get("target_type")
        or any(
            concept.get("concept_type") == "object" and concept.get("role") != "measure"
            for concept in concepts
        )
    ):
        # The question asks for a *kind* of place, which is what separates "the nearest bank"
        # from "which of these two is nearer". Structural: it reads the concept graph, not the
        # question's wording, and `target_type` is what the Analysis stage is for.
        score += 16
    if template.get("radius_literal") and re.search(
        r"(반경|이내|안에|내에)\s*|[\d,.]+\s*(?:km|m)\s*(?:이내|안|이하)", lowered_question
    ):
        # A stated radius is a sub-condition, and it decides the shape: the candidates are
        # everything inside it, not the k nearest. Without this a radius question asking for a
        # kind of place retrieved the ordinal shape, which ranks instead of counting.
        score += 16
    if template.get("network_literal") and re.search(
        r"(자동차|차량|운전|주행|도로|경로|route|driv)", lowered_question
    ):
        score += 8
    if template.get("listed_literal") and _lists_its_candidates(lowered_question):
        # The question names the candidates it is asking about, so the graph resolves those
        # rather than retrieving a neighbourhood. The two answer different questions: "how many
        # of these four are within 300 m" and "how many banks are within 300 m".
        score += 16
    if template.get("subtype_literal") and _states_a_narrowed_kind(analysis, lowered_question):
        # A kind written as a modifier plus a broad category -- 중식 음식점 -- needs the
        # narrowing step between the retrieval and the ranking. Without it the ranking answers
        # with the nearest place of the broad kind, which is what the other options are.
        score += 16
    if template.get("guidance_literal") and re.search(
        r"(안내에\s*따르|주행\s*안내|회전|turn|manoeuvr|maneuver)", lowered_question
    ):
        # The answer is read off the driving guidance rather than off the route's totals, which
        # is a different shape from "which route is shorter". Structural in the same way a
        # stated radius is: without it a turn-count question retrieved Multi-Route-Compare and
        # Filter-Aggregate-Measure, and never saw the shape that reads a step list at all.
        score += 16
    return score


def _lists_its_candidates(question: str) -> bool:
    """Does the question offer its own candidate list?  Structural, and read from the question."""

    from src.agent.spatial import _extract_listed_places

    return bool(_extract_listed_places(question))


def _states_a_narrowed_kind(analysis: dict[str, Any], question: str) -> bool:
    """Is the kind of place asked for a broad category with a narrowing modifier on it?"""

    from src.agent.spatial import _extract_target_type

    stated = _extract_target_type(question) or analysis.get("target_type")
    if not stated:
        return False
    _broad, attribute = split_place_type(str(stated))
    return bool(attribute)


def normalize_analysis(
    payload: dict[str, Any],
    question: str,
    fallback_intent: str,
    facts: Any = None,
) -> dict[str, Any]:
    """Normalize the Analysis stage's reply, completing it from `facts` where it came up short.

    The stage returns nothing usable on about a fifth of questions -- measured at 24% on the
    `af51e93` runs and 19% here, so this is long-standing and not a regression. What changed is
    the cost. Under the old architecture the planner copied place names out of the question
    itself, so a threadbare concept graph was decoration; under the semantic one, place identity
    travels through the concepts, and a `RESOLVE_PLACES` node with nothing to name geocodes the
    fallback -- which was the entire question text, as one "place".

    `facts` is what the deterministic extractors already read off the same question: the anchor,
    the kind of place, the stays. Building the fallback from those instead is not new evidence,
    it is evidence this pipeline had already gathered and was throwing away.
    """

    intent = str(payload.get("intent", "")).strip().lower() or fallback_intent
    raw_measure = payload.get("measure")
    measure = str(raw_measure).strip() if raw_measure not in (None, "") else None
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
        concepts = _concepts_from_facts(facts, measure)
    if measure is None:
        measure = next(
            (
                str(concept.get("text") or "").strip()
                for concept in concepts
                if concept.get("role") == "measure" and str(concept.get("text") or "").strip()
            ),
            "answer choice",
        )
    concepts = _complete_analysis_roles(concepts, question, measure)
    # The kind of place the question is asking for, which the Analysis stage may have had to
    # infer ("우산을 사야 합니다" -> 편의점). Grounding binds it when the question text does not
    # name a type outright; dropping it here is what made a need-shaped question unanswerable.
    target = payload.get("target_type") or payload.get("place_type")
    target_type = str(target).strip() if isinstance(target, str) and target.strip() else None
    if target_type is None and facts is not None:
        # The stage leaves this null on 44% of the questions that plainly name a kind of place,
        # in both revisions. The question scan already found it, and template retrieval reads
        # this field to tell "the nearest bank" from "which of these two is nearer".
        stated = getattr(facts, "target_type", None)
        target_type = str(stated).strip() if isinstance(stated, str) and stated.strip() else None
    return {
        "intent": intent,
        "concepts": concepts,
        "measure": measure,
        "target_type": target_type,
    }


def _concepts_from_facts(facts: Any, measure: str | None) -> list[dict[str, Any]]:
    """A concept graph from what the question was already read to state.

    Never the whole question as one concept. That is not a place, and a `RESOLVE_PLACES` node
    handed it geocodes a sentence.
    """

    def concept(cid: str, text: str, kind: str, role: str, depends: list[str]) -> dict[str, Any]:
        return {
            "id": cid,
            "text": text,
            "concept_type": kind,
            "role": role,
            "attributes": {},
            "depends_on": depends,
        }

    built: list[dict[str, Any]] = []
    anchor = getattr(facts, "anchor", None) if facts is not None else None
    if isinstance(anchor, str) and anchor.strip():
        built.append(concept("anchor", anchor.strip(), "location", "extent", []))
    target = getattr(facts, "target_type", None) if facts is not None else None
    if isinstance(target, str) and target.strip():
        built.append(
            concept("target_type", target.strip(), "object", "support",
                    ["anchor"] if built else [])
        )
    for index, stay in enumerate(getattr(facts, "stays", ()) or ()):
        name = stay[0]
        if str(name).strip():
            built.append(concept(f"stop_{index}", str(name).strip(), "location", "extent", []))
    pair = getattr(facts, "compared_pair", None) if facts is not None else None
    for index, name in enumerate(pair or ()):
        if str(name).strip():
            built.append(concept(f"compared_{index}", str(name).strip(), "location", "extent", []))
    if not built:
        # Nothing stated that any extractor could find. A measure alone is still a valid graph:
        # it says what is being asked for and nothing about where, which is honest.
        return [concept("requested_answer", measure or "answer choice", "amount", "measure", [])]
    built.append(
        concept(
            "requested_answer",
            measure or "answer choice",
            "amount",
            "measure",
            [built[0]["id"]],
        )
    )
    return built


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


def _static_flag(value: Any) -> bool | None:
    """A literal boolean when the planner wrote one; None for references or unknown spellings."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str) and not value.strip().startswith("$"):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0", "none", ""}:
            return False
    return None


def _static_int(value: Any, *, name: str) -> int | None:
    """Read an integer literal without pretending an unresolved reference already has a value."""

    if isinstance(value, str) and value.strip().startswith("$"):
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not a boolean")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer") from None
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer")
    return parsed


def _normalize_statically_known_argument_values(
    operator: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Canonicalize only exact literal equivalents needed by static contract validation."""

    if operator != "tsp_tw":
        return arguments
    normalized = dict(arguments)
    for alias in ARGUMENT_ALIASES["tsp_tw"]["fixed_order"]:
        if alias in normalized:
            normalized.setdefault("fixed_order", normalized.pop(alias))
    if "metric" in normalized:
        normalized["metric"] = normalize_tsp_metric(normalized["metric"])
    for name in ("fixed_order", "return_to_start"):
        if name in normalized and (flag := _static_flag(normalized[name])) is not None:
            normalized[name] = flag
    return normalized


def _statically_known_collection_length(
    value: Any, by_id: dict[str, dict[str, Any]]
) -> int | None:
    """Collection cardinality when literals or a literal batch_geocode node make it knowable."""

    if isinstance(value, list):
        total = 0
        for item in value:
            if isinstance(item, list):
                nested = _statically_known_collection_length(item, by_id)
                if nested is None:
                    return None
                total += nested
                continue
            if isinstance(item, str):
                reference = split_reference_arithmetic(canonical_reference(item))[0]
                if reference.startswith("$") and "." not in reference[1:]:
                    producer = by_id.get(reference[1:])
                    if producer and producer["operator"] == "batch_geocode":
                        names = producer["arguments"].get("place_names")
                        if not isinstance(names, list):
                            return None
                        total += len(names)
                        continue
            total += 1
        return total
    if isinstance(value, str):
        reference = split_reference_arithmetic(canonical_reference(value))[0]
        if reference.startswith("$") and "." not in reference[1:]:
            producer = by_id.get(reference[1:])
            if producer and producer["operator"] == "batch_geocode":
                names = producer["arguments"].get("place_names")
                return len(names) if isinstance(names, list) else None
        return None
    if isinstance(value, dict):
        return 1
    return None


def _validate_statically_known_argument_values(
    steps: list[dict[str, Any]], by_id: dict[str, dict[str, Any]]
) -> None:
    """Reject literal shapes and cross-argument combinations the implementation cannot run.

    References whose values do not exist until execution stay unknown and are not guessed at.
    The checks here either mirror a runtime limit/type exactly or compare literals already in the
    graph, so they belong to the ordinary repair round instead of becoming failed execution steps.
    """

    for step in steps:
        operator = step["operator"]
        arguments = step["arguments"]

        collection_limits = {
            "batch_geocode": {"place_names": 30},
            "batch_place_details": {"place_ids": 45},
            "distance_matrix": {"origins": 15, "destinations": 15, "pairs": 30},
        }.get(operator, {})
        for argument_name, maximum in collection_limits.items():
            if argument_name not in arguments:
                continue
            length = _statically_known_collection_length(arguments[argument_name], by_id)
            if length is not None and length > maximum:
                raise ValueError(
                    f"GeoFlow node {step['id']} gives {operator}.{argument_name} {length} items; "
                    f"the implementation accepts at most {maximum}"
                )

        if operator == "pairwise_distances":
            pairs = arguments.get("pairs")
            if isinstance(pairs, list) and any(
                not isinstance(pair, dict)
                and not (isinstance(pair, str) and pair.strip().startswith("$"))
                for pair in pairs
            ):
                raise ValueError(
                    f"GeoFlow node {step['id']} gives pairwise_distances.pairs items that are "
                    "not pair objects or pair references"
                )

        if operator in {"steps_analysis", "open_at_time"}:
            argument_name = "route" if operator == "steps_analysis" else "schedule"
            value = arguments.get(argument_name)
            if isinstance(value, list):
                raise ValueError(
                    f"GeoFlow node {step['id']} gives {operator}.{argument_name} a list; "
                    "the implementation accepts one object"
                )

        if operator in {"select_min", "select_max", "sort_by"} and "key" in arguments:
            key = arguments["key"]
            if not isinstance(key, str) or key.strip().startswith("$"):
                raise ValueError(
                    f"GeoFlow node {step['id']} gives {operator}.key a data value; "
                    "the implementation requires a literal field name"
                )

        if operator != "tsp_tw":
            continue
        metric = arguments.get("metric", "duration")
        if not isinstance(metric, str) or metric not in MATRIX_METRICS:
            raise ValueError(
                f"GeoFlow node {step['id']} gives tsp_tw.metric {metric!r}; "
                f"use one of {sorted(MATRIX_METRICS)}"
            )
        clock_arguments = ("service_times", "time_windows", "time_budget")
        if metric != "duration" and any(
            arguments.get(name) is not None for name in clock_arguments
        ):
            raise ValueError(
                f"GeoFlow node {step['id']} combines tsp_tw metric={metric!r} with clock "
                "arguments; service times, windows and budgets require metric='duration'"
            )

        node_count = _statically_known_collection_length(arguments.get("nodes"), by_id)
        if node_count is not None:
            if node_count > 9:
                raise ValueError(
                    f"GeoFlow node {step['id']} gives tsp_tw {node_count} nodes; "
                    "the deterministic implementation accepts at most 9"
                )
            for name in ("service_times", "time_windows"):
                value = arguments.get(name)
                if value is not None and isinstance(value, list) and len(value) != node_count:
                    raise ValueError(
                        f"GeoFlow node {step['id']} gives tsp_tw.{name} {len(value)} items for "
                        f"{node_count} nodes"
                    )
            for name in ("start_index", "end_index"):
                value = arguments.get(name)
                if value is None:
                    continue
                index = _static_int(value, name=f"tsp_tw.{name}")
                if index is not None and not 0 <= index < node_count:
                    raise ValueError(
                        f"GeoFlow node {step['id']} gives tsp_tw.{name}={index} for "
                        f"{node_count} nodes"
                    )

        start = _static_int(arguments.get("start_index", 0), name="tsp_tw.start_index")
        end_value = arguments.get("end_index")
        end = (
            _static_int(end_value, name="tsp_tw.end_index")
            if end_value is not None
            else None
        )
        fixed_order = _static_flag(arguments.get("fixed_order", False))
        if fixed_order and end is not None:
            raise ValueError(
                f"GeoFlow node {step['id']} cannot combine tsp_tw.fixed_order with "
                "end_index; the stated sequence already fixes its end"
            )
        returns = _static_flag(arguments.get("return_to_start", False))
        if returns and end is not None:
            if start == end:
                raise ValueError(
                    f"GeoFlow node {step['id']} redundantly sets tsp_tw.end_index to the "
                    "start while return_to_start is true; omit end_index"
                )
            raise ValueError(
                f"GeoFlow node {step['id']} cannot both return to the start and end at "
                "another node"
            )


def _validate_statically_known_reference_shapes(
    steps: list[dict[str, Any]], by_id: dict[str, dict[str, Any]], *, strict_types: bool = True
) -> None:
    """Refuse projections that cannot exist on an output whose shape is already known.

    Most operator output shapes are only known after provider execution, so validation must stay
    lenient about their paths.  A ``batch_geocode`` node is different: it returns one ordered
    record per literal ``place_names`` entry.  Its top-level cardinality and list shape are known
    before execution.  Letting ``$places.3`` through when the node was given three names degraded
    to the whole list in the executor and made the next spatial operator compute over the wrong
    evidence.  Likewise, ``$places.anchor_place`` can never name a field on a list.

    Two of the three refusals here are structural and apply to the final lenient pass as well as
    to strict validation, because each one silently substitutes *different* evidence and no legal
    spelling of the plan produces what the planner asked for: an out-of-range record index, and one
    whole batch reference repeated inside a single list.

    The third is not.  A named field on a batch list -- ``$places.place_ids`` -- degrades in the
    executor to the whole list, which is exactly what the legal spelling ``$places`` resolves to,
    and since ``batch_place_details`` learned to read ids off geocode rows that plan runs and
    answers.  So it is a spelling the repair round should still be asked to fix, and not a graph
    that cannot run: it steps aside on the lenient pass with the rest of this port's own rules.
    Leaving it on cost the four ``unanswerable_*`` families a third to a half of their rows, which
    is where a plan that geocodes options and then wants their details is the standard shape.

    Unknown paths below a valid record (for example Google's ``$places.0.geometry.location``
    spelling) remain deliberately lenient throughout.
    """

    def nested_lists(value: Any) -> list[list[Any]]:
        found: list[list[Any]] = []
        if isinstance(value, list):
            found.append(value)
            for item in value:
                found.extend(nested_lists(item))
        elif isinstance(value, dict):
            for item in value.values():
                found.extend(nested_lists(item))
        return found

    for step in steps:
        # A whole batch node is already a collection. Repeating that same reference inside one
        # list does not name its individual records: after resolution it becomes a list of the
        # same list several times, and the tool's ordinary one-level flattening multiplies N
        # places into N*N. Two benchmark plans wrote four copies of `locations`, producing 16
        # endpoints from four resolved places. Refuse the ambiguous plan rather than silently
        # changing its cardinality. A single whole-list wrapper and distinct/indexed references
        # remain valid, because both have one unambiguous flattening.
        for values in nested_lists(step["arguments"]):
            references = [
                split_reference_arithmetic(canonical_reference(item))[0]
                for item in values
                if isinstance(item, str) and item.strip().startswith("$")
            ]
            for reference in set(references):
                if references.count(reference) < 2:
                    continue
                root = reference[1:] if reference.startswith("$") else ""
                producer = by_id.get(root)
                if producer and producer["operator"] == "batch_geocode":
                    raise ValueError(
                        "Data availability violation: "
                        f"GeoFlow node {step['id']!r} repeats whole batch_geocode reference "
                        f"{reference!r} in one list; use it once or reference individual records"
                    )

        for reference in _reference_strings(step["arguments"]):
            parts = [part for part in reference.lstrip("$").split(".") if part]
            if len(parts) < 2:
                continue
            producer = by_id.get(parts[0])
            if not producer or producer["operator"] != "batch_geocode":
                continue
            projection = parts[1]
            if not projection.isdigit():
                if strict_types:
                    raise ValueError(
                        "Data availability violation: "
                        f"{reference} projects field {projection!r} from batch_geocode list "
                        f"{producer['id']!r}; use a numeric record index first"
                    )
                continue
            names = producer["arguments"].get("place_names")
            if isinstance(names, list) and int(projection) >= len(names):
                raise ValueError(
                    "Data availability violation: "
                    f"{reference} indexes batch_geocode node {producer['id']!r} at "
                    f"{projection}, but it has {len(names)} place_names"
                )


#: Which validation rule belongs to the paper and which is this port's. The distinction is not
#: cosmetic: upstream Spatial-Agent has no operator input-type table, no reference-shape rule and
#: no statically-known-argument check, so a graph refused by one of those is refused by
#: K-MapEval, not by GeoFlow. Any comparison against the paper's numbers has to say which.
#:
#: The paper's G1-G5 refuse on both passes. The port's heuristics refuse strictly, so the repair
#: round is told about them, and step aside on the last attempt -- measured, that is worth 5.7
#: points of pooled accuracy on graphs the executor runs correctly.
PAPER_CONSTRAINTS: dict[str, str] = {
    "G1": "acyclicity",
    "G2": "functional-role ordering on the operator graph",
    "G3": "operator output-type compatibility",
    "G4": "executable operators with their required arguments",
    "G5": "connectivity from contextual input through every node to a measure",
}

PORT_LOCAL_CHECKS: dict[str, str] = {
    "concept_role_ordering": (
        "role ordering over the *concept* graph, whose roles and edges the Analysis stage wrote. "
        "A violation reports disagreement with that stage's labelling, not a fault in the graph "
        "about to execute."
    ),
    "operator_input_types": (
        "OPERATOR_INPUT_TYPES, a declared table of what each operator accepts. Where it and an "
        "implementation disagree the implementation is right."
    ),
    "reference_shapes": (
        "data availability: whether a `$node.field` projection names a field that node produces."
    ),
    "statically_known_values": (
        "argument values derivable before execution, such as a list length that must match."
    ),
}


def normalize_and_validate_graph(
    payload: dict[str, Any], *, max_steps: int, strict_types: bool = True
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    """Normalize a planner graph and refuse the ones that cannot run.

    `strict_types=False` keeps every structural rule -- an unknown operator, a dependency that is
    not a node, a cycle, a graph with no Measure, and the formal constraints the validator
    reports -- and skips the four that are this port's own invention: declared output-type
    compatibility, functional-role ordering, the statically knowable argument values, and a named
    field projected off a batch list, which the executor degrades to the whole list the legal
    spelling would have named anyway.
    Upstream has none of them (there is no type check anywhere in `spatial-agent`), so a graph
    they reject is a graph upstream would have executed, and refusing it measures our validator
    rather than the architecture. They stay on by default because their message is what the
    repair round is given to work with; the lenient pass is the last thing tried before a
    question is given up on.
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
        arguments = _normalize_statically_known_argument_values(operator, arguments)
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
    _validate_statically_known_reference_shapes(steps, by_id, strict_types=strict_types)
    if strict_types:
        # These predict one step's refusal, not a graph that cannot run. The executor already
        # records a step that raises and carries on, and generation answers from whatever did
        # resolve -- four of eighteen terminal failures over three passes were given up on here
        # for one bad argument the run could have absorbed. So they inform the repair round and
        # then step aside on the lenient pass, exactly like the other two rules upstream
        # does not have.
        _validate_statically_known_argument_values(steps, by_id)
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
            # G2 on the executable graph, and never skipped. This is the paper's constraint: the
            # roles here are the ones factorization assigned to operator nodes, so a violation
            # says the graph computes a sub-condition out of a measure -- a real ordering fault,
            # not a disagreement about labels. Over 2,032 recorded questions it was never the
            # last blocker before a lenient pass, so making it unconditional costs nothing and
            # stops a genuine G2 failure from being executed anyway.
            if _violates_procedural_order(by_id[dependency]["role"], step["role"]):
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
            # The *concept* graph's ordering, which is a different claim from the one above: the
            # roles are what the Analysis stage assigned and the edges are what it drew, so a
            # violation reports a disagreement with that stage's labelling rather than a fault in
            # the graph that will execute. It is the single largest reason a question needs the
            # lenient attempt -- 62 of the 122 rescues over 2,032 recorded questions, and those
            # answer 93% correctly. Kept as a repair signal, stepped aside from on the last try.
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
