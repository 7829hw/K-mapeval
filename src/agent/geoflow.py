from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

CORE_CONCEPTS = frozenset(
    {"location", "object", "field", "event", "network", "amount", "proportion"}
)
FUNCTIONAL_ROLES = frozenset(
    {"extent", "temporal_extent", "sub_condition", "condition", "support", "measure"}
)
ROLE_PRIORITY = {
    "extent": 0,
    "temporal_extent": 0,
    "sub_condition": 1,
    "condition": 2,
    "support": 3,
    "measure": 4,
}


@dataclass(frozen=True)
class OperatorContract:
    output_type: str
    required_arguments: tuple[str, ...] = ()


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

    def as_dict(self) -> dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "input_concepts": list(self.input_concepts),
            "output_bindings": list(self.output_bindings),
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
    "identity_measure": OperatorContract("object", ("value",)),
    "place_search": OperatorContract("object", ("query",)),
    "batch_geocode": OperatorContract("object", ("place_names",)),
    "geocode": OperatorContract("location", ("address",)),
    "place_details": OperatorContract("object", ("place_id",)),
    "nearby_places": OperatorContract("object", ("center",)),
    "recover_option_places": OperatorContract(
        "object", ("options", "candidates", "anchor")
    ),
    "directions": OperatorContract("field", ("origin", "destination")),
    "travel_time": OperatorContract("field", ("origin", "destination")),
    "distance_matrix": OperatorContract("field"),
    "haversine_distance": OperatorContract("amount"),
    "pairwise_distances": OperatorContract("field", ("pairs",)),
    "bearing_to_direction": OperatorContract("field"),
    "filter_by_direction": OperatorContract("object", ("center", "places", "direction")),
    "nearest": OperatorContract("object", ("anchor", "candidates")),
    "within_radius": OperatorContract("object", ("center", "candidates", "radius_m")),
    "select_min": OperatorContract("object", ("items", "key")),
    "select_max": OperatorContract("object", ("items", "key")),
    "sort_by": OperatorContract("object", ("items", "key")),
    "compare_routes": OperatorContract("object", ("routes",)),
    "sum_route_metrics": OperatorContract("amount", ("routes",)),
    "aggregate_route_groups": OperatorContract("amount", ("routes", "groups")),
    "merge_places": OperatorContract("object", ("items",)),
    "match_options": OperatorContract("object", ("options", "places")),
    "match_distance_options": OperatorContract("object", ("distance", "options")),
    "match_type_options": OperatorContract("object", ("place", "options")),
    "events_from_objects": OperatorContract("event", ("objects",)),
    "filter_events": OperatorContract("event", ("events", "field", "operator", "value")),
    "build_route_network": OperatorContract("network", ("nodes", "edges")),
    "calculate_proportion": OperatorContract("proportion", ("numerator", "denominator")),
    "open_at_time": OperatorContract("event", ("schedule", "local_time", "timezone")),
    "timezone_convert": OperatorContract("event", ("local_time", "from_timezone", "to_timezone")),
    "calculate_finish_time": OperatorContract(
        "event", ("start_time", "duration_s", "timezone")
    ),
    "tsp_tw": OperatorContract("network", ("nodes", "distance_matrix")),
}

OPERATOR_INPUT_TYPES: dict[str, dict[str, frozenset[str]]] = {
    "identity_measure": {
        "value": frozenset(CORE_CONCEPTS),
    },
    "batch_geocode": {"anchor": frozenset({"location", "object"})},
    "place_details": {"place_id": frozenset({"object"})},
    "nearby_places": {"center": frozenset({"location", "object"})},
    "recover_option_places": {
        "candidates": frozenset({"object"}),
        "anchor": frozenset({"location", "object"}),
    },
    "directions": {
        "origin": frozenset({"location", "object"}),
        "destination": frozenset({"location", "object"}),
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
    },
    "within_radius": {
        "center": frozenset({"location", "object"}),
        "candidates": frozenset({"object"}),
    },
    "select_min": {"items": frozenset({"object", "field", "amount"})},
    "select_max": {"items": frozenset({"object", "field", "amount"})},
    "sort_by": {"items": frozenset({"object", "field", "amount"})},
    "compare_routes": {"routes": frozenset({"field"})},
    "sum_route_metrics": {"routes": frozenset({"field"})},
    "aggregate_route_groups": {"routes": frozenset({"field"})},
    "merge_places": {"items": frozenset({"object"})},
    "match_options": {
        "places": frozenset({"object"}),
        "anchor": frozenset({"object", "location"}),
    },
    "match_distance_options": {"distance": frozenset({"amount"})},
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
        "start_time": frozenset({"event", "field"}),
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
        "intents": {"poi", "type"},
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
                    "role": "measure",
                }
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
                    "role": "support",
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
                    "role": "support",
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
    "radius": {
        "name": "Filter-Aggregate-Measure",
        "intents": {"nearby", "radius"},
        "keywords": ("반경", "이내", "within", "radius"),
        "pattern": "nearby_places(center, exact radius and category/keyword) -> Measure",
        "example": {
            "graph": [
                {
                    "id": "nearby",
                    "operator": "nearby_places",
                    "arguments": {
                        "center": "서울역",
                        "query": "편의점",
                        "radius_m": 500,
                        "limit": 45,
                    },
                    "depends_on": [],
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
                    "role": "measure",
                }
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
                    "role": "support",
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
        "pattern": "locations -> route network -> tsp_tw(time windows) -> Measure",
        "example": {
            "graph": [
                {
                    "id": "locations",
                    "operator": "batch_geocode",
                    "arguments": {"place_names": ["출발지", "방문지 1", "방문지 2"]},
                    "role": "extent",
                },
                {
                    "id": "optimized",
                    "operator": "tsp_tw",
                    "arguments": {
                        "nodes": "$locations",
                        "distance_matrix": [[0, 10, 20], [10, 0, 5], [20, 5, 0]],
                    },
                    "depends_on": ["locations"],
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
                    },
                    "depends_on": ["endpoints"],
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
                    "id": "finish",
                    "operator": "calculate_finish_time",
                    "arguments": {
                        "start_time": "$time_context.converted_time",
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
    return [
        {"name": template["name"], "pattern": template["pattern"], "example": template["example"]}
        for _, _, template in ranked[:limit]
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
    return {"intent": intent, "concepts": concepts, "measure": payload.get("measure", intent)}


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
    if not edges:
        edges = _infer_concept_edges(nodes)
    return ConceptGraph(nodes=nodes, edges=tuple(dict.fromkeys(edges)))


def factorize_geoflow(analysis: dict[str, Any], payload: dict[str, Any]) -> FactorizedGeoFlow:
    """Map concept graph G to an executable operator-concept hypergraph G'.

    Each operator hyperedge records all input concepts and one or more output bindings. Literal
    source operators are explicitly bound to EXTENT/TEXTENT concepts instead of being implicit
    roots, and every original analysis concept remains bound to at least one operator output.
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
        dependencies.extend(_reference_roots(raw.get("arguments") or raw.get("params") or {}))
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
        role = _factorized_role(
            step_id,
            operator,
            str(step.get("role") or ""),
            source_ids,
            sink_ids,
        )
        step["role"] = role
        explicit_ids = step.get("concept_ids") or []
        concept_ids = [str(value) for value in explicit_ids if str(value) in concepts]
        if not concept_ids:
            available = [value for value in role_buckets[role] if value not in bound_concepts]
            concept_ids = available or list(role_buckets[role][:1])
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

    for concept_id, concept in concepts.items():
        if concept_id in bound_concepts:
            continue
        candidates = [step for step in graph if step["role"] == concept.role]
        if not candidates:
            candidates = graph[-1:] if concept.role == "measure" else graph[:1]
        candidates[0]["concept_ids"].append(concept_id)
        step_concepts[candidates[0]["id"]].append(concept_id)
        bound_concepts.add(concept_id)

    hyperedges: list[OperatorHyperedge] = []
    concept_edges = list(concept_graph.edges)
    for step in graph:
        input_concepts = list(
            dict.fromkeys(
                concept_id
                for dependency in step["depends_on"]
                for concept_id in step_concepts.get(dependency, [])
            )
        )
        bindings = step.get("output_bindings")
        if not isinstance(bindings, list) or not bindings:
            bindings = [
                {"concept_id": concept_id, "path": "$"}
                for concept_id in step["concept_ids"]
            ]
        step["input_concepts"] = input_concepts
        step["output_bindings"] = bindings
        hyperedges.append(
            OperatorHyperedge(
                operator_id=step["id"],
                input_concepts=tuple(input_concepts),
                output_bindings=tuple(dict(binding) for binding in bindings),
            )
        )
        for source in input_concepts:
            concept_edges.extend((source, target) for target in step["concept_ids"])

    complete_graph = ConceptGraph(
        nodes=tuple(concepts.values()),
        edges=tuple(dict.fromkeys(concept_edges)),
    )
    return FactorizedGeoFlow(complete_graph, tuple(graph), tuple(hyperedges))


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
    if not any(concept.get("depends_on") for concept in completed):
        nodes = tuple(
            ConceptNode(
                id=str(concept["id"]),
                text=str(concept.get("text") or ""),
                concept_type=str(concept["concept_type"]),
                role=str(concept["role"]),
                attributes=dict(concept.get("attributes") or {}),
            )
            for concept in completed
        )
        inferred = _infer_concept_edges(nodes)
        incoming: dict[str, list[str]] = {node.id: [] for node in nodes}
        for source, target in inferred:
            incoming[target].append(source)
        for concept in completed:
            concept["depends_on"] = incoming[str(concept["id"])]
    return completed


def _infer_concept_edges(nodes: tuple[ConceptNode, ...]) -> list[tuple[str, str]]:
    levels: dict[int, list[str]] = {}
    for node in nodes:
        levels.setdefault(ROLE_PRIORITY[node.role], []).append(node.id)
    ordered_levels = sorted(levels)
    edges: list[tuple[str, str]] = []
    for position in range(1, len(ordered_levels)):
        sources = levels[ordered_levels[position - 1]]
        targets = levels[ordered_levels[position]]
        edges.extend((source, target) for source in sources for target in targets)
    return edges


def _factorized_role(
    step_id: str,
    operator: str,
    declared_role: str,
    source_ids: set[str],
    sink_ids: set[str],
) -> str:
    if step_id in source_ids:
        if operator in {"open_at_time", "calculate_finish_time", "timezone_convert"}:
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


def normalize_and_validate_graph(
    payload: dict[str, Any], *, max_steps: int
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
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
        contract = OPERATOR_CONTRACTS.get(operator)
        if contract is None:
            raise ValueError(f"Unknown GeoFlow operator: {operator}")
        arguments = raw.get("arguments")
        if arguments is None:
            arguments = raw.get("params")
        if not isinstance(arguments, dict):
            raise ValueError(f"GeoFlow node {step_id} arguments must be an object")
        missing = [name for name in contract.required_arguments if name not in arguments]
        if missing:
            raise ValueError(f"GeoFlow node {step_id} is missing arguments: {', '.join(missing)}")
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
        declared = raw.get("depends_on") or raw.get("before") or []
        if not isinstance(declared, list):
            raise ValueError(f"GeoFlow node {step_id} depends_on must be a list")
        declared_dependencies = [_normalize_dependency(value, raw_ids) for value in declared]
        arguments = _rewrite_placeholder_references(
            arguments,
            known_ids=known_ids,
            declared_dependencies=declared_dependencies,
        )
        inferred = _reference_roots(arguments)
        dependencies = list(dict.fromkeys([*declared_dependencies, *inferred]))
        output_type = str(raw.get("output_type") or contract.output_type).lower()
        if output_type != contract.output_type:
            raise ValueError(
                f"GeoFlow node {step_id} declares {output_type}, but {operator} outputs "
                f"{contract.output_type}"
            )
        default_role = "measure" if index == len(raw_steps) - 1 else "support"
        role = str(raw.get("role") or default_role).lower()
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
    for step in steps:
        for dependency in step["depends_on"]:
            if dependency not in by_id:
                raise ValueError(f"Unknown dependency {dependency!r} on GeoFlow node {step['id']}")
            if ROLE_PRIORITY[by_id[dependency]["role"]] > ROLE_PRIORITY[step["role"]]:
                raise ValueError(
                    f"Role ordering violation: {dependency} ({by_id[dependency]['role']}) -> "
                    f"{step['id']} ({step['role']})"
                )
        accepted_inputs = OPERATOR_INPUT_TYPES.get(step["operator"], {})
        for argument_name, references in _references_by_argument(step["arguments"]).items():
            accepted = accepted_inputs.get(argument_name)
            if not accepted:
                continue
            for dependency in references:
                dependency_type = by_id[dependency]["output_type"]
                if dependency_type not in accepted:
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
        bound_ids = {concept_id for step in steps for concept_id in step["concept_ids"]}
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
        for edge in concept_graph_payload.get("edges") or []:
            if not isinstance(edge, dict):
                continue
            source, target = str(edge.get("source") or ""), str(edge.get("target") or "")
            if source not in concept_by_id or target not in concept_by_id:
                raise ValueError(
                    f"Concept graph edge references an unknown node: {source}->{target}"
                )
            source_role = str(concept_by_id[source].get("role") or "support")
            target_role = str(concept_by_id[target].get("role") or "support")
            if ROLE_PRIORITY[source_role] > ROLE_PRIORITY[target_role]:
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


def _reference_roots(value: Any) -> list[str]:
    roots: list[str] = []
    if isinstance(value, dict):
        for item in value.values():
            roots.extend(_reference_roots(item))
    elif isinstance(value, list):
        for item in value:
            roots.extend(_reference_roots(item))
    elif isinstance(value, str):
        reference = canonical_reference(value)
        if reference.startswith("$"):
            roots.append(reference[1:].split(".", 1)[0])
    return list(dict.fromkeys(roots))


def _references_by_argument(arguments: dict[str, Any]) -> dict[str, list[str]]:
    return {name: _reference_roots(value) for name, value in arguments.items()}


def canonical_reference(value: str) -> str:
    """Normalize common LLM reference spellings into the GeoFlow `$node.path` form."""

    reference = value.strip()
    if reference.startswith("${") and reference.endswith("}"):
        reference = "$" + reference[2:-1]
    return re.sub(r"\[(\d+)]", r".\1", reference)


def _normalize_dependency(value: Any, raw_ids: list[str]) -> str:
    if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
        index = int(value)
        if 0 <= index < len(raw_ids):
            return raw_ids[index]
    return str(value)


def _rewrite_placeholder_references(
    value: Any,
    *,
    known_ids: set[str],
    declared_dependencies: list[str],
) -> Any:
    if isinstance(value, dict):
        return {
            key: _rewrite_placeholder_references(
                item,
                known_ids=known_ids,
                declared_dependencies=declared_dependencies,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _rewrite_placeholder_references(
                item,
                known_ids=known_ids,
                declared_dependencies=declared_dependencies,
            )
            for item in value
        ]
    if not isinstance(value, str):
        return value
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
                ROLE_PRIORITY[by_id[step_id]["role"]],
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
