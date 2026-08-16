from __future__ import annotations

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


OPERATOR_CONTRACTS: dict[str, OperatorContract] = {
    "place_search": OperatorContract("object", ("query",)),
    "batch_geocode": OperatorContract("object", ("place_names",)),
    "geocode": OperatorContract("location", ("address",)),
    "place_details": OperatorContract("object", ("place_id",)),
    "nearby_places": OperatorContract("object", ("center",)),
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
}

OPERATOR_INPUT_TYPES: dict[str, dict[str, frozenset[str]]] = {
    "batch_geocode": {"anchor": frozenset({"location", "object"})},
    "place_details": {"place_id": frozenset({"object"})},
    "nearby_places": {"center": frozenset({"location", "object"})},
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
}


TEMPLATES = {
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
                        "limit": 15,
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
            },
            {
                "id": "requested_answer",
                "text": str(payload.get("measure") or "answer choice"),
                "concept_type": "amount",
                "role": "measure",
                "attributes": {},
            },
        ]
    return {"intent": intent, "concepts": concepts, "measure": payload.get("measure", intent)}


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
        inferred = _reference_roots(arguments)
        declared = raw.get("depends_on") or raw.get("before") or []
        if not isinstance(declared, list):
            raise ValueError(f"GeoFlow node {step_id} depends_on must be a list")
        dependencies = list(dict.fromkeys([*(str(value) for value in declared), *inferred]))
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
    measures = {step["id"] for step in steps if step["role"] == "measure"}
    if not measures:
        raise ValueError("GeoFlow graph has no Measure node")
    outgoing: dict[str, set[str]] = {step["id"]: set() for step in steps}
    for step in steps:
        for dependency in step["depends_on"]:
            outgoing[dependency].add(step["id"])
    for step in steps:
        if not _reaches_measure(step["id"], outgoing, measures):
            raise ValueError(
                f"Disconnected GeoFlow node does not contribute to a Measure: {step['id']}"
            )

    constraints = {
        "acyclicity": True,
        "role_ordering": True,
        "type_compatibility": True,
        "data_availability": True,
        "connectivity": True,
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
    elif isinstance(value, str) and value.startswith("$"):
        roots.append(value[1:].split(".", 1)[0])
    return list(dict.fromkeys(roots))


def _references_by_argument(arguments: dict[str, Any]) -> dict[str, list[str]]:
    return {name: _reference_roots(value) for name, value in arguments.items()}


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
