"""Re-derive every gold answer through the tool path the agents themselves use.

A benchmark whose answers cannot be reached with `ToolRegistry` + `SpatialOperatorRegistry` is
measuring the evidence, not the architecture. This runs the deterministic pipeline each family
implies and reports the questions where the tools disagree with the stored gold.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings  # noqa: E402
from src.tools.kakao import KakaoMapProvider  # noqa: E402
from src.tools.registry import ToolRegistry  # noqa: E402
from src.tools.spatial import SpatialOperatorRegistry  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_core import wa  # noqa: E402

DATASET = Path(__file__).resolve().parents[1] / "dataset" / "seoul_kmapeval_v2_mcq_100.jsonl"

CATEGORY_OF_NOUN = {
    "관광명소": "AT4", "문화시설": "CT1", "숙박시설": "AD5", "지하철역": "SW8",
    "카페": "CE7", "음식점": "FD6", "대형마트": "MT1", "편의점": "CS2",
    "은행": "BK9", "공공기관": "PO3", "학교": "SC4", "병원": "HP8", "주차장": "PK6",
}


def call(registry: ToolRegistry, name: str, **arguments: Any) -> Any:
    execution = registry.invoke(name, arguments)
    if execution.status != "ok":
        raise RuntimeError(f"{name}: {execution.error}")
    return execution.output


def geocode_all(registry: ToolRegistry, names: list[str], anchor: str | None = None) -> list[Any]:
    payload: dict[str, Any] = {"place_names": names}
    if anchor:
        payload["anchor"] = anchor
    return call(registry, "batch_geocode", **payload)


def _resolved(entry: Any) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        return None
    place = entry.get("place") or entry
    return place if isinstance(place, dict) and "latitude" in place else None


def verify_option_match(
    ops: SpatialOperatorRegistry, options: list[str], winner: dict[str, Any]
) -> int | None:
    """Score the winning place against the option texts the way the agents' Measure stage does."""

    matched = ops.invoke("match_options", {"options": options, "places": [winner]})
    index = matched.get("best_option") if isinstance(matched, dict) else None
    return index if isinstance(index, int) else None


def _verify_feasible_count(
    registry: ToolRegistry,
    ops: SpatialOperatorRegistry,
    evidence: dict[str, Any],
    options: list[str],
    gold: int,
) -> str:
    """Re-derive "how many places fit" with `distance_matrix` + `tsp_tw`, the agents' own path."""

    base = evidence["base"]
    stops = evidence.get("stops")
    if not stops:
        return "no stops recorded"
    nodes = [base, *stops]
    pairs = [
        {"origin": origin, "destination": destination}
        for origin in nodes
        for destination in nodes
        if origin != destination
    ]
    matrix_result = call(registry, "distance_matrix", pairs=pairs, priority="TIME")
    by_pair = {
        (entry["origin"], entry["destination"]): entry.get("duration_s")
        for entry in matrix_result["routes"]
        if entry.get("status") == "ok"
    }
    size = len(nodes)
    matrix = [
        [
            0.0 if i == j else by_pair.get((nodes[i], nodes[j]), float("inf"))
            for j in range(size)
        ]
        for i in range(size)
    ]
    if any(value == float("inf") for row in matrix for value in row):
        return "unroutable legs"
    stays = [0.0, *evidence["stay_s"]]
    budget = evidence["budget_s"]
    best = 0
    for count in range(1, size):
        for subset in itertools.combinations(range(1, size), count):
            plan = ops.invoke(
                "tsp_tw",
                {
                    "nodes": [{"name": nodes[index]} for index in (0, *subset)],
                    "distance_matrix": [
                        [matrix[i][j] for j in (0, *subset)] for i in (0, *subset)
                    ],
                    "service_times": [stays[index] for index in (0, *subset)],
                    "start_index": 0,
                    "time_budget": budget,
                },
            )
            if plan.get("feasible") and len(plan.get("order", [])) == count + 1:
                best = max(best, count)
    return "ok" if options[gold] == ("한 곳", "두 곳", "세 곳", "네 곳")[best - 1] else (
        f"tools fit {best}"
    )


# `Builder.route` routes with DISTANCE, and every routing family's gold is a property of *that*
# route: which turns come in which order, and how many of them are left. Verifying through the
# tool's own RECOMMEND default asked about a different route, one that re-optimizes against live
# traffic — so eleven rows drifted out of agreement the first time traffic moved, none of them
# because an answer had changed. Ask for the route the gold was built on.
def verify(row: dict[str, Any], registry: ToolRegistry, ops: SpatialOperatorRegistry) -> str:
    template = row["template_id"]
    options = row["options"]
    gold = row["answer"]
    evidence = row["gold_evidence"]
    question = row["question"]

    if template in ("nearby_nearest_by_type", "direction_nearest_by_type", "radius_within_by_type"):
        anchor_name = question.split("에서")[0]
        anchor_hits = call(registry, "place_search", query=anchor_name, limit=1)
        if not anchor_hits:
            return "anchor unresolved"
        anchor = anchor_hits[0]
        found = call(
            registry,
            "nearby_places",
            center=anchor,
            category_code=evidence["category"],
            radius_m=3000,
            limit=15,
        )
        if template == "direction_nearest_by_type":
            found = ops.invoke(
                "filter_by_direction",
                {"center": anchor, "places": found, "direction": evidence["direction"]},
            )
        if template == "radius_within_by_type":
            found = ops.invoke(
                "within_radius",
                {"center": anchor, "candidates": found, "radius_m": evidence["radius_m"]},
            )
        if not found:
            return "no candidates"
        ranked = ops.invoke("nearest", {"anchor": anchor, "candidates": found})
        winner = ranked[0] if isinstance(ranked, list) else ranked
        index = verify_option_match(ops, options, winner)
        return "ok" if index == gold else f"tools chose {index} ({winner.get('name')})"

    if template == "poi_farthest_pair":
        pairs = evidence["pairs"]
        names = [name for pair in pairs for name in pair]
        places = geocode_all(registry, names)
        resolved = [_resolved(entry) for entry in places]
        if any(place is None for place in resolved):
            return "unresolved endpoint"
        best, best_index = -1.0, None
        for index in range(4):
            a, b = resolved[index * 2], resolved[index * 2 + 1]
            value = ops.invoke("haversine_distance", {"place_a": a, "place_b": b})["distance_m"]
            if value > best:
                best, best_index = value, index
        # `pairs` is stored in the order the gold was scored in, before options were shuffled, so
        # an index into it is not an option index. The rendered text is what identifies the option.
        chosen = pairs[best_index]
        rendered = f"{wa(chosen[0])} {chosen[1]}"
        return "ok" if options[gold] == rendered else f"tools chose {rendered!r}"

    if template in ("poi_between", "poi_common_nearby"):
        head = question.split("에서")[0] if template == "poi_between" else None
        if template == "poi_between":
            anchors = [head, question.split("에서")[1].split("까지")[0]]
        else:
            joiner = "과 " if "과 " in question else "와 "
            first, rest = question.split(joiner, 1)
            anchors = [first, rest.split(" 양쪽")[0]]
        places = geocode_all(registry, [*anchors, *options])
        resolved = [_resolved(entry) for entry in places]
        if any(place is None for place in resolved):
            return "unresolved endpoint"
        a, b = resolved[0], resolved[1]
        candidates = resolved[2:]
        def gap(left: dict[str, Any], right: dict[str, Any]) -> float:
            return ops.invoke(
                "haversine_distance", {"place_a": left, "place_b": right}
            )["distance_m"]

        scores = [(gap(a, candidate), gap(b, candidate)) for candidate in candidates]
        if template == "poi_between":
            span = gap(a, b)
            best_index = min(range(4), key=lambda i: (scores[i][0] + scores[i][1]) / span)
        else:
            radius = evidence["radius_m"]
            inside = [i for i, (da, db) in enumerate(scores) if da <= radius and db <= radius]
            best_index = inside[0] if len(inside) == 1 else None
        return "ok" if best_index == gold else f"tools chose {best_index}"

    if template == "routing_via_compare":
        origin, rest = question.split("에서", 1)
        destination = rest.split("까지")[0]
        durations = []
        for via in options:
            route = call(
                registry,
                "directions",
                origin=origin,
                destination=destination,
                waypoints=[via],
                priority="TIME",
            )
            durations.append(route["duration_s"])
        best_index = min(range(4), key=lambda i: durations[i])
        return "ok" if best_index == gold else f"tools chose {best_index} {durations}"

    if template == "routing_next_turn":
        origin, rest = question.split("에서", 1)
        destination = rest.split("까지")[0]
        route = call(
            registry,
            "directions",
            origin=origin,
            destination=destination,
            include_steps=True,
            priority="DISTANCE",
        )
        steps = route["steps"]
        target = evidence["step_index"]
        if target >= len(steps):
            return "route changed shape"
        step = steps[target]
        text = step["instruction"]
        if step["road_name"]:
            text = f"{text} ({step['road_name']})"
        return "ok" if options[gold] == text else f"tools read {text!r}"

    if template == "routing_turn_count":
        origin, rest = question.split("에서", 1)
        via = rest.split("을 경유")[0].split("를 경유")[0]
        destination = rest.split("경유하여 ")[1].split("까지")[0]
        route = call(
            registry,
            "directions",
            origin=origin,
            destination=destination,
            waypoints=[via],
            include_steps=True,
            priority="DISTANCE",
        )
        turns = sum(1 for step in route["steps"] if "좌회전" in step["instruction"])
        return "ok" if options[gold] == f"{turns}번" else f"tools counted {turns}"

    if template in ("trip_optimal_order", "trip_feasible_count"):
        base = evidence["base"]
        if template == "trip_feasible_count":
            return _verify_feasible_count(registry, ops, evidence, options, gold)
        pairs = []
        labels = []
        for order in options:
            names = order.split(" → ")
            path = [base, *names]
            for index in range(3):
                pairs.append({"origin": path[index], "destination": path[index + 1]})
                labels.append((order, index))
        # `Builder.duration_s` routes with TIME, so the gold is the fastest route's duration.
        matrix = call(registry, "distance_matrix", pairs=pairs, priority="TIME")
        by_pair = {
            (entry["origin"], entry["destination"]): entry.get("duration_s")
            for entry in matrix["routes"]
            if entry.get("status") == "ok"
        }
        totals = []
        for order in options:
            path = [base, *order.split(" → ")]
            legs = [by_pair.get((path[i], path[i + 1])) for i in range(3)]
            totals.append(None if any(leg is None for leg in legs) else sum(legs))
        if any(total is None for total in totals):
            return "unroutable legs"
        best_index = min(range(4), key=lambda i: totals[i])
        return "ok" if best_index == gold else f"tools chose {best_index} {totals}"

    return "no verifier"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    settings = Settings()
    provider = KakaoMapProvider(
        settings.kakao_rest_api_key,
        timeout=settings.kakao_timeout_seconds,
        cache_path=settings.kakao_cache_db_path,
        cache_ttl_seconds=settings.kakao_cache_ttl_seconds,
        search_center=settings.search_center(),
        search_radius_m=settings.kakao_search_radius_m,
    )
    registry = ToolRegistry(provider)
    ops = SpatialOperatorRegistry()
    lines = Path(args.dataset).read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    if args.limit:
        rows = rows[: args.limit]

    tally: Counter[str] = Counter()
    failures: list[tuple[str, str, str]] = []
    for row in rows:
        try:
            verdict = verify(row, registry, ops)
        except Exception as error:  # noqa: BLE001 - a verifier crash is a verification failure
            verdict = f"{type(error).__name__}: {error}"
        bucket = "ok" if verdict.startswith(("ok", "checked")) else "MISMATCH"
        tally[bucket] += 1
        tally[f"{row['template_id']}:{bucket}"] += 1
        if bucket == "MISMATCH":
            failures.append((row["id"], row["template_id"], verdict))
        print(f"{row['id']} {row['template_id']:26s} {verdict}", flush=True)

    print(f"\nderivable {tally['ok']}/{len(rows)}")
    for question_id, template, verdict in failures:
        print(f"  ! {question_id} [{template}] {verdict}")
    provider.close()


if __name__ == "__main__":
    main()
