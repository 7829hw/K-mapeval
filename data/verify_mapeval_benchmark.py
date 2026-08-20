"""Re-derive every v4 gold through the tools the agents themselves call.

Same contract as `data/verify_benchmark.py`, with one addition the other benchmarks have no need
for: the `unanswerable` rows are verified *in the negative*. Their gold is "the map does not say",
so the check is that the provider really does not say it — every candidate in the neighbourhood
carries `None` for the field the question asks about. A row that became answerable because Kakao
started publishing ratings would be a broken question, and this is what catches it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings  # noqa: E402
from src.tools.kakao import KakaoMapProvider  # noqa: E402
from src.tools.registry import ToolRegistry  # noqa: E402
from src.tools.spatial import SpatialOperatorRegistry, haversine_meters  # noqa: E402

DATASET = Path(__file__).resolve().parents[1] / "dataset" / "seoul_kmapeval_v4_mcq_100.jsonl"


def call(registry: ToolRegistry, name: str, **arguments: Any) -> Any:
    execution = registry.invoke(name, arguments)
    if execution.status != "ok":
        raise RuntimeError(f"{name}: {execution.error}")
    return execution.output


def _place(registry: ToolRegistry, name: str) -> dict[str, Any]:
    found = call(registry, "place_search", query=name, limit=1)
    if not found:
        raise RuntimeError(f"unresolved: {name}")
    return found[0]


def _number(text: str) -> float:
    match = re.search(r"([\d.]+)", text)
    return float(match.group(1)) if match else float("nan")


def _minutes(text: str) -> float:
    match = re.search(r"(오전|오후)\s*(\d+)시\s*(\d+)분", text)
    if not match:
        return float("nan")
    hour = int(match.group(2)) % 12
    if match.group(1) == "오후":
        hour += 12
    return hour * 60 + int(match.group(3))


def _nearest_option(options: list[str], value: float, read: Any) -> int | None:
    """The option a solver would pick: the numerically closest one.

    An exact string match would test our rounding rather than the question. The refusal option
    carries no number and is skipped, which is also the right behaviour: a measured answer never
    lands on "the map does not say".
    """

    scored = [(abs(read(option) - value), index) for index, option in enumerate(options)]
    scored = [(gap, index) for gap, index in scored if gap == gap]
    if not scored:
        return None
    scored.sort()
    return scored[0][1]


def _match_name(options: list[str], name: str) -> int | None:
    if not name:
        return None
    for index, option in enumerate(options):
        if option == name:
            return index
    for index, option in enumerate(options):
        if option in name or name in option:
            return index
    return None


def verify(row: dict[str, Any], registry: ToolRegistry, ops: SpatialOperatorRegistry) -> str:
    template = row["template_id"]
    options, gold, evidence = row["options"], row["answer"], row["gold_evidence"]

    if template in ("nearby_clinic_subtype", "nearby_cuisine_subtype"):
        anchor = _place(registry, evidence["anchor"])
        found = call(
            registry,
            "nearby_places",
            center=anchor,
            category_code=evidence["category_code"],
            radius_m=2500,
            limit=45,
        )
        wanted = ops.invoke(
            "filter_places", {"places": found, "required_types": [evidence["required_subtype"]]}
        )
        ranked = ops.invoke("nearest", {"anchor": anchor, "candidates": wanted})
        winner = ranked["nearest"] if isinstance(ranked, dict) else ranked[0]
        index = _match_name(options, winner.get("name", ""))
        return "ok" if index == gold else f"tools chose {index} ({winner.get('name')})"

    if template == "poi_direction_distance":
        start, end = _place(registry, evidence["from"]), _place(registry, evidence["to"])
        heading = ops.invoke("bearing_to_direction", {"place_a": start, "place_b": end})
        metres = haversine_meters(
            start["latitude"], start["longitude"], end["latitude"], end["longitude"]
        )
        expected = f"{heading['direction_ko']}, {round(metres / 1000, 2):g}km"
        index = _nearest_option(
            [option if heading["direction_ko"] in option else "" for option in options],
            metres / 1000,
            _number,
        )
        return "ok" if index == gold else f"tools read {expected} -> option {index}"

    if template == "poi_which_is_closer":
        anchor = _place(registry, evidence["anchor"])
        pair = {key: _place(registry, evidence[key]) for key in ("near", "far")}
        spans = {
            key: haversine_meters(
                anchor["latitude"], anchor["longitude"], place["latitude"], place["longitude"]
            )
            for key, place in pair.items()
        }
        winner = min(spans, key=lambda key: spans[key])
        index = _match_name(options, evidence[winner])
        return "ok" if index == gold else f"tools chose {index} ({evidence[winner]})"

    if template == "poi_straight_distance":
        start, end = _place(registry, evidence["from"]), _place(registry, evidence["to"])
        metres = haversine_meters(
            start["latitude"], start["longitude"], end["latitude"], end["longitude"]
        )
        index = _nearest_option(options, metres / 1000, _number)
        return "ok" if index == gold else f"tools read {metres / 1000:.1f}km -> option {index}"

    if template == "routing_distance_via":
        route = call(
            registry,
            "directions",
            origin=_place(registry, evidence["origin"]),
            destination=_place(registry, evidence["destination"]),
            waypoints=[_place(registry, evidence["via"])],
            priority="DISTANCE",
        )
        kilometres = route["distance_m"] / 1000
        index = _nearest_option(options, kilometres, _number)
        return "ok" if index == gold else f"tools read {kilometres:.1f}km -> option {index}"

    if template == "routing_turn_count":
        route = call(
            registry,
            "directions",
            origin=_place(registry, evidence["origin"]),
            destination=_place(registry, evidence["destination"]),
            priority="DISTANCE",
        )
        analysis = ops.invoke("steps_analysis", {"route": route})
        turns = int(analysis.get("left_turn_count") or 0)
        index = _nearest_option(options, turns, _number)
        return "ok" if index == gold else f"tools counted {turns} -> option {index}"

    if template == "routing_next_turn":
        route = call(
            registry,
            "directions",
            origin=_place(registry, evidence["origin"]),
            destination=_place(registry, evidence["destination"]),
            priority="DISTANCE",
        )
        steps = route.get("steps") or []
        position = evidence["step_index"]
        if position >= len(steps):
            return f"route has {len(steps)} steps, gold names {position}"
        step = steps[position]
        road = (step.get("road_name") or "").strip()
        text = (step.get("instruction") or "").strip()
        rendered = f"{text} ({road})" if road else text
        index = _match_name(options, rendered)
        return "ok" if index == gold else f"tools read {rendered!r} -> option {index}"

    if template == "trip_arrival_clock":
        chain = [evidence["base"], *evidence["stops"]]
        result = call(
            registry,
            "calculate_finish_time",
            start_time="2026-05-12T09:00:00",
            locations=chain,
            stay_durations_s=[0, *evidence["stay_s"]],
            timezone="Asia/Seoul",
            priority="DISTANCE",
        )
        finish = datetime.fromisoformat(result["finish_time"])
        index = _nearest_option(options, finish.hour * 60 + finish.minute, _minutes)
        return "ok" if index == gold else f"tools read {finish:%H:%M} -> option {index}"

    if template == "trip_total_distance":
        chain = evidence["chain"]
        total = 0.0
        for origin, destination in zip(chain, chain[1:], strict=False):
            route = call(
                registry,
                "directions",
                origin=_place(registry, origin),
                destination=_place(registry, destination),
                priority="DISTANCE",
            )
            total += route["distance_m"]
        index = _nearest_option(options, total / 1000, _number)
        return "ok" if index == gold else f"tools read {total / 1000:.1f}km -> option {index}"

    if template == "trip_feasible_count":
        chain = [evidence["base"], *evidence["stops"]]
        spent = 0.0
        count = 0
        for index, (origin, destination) in enumerate(
            zip(chain, chain[1:], strict=False)
        ):
            route = call(
                registry,
                "directions",
                origin=_place(registry, origin),
                destination=_place(registry, destination),
                priority="DISTANCE",
            )
            spent += route["duration_s"] + evidence["stay_s"][index]
            if spent > evidence["budget_s"]:
                break
            count += 1
        return "ok" if count - 1 == gold else f"tools fit {count} -> option {count - 1}"

    if template.startswith("unanswerable_"):
        anchor = _place(registry, evidence["anchor"])
        found = call(
            registry,
            "nearby_places",
            center=anchor,
            category_code=evidence["category_code"],
            radius_m=1500,
            limit=15,
        )
        field = evidence["missing_field"]
        published = [
            place.get("name")
            for place in found
            if field in place and place.get(field) is not None
        ]
        if published:
            return f"provider now publishes {field}: {published[:3]}"
        return "ok" if "알 수 없음" in options[gold] else "gold is not the refusal option"

    return "no verifier"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DATASET))
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
    rows = [
        json.loads(line)
        for line in Path(args.dataset).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    tally: Counter[str] = Counter()
    failures: list[tuple[str, str, str]] = []
    for row in rows:
        try:
            verdict = verify(row, registry, ops)
        except Exception as error:  # noqa: BLE001 - a verifier crash is a verification failure
            verdict = f"{type(error).__name__}: {error}"
        bucket = "ok" if verdict == "ok" else "MISMATCH"
        tally[bucket] += 1
        if bucket == "MISMATCH":
            failures.append((row["id"], row["template_id"], verdict))
        print(f"{row['id']} {row['template_id']:26s} {verdict}", flush=True)

    print(f"\nderivable {tally['ok']}/{len(rows)}")
    for question_id, template, verdict in failures:
        print(f"  ! {question_id} [{template}] {verdict}")
    provider.close()


if __name__ == "__main__":
    main()
