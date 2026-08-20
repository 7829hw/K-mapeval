"""Re-derive every v3 gold answer through the tools the agents themselves call.

The compositional families are the ones whose paths were broken longest (`tsp_tw` could not be
fed at all until recently), so a v3 answer is only trustworthy once the operator chain it names
actually produces it.
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
from src.tools.spatial import SpatialOperatorRegistry  # noqa: E402

DATASET = Path(__file__).resolve().parents[1] / "dataset" / "seoul_kmapeval_v3_mcq_100.jsonl"


def _nearest_option(options: list[str], value: float, read: Any) -> int | None:
    """Pick the option a solver would pick: the numerically closest one.

    Routing priority changes a duration by minutes — `calculate_finish_time` defaults to TIME
    while a bare `directions` call defaults to RECOMMEND — so an exact string match tests the
    provider's tie-breaking, not the question. The options are tens of minutes apart precisely
    so that difference cannot reach the answer.
    """

    scored = [(abs(read(option) - value), index) for index, option in enumerate(options)]
    scored = [(gap, index) for gap, index in scored if gap == gap]
    if not scored:
        return None
    scored.sort()
    return scored[0][1]


def _minutes(text: str) -> float:
    match = re.search(r"(오전|오후)\s*(\d+)시\s*(\d+)분", text)
    if not match:
        return float("nan")
    hour = int(match.group(2)) % 12
    if match.group(1) == "오후":
        hour += 12
    return hour * 60 + int(match.group(3))


def _number(text: str) -> float:
    match = re.search(r"([\d.]+)", text)
    return float(match.group(1)) if match else float("nan")


def call(registry: ToolRegistry, name: str, **arguments: Any) -> Any:
    execution = registry.invoke(name, arguments)
    if execution.status != "ok":
        raise RuntimeError(f"{name}: {execution.error}")
    return execution.output


def _clock(moment: datetime) -> str:
    hour, minute = moment.hour, moment.minute
    period = "오전" if hour < 12 else "오후"
    return f"{period} {hour if hour <= 12 else hour - 12}시 {minute:02d}분"


def _ref(registry: ToolRegistry, name: str) -> Any:
    """Resolve a name the way an agent must: `place_search` first, then pass what came back.

    A place argument on a baseline tool is a reference the provider issued, so the verifier
    threads the resolved place rather than the name — the same discipline
    `mapeval-api/FormattedTools.py` imposes on its own tools.
    """

    found = call(registry, "place_search", query=name.strip(), limit=1)
    if not found:
        raise RuntimeError(f"unresolved: {name}")
    # `place_search` hands back a reference, as upstream's PlaceSearchTool does.
    return found[0]["place_id"]


def verify(row: dict[str, Any], registry: ToolRegistry, ops: SpatialOperatorRegistry) -> str:
    from src.agent.spatial import _extract_route_priority

    template = row["template_id"]
    # The question names its route; grounding binds that for the agent, so the check does too.
    priority = _extract_route_priority(row["question"]) or "RECOMMEND"
    options, gold, evidence = row["options"], row["answer"], row["gold_evidence"]

    if template == "trip_finish_time":
        chain = [evidence["base"], *evidence["stops"], evidence["base"]]
        result = call(
            registry,
            "calculate_finish_time",
            start_time=evidence["depart"],
            locations=chain,
            # One stay per location, in order: nothing at the start, the stated stay at each
            # stop, nothing on the way back. A shorter list is genuinely ambiguous about which
            # end the missing entries belong to, which is why the tool refuses it.
            stay_durations_s=[0, *evidence["stay_s"], 0],
            timezone="Asia/Seoul",
            priority=priority,
        )
        finish = datetime.fromisoformat(result["finish_time"]).replace(second=0, microsecond=0)
        choice = _nearest_option(options, finish.hour * 60 + finish.minute, _minutes)
        return "ok" if choice == gold else f"tools read {_clock(finish)} -> option {choice}"

    if template == "trip_latest_departure":
        chain = [evidence["base"], *evidence["errands"], evidence["target"]]
        matrix = call(
            registry,
            "distance_matrix",
            pairs=[
                {"origin": a, "destination": b}
                for a, b in zip(chain, chain[1:], strict=False)
            ],
            priority=priority,
        )
        legs = [entry.get("duration_s") for entry in matrix["routes"]]
        if any(value is None for value in legs):
            return "unroutable legs"
        total = sum(legs) + sum(evidence["stay_s"])
        start = ops.invoke(
            "calculate_start_time",
            {
                "arrival_time": evidence["arrival"],
                "duration_s": total,
                "timezone": "Asia/Seoul",
            },
        )
        moment = datetime.fromisoformat(start["start_time"]).replace(second=0, microsecond=0)
        choice = _nearest_option(options, moment.hour * 60 + moment.minute, _minutes)
        return "ok" if choice == gold else f"tools read {_clock(moment)} -> option {choice}"

    if template == "multisegment_total":
        chain = evidence["chain"]
        matrix = call(
            registry,
            "distance_matrix",
            pairs=[
                {"origin": a, "destination": b}
                for a, b in zip(chain, chain[1:], strict=False)
            ],
            priority=priority,
        )
        totals = ops.invoke("sum_route_metrics", {"routes": matrix["routes"]})
        kilometres = round(totals["distance_m"] / 1000, 1)
        choice = _nearest_option(options, kilometres, _number)
        return "ok" if choice == gold else f"tools read {kilometres}km -> option {choice}"

    if template == "poi_brand_share":
        anchor_name = row["question"].split("에서")[0]
        anchor = call(registry, "place_search", query=anchor_name, limit=1)
        if not anchor:
            return "anchor unresolved"
        shops = call(
            registry,
            "nearby_places",
            center=anchor[0],
            category_code="CS2",
            radius_m=evidence["radius_m"],
            limit=45,
        )
        inside = ops.invoke(
            "within_radius",
            {"center": anchor[0], "candidates": shops, "radius_m": evidence["radius_m"]},
        )
        matching = [p for p in inside if str(p.get("name", "")).startswith(evidence["brand"])]
        if not inside:
            return "no shops"
        share = ops.invoke(
            "calculate_proportion", {"numerator": len(matching), "denominator": len(inside)}
        )
        percent = round(share["percentage"])
        choice = _nearest_option(options, percent, _number)
        return "ok" if choice == gold else f"tools read {percent}% -> option {choice}"

    if template == "routing_turns_before_road":
        origin, rest = row["question"].split("에서", 1)
        destination = rest.split("까지")[0]
        route = call(
            registry,
            "directions",
            origin=_ref(registry, origin),
            destination=_ref(registry, destination),
            include_steps=True,
            priority=priority,
        )
        steps = route["steps"]
        boundary = evidence["boundary_road"]
        index = next(
            (i for i, step in enumerate(steps) if step["road_name"] == boundary), None
        )
        if index is None:
            return "boundary road absent"
        before = sum(1 for step in steps[:index] if "좌회전" in step["instruction"])
        return "ok" if options[gold] == f"{before}번" else f"tools counted {before}"

    if template == "poi_bearing_and_distance":
        head, rest = row["question"].split("을 기준으로 볼 때 ", 1) if "을 기준으로" in row[
            "question"
        ] else row["question"].split("를 기준으로 볼 때 ", 1)
        other = rest.split(" 어느 방향")[0]
        for suffix in ("은", "는"):
            if other.endswith(suffix):
                other = other[: -len(suffix)]
                break
        found = call(registry, "batch_geocode", place_names=[head, other])
        places = [entry.get("place") for entry in found]
        if any(place is None for place in places):
            return "unresolved endpoint"
        bearing = ops.invoke(
            "bearing_to_direction", {"place_a": places[0], "place_b": places[1]}
        )
        span = ops.invoke("haversine_distance", {"place_a": places[0], "place_b": places[1]})
        text = f"{bearing['direction_ko']}, 약 {round(span['distance_m'] / 1000, 1):g}km"
        return "ok" if options[gold] == text else f"tools read {text}"

    if template == "nearby_from_need":
        anchor_name = row["question"].split("지금 ")[1].split("에 있습니다")[0]
        anchor = call(registry, "place_search", query=anchor_name, limit=1)
        if not anchor:
            return "anchor unresolved"
        found = call(
            registry,
            "nearby_places",
            center=anchor[0],
            category_code=evidence["need_category"],
            radius_m=2000,
            limit=15,
        )
        ranked = ops.invoke("nearest", {"anchor": anchor[0], "candidates": found})
        winner = ranked[0] if isinstance(ranked, list) else ranked
        matched = ops.invoke("match_options", {"options": options, "places": [winner]})
        index = matched.get("best_option")
        return "ok" if index == gold else f"tools chose {index} ({winner.get('name')})"

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
    lines = Path(args.dataset).read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]

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
