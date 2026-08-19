"""Generate the Kakao-grounded Korean benchmark.

Class mix mirrors MapEval-API's answerable half (nearby 30%, trip 24%, routing 23%, poi 23%),
because that is where the Spatial-Agent paper's gains are reported: trip and routing are the
multi-hop families a single retrieval cannot answer. `unanswerable` is deliberately absent —
MapEval encodes it as the sentinel `answer = 0` meaning "no option is right", a refusal channel
this MCQ format does not have.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from collections.abc import Callable
from pathlib import Path

from benchmark_core import (
    CATEGORY_NOUNS,
    Builder,
    Place,
    RouteStep,
    distance_m,
    eul,
    eun,
    load_pool,
    plausible_name,
    take_resolvable,
    to_place,
    wa,
)

from src.tools.spatial import SpatialOperatorRegistry

SEED = 20260818
OUT_PATH = Path(__file__).resolve().parents[1] / "dataset" / "seoul_kmapeval_v2_mcq_100.jsonl"


class Pool:
    def __init__(self) -> None:
        records = load_pool()
        self.by_code: dict[str, list[Place]] = {}
        seen: set[str] = set()
        for record in records:
            if not plausible_name(record["name"]) or record["name"] in seen:
                continue
            seen.add(record["name"])
            self.by_code.setdefault(record["category_code"], []).append(to_place(record))

    def of(self, *codes: str) -> list[Place]:
        return [place for code in codes for place in self.by_code.get(code, [])]


def _distinct(places: list[Place], minimum_gap_m: float) -> bool:
    return all(
        distance_m(a, b) >= minimum_gap_m for a, b in itertools.combinations(places, 2)
    )


# --------------------------------------------------------------------------- trip


def trip_optimal_order(builder: Builder, pool: Pool, rng: random.Random, count: int) -> list[dict]:
    bases = [p for p in pool.of("AD5")]
    sights = [p for p in pool.of("AT4", "CT1")]
    rng.shuffle(bases)
    rng.shuffle(sights)
    made: list[dict] = []
    used_bases: set[str] = set()
    for base in bases:
        if len(made) >= count:
            break
        if base.place_id in used_bases or not builder.resolves_to(base):
            continue
        near = [s for s in sights if 2000 <= distance_m(base, s) <= 12000]
        rng.shuffle(near)
        chosen: list[Place] = []
        for sight in near:
            if len(chosen) == 3:
                break
            if _distinct([*chosen, sight], 1500) and builder.resolves_to(sight):
                chosen.append(sight)
        if len(chosen) < 3:
            continue
        legs: dict[tuple[str, str], int] = {}
        broken = False
        for a, b in itertools.permutations([base, *chosen], 2):
            if a.place_id == base.place_id or b.place_id != base.place_id:
                seconds = builder.duration_s(a, b)
                if seconds is None:
                    broken = True
                    break
                legs[(a.place_id, b.place_id)] = seconds
        if broken:
            continue
        totals: list[tuple[int, tuple[Place, ...]]] = []
        for order in itertools.permutations(chosen):
            path = (base, *order)
            total = sum(legs[(path[i].place_id, path[i + 1].place_id)] for i in range(3))
            totals.append((total, order))
        totals.sort(key=lambda item: item[0])
        if totals[1][0] - totals[0][0] < 180:  # no decisive answer
            continue
        stays = [rng.choice([1.0, 1.5, 2.0]) for _ in chosen]
        budget = int(sum(stays)) + 4
        depart = rng.choice([8, 9, 10])
        options = [totals[0][1], *[order for _, order in totals[1:]][:3]]
        rendered = [" → ".join(place.name for place in order) for order in options]
        stay_text = ", ".join(
            f"{eul(place.name)} {stay:g}시간" for place, stay in zip(chosen, stays, strict=True)
        )
        made.append(
            {
                "question": (
                    f"저는 지금 {base.name}에 머물고 있습니다. {stay_text} 동안 둘러보려고 합니다. "
                    f"총 {budget}시간이 있고 오전 {depart}시에 출발합니다. 자동차로 이동할 때 "
                    "가장 효율적인 방문 순서는 무엇인가요?"
                ),
                "options": rendered,
                "answer": 0,
                "classification": "trip",
                "template_id": "trip_optimal_order",
                "gold_evidence": {
                    "base": base.name,
                    "travel_seconds": [total for total, _ in totals[:4]],
                    "margin_s": totals[1][0] - totals[0][0],
                },
            }
        )
        used_bases.add(base.place_id)
    return made


def trip_feasible_count(builder: Builder, pool: Pool, rng: random.Random, count: int) -> list[dict]:
    bases = [p for p in pool.of("AD5")]
    sights = [p for p in pool.of("AT4", "CT1")]
    rng.shuffle(bases)
    rng.shuffle(sights)
    made: list[dict] = []
    wanted = itertools.cycle([2, 3, 1, 2, 3])
    for base in bases:
        if len(made) >= count:
            break
        if not builder.resolves_to(base):
            continue
        near = [s for s in sights if 2000 <= distance_m(base, s) <= 15000]
        rng.shuffle(near)
        chosen: list[Place] = []
        for sight in near:
            if len(chosen) == 4:
                break
            if _distinct([*chosen, sight], 1500) and builder.resolves_to(sight):
                chosen.append(sight)
        if len(chosen) < 4:
            continue
        legs: dict[tuple[str, str], int] = {}
        broken = False
        for a, b in itertools.permutations([base, *chosen], 2):
            if b.place_id == base.place_id:
                continue
            seconds = builder.duration_s(a, b)
            if seconds is None:
                broken = True
                break
            legs[(a.place_id, b.place_id)] = seconds
        if broken:
            continue
        stays = [rng.choice([1.0, 1.5]) for _ in chosen]
        stay_by_id = {
            place.place_id: stay * 3600
            for place, stay in zip(chosen, stays, strict=True)
        }

        def best_finish(
            subset: tuple[Place, ...],
            *,
            start: Place = base,
            legs: dict[tuple[str, str], int] = legs,
            stay_by_id: dict[str, float] = stay_by_id,
        ) -> float:
            best = float("inf")
            for order in itertools.permutations(subset):
                path = (start, *order)
                total = sum(
                    legs[(path[i].place_id, path[i + 1].place_id)] for i in range(len(order))
                )
                total += sum(stay_by_id[place.place_id] for place in order)
                best = min(best, total)
            return best

        feasible_at = {
            size: min(
                best_finish(subset) for subset in itertools.combinations(chosen, size)
            )
            for size in range(1, 5)
        }
        target = next(wanted)
        budget_s = (feasible_at[target] + feasible_at[target + 1]) / 2 if target < 4 else None
        if budget_s is None or budget_s <= feasible_at[target]:
            continue
        budget_h = round(budget_s / 3600 * 2) / 2
        if budget_h * 3600 <= feasible_at[target] or budget_h * 3600 >= feasible_at[target + 1]:
            continue
        depart = rng.choice([8, 9, 10])
        stay_text = ", ".join(
            f"{eul(place.name)} {stay:g}시간" for place, stay in zip(chosen, stays, strict=True)
        )
        made.append(
            {
                "question": (
                    f"저는 지금 {base.name}에 머물고 있습니다. {stay_text} 동안 둘러보려고 합니다. "
                    f"총 {budget_h:g}시간이 있고 오전 {depart}시에 자동차로 출발합니다. "
                    "몇 곳을 방문할 수 있나요?"
                ),
                "options": ["한 곳", "두 곳", "세 곳", "네 곳"],
                "answer": target - 1,
                "classification": "trip",
                "template_id": "trip_feasible_count",
                "ordinal_options": True,
                "gold_evidence": {
                    "base": base.name,
                    "budget_s": budget_h * 3600,
                    "best_finish_by_size": feasible_at,
                    # Ordinal options name no places, so the stops and stays live here — without
                    # them the row cannot be re-derived through the tools after the fact.
                    "stops": [place.name for place in chosen],
                    "stay_s": [stay * 3600 for stay in stays],
                },
            }
        )
    return made


# ------------------------------------------------------------------------- routing


def routing_via_compare(builder: Builder, pool: Pool, rng: random.Random, count: int) -> list[dict]:
    """Which of four detours is fastest — four waypoint routes, no single retrieval answers it."""

    endpoints = pool.of("SW8", "AT4")
    vias = pool.of("AT4", "CT1", "MT1")
    rng.shuffle(endpoints)
    rng.shuffle(vias)
    made: list[dict] = []
    used: set[str] = set()
    for origin, destination in itertools.combinations(endpoints, 2):
        if len(made) >= count:
            break
        if origin.place_id in used or destination.place_id in used:
            continue
        if not 8000 <= distance_m(origin, destination) <= 20000:
            continue
        if not (builder.resolves_to(origin) and builder.resolves_to(destination)):
            continue
        candidates = [
            via
            for via in vias
            if via.place_id not in (origin.place_id, destination.place_id)
            and distance_m(origin, via) > 1500
            and distance_m(destination, via) > 1500
        ]
        rng.shuffle(candidates)
        scored: list[tuple[int, Place]] = []
        for via in candidates:
            if len(scored) == 4:
                break
            if not builder.resolves_to(via):
                continue
            route = builder.route(origin, destination, (via,))
            if route is None:
                continue
            scored.append((route.duration_s, via))
        if len(scored) < 4:
            continue
        scored.sort(key=lambda item: item[0])
        if scored[1][0] - scored[0][0] < 120:
            continue
        made.append(
            {
                "question": (
                    f"{origin.name}에서 {destination.name}까지 자동차로 이동하려고 합니다. "
                    "다음 경유지 중 한 곳을 반드시 거쳐야 한다면, 어디를 경유하는 것이 "
                    "가장 빠릅니까?"
                ),
                "options": [via.name for _, via in scored],
                "answer": 0,
                "classification": "routing",
                "template_id": "routing_via_compare",
                "gold_evidence": {
                    "durations_s": [seconds for seconds, _ in scored],
                    "margin_s": scored[1][0] - scored[0][0],
                },
            }
        )
        used.update((origin.place_id, destination.place_id))
    return made


def _guide_text(step: RouteStep) -> str:
    return f"{step.instruction} ({step.road_name})" if step.road_name else step.instruction


def routing_next_turn(builder: Builder, pool: Pool, rng: random.Random, count: int) -> list[dict]:
    """What comes after a named road — only the turn-by-turn guides carry it."""

    endpoints = pool.of("SW8", "AT4", "CT1")
    rng.shuffle(endpoints)
    made: list[dict] = []
    used: set[str] = set()
    distractor_pool: list[str] = []
    for origin, destination in itertools.combinations(endpoints, 2):
        if len(made) >= count:
            break
        if origin.place_id in used or destination.place_id in used:
            continue
        if not 6000 <= distance_m(origin, destination) <= 18000:
            continue
        if not (builder.resolves_to(origin) and builder.resolves_to(destination)):
            continue
        route = builder.route(origin, destination)
        if route is None or len(route.steps) < 6:
            continue
        labelled = [
            (index, step)
            for index, step in enumerate(route.steps)
            if step.road_name and step.instruction and index + 1 < len(route.steps)
        ]
        # A road that appears twice cannot identify a moment in the drive.
        road_counts = {step.road_name: 0 for _, step in labelled}
        for step in route.steps:
            if step.road_name in road_counts:
                road_counts[step.road_name] += 1
        usable = [
            (index, step)
            for index, step in labelled
            if road_counts[step.road_name] == 1 and route.steps[index + 1].instruction
        ]
        if not usable:
            continue
        index, step = usable[len(usable) // 2]
        gold_step = route.steps[index + 1]
        gold = _guide_text(gold_step)
        others = [
            _guide_text(other)
            for position, other in enumerate(route.steps)
            if position not in (index, index + 1) and other.instruction
        ]
        others = [text for text in dict.fromkeys(others) if text != gold]
        rng.shuffle(others)
        pool_extra = [text for text in distractor_pool if text != gold and text not in others]
        options = [gold, *others[:3]]
        while len(options) < 4 and pool_extra:
            options.append(pool_extra.pop())
        if len(options) < 4:
            continue
        distractor_pool.extend(others[:6])
        made.append(
            {
                "question": (
                    f"{origin.name}에서 {destination.name}까지 자동차로 운전하고 있습니다. "
                    f"{step.road_name} 구간에 진입한 뒤 이어지는 주행 안내는 무엇인가요?"
                ),
                "options": options,
                "answer": 0,
                "classification": "routing",
                "template_id": "routing_next_turn",
                "gold_evidence": {"after_road": step.road_name, "step_index": index + 1},
            }
        )
        used.update((origin.place_id, destination.place_id))
    return made


def routing_turn_count(builder: Builder, pool: Pool, rng: random.Random, count: int) -> list[dict]:
    """How many left turns on a via-route — a count over the guides of a two-leg drive."""

    endpoints = pool.of("SW8", "AT4", "CT1")
    vias = pool.of("AT4", "MT1", "CT1")
    rng.shuffle(endpoints)
    rng.shuffle(vias)
    made: list[dict] = []
    used: set[str] = set()
    for origin, destination in itertools.combinations(endpoints, 2):
        if len(made) >= count:
            break
        if origin.place_id in used or destination.place_id in used:
            continue
        if not 7000 <= distance_m(origin, destination) <= 18000:
            continue
        if not (builder.resolves_to(origin) and builder.resolves_to(destination)):
            continue
        via = next(
            (
                candidate
                for candidate in vias
                if distance_m(origin, candidate) > 2000
                and distance_m(destination, candidate) > 2000
                and builder.resolves_to(candidate)
            ),
            None,
        )
        if via is None:
            continue
        route = builder.route(origin, destination, (via,))
        if route is None or not route.steps:
            continue
        turns = sum(1 for step in route.steps if "좌회전" in step.instruction)
        if not 1 <= turns <= 8:
            continue
        choices = sorted({turns, max(0, turns - 1), turns + 1, turns + 2})[:4]
        while len(choices) < 4:
            choices.append(max(choices) + 1)
        made.append(
            {
                "question": (
                    f"{origin.name}에서 {eul(via.name)} 경유하여 {destination.name}까지 "
                    "자동차로 이동합니다. 주행 안내에 따르면 좌회전을 몇 번 해야 하나요?"
                ),
                "options": [str(value) + "번" for value in choices],
                "answer": choices.index(turns),
                "classification": "routing",
                "template_id": "routing_turn_count",
                "gold_evidence": {"left_turns": turns, "steps": len(route.steps)},
            }
        )
        used.update((origin.place_id, destination.place_id))
    return made


# ----------------------------------------------------------------------------- poi


def poi_farthest_pair(builder: Builder, pool: Pool, rng: random.Random, count: int) -> list[dict]:
    """Which of four pairs is farthest apart — eight lookups and four comparisons."""

    landmarks = pool.of("AT4", "CT1", "SW8")
    rng.shuffle(landmarks)
    resolvable = [place for place in landmarks[:400] if builder.resolves_to(place)]
    made: list[dict] = []
    guard = 0
    while len(made) < count and guard < 4000:
        guard += 1
        picked = rng.sample(resolvable, 8)
        pairs = [(picked[i], picked[i + 1]) for i in range(0, 8, 2)]
        scored = sorted(((distance_m(a, b), a, b) for a, b in pairs), reverse=True)
        if scored[0][0] - scored[1][0] < 1500:
            continue
        made.append(
            {
                "question": "다음 중 서로 가장 멀리 떨어진 두 장소의 조합은 무엇인가요?",
                "options": [f"{wa(a.name)} {b.name}" for _, a, b in scored],
                "answer": 0,
                "classification": "poi",
                "template_id": "poi_farthest_pair",
                "gold_evidence": {
                    "distances_m": [round(value) for value, _, _ in scored],
                    "margin_m": round(scored[0][0] - scored[1][0]),
                    # The option text joins the names with a particle that attaches to the first
                    # one (…마을과 …), so the pair cannot be recovered by splitting it.
                    "pairs": [[a.name, b.name] for _, a, b in scored],
                },
            }
        )
    return made


def poi_between(builder: Builder, pool: Pool, rng: random.Random, count: int) -> list[dict]:
    """Which candidate actually sits between two landmarks — a corridor test, not a distance."""

    anchors = pool.of("AT4", "CT1", "SW8")
    rng.shuffle(anchors)
    made: list[dict] = []
    used: set[str] = set()
    for a, b in itertools.combinations(anchors[:250], 2):
        if len(made) >= count:
            break
        if a.place_id in used or b.place_id in used:
            continue
        span = distance_m(a, b)
        if not 2500 <= span <= 7000:
            continue
        if not (builder.resolves_to(a) and builder.resolves_to(b)):
            continue
        code = rng.choice(["CE7", "MT1", "CT1"])
        candidates = [
            place
            for place in pool.by_code.get(code, [])
            if place.place_id not in (a.place_id, b.place_id)
        ]
        detours = sorted(
            (
                ((distance_m(a, place) + distance_m(place, b)) / span, place)
                for place in candidates
            ),
            key=lambda item: item[0],
        )
        between = [(ratio, place) for ratio, place in detours if ratio <= 1.08]
        outside = [(ratio, place) for ratio, place in detours if ratio >= 1.6]
        if not between or len(outside) < 3:
            continue
        # Sorted by how little the detour costs, so the head of the list is the best evidence
        # and a bounded prefix keeps a hopeless neighbourhood from costing a thousand lookups.
        gold = next(iter(take_resolvable(builder, (p for _, p in between[:25]), 1)), None)
        if gold is None:
            continue
        rng.shuffle(outside)
        wrong = take_resolvable(builder, (place for _, place in outside), 3)
        if len(wrong) < 3:
            continue
        made.append(
            {
                "question": (
                    f"{a.name}에서 {b.name}까지 이동하는 경로 위에 있다고 볼 수 있는 "
                    f"{eun(CATEGORY_NOUNS[code])} 다음 중 어디인가요?"
                ),
                "options": [gold.name, *[place.name for place in wrong]],
                "answer": 0,
                "classification": "poi",
                "template_id": "poi_between",
                "gold_evidence": {"span_m": round(span), "category": code},
            }
        )
        used.update((a.place_id, b.place_id))
    return made


def poi_common_nearby(builder: Builder, pool: Pool, rng: random.Random, count: int) -> list[dict]:
    """Which place is close to both anchors — an intersection of two neighbourhoods."""

    anchors = pool.of("AT4", "CT1", "SW8")
    rng.shuffle(anchors)
    made: list[dict] = []
    used: set[str] = set()
    for a, b in itertools.combinations(anchors[:250], 2):
        if len(made) >= count:
            break
        if a.place_id in used or b.place_id in used:
            continue
        if not 1200 <= distance_m(a, b) <= 2600:
            continue
        if not (builder.resolves_to(a) and builder.resolves_to(b)):
            continue
        code = rng.choice(["CE7", "MT1", "BK9", "HP8"])
        radius = 1500
        candidates = [
            place
            for place in pool.by_code.get(code, [])
            if place.place_id not in (a.place_id, b.place_id)
        ]
        both = [
            place
            for place in candidates
            if distance_m(a, place) <= radius and distance_m(b, place) <= radius
        ]
        one_only = [
            place
            for place in candidates
            if (distance_m(a, place) <= radius) != (distance_m(b, place) <= radius)
        ]
        if not both or len(one_only) < 3:
            continue
        gold = next(iter(take_resolvable(builder, both[:25], 1)), None)
        if gold is None:
            continue
        rng.shuffle(one_only)
        wrong = take_resolvable(builder, one_only, 3)
        if len(wrong) < 3:
            continue
        made.append(
            {
                "question": (
                    f"{wa(a.name)} {b.name} 양쪽 모두에서 직선거리 {radius}m 이내에 있는 "
                    f"{eun(CATEGORY_NOUNS[code])} 다음 중 어디인가요?"
                ),
                "options": [gold.name, *[place.name for place in wrong]],
                "answer": 0,
                "classification": "poi",
                "template_id": "poi_common_nearby",
                "gold_evidence": {"radius_m": radius, "category": code},
            }
        )
        used.update((a.place_id, b.place_id))
    return made


# -------------------------------------------------------------------------- nearby


def _kakao_neighbours(
    builder: Builder, anchor: Place, code: str, radius_m: int, limit: int = 15
) -> list[Place]:
    """Ask Kakao itself, so the gold is the agent's own retrieval rather than our arithmetic."""

    try:
        found = builder.provider.nearby_search(
            anchor, category_code=code, radius_m=radius_m, limit=limit
        )
    except Exception:  # noqa: BLE001 - an empty neighbourhood is simply not usable
        return []
    return [
        place
        for place in found
        if place.place_id != anchor.place_id and distance_m(anchor, place) > 5
    ]


def nearby_nearest_by_type(
    builder: Builder, pool: Pool, rng: random.Random, count: int
) -> list[dict]:
    anchors = pool.of("AT4", "CT1", "SW8", "AD5")
    rng.shuffle(anchors)
    made: list[dict] = []
    for anchor in anchors:
        if len(made) >= count:
            break
        if not builder.resolves_to(anchor):
            continue
        code = rng.choice(["CE7", "MT1", "BK9", "HP8", "CS2"])
        found = _kakao_neighbours(builder, anchor, code, 2000)
        if len(found) < 6:
            continue
        gold = found[0]
        if distance_m(anchor, found[1]) - distance_m(anchor, gold) < 120:
            continue  # a tie is not a question
        if not builder.resolves_to(gold):
            continue
        # The runner-up is the distractor that makes the question a question; a field of only
        # far-away places is answerable by rough position alone.
        wrong = take_resolvable(builder, found[1:], 3)
        if len(wrong) < 3:
            continue
        made.append(
            {
                "question": (
                    f"{anchor.name}에서 가장 가까운 {eun(CATEGORY_NOUNS[code])} 다음 중 "
                    "어디인가요?"
                ),
                "options": [gold.name, *[place.name for place in wrong]],
                "answer": 0,
                "classification": "nearby",
                "template_id": "nearby_nearest_by_type",
                "gold_evidence": {
                    "category": code,
                    "gold_distance_m": round(distance_m(anchor, gold)),
                    "runner_up_m": round(distance_m(anchor, found[1])),
                },
            }
        )
    return made


def nearby_direction(builder: Builder, pool: Pool, rng: random.Random, count: int) -> list[dict]:
    ops = SpatialOperatorRegistry()
    anchors = pool.of("AT4", "CT1", "SW8", "AD5")
    rng.shuffle(anchors)
    made: list[dict] = []
    for anchor in anchors:
        if len(made) >= count:
            break
        if not builder.resolves_to(anchor):
            continue
        code = rng.choice(["CE7", "MT1", "BK9", "HP8"])
        found = _kakao_neighbours(builder, anchor, code, 2500, limit=15)
        if len(found) < 6:
            continue
        want = rng.choice(["북", "남", "동", "서"])
        # `filter_by_direction` bins bearings into 90-degree quadrants; an eight-point compass
        # disagrees with it on everything between 22.5 and 45 degrees off the axis. The operator
        # the agent runs is the one that decides, so the gold cannot drift from it.
        in_ids = {
            place["place_id"]
            for place in ops.invoke(
                "filter_by_direction",
                {
                    "center": anchor.model_dump(),
                    "places": [place.model_dump() for place in found],
                    "direction": want,
                },
            )
        }
        in_sector = [place for place in found if place.place_id in in_ids]
        off_sector = [place for place in found if place.place_id not in in_ids]
        if not in_sector or len(off_sector) < 3:
            continue
        # The nearest in-sector place is the answer, so it is the only one that may be gold. Taking
        # the first *resolvable* one instead silently promotes the runner-up, and the tools then
        # correctly return a place that is not among the options at all.
        gold = in_sector[0] if builder.resolves_to(in_sector[0]) else None
        if gold is None:
            continue
        # A nearer place in the wrong direction is what makes the constraint bite.
        closer_off = [
            place for place in off_sector if distance_m(anchor, place) < distance_m(anchor, gold)
        ]
        if not closer_off:
            continue
        rng.shuffle(off_sector)
        wrong = take_resolvable(builder, closer_off, 2)
        wrong += take_resolvable(
            builder, (place for place in off_sector if place not in wrong), 3 - len(wrong)
        )
        if len(wrong) < 3:
            continue
        made.append(
            {
                "question": (
                    f"{anchor.name}에서 {want}쪽 방향에 있는 {CATEGORY_NOUNS[code]} 중 "
                    "가장 가까운 곳은 다음 중 어디인가요?"
                ),
                "options": [gold.name, *[place.name for place in wrong]],
                "answer": 0,
                "classification": "direction",
                "template_id": "direction_nearest_by_type",
                "gold_evidence": {
                    "direction": want,
                    "category": code,
                    "gold_distance_m": round(distance_m(anchor, gold)),
                },
            }
        )
    return made


def nearby_within_radius(
    builder: Builder, pool: Pool, rng: random.Random, count: int
) -> list[dict]:
    anchors = pool.of("AT4", "CT1", "SW8", "AD5")
    rng.shuffle(anchors)
    made: list[dict] = []
    for anchor in anchors:
        if len(made) >= count:
            break
        if not builder.resolves_to(anchor):
            continue
        code = rng.choice(["CE7", "MT1", "BK9", "HP8", "SC4"])
        radius = rng.choice([400, 600, 800])
        found = _kakao_neighbours(builder, anchor, code, 3000, limit=15)
        inside = [place for place in found if distance_m(anchor, place) <= radius]
        outside = [place for place in found if distance_m(anchor, place) >= radius * 2]
        if len(inside) != 1 or len(outside) < 3:
            continue
        gold = inside[0]
        if not builder.resolves_to(gold):
            continue
        rng.shuffle(outside)
        wrong = take_resolvable(builder, outside, 3)
        if len(wrong) < 3:
            continue
        made.append(
            {
                "question": (
                    f"{anchor.name}에서 직선거리 {radius}m 이내에 있는 "
                    f"{eun(CATEGORY_NOUNS[code])} 다음 중 어디인가요?"
                ),
                "options": [gold.name, *[place.name for place in wrong]],
                "answer": 0,
                "classification": "radius",
                "template_id": "radius_within_by_type",
                "gold_evidence": {
                    "radius_m": radius,
                    "category": code,
                    "gold_distance_m": round(distance_m(anchor, gold)),
                },
            }
        )
    return made


# ------------------------------------------------------------------------ assembly

FAMILIES: list[tuple[str, Callable[..., list[dict]], int]] = [
    ("trip_optimal_order", trip_optimal_order, 14),
    ("trip_feasible_count", trip_feasible_count, 10),
    ("routing_via_compare", routing_via_compare, 8),
    ("routing_next_turn", routing_next_turn, 8),
    ("routing_turn_count", routing_turn_count, 7),
    ("poi_farthest_pair", poi_farthest_pair, 8),
    ("poi_between", poi_between, 8),
    ("poi_common_nearby", poi_common_nearby, 7),
    ("nearby_nearest_by_type", nearby_nearest_by_type, 12),
    ("nearby_direction", nearby_direction, 9),
    ("nearby_within_radius", nearby_within_radius, 9),
]


def finalize(rows: list[dict]) -> list[dict]:
    """Assign ids and shuffle options per row, so option position is never evidence."""

    finished: list[dict] = []
    for index, row in enumerate(rows):
        question_id = f"seoul_kmapeval_v2_{index:03d}"
        options = list(row["options"])
        answer = row["answer"]
        if not row.pop("ordinal_options", False):
            order = list(range(len(options)))
            random.Random(f"{question_id}:{SEED}").shuffle(order)
            options = [row["options"][position] for position in order]
            answer = order.index(row["answer"])
        finished.append(
            {
                "id": question_id,
                "question": row["question"],
                "options": options,
                "answer": answer,
                "classification": row["classification"],
                "region": "서울",
                "template_id": row["template_id"],
                "source_dataset": "kakao_local_mobility",
                "gold_evidence": row["gold_evidence"],
            }
        )
    return finished


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    builder = Builder.open()
    pool = Pool()
    rows: list[dict] = []
    try:
        for name, function, quota in FAMILIES:
            if args.families and name not in args.families:
                continue
            wanted = max(1, round(quota * args.scale))
            rng = random.Random(f"{SEED}:{name}")
            produced = function(builder, pool, rng, wanted)
            print(f"{name}: {len(produced)}/{wanted} (api={builder.provider.api_call_count})",
                  flush=True)
            rows.extend(produced)
    finally:
        builder.close()

    finished = finalize(rows)
    out = Path(args.out)
    out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in finished) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {out} rows={len(finished)}")


if __name__ == "__main__":
    main()
