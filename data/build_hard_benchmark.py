"""Generate the compositional Korean benchmark — the families GeoFlow exists for.

`seoul_kmapeval_v2` covers six of the ten Appendix E macro families and leaves the compositional
ones untouched: **Time-Window-Reverse**, **Multi-Segment-Aggregate** and **Object-Field-Measure**
were never exercised by a single question. Those are where a typed operator graph should pay for
its overhead, and where a ReAct loop has to carry the most state across turns.

Two design rules follow from why v2 failed to separate the architectures:

1. **Depth.** Every family here needs at least four dependent stages, so no single retrieval and
   no single arithmetic step finishes the question.
2. **Inference.** Where a question can name the kind of place it wants, it does not. v2 stated the
   category, the radius and the orderings to compare, which is exactly the work MapEval leaves to
   the agent ("I am at X and hungry, where can I eat quickly?").

Place-Attribute-Query stays unported: it asks for ratings, price levels and opening hours, and
Kakao Local exposes none of them. Inventing those values would be fabricating evidence.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

from benchmark_core import (
    Builder,
    Place,
    distance_m,
    eul,
    eun,
    load_pool,
    plausible_name,
    take_resolvable,
    to_place,
)

SEED = 20260819
OUT_PATH = Path(__file__).resolve().parents[1] / "dataset" / "seoul_kmapeval_v3_mcq_100.jsonl"

# A need, not a category. The agent has to work out what kind of place answers it — the step v2
# removed by naming the category outright. Generic over place types, never over questions.
NEEDS: list[tuple[str, str, str]] = [
    ("CS2", "갑자기 비가 쏟아져서 우산을 사야 합니다", "편의점"),
    ("PM9", "두통이 너무 심해서 두통약을 사야 합니다", "약국"),
    ("OL7", "연료 경고등이 켜져서 기름을 넣어야 합니다", "주유소"),
    ("BK9", "현금이 필요해서 돈을 찾아야 합니다", "은행"),
    ("CE7", "노트북으로 잠깐 일할 조용한 자리가 필요합니다", "카페"),
    ("HP8", "발목을 접질려서 진료를 받아야 합니다", "병원"),
    ("PK6", "차를 한동안 세워둘 곳이 필요합니다", "주차장"),
    ("SW8", "지하철을 타고 이동해야 합니다", "지하철역"),
    ("MT1", "저녁에 쓸 장을 봐야 합니다", "대형마트"),
    ("FD6", "배가 고파서 끼니를 해결해야 합니다", "음식점"),
]

DISTRACTOR_CODES = ("CE7", "CS2", "BK9", "HP8", "MT1", "PK6", "FD6", "SC4")


class Pool:
    def __init__(self) -> None:
        self.by_code: dict[str, list[Place]] = {}
        seen: set[str] = set()
        for record in load_pool():
            if not plausible_name(record["name"]) or record["name"] in seen:
                continue
            seen.add(record["name"])
            self.by_code.setdefault(record["category_code"], []).append(to_place(record))

    def of(self, *codes: str) -> list[Place]:
        return [place for code in codes for place in self.by_code.get(code, [])]


def _spread(places: list[Place], gap_m: float) -> bool:
    return all(distance_m(a, b) >= gap_m for a, b in itertools.combinations(places, 2))


def _clock(moment: datetime) -> str:
    hour, minute = moment.hour, moment.minute
    period = "오전" if hour < 12 else "오후"
    display = hour if hour <= 12 else hour - 12
    return f"{period} {display}시 {minute:02d}분"


def _time_options(
    gold: datetime, rng: random.Random, offsets_min: tuple[int, ...] = (-150, -90, 85, 145, 200)
) -> list[str]:
    """Times far enough apart that a traffic estimate cannot flip them.

    A duration is a live estimate even at a fixed routing priority — the identical route came
    back as 3,243 s and then 4,337 s, a third of itself. Over four legs that is tens of minutes
    on the clock, so the nearest wrong option has to sit further away than that or the question
    grades the traffic rather than the plan.
    """

    picked = rng.sample(list(offsets_min), 3)
    return [_clock(gold), *[_clock(gold + timedelta(minutes=value)) for value in picked]]


# -------------------------------------------------------- Time-Window-Reverse


def _survives_traffic(
    builder: Builder,
    chain: list[Place],
    fixed_seconds: float,
    gold: datetime,
    options: list[str],
    depart: datetime,
    sign: int = 1,
) -> bool:
    """Keep only rows whose answer holds under a second, differently-routed traffic estimate.

    A duration is a live estimate: the same fixed route came back as 3,243 s and then 4,337 s.
    Widening the options absorbs most of that, but not on every chain, and a row whose answer
    flips with the traffic grades the hour it was asked in rather than the plan. Measured here
    against RECOMMEND, which re-optimizes against current speeds, so a row that agrees under both
    is one whose answer does not depend on which route the agent is given.
    """

    legs = [
        builder.route(a, b, priority="RECOMMEND")
        for a, b in zip(chain, chain[1:], strict=False)
    ]
    if any(route is None for route in legs):
        return False
    alternative = depart + sign * timedelta(
        seconds=sum(route.duration_s for route in legs) + fixed_seconds
    )
    alternative = alternative.replace(second=0, microsecond=0)

    def minutes(text: str) -> int:
        match = re.search(r"(오전|오후)\s*(\d+)시\s*(\d+)분", text)
        hour = int(match.group(2)) % 12 + (12 if match.group(1) == "오후" else 0)
        return hour * 60 + int(match.group(3))

    target = alternative.hour * 60 + alternative.minute
    nearest = min(range(len(options)), key=lambda i: abs(minutes(options[i]) - target))
    return options[nearest] == _clock(gold)


def trip_finish_time(builder: Builder, pool: Pool, rng: random.Random, count: int) -> list[dict]:
    """When does a fixed itinerary end — four legs, three stays, and clock arithmetic."""

    bases = pool.of("AD5")
    sights = pool.of("AT4", "CT1")
    rng.shuffle(bases)
    rng.shuffle(sights)
    made: list[dict] = []
    for base in bases:
        if len(made) >= count:
            break
        if not builder.resolves_to(base):
            continue
        near = [s for s in sights if 2000 <= distance_m(base, s) <= 11000]
        rng.shuffle(near)
        stops: list[Place] = []
        for sight in near:
            if len(stops) == 3:
                break
            if _spread([*stops, sight], 1500) and builder.resolves_to(sight):
                stops.append(sight)
        if len(stops) < 3:
            continue
        legs = [
            builder.duration_s(a, b)
            for a, b in zip([base, *stops], [*stops, base], strict=True)
        ]
        if any(value is None for value in legs):
            continue
        stays = [rng.choice([1.0, 1.5]) for _ in stops]
        depart = datetime(2026, 8, 19, rng.choice([9, 10]), 0)
        gold = depart + timedelta(
            seconds=sum(legs) + sum(stay * 3600 for stay in stays)
        )
        gold = gold.replace(second=0, microsecond=0)
        plan = ", ".join(
            f"{eul(place.name)} {stay:g}시간" for place, stay in zip(stops, stays, strict=True)
        )
        options = _time_options(gold, rng)
        if not _survives_traffic(
            builder, [base, *stops, base], sum(stay * 3600 for stay in stays), gold, options,
            depart,
        ):
            continue
        made.append(
            {
                "question": (
                    f"{_clock(depart)}에 {base.name}에서 자동차로 출발해 {plan} 동안 차례로 "
                    f"둘러본 뒤 {base.name}로 돌아옵니다. 방문 순서는 적은 그대로이고 구간마다 "
                    "가장 빠른 경로로 이동합니다. 몇 시에 돌아오게 되나요?"
                ),
                "options": options,
                "answer": 0,
                "classification": "trip",
                "template_id": "trip_finish_time",
                "gold_evidence": {
                    "base": base.name,
                    "stops": [place.name for place in stops],
                    "stay_s": [stay * 3600 for stay in stays],
                    "depart": depart.isoformat(),
                    "leg_s": legs,
                },
            }
        )
    return made


def trip_latest_departure(
    builder: Builder, pool: Pool, rng: random.Random, count: int
) -> list[dict]:
    """Work the clock backwards: an arrival deadline, two errands, one departure time."""

    bases = pool.of("AD5", "SW8")
    stops_pool = pool.of("MT1", "BK9", "CE7")
    targets = pool.of("AT4", "CT1")
    rng.shuffle(bases)
    rng.shuffle(stops_pool)
    rng.shuffle(targets)
    made: list[dict] = []
    for base in bases:
        if len(made) >= count:
            break
        if not builder.resolves_to(base):
            continue
        target = next(
            (
                place
                for place in targets
                if 5000 <= distance_m(base, place) <= 15000 and builder.resolves_to(place)
            ),
            None,
        )
        if target is None:
            continue
        errands: list[Place] = []
        for place in stops_pool:
            if len(errands) == 2:
                break
            if (
                distance_m(base, place) > 1000
                and distance_m(target, place) > 1000
                and _spread([*errands, place], 1000)
                and builder.resolves_to(place)
            ):
                errands.append(place)
        if len(errands) < 2:
            continue
        chain = [base, *errands, target]
        legs = [builder.duration_s(a, b) for a, b in zip(chain, chain[1:], strict=False)]
        if any(value is None for value in legs):
            continue
        stays = [rng.choice([15, 30, 45]) for _ in errands]
        arrival = datetime(2026, 8, 19, rng.choice([17, 18, 19]), 0)
        gold = arrival - timedelta(seconds=sum(legs) + sum(stay * 60 for stay in stays))
        gold = gold.replace(second=0, microsecond=0)
        errand_text = ", ".join(
            f"{place.name}에서 {stay}분" for place, stay in zip(errands, stays, strict=True)
        )
        options = _time_options(gold, rng, offsets_min=(-155, -95, 80, 140, 195))
        if not _survives_traffic(
            builder, chain, sum(stay * 60 for stay in stays), gold, options, arrival, sign=-1
        ):
            continue
        made.append(
            {
                "question": (
                    f"{target.name}에서 {_clock(arrival)}에 약속이 있습니다. 가는 길에 "
                    f"{errand_text}씩 들러야 하고, 이동은 모두 자동차로 가장 빠른 경로를 "
                    f"이용합니다. {base.name}에서 늦어도 몇 시에 출발해야 하나요?"
                ),
                "options": options,
                "answer": 0,
                "classification": "trip",
                "template_id": "trip_latest_departure",
                "gold_evidence": {
                    "base": base.name,
                    "errands": [place.name for place in errands],
                    "target": target.name,
                    "stay_s": [stay * 60 for stay in stays],
                    "arrival": arrival.isoformat(),
                    "leg_s": legs,
                },
            }
        )
    return made


# --------------------------------------------------- Multi-Segment-Aggregate


def multisegment_total(builder: Builder, pool: Pool, rng: random.Random, count: int) -> list[dict]:
    """Total driving distance over a stated four-stop chain — every leg must be looked up."""

    starts = pool.of("SW8", "AD5")
    stops_pool = pool.of("AT4", "CT1", "MT1")
    rng.shuffle(starts)
    rng.shuffle(stops_pool)
    made: list[dict] = []
    for start in starts:
        if len(made) >= count:
            break
        if not builder.resolves_to(start):
            continue
        stops: list[Place] = []
        for place in stops_pool:
            if len(stops) == 3:
                break
            if (
                2000 <= distance_m(start, place) <= 13000
                and _spread([*stops, place], 2000)
                and builder.resolves_to(place)
            ):
                stops.append(place)
        if len(stops) < 3:
            continue
        chain = [start, *stops]
        legs = [builder.distance_m_driving(a, b) for a, b in zip(chain, chain[1:], strict=False)]
        if any(value is None for value in legs):
            continue
        total_km = sum(legs) / 1000
        gold = round(total_km, 1)
        # Wrong options are plausible totals, not round numbers, so the shape of the answer cannot
        # be guessed. The gaps are wide because the agent may route with Kakao's traffic-aware
        # RECOMMEND while the gold is built from the stable DISTANCE priority: the two disagreed
        # by up to 15% on a whole chain, so the nearest wrong option sits further away than that.
        candidates = {gold}
        for factor in (0.58, 1.42, 1.85):
            candidates.add(round(total_km * factor, 1))
        others = [value for value in sorted(candidates) if value != gold]
        rng.shuffle(others)
        chosen = others[:3]
        if len(chosen) < 3:
            continue
        order = " → ".join(place.name for place in chain)
        made.append(
            {
                "question": (
                    f"{order} 순서로 자동차로 이동합니다. 각 구간을 거리가 가장 짧은 경로로 "
                    "갈 때 총 주행 거리는 다음 중 어디에 가장 가깝나요?"
                ),
                "options": [f"약 {value:g}km" for value in [gold, *chosen]],
                "answer": 0,
                "classification": "routing",
                "template_id": "multisegment_total",
                "gold_evidence": {
                    "chain": [place.name for place in chain],
                    "leg_m": legs,
                    "total_km": gold,
                },
            }
        )
    return made


# ------------------------------------------------------- Object-Field-Measure


def poi_brand_share(builder: Builder, pool: Pool, rng: random.Random, count: int) -> list[dict]:
    """What share of a neighbourhood's shops belong to one brand — retrieve, filter, divide."""

    anchors = pool.of("SW8", "AT4", "CT1", "AD5")
    rng.shuffle(anchors)
    brands = ("GS25", "CU", "세븐일레븐")
    made: list[dict] = []
    for anchor in anchors:
        if len(made) >= count:
            break
        if not builder.resolves_to(anchor):
            continue
        radius = rng.choice([400, 500, 600])
        try:
            found = builder.provider.nearby_search(
                anchor, category_code="CS2", radius_m=radius, limit=45
            )
        except Exception:  # noqa: BLE001 - an empty neighbourhood is simply not usable
            continue
        shops = [place for place in found if distance_m(anchor, place) <= radius]
        if not 6 <= len(shops) <= 40:
            continue
        brand = rng.choice(brands)
        matching = [place for place in shops if place.name.startswith(brand)]
        if not 1 <= len(matching) < len(shops):
            continue
        share = round(100 * len(matching) / len(shops))
        # Distractors are the shares the other brands would give, plus the complement — every
        # wrong option is a real number about this neighbourhood, just not the one asked for.
        pool_values = {share}
        for other in brands:
            if other != brand:
                count_other = sum(1 for place in shops if place.name.startswith(other))
                pool_values.add(round(100 * count_other / len(shops)))
        pool_values.add(100 - share)
        pool_values.add(round(100 * len(matching) / max(1, len(shops) - 1)))
        # Two options five points apart are one option: 2/14 and 2/13 both read "약 14%", and no
        # amount of correct retrieval can choose between them.
        others: list[int] = []
        for value in sorted(pool_values):
            if value == share or not 0 <= value <= 100:
                continue
            if all(abs(value - kept) >= 6 for kept in [share, *others]):
                others.append(value)
        rng.shuffle(others)
        if len(others) < 3:
            continue
        made.append(
            {
                "question": (
                    f"{anchor.name}에서 직선거리 {radius}m 이내에 있는 편의점 가운데 "
                    f"{eun(brand)} 몇 퍼센트를 차지하나요? 가장 가까운 값을 고르세요."
                ),
                "options": [f"약 {value}%" for value in [share, *others[:3]]],
                "answer": 0,
                # `classification` is the intent the agent routes on (it is SUPPORTED_INTENTS),
                # not the Appendix E family, which `template_id` records. This question searches
                # a stated radius around one anchor, so that is what it is.
                "classification": "radius",
                "template_id": "poi_brand_share",
                "gold_evidence": {
                    "radius_m": radius,
                    "brand": brand,
                    "matching": len(matching),
                    "total": len(shops),
                },
            }
        )
    return made


# --------------------------------------------- Route-Step-Extract, but bounded


def routing_turns_before_road(
    builder: Builder, pool: Pool, rng: random.Random, count: int
) -> list[dict]:
    """Count left turns *before* a named road — find the boundary first, then aggregate."""

    endpoints = pool.of("SW8", "AT4", "CT1")
    rng.shuffle(endpoints)
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
        route = builder.route(origin, destination, priority="DISTANCE")
        if route is None or len(route.steps) < 8:
            continue
        counts: dict[str, int] = {}
        for step in route.steps:
            if step.road_name:
                counts[step.road_name] = counts.get(step.road_name, 0) + 1
        usable = [
            (index, step)
            for index, step in enumerate(route.steps)
            if step.road_name and counts[step.road_name] == 1 and 4 <= index < len(route.steps) - 1
        ]
        if not usable:
            continue
        index, step = usable[len(usable) // 2]
        before = sum(1 for earlier in route.steps[:index] if "좌회전" in earlier.instruction)
        total = sum(1 for every in route.steps if "좌회전" in every.instruction)
        # A bound that changes nothing is not a bound: if every left turn happens before the
        # road anyway, the question collapses into the unbounded count v2 already asks.
        if before == total or not 1 <= before <= 7:
            continue
        options = sorted({before, total, max(0, before - 1), before + 1, before + 2})[:4]
        while len(options) < 4:
            options.append(max(options) + 1)
        made.append(
            {
                "question": (
                    f"{origin.name}에서 {destination.name}까지 자동차로, 거리가 가장 짧은 "
                    f"경로로 운전합니다. {step.road_name} 구간에 진입하기 전까지 좌회전을 "
                    "몇 번 하게 되나요?"
                ),
                "options": [f"{value}번" for value in options],
                "answer": options.index(before),
                "classification": "routing",
                "template_id": "routing_turns_before_road",
                "gold_evidence": {
                    "boundary_road": step.road_name,
                    "left_turns_before": before,
                    "left_turns_total": total,
                },
            }
        )
        used.update((origin.place_id, destination.place_id))
    return made


# ------------------------------------ Location-Bearing-Classify + distance, jointly


def poi_bearing_and_distance(
    builder: Builder, pool: Pool, rng: random.Random, count: int
) -> list[dict]:
    """Direction *and* straight-line distance in one answer, so one right half is not enough."""

    from src.tools.spatial import SpatialOperatorRegistry

    ops = SpatialOperatorRegistry()
    landmarks = pool.of("AT4", "CT1", "SW8")
    rng.shuffle(landmarks)
    made: list[dict] = []
    used: set[str] = set()
    opposite = {
        "북쪽": "남쪽", "남쪽": "북쪽", "동쪽": "서쪽", "서쪽": "동쪽",
        "북동쪽": "남서쪽", "남서쪽": "북동쪽", "남동쪽": "북서쪽", "북서쪽": "남동쪽",
    }
    for a, b in itertools.combinations(landmarks[:300], 2):
        if len(made) >= count:
            break
        if a.place_id in used or b.place_id in used:
            continue
        if not 1500 <= distance_m(a, b) <= 9000:
            continue
        if not (builder.resolves_to(a) and builder.resolves_to(b)):
            continue
        bearing = ops.invoke(
            "bearing_to_direction", {"place_a": a.model_dump(), "place_b": b.model_dump()}
        )
        heading = bearing["direction_ko"]
        kilometres = round(distance_m(a, b) / 1000, 1)
        wrong_distance = round(kilometres * rng.choice([0.55, 1.6]), 1)
        if wrong_distance == kilometres:
            continue
        options = [
            f"{heading}, 약 {kilometres:g}km",
            f"{opposite[heading]}, 약 {kilometres:g}km",
            f"{heading}, 약 {wrong_distance:g}km",
            f"{opposite[heading]}, 약 {wrong_distance:g}km",
        ]
        if len(set(options)) < 4:
            continue
        made.append(
            {
                "question": (
                    f"{eul(a.name)} 기준으로 볼 때 {eun(b.name)} 어느 방향에 있고 "
                    "직선거리는 약 얼마인가요?"
                ),
                "options": options,
                "answer": 0,
                "classification": "direction",
                "template_id": "poi_bearing_and_distance",
                "gold_evidence": {"direction_ko": heading, "distance_km": kilometres},
            }
        )
        used.update((a.place_id, b.place_id))
    return made


# ------------------------------------ the category is inferred, not stated


def nearby_from_need(builder: Builder, pool: Pool, rng: random.Random, count: int) -> list[dict]:
    """A need, not a category — and a nearer place of the wrong kind among the options.

    v2 named the category, which is the inference MapEval leaves to the agent. Naming a need
    instead only bites if guessing wrong is punished, so every option set holds a *closer* place
    of a different kind: an agent that retrieves the wrong category finds it and answers with it.
    """

    anchors = pool.of("AT4", "CT1", "SW8", "AD5")
    rng.shuffle(anchors)
    made: list[dict] = []
    used: set[str] = set()
    for anchor in anchors:
        if len(made) >= count:
            break
        if anchor.place_id in used or not builder.resolves_to(anchor):
            continue
        code, need, noun = NEEDS[len(made) % len(NEEDS)]
        try:
            wanted = [
                place
                for place in builder.provider.nearby_search(
                    anchor, category_code=code, radius_m=2000, limit=15
                )
                if place.place_id != anchor.place_id and distance_m(anchor, place) > 5
            ]
        except Exception:  # noqa: BLE001 - an empty neighbourhood is simply not usable
            continue
        if len(wanted) < 3:
            continue
        gold = wanted[0]
        if distance_m(anchor, wanted[1]) - distance_m(anchor, gold) < 100:
            continue
        if not builder.resolves_to(gold):
            continue
        decoy_code = next(
            other for other in DISTRACTOR_CODES if other != code
        )
        decoys: list[Place] = []
        for other in DISTRACTOR_CODES:
            if other == code or len(decoys) >= 2:
                continue
            try:
                found = builder.provider.nearby_search(
                    anchor, category_code=other, radius_m=int(distance_m(anchor, gold)), limit=5
                )
            except Exception:  # noqa: BLE001
                continue
            closer = [
                place
                for place in found
                if place.place_id != anchor.place_id
                and 5 < distance_m(anchor, place) < distance_m(anchor, gold)
            ]
            decoys += take_resolvable(builder, closer, 1)
            decoy_code = other
        if len(decoys) < 2:
            continue
        same_kind = take_resolvable(builder, wanted[1:], 1)
        if not same_kind:
            continue
        options = [gold, *decoys[:2], same_kind[0]]
        if len({place.name for place in options}) < 4:
            continue
        made.append(
            {
                "question": (
                    f"지금 {anchor.name}에 있습니다. {need}. 다음 중 걸어가기에 가장 가까운 곳은 "
                    "어디인가요?"
                ),
                "options": [place.name for place in options],
                "answer": 0,
                "classification": "nearby",
                "template_id": "nearby_from_need",
                "gold_evidence": {
                    "need_category": code,
                    "need_noun": noun,
                    "gold_distance_m": round(distance_m(anchor, gold)),
                    "nearer_wrong_kind_m": [round(distance_m(anchor, d)) for d in decoys[:2]],
                    "decoy_category": decoy_code,
                },
            }
        )
        used.add(anchor.place_id)
    return made


# ------------------------------------------------------------------ assembly

FAMILIES: list[tuple[str, Callable[..., list[dict]], int]] = [
    ("trip_finish_time", trip_finish_time, 16),
    ("trip_latest_departure", trip_latest_departure, 14),
    ("multisegment_total", multisegment_total, 14),
    ("poi_brand_share", poi_brand_share, 14),
    ("routing_turns_before_road", routing_turns_before_road, 14),
    ("poi_bearing_and_distance", poi_bearing_and_distance, 14),
    ("nearby_from_need", nearby_from_need, 14),
]


def finalize(rows: list[dict]) -> list[dict]:
    """Place each gold at an assigned index rather than a randomly drawn one.

    A per-row shuffle is uniform in expectation and lumpy in practice: one family drew index 0
    eight times out of fourteen. Per-family accuracy is a reported number, so the position an
    answer sits at is balanced by construction instead of left to the seed.
    """

    slots: dict[str, list[int]] = {}
    for row in rows:
        family = row["template_id"]
        if family not in slots:
            size = sum(1 for other in rows if other["template_id"] == family)
            positions = [index % 4 for index in range(size)]
            random.Random(f"{family}:{SEED}").shuffle(positions)
            slots[family] = positions

    finished: list[dict] = []
    for index, row in enumerate(rows):
        question_id = f"seoul_kmapeval_v3_{index:03d}"
        target = slots[row["template_id"]].pop()
        remaining = [
            position for position in range(len(row["options"])) if position != row["answer"]
        ]
        random.Random(f"{question_id}:{SEED}").shuffle(remaining)
        order = list(remaining)
        order.insert(target, row["answer"])
        finished.append(
            {
                "id": question_id,
                "question": row["question"],
                "options": [row["options"][position] for position in order],
                "answer": order.index(row["answer"]),
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
            print(
                f"{name}: {len(produced)}/{wanted} (api={builder.provider.api_call_count})",
                flush=True,
            )
            rows.extend(produced)
    finally:
        builder.close()

    finished = finalize(rows)
    Path(args.out).write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in finished) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {args.out} rows={len(finished)}")


if __name__ == "__main__":
    main()
