"""Generate v6: v5's families, each raised by one step, and the radius family's word order fixed.

v5 put the floor at the chance rate and separated the architectures on two families. What it did
not do is stop being *answerable in one hop*. Read against the v5 run
(`docs/REFERENCE_MAPPING.md`), seven of its fourteen families were saturated by both agents:
`poi_straight_distance_tight` 11/11 and 11/11, `poi_direction_distance_straddled` 10/10 and 10/10,
`routing_next_turn` 7/7 and 7/7, `trip_feasible_count` 7/7 and 7/7, `nearby_second_nearest` 5/6 and
6/6, `routing_turn_count` 6/7 and 7/7, `nearby_within_radius` 4/4 and 0/4.

A saturated family cannot show a difference, and tightening its options does not help when the
measure is exact: a haversine over two resolved coordinates is arithmetic, not an estimate, so no
option spacing defeats it. The way up is **composition** — a question whose answer needs two
measurements and an operation between them — and **ordinality**, which denies the agent the first
row of a ranking. Every family here moves along one of those two axes:

| v5 family | v6 family | what changed |
| --- | --- | --- |
| `poi_straight_distance_tight` | `poi_distance_difference` | two haversines and a subtraction |
| `poi_direction_distance_straddled` | `poi_farthest_of_three` | three haversines and a maximum |
| `nearby_second_nearest` | `nearby_kth_nearest` | k drawn from 2..4, options from ranks 1..6 |
| `nearby_clinic_subtype` | `nearby_subtype_kth` | the k-th of a named subtype, not the nearest |
| `routing_next_turn` | `routing_nth_turn` | count into the guidance list, not match a road |
| `routing_turn_count` | `routing_turn_count_via` | counted on a route through a waypoint |
| `routing_distance_via` | `routing_detour_cost` | the detour's *cost*: via length minus direct |
| `trip_optimal_order` | `trip_optimal_order_four` | four stops, 24 orders instead of 6 |
| `trip_total_distance` | `trip_total_distance_four` | four stops, five legs |
| `nearby_within_radius` | `nearby_within_radius_count` | word order fixed, radius 300/500/800 |

The word-order fix is not a difficulty change and is the reason Spatial-Agent scored 0/4 on the v5
family. v5 asked "다음 네 약국 중 <anchor>에서 반경 500m …", and `_extract_anchor` splits a radius
question on " 반경" and takes everything before it — so the anchor it handed `batch_geocode` was
the whole clause `"다음 네 약국 중 신이문역 1호선에서"`, which resolves to nothing and fails every
step downstream. All four failures are that, in all four traces. The question was at fault, not the
agent: no Korean speaker puts the list before the landmark. v6 asks
"<anchor>에서 반경 500m 이내에 있는 약국은 다음 네 곳 중 몇 곳인가요? (…)", which is both the
ordinary word order and the one `_extract_anchor` was written for.

`nearby_cuisine_subtype`, `trip_feasible_count`, `unanswerable` and `unanswerable_subjective` are
imported unchanged: the first two were not saturated (5/8 and 6/8; 7/7 and 7/7 but on a count the
agents reach differently), and the unanswerable pair is where Spatial-Agent already loses 4 rows to
ReAct. Raising a family that is already discriminating only spends rows.

Class proportions stay MapEval-API's: nearby 28, poi 21, routing 22, trip 22, unanswerable 7.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from collections.abc import Callable
from pathlib import Path

from benchmark_core import (
    Builder,
    Place,
    distance_m,
    eul,
    eun,
    take_resolvable,
)
from build_mapeval_benchmark import (
    COMPLAINTS,
    Pool,
    _distance_options,
    _guidance,
    _subtype,
    finalize,
    nearby_cuisine,
    straddling_multipliers,
    unanswerable,
)
from build_mapeval_v5_benchmark import (
    ABOVE_MULTIPLIERS,
    BELOW_MULTIPLIERS,
    NOUNS,
    ORDINAL_MARGIN_M,
    _rotate,
    _spread,
    unanswerable_subjective,
)

SEED = 20260821_06
OUT_PATH = Path(__file__).resolve().parents[1] / "dataset" / "seoul_kmapeval_v6_mcq_100.jsonl"

ORDINALS = {
    2: "두 번째",
    3: "세 번째",
    4: "네 번째",
    5: "다섯 번째",
    6: "여섯 번째",
    7: "일곱 번째",
}

# Radii a question may state. One fixed radius lets an agent learn the boundary instead of reading
# it, and the three here are far enough apart that the same neighbourhood gives different answers.
RADII = (300, 500, 800)

# How far past the stated radius an outside place has to sit, and how far inside an inside one.
# Both the pool's coordinates and round-trip name resolution move a place by tens of metres, so a
# place within this band of the boundary is not reliably on either side of it.
BOUNDARY_MARGIN_M = 70.0
# How many anchors the radius-count family may resolve while hunting for an uncovered rung.
RADIUS_SCAN_LIMIT = 24
# How many anchors the k-th nearest family may build while hunting for a scarce ordinal.
ORDINAL_SCAN_LIMIT = 24


# ------------------------------------------------------------------ nearby


def nearby_kth_nearest(
    builder: Builder, pool: Pool, rng: random.Random, count: int
) -> list[dict]:
    """v5's second-nearest question with the ordinal drawn and the option ranks drawn.

    v5 always asked for the second and always offered ranks 1 through 4, so "the second option in
    the retrieval" was a constant recipe. Here k is 2, 3 or 4 and the three wrong options are drawn
    from ranks 1 through 6, which means the agent has to hold an ordered list rather than reach a
    fixed row — and the nearest place is only sometimes among the choices.
    """

    anchors = pool.of("AT4", "CT1", "SW8", "AD5")
    rng.shuffle(anchors)
    codes = ["CE7", "BK9", "PM9", "CS2"]
    produced_counts: dict[int, int] = {2: 0, 3: 0, 4: 0}
    made: list[tuple[int, dict]] = []
    used: set[str] = set()
    for index, anchor in enumerate(anchors):
        # Least-used-first only spreads k across what the anchors in hand can offer, and they
        # cannot offer much: under a 90 m ordinal margin six of one draw's eight anchors were
        # separable at k=2 only, because ranks three through five of a dense neighbourhood sit
        # within 90 m of each other. So keep scanning while a value is still short, and pick the
        # rows at the end.
        if len(made) >= count and min(produced_counts.values()) >= count // 4:
            break
        if len(made) >= ORDINAL_SCAN_LIMIT:
            break
        if anchor.place_id in used:
            continue
        resolved = builder.as_resolved(anchor)
        if resolved is None:
            continue
        code = _rotate(codes, index)
        try:
            found = [
                place
                for place in builder.provider.nearby_search(
                    resolved, category_code=code, radius_m=1800, limit=25
                )
                if place.place_id != resolved.place_id and distance_m(resolved, place) > 5
            ]
        except Exception:  # noqa: BLE001 - an empty neighbourhood is simply not usable
            continue
        ranked = sorted(found, key=lambda place: distance_m(resolved, place))[:6]

        # Only the ranks the ordinal depends on have to be separable: ranks 1..k decide which
        # place is k-th, and rank k+1 has to stay behind it. Demanding the margin between every
        # one of six neighbours -- and that all six round-trip -- produced 0 rows in 8 over 5,889
        # Kakao calls, because a dense city puts four cafes 30 m apart.
        feasible: list[int] = []
        for candidate_k in (2, 3, 4):
            if len(found) < candidate_k + 2:
                continue
            gaps = [
                distance_m(resolved, ranked[position + 1])
                - distance_m(resolved, ranked[position])
                for position in range(min(candidate_k, len(ranked) - 1))
            ]
            if gaps and min(gaps) >= ORDINAL_MARGIN_M:
                feasible.append(candidate_k)

        # `kth = 2 + (index % 3)` keyed the ordinal on the anchor loop index, so k was spent
        # wherever the loop happened to succeed rather than across its three values: seven of
        # v7's eight rows came out k=2. That matters because ranking the four options against
        # each other -- which is what the agent does when it does not retrieve -- answers a k-th
        # question whenever all k-1 nearer places happen to be among the three decoys, and that
        # is C(m-k, 4-k) / C(m-1, 3): 60% at k=2, 30% at k=3, 10% at k=4. A family that is 56%
        # answerable without retrieving cannot show whether an agent retrieved. Spend k the way
        # `trip_feasible_count` spends its rungs instead.
        if not feasible:
            continue
        kth = min(feasible, key=lambda k: (produced_counts[k], k))
        gold = ranked[kth - 1]
        decoys = rng.sample([place for place in ranked if place is not gold], 3)
        options = [gold, *decoys]
        if len({place.name for place in options}) < 4:
            continue
        # Resolvability is required of the four names the question prints, not of the ranking they
        # were drawn from: an option an agent cannot turn back into a place is unanswerable, but a
        # neighbour that never reaches the page costs nothing.
        if len(take_resolvable(builder, options, 4)) < 4:
            continue
        row = {
            "question": (
                f"{anchor.name}에서 {ORDINALS[kth]}로 가까운 {NOUNS[code]}은 "
                "다음 중 어디인가요?"
            ),
            "options": [place.name for place in options],
            "answer": 0,
            "classification": "nearby",
            "mapeval_class": "nearby",
            "template_id": "nearby_kth_nearest",
            "gold_evidence": {
                "anchor": anchor.name,
                "category_code": code,
                "k": kth,
                "ranked_m": [round(distance_m(resolved, place)) for place in ranked],
            },
        }
        made.append((kth, row))
        produced_counts[kth] += 1
        used.add(anchor.place_id)

    # Spend the rows across the values rather than handing them all to whichever the anchors were
    # generous with. Quotas are equal, and the remainder goes to k=2 because it is the value the
    # city can always supply.
    base, remainder = divmod(count, 3)
    quota = {2: base + remainder, 3: base, 4: base}
    selected: list[dict] = []
    for value in (4, 3, 2):
        for k, row in made:
            if k == value and quota[value] > 0 and len(selected) < count:
                quota[value] -= 1
                selected.append(row)
    for _, row in made:
        if len(selected) >= count:
            break
        if row not in selected:
            selected.append(row)
    return selected[:count]


def nearby_subtype_kth(
    builder: Builder, pool: Pool, rng: random.Random, count: int
) -> list[dict]:
    """The *k-th* nearest clinic of a named specialty, both constraints at once.

    v4's version asked for the nearest of a subtype, which a correct retrieval answers by filtering
    and taking row 0 — 9/10 and 10/10 on v5. Asking for the second or third denies that row while
    keeping the subtype trap: the option set still holds a nearer clinic of the wrong specialty, so
    reading names alone and reading distances alone both pick wrong, and now taking the first
    correctly-filtered row does too.
    """

    anchors = pool.of("AT4", "CT1", "SW8", "AD5")
    rng.shuffle(anchors)
    subtype_counts: dict[int, int] = {2: 0, 3: 0}
    made: list[dict] = []
    used: set[str] = set()
    for index, pooled in enumerate(anchors):
        if len(made) >= count:
            break
        anchor = builder.as_resolved(pooled)
        if pooled.place_id in used or anchor is None:
            continue
        prompt, token = COMPLAINTS[index % len(COMPLAINTS)]
        try:
            found = [
                place
                for place in builder.provider.nearby_search(
                    anchor, category_code="HP8", radius_m=3000, limit=45
                )
                if place.place_id != anchor.place_id and distance_m(anchor, place) > 5
            ]
        except Exception:  # noqa: BLE001
            continue
        wanted = [place for place in found if _subtype(place, token)]
        if len(wanted) < 4:
            continue
        gaps = [
            distance_m(anchor, wanted[position + 1]) - distance_m(anchor, wanted[position])
            for position in range(min(len(wanted), 4) - 1)
        ]
        if min(gaps) < 60:
            continue
        # Same defect as the family above: the ordinal was keyed on the anchor loop index.
        # Both values are always available once the four-deep margin holds, so spend the
        # least-used one.
        kth = min((2, 3), key=lambda k: (subtype_counts[k], k))
        gold = wanted[kth - 1]
        others = [place for place in wanted[:4] if place is not gold]
        trap = take_resolvable(
            builder,
            [
                place
                for place in found
                if not _subtype(place, token)
                and distance_m(anchor, place) < distance_m(anchor, gold) - 5
            ],
            1,
        )
        if len(others) < 2 or not trap:
            continue
        if len(take_resolvable(builder, [gold, *others[:2]], 3)) < 3:
            continue
        options = [gold, *others[:2], *trap]
        if len({place.name for place in options}) < 4:
            continue
        made.append(
            {
                "question": (
                    f"지금 {anchor.name}에 있습니다. {prompt}. "
                    f"여기서 {ORDINALS[kth]}로 가까운 {eun(token)} 다음 중 어디인가요?"
                ),
                "options": [place.name for place in options],
                "answer": 0,
                "classification": "nearby",
                "mapeval_class": "nearby",
                "template_id": "nearby_subtype_kth",
                "gold_evidence": {
                    "anchor": anchor.name,
                    "required_subtype": token,
                    "k": kth,
                    "subtype_ranked_m": [round(distance_m(anchor, place)) for place in wanted[:4]],
                    "nearer_wrong_subtype_m": [round(distance_m(anchor, place)) for place in trap],
                },
            }
        )
        subtype_counts[kth] += 1
        used.add(pooled.place_id)
    return made


def nearby_within_radius_count(
    builder: Builder, pool: Pool, rng: random.Random, count: int
) -> list[dict]:
    """v5's membership count, asked in Korean word order and with the radius drawn.

    Two changes. The anchor comes first, because v5 put the option list ahead of it and that is
    what made `_extract_anchor` hand `batch_geocode` the phrase `"다음 네 약국 중 신이문역 1호선
    에서"`; every one of Spatial-Agent's four failures on the family is that unresolved anchor, not
    a miscount. And the radius is 300, 500 or 800 rather than always 500, so the boundary is a
    number the question states rather than one an agent can settle on.
    """

    anchors = pool.of("AT4", "CT1", "SW8")
    rng.shuffle(anchors)
    codes = ["BK9", "PM9", "CE7", "CS2"]
    produced_counts: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}
    made: list[tuple[int, dict]] = []
    used: set[str] = set()
    scanned = 0
    # Coverage cannot be left to the draw. Picking the least-used rung only spreads the rungs the
    # first `count` anchors happen to offer, and under holdout seed 927451 three of them sat in
    # blocks dense enough that `outside` came back empty -- which leaves "네 곳" the only feasible
    # rung, since 1 through 3 each need a place beyond the radius and 4 needs none. The draw
    # shipped a four-rung ladder that could only ever answer three of them, and
    # `audit_dataset.py` failed it. So keep scanning anchors past `count` while a rung is still
    # uncovered, and select for coverage afterwards.
    for index, anchor in enumerate(anchors):
        covered = {rung for rung, _ in made}
        if len(made) >= count and len(covered) >= min(count, 4):
            break
        if scanned >= RADIUS_SCAN_LIMIT:
            break
        if anchor.place_id in used:
            continue
        resolved = builder.as_resolved(anchor)
        if resolved is None:
            continue
        scanned += 1
        code = _rotate(codes, index)
        radius = RADII[index % len(RADII)]
        try:
            found = [
                place
                for place in builder.provider.nearby_search(
                    resolved, category_code=code, radius_m=radius * 3, limit=25
                )
                if place.place_id != resolved.place_id and distance_m(resolved, place) > 5
            ]
        except Exception:  # noqa: BLE001
            continue
        inside = [
            place
            for place in found
            if distance_m(resolved, place) <= radius - BOUNDARY_MARGIN_M
        ]
        outside = [
            place
            for place in found
            if distance_m(resolved, place) >= radius + BOUNDARY_MARGIN_M
        ]
        # 1 through 4, not 1 through 3. v5 drew from three counts against a four-rung ladder, so
        # "네 곳" was never the answer on any row of the family and the question was a three-way
        # choice wearing a fourth option. Every rung a question offers has to be reachable --
        # and reached: keying the count on the anchor's position in a list that mostly fails
        # produced 1, 2, 4, 1 over four rows, so the count is spent on whichever rung the family
        # has least of, the way `trip_feasible_count` spends its budget.
        feasible = [
            value
            for value in (1, 2, 3, 4)
            if len(inside) >= value and len(outside) >= 4 - value
        ]
        if not feasible:
            continue
        wanted_inside = min(feasible, key=lambda value: (produced_counts[value], value))
        near = take_resolvable(builder, inside, wanted_inside)
        far = take_resolvable(builder, outside, 4 - wanted_inside)
        if len(near) < wanted_inside or len(far) < 4 - wanted_inside:
            continue
        listed = [*near, *far]
        if len({place.name for place in listed}) < 4:
            continue
        rng.shuffle(listed)
        produced_counts[wanted_inside] += 1
        row = {
            "question": (
                f"{anchor.name}에서 반경 {radius}m 이내에 있는 {NOUNS[code]}은 아래 목록 중 "
                f"몇 곳인가요? ({', '.join(place.name for place in listed)})"
            ),
            "options": ["한 곳", "두 곳", "세 곳", "네 곳"],
            "answer": wanted_inside - 1,
            "classification": "radius",
            "mapeval_class": "nearby",
            "template_id": "nearby_within_radius_count",
            "gold_evidence": {
                "anchor": anchor.name,
                "category_code": code,
                "radius_m": radius,
                "listed": [place.name for place in listed],
                "listed_m": [round(distance_m(resolved, place)) for place in listed],
                "inside": wanted_inside,
            },
        }
        made.append((wanted_inside, row))
        used.add(anchor.place_id)

    # One row per distinct rung first, then fill the remaining slots in draw order.
    selected: list[dict] = []
    seen: set[int] = set()
    for rung, row in made:
        if rung not in seen:
            seen.add(rung)
            selected.append(row)
    for _, row in made:
        if len(selected) >= count:
            break
        if row not in selected:
            selected.append(row)
    return selected[:count]


# --------------------------------------------------------------------- poi


def poi_distance_difference(
    builder: Builder, pool: Pool, rng: random.Random, count: int
) -> list[dict]:
    """How much *farther* one place is than another — two haversines and a subtraction.

    A single straight-line distance is exact arithmetic over two resolved coordinates, which is why
    v5's version stayed at 11/11 for both agents no matter how tight its options were. A difference
    is exact too, but it needs three names resolved instead of two and one operation between the
    two measurements, and it fails whenever any of the three resolutions does. That is composition,
    which is the axis the paper's claim lives on.
    """

    landmarks = pool.of("AT4", "CT1", "SW8", "MT1")
    rng.shuffle(landmarks)
    made: list[dict] = []
    used: set[str] = set()
    for anchor, near, far in itertools.combinations(landmarks[:150], 3):
        if len(made) >= count:
            break
        if any(place.place_id in used for place in (anchor, near, far)):
            continue
        first, second = distance_m(anchor, near), distance_m(anchor, far)
        gap = abs(first - second)
        if not (900 <= gap <= 9000):
            continue
        if not all(1000 <= value <= 15000 for value in (first, second)):
            continue
        if not all(builder.resolves_to(place) for place in (anchor, near, far)):
            continue
        options = _distance_options(
            gap, straddling_multipliers(rng, BELOW_MULTIPLIERS, ABOVE_MULTIPLIERS)
        )
        if options is None:
            continue
        made.append(
            {
                "question": (
                    f"{anchor.name}에서 {near.name}까지의 직선거리와 {anchor.name}에서 "
                    f"{far.name}까지의 직선거리는 얼마나 차이가 나나요?"
                ),
                "options": options,
                "answer": 0,
                "classification": "distance",
                "mapeval_class": "poi",
                "template_id": "poi_distance_difference",
                "gold_evidence": {
                    "anchor": anchor.name,
                    "first": near.name,
                    "second": far.name,
                    "first_m": round(first),
                    "second_m": round(second),
                    "difference_m": round(gap),
                },
            }
        )
        used.update(place.place_id for place in (anchor, near, far))
    return made


def poi_farthest_of_three(
    builder: Builder, pool: Pool, rng: random.Random, count: int
) -> list[dict]:
    """The distance to whichever of three places is farthest — three measurements and a maximum.

    v5's direction-and-distance family asked for one bearing and one length, both computed from one
    pair, and both agents took 10/10. Here the agent has to measure three pairs and then select
    among them, and the answer is the *value*, so identifying the right place is necessary and not
    sufficient. The runner-up is kept 500 m back, which is far enough that a resolution landing
    within `Builder`'s 200 m tolerance cannot change which place is farthest.
    """

    landmarks = pool.of("AT4", "CT1", "SW8", "MT1")
    rng.shuffle(landmarks)
    made: list[dict] = []
    used: set[str] = set()
    for anchor, a, b, c in itertools.combinations(landmarks[:55], 4):
        if len(made) >= count:
            break
        if any(place.place_id in used for place in (anchor, a, b, c)):
            continue
        targets = [a, b, c]
        distances = [distance_m(anchor, place) for place in targets]
        ordered = sorted(distances, reverse=True)
        if not (2000 <= ordered[0] <= 15000) or ordered[0] - ordered[1] < 500:
            continue
        if min(distances) < 800:
            continue
        if not all(builder.resolves_to(place) for place in (anchor, a, b, c)):
            continue
        options = _distance_options(
            ordered[0], straddling_multipliers(rng, BELOW_MULTIPLIERS, ABOVE_MULTIPLIERS)
        )
        if options is None:
            continue
        listed = ", ".join(place.name for place in targets)
        made.append(
            {
                "question": (
                    f"{anchor.name}에서 다음 세 곳({listed}) 가운데 직선거리가 가장 먼 곳까지는 "
                    "얼마인가요?"
                ),
                "options": options,
                "answer": 0,
                "classification": "distance",
                "mapeval_class": "poi",
                "template_id": "poi_farthest_of_three",
                "gold_evidence": {
                    "anchor": anchor.name,
                    "targets": [place.name for place in targets],
                    "target_m": [round(value) for value in distances],
                    "farthest_m": round(ordered[0]),
                },
            }
        )
        used.update(place.place_id for place in (anchor, a, b, c))
    return made


# ----------------------------------------------------------------- routing


def routing_detour_cost(
    builder: Builder, pool: Pool, rng: random.Random, count: int
) -> list[dict]:
    """What the detour costs: the via route's length minus the direct route's.

    v5 asked for the via route's length outright, which is one `directions` call and a read.
    Subtracting the direct route makes it two routes and an operation, and it punishes an agent
    that routes only one of them — which is the same failure the trip families punish, on two legs
    instead of five. Both routes at DISTANCE priority, so both are facts about the road network.
    """

    origins = pool.of("SW8", "AD5")
    waypoints = pool.of("AT4", "CT1", "MT1")
    rng.shuffle(origins)
    rng.shuffle(waypoints)
    made: list[dict] = []
    used: set[str] = set()
    for origin, destination in itertools.combinations(origins[:80], 2):
        if len(made) >= count:
            break
        if origin.place_id in used or destination.place_id in used:
            continue
        if not 4000 <= distance_m(origin, destination) <= 14000:
            continue
        if not (builder.resolves_to(origin) and builder.resolves_to(destination)):
            continue
        direct = builder.route(origin, destination, priority="DISTANCE")
        if direct is None:
            continue
        detour = None
        via = None
        for candidate in waypoints:
            if candidate.place_id in {origin.place_id, destination.place_id}:
                continue
            if not 2000 <= distance_m(origin, candidate) <= 12000:
                continue
            if not builder.resolves_to(candidate):
                continue
            routed = builder.route(
                origin, destination, waypoints=(candidate,), priority="DISTANCE"
            )
            if routed is None:
                continue
            extra = routed.distance_m - direct.distance_m
            if extra >= 1200:
                detour, via = extra, candidate
                break
        if via is None or detour is None:
            continue
        options = _distance_options(
            detour, straddling_multipliers(rng, BELOW_MULTIPLIERS, ABOVE_MULTIPLIERS)
        )
        if options is None:
            continue
        made.append(
            {
                "question": (
                    f"{origin.name}에서 {destination.name}까지 자동차로 갈 때, 거리가 가장 짧은 "
                    f"경로로 곧장 가는 경우와 {eul(via.name)} 경유해서 가는 경우의 총 주행거리는 "
                    "얼마나 차이가 나나요?"
                ),
                "options": options,
                "answer": 0,
                "classification": "routing",
                "mapeval_class": "routing",
                "template_id": "routing_detour_cost",
                "gold_evidence": {
                    "origin": origin.name,
                    "destination": destination.name,
                    "via": via.name,
                    "direct_m": direct.distance_m,
                    "via_m": direct.distance_m + detour,
                    "detour_m": detour,
                },
            }
        )
        used.update((origin.place_id, destination.place_id))
    return made


def routing_nth_turn(
    builder: Builder, pool: Pool, rng: random.Random, count: int
) -> list[dict]:
    """The n-th instruction on the drive, counted rather than matched.

    v5 named a road and asked what follows it, so an agent that finds the road in the guidance list
    reads the next line. Counting to the n-th instruction requires the whole list in order, and a
    single dropped or merged step moves the answer — which is exactly what a guidance list tests.
    """

    places = pool.of("SW8", "AT4", "CT1")
    rng.shuffle(places)
    made: list[dict] = []
    used: set[str] = set()
    routes: list[tuple[Place, Place, list]] = []
    for origin, destination in itertools.combinations(places[:140], 2):
        if len(routes) >= count * 3:
            break
        if origin.place_id in used or destination.place_id in used:
            continue
        if not 3000 <= distance_m(origin, destination) <= 13000:
            continue
        if not (builder.resolves_to(origin) and builder.resolves_to(destination)):
            continue
        route = builder.route(origin, destination, priority="DISTANCE")
        if route is None or len(route.steps or []) < 9:
            continue
        routes.append((origin, destination, list(route.steps)))
        used.update((origin.place_id, destination.place_id))
    for index, (origin, destination, steps) in enumerate(routes):
        if len(made) >= count:
            break
        position = 3 + (index % 3)
        if position >= len(steps):
            continue
        gold_text = _guidance(steps[position])
        if not gold_text:
            continue
        elsewhere: list[str] = []
        for step in steps:
            text = _guidance(step)
            if text and text != gold_text and text not in elsewhere:
                elsewhere.append(text)
        for other, (_, _, other_steps) in enumerate(routes):
            if other == index or len(elsewhere) >= 6:
                continue
            for step in other_steps:
                text = _guidance(step)
                if text and text != gold_text and text not in elsewhere:
                    elsewhere.append(text)
        if len(elsewhere) < 3:
            continue
        made.append(
            {
                "question": (
                    f"{origin.name}에서 {destination.name}까지 자동차로, 거리가 가장 짧은 경로로 "
                    f"운전합니다. 주행 안내를 처음부터 세었을 때 {ORDINALS[position + 1]} 안내는 "
                    "무엇인가요?"
                ),
                "options": [gold_text, *rng.sample(elsewhere, 3)],
                "answer": 0,
                "classification": "routing",
                "mapeval_class": "routing",
                "template_id": "routing_nth_turn",
                "gold_evidence": {
                    "origin": origin.name,
                    "destination": destination.name,
                    "step_index": position,
                    "steps": len(steps),
                },
            }
        )
    return made


def routing_turn_count_via(
    builder: Builder, pool: Pool, rng: random.Random, count: int
) -> list[dict]:
    """Left turns on a route *through a waypoint*, which is a route no single call returns first.

    v5 counted turns on a direct route. Adding a waypoint means the count belongs to a route the
    agent has to ask for correctly — origin, waypoint and destination in one call — and an agent
    that routes the two legs separately and adds the counts gets a different number, because the
    turn joining them is not in either leg.
    """

    from src.tools.spatial import SpatialOperatorRegistry

    ops = SpatialOperatorRegistry()
    places = pool.of("SW8", "AT4", "CT1")
    stops = pool.of("MT1", "AD5")
    rng.shuffle(places)
    rng.shuffle(stops)
    made: list[dict] = []
    used: set[str] = set()
    for origin, destination in itertools.combinations(places[:110], 2):
        if len(made) >= count:
            break
        if origin.place_id in used or destination.place_id in used:
            continue
        if not 3000 <= distance_m(origin, destination) <= 12000:
            continue
        if not (builder.resolves_to(origin) and builder.resolves_to(destination)):
            continue
        via = next(
            (
                place
                for place in stops
                if place.place_id not in {origin.place_id, destination.place_id}
                and 1500 <= distance_m(origin, place) <= 10000
                and 1500 <= distance_m(place, destination) <= 10000
                and builder.resolves_to(place)
            ),
            None,
        )
        if via is None:
            continue
        route = builder.route(origin, destination, waypoints=(via,), priority="DISTANCE")
        if route is None or not route.steps:
            continue
        analysis = ops.invoke("steps_analysis", {"route": route.model_dump()})
        turns = int(analysis.get("left_turn_count") or 0)
        if turns < 3:
            continue
        # How many wrong counts fall below the gold is drawn, not fixed. v4 sampled three of
        # `turns +/- 1, +/- 2`, which can never leave the gold as the smallest or the largest of
        # the four -- so "never pick an extreme" narrowed it to two options with no map. Measured
        # on the shipped v4 file, that family's gold sat at sorted rank 2 on all five numeric rows.
        below = [value for value in (turns - 3, turns - 2, turns - 1) if value >= 0]
        above = [turns + 1, turns + 2, turns + 3]
        picked = rng.randint(max(0, 3 - len(above)), min(3, len(below)))
        wrong = [*rng.sample(below, picked), *rng.sample(above, 3 - picked)]
        options = [f"{turns}번", *[f"{value}번" for value in wrong]]
        if len(set(options)) < 4:
            continue
        made.append(
            {
                "question": (
                    f"{origin.name}에서 {eul(via.name)} 들러 {destination.name}까지 자동차로, "
                    "거리가 가장 짧은 경로로 이동합니다. 주행 안내에 따르면 좌회전을 몇 번 "
                    "해야 하나요?"
                ),
                "options": options,
                "answer": 0,
                "classification": "routing",
                "mapeval_class": "routing",
                "template_id": "routing_turn_count_via",
                "gold_evidence": {
                    "origin": origin.name,
                    "via": via.name,
                    "destination": destination.name,
                    "left_turns": turns,
                    "steps": len(route.steps),
                },
            }
        )
        used.update((origin.place_id, destination.place_id))
    return made


# -------------------------------------------------------------------- trip


def _tour_totals(
    builder: Builder, base: Place, stops: list[Place]
) -> list[tuple[int, tuple[Place, ...]]] | None:
    """Every visiting order of `stops` from and back to `base`, by road length at DISTANCE."""

    legs: dict[tuple[str, str], int] = {}
    for a, b in itertools.permutations([base, *stops], 2):
        metres = builder.distance_m_driving(a, b)
        if metres is None:
            return None
        legs[(a.place_id, b.place_id)] = metres
    totals: list[tuple[int, tuple[Place, ...]]] = []
    for order in itertools.permutations(stops):
        path = (base, *order, base)
        totals.append(
            (
                sum(legs[(path[i].place_id, path[i + 1].place_id)] for i in range(len(path) - 1)),
                order,
            )
        )
    totals.sort(key=lambda item: item[0])
    return totals


def trip_optimal_order_four(
    builder: Builder, pool: Pool, rng: random.Random, count: int
) -> list[dict]:
    """Four stops instead of three: 24 orders to compare rather than 6.

    Both agents scored 6/8 on the three-stop version, which is the shape upstream reports 55.2% on,
    so the family was already discriminating; four stops raise it without changing what it asks.
    The three wrong orders are drawn from those at least 500 m worse than the gold — nearer than
    that and the two drives are the same drive, and the option is not wrong so much as tied.
    """

    bases = pool.of("AD5")
    sights = pool.of("AT4", "CT1", "MT1")
    rng.shuffle(bases)
    rng.shuffle(sights)
    made: list[dict] = []
    used: set[str] = set()
    for base in bases:
        if len(made) >= count:
            break
        if base.place_id in used or not builder.resolves_to(base):
            continue
        near = [place for place in sights if 2000 <= distance_m(base, place) <= 12000]
        rng.shuffle(near)
        chosen: list[Place] = []
        for sight in near:
            if len(chosen) == 4:
                break
            if _spread([*chosen, sight], 1500) and builder.resolves_to(sight):
                chosen.append(sight)
        if len(chosen) < 4:
            continue
        totals = _tour_totals(builder, base, chosen)
        if totals is None:
            continue
        gold_total, gold_order = totals[0]
        worse = [order for total, order in totals[1:] if total - gold_total >= 500]
        if len(worse) < 3:
            continue
        rendered = [
            " → ".join(place.name for place in order)
            for order in (gold_order, *rng.sample(worse, 3))
        ]
        if len(set(rendered)) < 4:
            continue
        stays = [rng.choice([1.0, 1.5]) for _ in chosen]
        stay_text = ", ".join(
            f"{eul(place.name)} {stay:g}시간" for place, stay in zip(chosen, stays, strict=True)
        )
        made.append(
            {
                "question": (
                    f"{base.name}에서 출발해 {stay_text} 동안 둘러본 뒤 다시 {base.name}로 "
                    "돌아옵니다. 자동차 총 주행거리가 가장 짧은 방문 순서는 다음 중 무엇인가요?"
                ),
                "options": rendered,
                "answer": 0,
                "classification": "trip",
                "mapeval_class": "trip",
                "template_id": "trip_optimal_order_four",
                "gold_evidence": {
                    "base": base.name,
                    "stops": [place.name for place in chosen],
                    "best_m": gold_total,
                    "orders_compared": len(totals),
                },
            }
        )
        used.add(base.place_id)
    return made


def trip_total_distance_four(
    builder: Builder, pool: Pool, rng: random.Random, count: int
) -> list[dict]:
    """Five legs to add instead of four, on the family that already separated the architectures.

    v5 ran ReAct 2/7 against Spatial-Agent 7/7 here: accumulating legs by hand across tool calls
    drifts, and one `distance_matrix` does not. A fifth leg is one more chance to drift, and the
    order is stated so the question is arithmetic rather than optimization.
    """

    bases = pool.of("AD5", "SW8")
    sights = pool.of("AT4", "CT1", "MT1")
    rng.shuffle(bases)
    rng.shuffle(sights)
    made: list[dict] = []
    used: set[str] = set()
    for base in bases:
        if len(made) >= count:
            break
        if base.place_id in used or not builder.resolves_to(base):
            continue
        near = [place for place in sights if 2000 <= distance_m(base, place) <= 11000]
        rng.shuffle(near)
        chosen: list[Place] = []
        for sight in near:
            if len(chosen) == 4:
                break
            if _spread([*chosen, sight], 1500) and builder.resolves_to(sight):
                chosen.append(sight)
        if len(chosen) < 4:
            continue
        chain = [base, *chosen, base]
        legs = [
            builder.distance_m_driving(a, b)
            for a, b in zip(chain, chain[1:], strict=False)
        ]
        if any(value is None for value in legs):
            continue
        total = sum(legs)
        options = _distance_options(
            total, straddling_multipliers(rng, BELOW_MULTIPLIERS, ABOVE_MULTIPLIERS)
        )
        if options is None:
            continue
        listed = ", ".join(place.name for place in chosen)
        made.append(
            {
                "question": (
                    f"{base.name}에서 출발해 {listed} 순서로 들른 뒤 {base.name}로 돌아옵니다. "
                    "구간마다 거리가 가장 짧은 경로를 쓴다면 총 주행거리는 얼마인가요?"
                ),
                "options": options,
                "answer": 0,
                "classification": "trip",
                "mapeval_class": "trip",
                "template_id": "trip_total_distance_four",
                "gold_evidence": {
                    "base": base.name,
                    "stops": [place.name for place in chosen],
                    "leg_m": legs,
                    "total_m": total,
                },
            }
        )
        used.add(base.place_id)
    return made



def trip_feasible_count_five(
    builder: Builder, pool: Pool, rng: random.Random, count: int
) -> list[dict]:
    """How many of *five* stops fit in the budget, so that every rung of the ladder is reachable.

    v4's four-stop version cannot ever answer "네 곳", and the reason is structural rather than
    unlucky. It rejects any row whose count the stay times alone already give, because such a row
    is answerable without the map. For the *maximum* count that rejection always fires: if travel
    plus stays fits every stop, then stays alone fits every stop too. So the top rung was dead on
    all three files built from it — v4, v5 and the holdout — and the family was a three-way choice
    printed as a four-way one.

    Offering five stops against the same four-rung ladder puts the dead count at five, which is not
    on the ladder. One more leg to route is also one more chance to drift, which is the direction
    this benchmark is moving anyway.
    """

    bases = pool.of("AD5")
    sights = pool.of("AT4", "CT1", "MT1")
    rng.shuffle(bases)
    rng.shuffle(sights)
    words = ["한 곳", "두 곳", "세 곳", "네 곳"]
    produced_counts: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}
    made: list[dict] = []
    used: set[str] = set()
    for base in bases:
        if len(made) >= count:
            break
        if base.place_id in used or not builder.resolves_to(base):
            continue
        near = [place for place in sights if 1500 < distance_m(base, place) < 9000]
        chosen = take_resolvable(builder, near, 5)
        if len(chosen) < 5:
            continue
        stays = [rng.choice([1.0, 1.5]) * 3600 for _ in chosen]
        chain = [base, *chosen]
        legs = [
            builder.route(a, b, priority="DISTANCE")
            for a, b in zip(chain, chain[1:], strict=False)
        ]
        if any(route is None for route in legs):
            continue

        def fits(budget: float, factor: float, legs=legs, stays=stays) -> int:
            spent = 0.0
            for index, route in enumerate(legs):
                spent += route.duration_s * factor + stays[index]
                if spent > budget:
                    return index
            return len(legs)

        by_count: dict[int, tuple[float, int]] = {}
        for candidate in range(int(2.0 * 3600), int(13 * 3600), 900):
            true_count = fits(candidate, 1.0)
            if not 1 <= true_count <= 4 or true_count in by_count:
                continue
            if fits(candidate, 0.7) != true_count or fits(candidate, 1.3) != true_count:
                continue  # the answer would depend on the traffic
            stays_only = 0.0
            no_travel = 0
            for stay in stays:
                stays_only += stay
                if stays_only > candidate:
                    break
                no_travel += 1
            if no_travel == true_count:
                continue  # answerable without the map, which is the flaw being repaired
            by_count[true_count] = (candidate, no_travel)
        if not by_count:
            continue
        chosen_count = min(by_count, key=lambda value: (produced_counts[value], value))
        budget, no_travel = by_count[chosen_count]
        produced_counts[chosen_count] += 1
        visits = ", ".join(
            f"{eul(place.name)} {stay / 3600:g}시간"
            for place, stay in zip(chosen, stays, strict=False)
        )
        made.append(
            {
                "question": (
                    f"지금 {base.name}에 있습니다. {visits} 동안 적힌 순서대로 둘러보려 합니다. "
                    f"총 {budget / 3600:g}시간이 있고 자동차로 이동합니다. 몇 곳을 방문할 수 "
                    "있나요?"
                ),
                "options": words,
                "answer": chosen_count - 1,
                "classification": "trip",
                "mapeval_class": "trip",
                "template_id": "trip_feasible_count_five",
                "gold_evidence": {
                    "base": base.name,
                    "stops": [place.name for place in chosen],
                    "stay_s": stays,
                    "travel_s": [route.duration_s for route in legs],
                    "budget_s": budget,
                    "count_without_travel": no_travel,
                },
            }
        )
        used.add(base.place_id)
    return made


# --------------------------------------------------------------- assembly

FAMILIES: list[tuple[str, Callable[..., list[dict]], int]] = [
    # nearby 28
    ("nearby_subtype_kth", nearby_subtype_kth, 10),
    ("nearby_kth_nearest", nearby_kth_nearest, 8),
    ("nearby_cuisine_subtype", nearby_cuisine, 6),
    ("nearby_within_radius_count", nearby_within_radius_count, 4),
    # poi 21
    ("poi_distance_difference", poi_distance_difference, 11),
    ("poi_farthest_of_three", poi_farthest_of_three, 10),
    # routing 22
    ("routing_detour_cost", routing_detour_cost, 8),
    ("routing_nth_turn", routing_nth_turn, 7),
    ("routing_turn_count_via", routing_turn_count_via, 7),
    # trip 22
    ("trip_optimal_order_four", trip_optimal_order_four, 8),
    ("trip_total_distance_four", trip_total_distance_four, 7),
    ("trip_feasible_count_five", trip_feasible_count_five, 7),
    # unanswerable 7
    ("unanswerable_subjective", unanswerable_subjective, 4),
    ("unanswerable", unanswerable, 3),
]

V6_ORDERED_FAMILIES = frozenset(
    {"trip_feasible_count_five", "nearby_within_radius_count"}
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--id-prefix", default="seoul_kmapeval_v6")
    args = parser.parse_args()
    if args.id_prefix != "seoul_kmapeval_v6" and args.seed == SEED:
        raise SystemExit(
            f"--seed {SEED} is this builder's default, so --id-prefix {args.id_prefix} would "
            "relabel the tuned set rather than draw a new sample. Pick another seed."
        )

    builder = Builder.open()
    pool = Pool()
    rows: list[dict] = []
    try:
        for name, function, quota in FAMILIES:
            if args.families and name not in args.families:
                continue
            wanted = max(1, round(quota * args.scale))
            rng = random.Random(f"{args.seed}:{name}")
            produced = function(builder, pool, rng, wanted)
            print(
                f"{name}: {len(produced)}/{wanted} (api={builder.provider.api_call_count})",
                flush=True,
            )
            rows.extend(produced)
    finally:
        builder.close()

    finished = finalize(
        rows,
        seed=args.seed,
        prefix=args.id_prefix,
        ordered=V6_ORDERED_FAMILIES,
    )
    Path(args.out).write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in finished) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {args.out} rows={len(finished)}")


if __name__ == "__main__":
    main()
