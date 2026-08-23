"""Generate v5: v4's MapEval method, at MapEval's own difficulty.

`dataset/seoul_kmapeval_v4_mcq_100.jsonl` already reproduces MapEval-API's *method* (evidence
first, options as values, a refusal that carries no signal) and its *class proportions* exactly —
nearby 28 / trip 22 / routing 22 / poi 21 / unanswerable 7 against upstream's 27.7 / 22.3 / 22.0 /
21.3 / 6.7. What it does not reproduce is what makes upstream hard. Read against
`MapEval-API.jsonl` itself, three properties separate the two sets:

1. **Upstream's options sit inside the measurement, not outside it.** Real rows read
   `['18 mins', '19 mins', '20 mins', '21 mins']`, `['99 mins', '98 mins', '95 mins', '100 mins']`,
   `['2:23 PM', '2:37 PM', '1:23 PM', '1:37 PM']`. v4's clock options are 90 to 195 minutes apart
   and its distance options 28% to 75% apart, so any agent that computes at all lands on the gold.
   The spacing was not arbitrary — a Kakao driving *duration* is a live estimate, and the identical
   route came back as 3,243 s and then 4,337 s — but the consequence is that the families resting
   on a duration cannot be asked at upstream's precision and are the ones both agents saturate.
   v5 answers that by **building the tight families out of the reproducible measures only**:
   straight-line distance, DISTANCE-priority road length, turn counts, and orderings by road
   length. `trip_arrival_clock` is dropped rather than widened; the write-up says why. Tight
   options need `straddling_multipliers` underneath them — a fixed multiplier set puts the gold
   at a constant position in the sorted option list, which is a second answer key.

2. **Upstream asks ordinal and membership questions.** "What is the second nearest park to the
   Tower of London?", "Which of the following bank is within 500 meters of the Pantheon?" — the
   nearest place of the right kind is a *distractor* there. v4's nearby families are all
   "the nearest one that satisfies a subtype", which a correct retrieval answers by taking the
   first row. `nearby_second_nearest` and `nearby_within_radius` port the two shapes that punish
   exactly that.

3. **Upstream's unanswerable rows are subjective, not schema-shaped.** "What is the most beautiful
   route…", "Which restaurant nearby is considered the best for fresh seafood?" — the options are
   four real places and none of them is the answer. v4's are all "Kakao has no rating / no opening
   hours / no price level", which is one fact an agent learns once and then applies to every row of
   the family; ReAct scored 7/7 on them. `unanswerable_subjective` asks for a ranking no map
   publishes at all, over real neighbours of the anchor.

Everything else is v4's, imported rather than copied, so a v4-versus-v5 difference is a difference
in these families and nothing else.

Two things this file deliberately does not port, both because Kakao cannot serve them:

- **Route-label options.** Upstream's routing options are `Via Av. Ing. Huergo` and friends,
  because Google returns alternative routes with summaries. Kakao Mobility returns one route per
  priority; `normalize_route` picks the successful one. A "which route" question here would be a
  question about priority flags, not about routes.
- **Attribute questions.** Ratings, price levels and opening hours have no Kakao counterpart, which
  is why they live in the unanswerable class rather than the poi class.

Run it with `--seed` set to something other than the default to get a **held-out** set: nothing
under `src/` has been tuned against a sample it has never seen, and an accuracy measured there is
the first number on this benchmark that is not also a training-set number.
"""

from __future__ import annotations

import itertools
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
    wa,
)
from build_mapeval_benchmark import (
    ORDERED_FAMILIES,
    UNANSWERABLE_TEXT,
    Pool,
    _distance_options,
    finalize,
    nearby_clinic,
    nearby_cuisine,
    routing_distance_via,
    routing_next_turn,
    routing_turn_count,
    straddling_multipliers,
    trip_feasible_count,
    trip_total_distance,
    unanswerable,
)
from builder_cli import run_builder

SEED = 20260821
OUT_PATH = Path(__file__).resolve().parents[1] / "dataset" / "seoul_kmapeval_v5_mcq_100.jsonl"

# The Korean noun for each Kakao category code this file asks by, so a question names a *kind* of
# place the way upstream names "park" or "bank" rather than naming a code.
NOUNS = {
    "CE7": "카페",
    "BK9": "은행",
    "PM9": "약국",
    "CS2": "편의점",
    "PK6": "주차장",
    "FD6": "음식점",
}

# How much closer the gold has to be than the next candidate before an ordinal question is
# decidable. Kakao's own coordinates and the round-trip resolution both move a place by tens of
# metres, so two neighbours 20 m apart are not "first" and "second" in any stable order.
ORDINAL_MARGIN_M = 90.0

# A membership question needs the same guard on both sides of its boundary.
RADIUS_M = 500
INSIDE_CEILING_M = 430.0
OUTSIDE_FLOOR_M = 590.0


# ------------------------------------------------------------------ nearby


# Which category an anchor is asked about, rotated by the anchor rather than by how many rows the
# family has already produced. Keying it on `len(made)` deadlocks: a category with no usable
# neighbourhood anywhere pins every remaining anchor to itself and the family stops growing —
# `nearby_second_nearest` returned 2 of 6 that way, having spent 3,389 Kakao calls to do it.
def _rotate(codes: list[str], index: int) -> str:
    return codes[index % len(codes)]


def nearby_second_nearest(
    builder: Builder, pool: Pool, rng: random.Random, count: int
) -> list[dict]:
    """Upstream's "What is the second nearest park to X?", where the nearest is a distractor.

    Every option is a real place of the requested kind, in a ring around the anchor, so geocoding
    the options tells the agent nothing: they all resolve, they are all the right type, and only
    the ranking separates them. An agent that retrieves correctly and then reports row 0 is wrong
    here, which is the whole point of the shape.
    """

    anchors = pool.of("AT4", "CT1", "SW8", "AD5")
    rng.shuffle(anchors)
    codes = ["CE7", "BK9", "PM9", "CS2"]
    made: list[dict] = []
    used: set[str] = set()
    for index, anchor in enumerate(anchors):
        if len(made) >= count:
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
                    resolved, category_code=code, radius_m=1500, limit=15
                )
                if place.place_id != resolved.place_id and distance_m(resolved, place) > 5
            ]
        except Exception:  # noqa: BLE001 - an empty neighbourhood is simply not usable
            continue
        if len(found) < 4:
            continue
        ranked = sorted(found, key=lambda place: distance_m(resolved, place))[:4]
        gaps = [
            distance_m(resolved, ranked[index + 1]) - distance_m(resolved, ranked[index])
            for index in range(3)
        ]
        if min(gaps) < ORDINAL_MARGIN_M:
            # Two neighbours inside the resolution jitter are not first and second.
            continue
        if len(take_resolvable(builder, ranked, 4)) < 4:
            continue
        if len({place.name for place in ranked}) < 4:
            continue
        noun = NOUNS[code]
        made.append(
            {
                "question": (
                    f"{anchor.name}에서 두 번째로 가까운 {noun}은 다음 중 어디인가요?"
                ),
                "options": [
                    ranked[1].name,
                    ranked[0].name,
                    ranked[2].name,
                    ranked[3].name,
                ],
                "answer": 0,
                "classification": "nearby",
                "mapeval_class": "nearby",
                "template_id": "nearby_second_nearest",
                "gold_evidence": {
                    "anchor": anchor.name,
                    "category_code": code,
                    "ranked_m": [round(distance_m(resolved, place)) for place in ranked],
                },
            }
        )
        used.add(anchor.place_id)
    return made


def nearby_within_radius(
    builder: Builder, pool: Pool, rng: random.Random, count: int
) -> list[dict]:
    """Upstream's "Which of the following bank is within 500 meters of the Pantheon?", as a count.

    Asked as *which one*, the family collapses into the nearest-of-a-type question it was added to
    replace: with exactly one option inside the radius, the one inside is necessarily the nearest,
    and an agent that ranks and reports row 0 is right without ever reading the radius. Asked as
    *how many of these four*, every option has to be measured against the stated number and no
    ranking answers it. The options become a value, which is also the form upstream's non-`nearby`
    classes overwhelmingly take.

    Both sides of the boundary are guarded: an inside place sits under `INSIDE_CEILING_M` and an
    outside one past `OUTSIDE_FLOOR_M`, so a hundred metres of resolution jitter cannot move a
    place across it and change the count.
    """

    anchors = pool.of("AT4", "CT1", "SW8")
    rng.shuffle(anchors)
    codes = ["BK9", "PM9", "CE7", "CS2"]
    made: list[dict] = []
    used: set[str] = set()
    for index, anchor in enumerate(anchors):
        if len(made) >= count:
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
                    resolved, category_code=code, radius_m=1800, limit=15
                )
                if place.place_id != resolved.place_id and distance_m(resolved, place) > 5
            ]
        except Exception:  # noqa: BLE001
            continue
        inside = [place for place in found if distance_m(resolved, place) <= INSIDE_CEILING_M]
        outside = [place for place in found if distance_m(resolved, place) >= OUTSIDE_FLOOR_M]
        # One, two or three of the four inside: never zero and never all four, so neither "none of
        # them" nor "all of them" is ever the answer and the count has to be taken.
        wanted_inside = 1 + (index % 3)
        if len(inside) < wanted_inside or len(outside) < 4 - wanted_inside:
            continue
        near = take_resolvable(builder, inside, wanted_inside)
        far = take_resolvable(builder, outside, 4 - wanted_inside)
        if len(near) < wanted_inside or len(far) < 4 - wanted_inside:
            continue
        listed = [*near, *far]
        if len({place.name for place in listed}) < 4:
            continue
        rng.shuffle(listed)
        noun = NOUNS[code]
        counts = ["한 곳", "두 곳", "세 곳", "네 곳"]
        made.append(
            {
                "question": (
                    f"다음 네 {noun} 중 {anchor.name}에서 반경 {RADIUS_M}m 이내에 있는 곳은 "
                    f"몇 곳인가요? ({', '.join(place.name for place in listed)})"
                ),
                "options": counts,
                "answer": wanted_inside - 1,
                "classification": "radius",
                "mapeval_class": "nearby",
                "template_id": "nearby_within_radius",
                "gold_evidence": {
                    "anchor": anchor.name,
                    "category_code": code,
                    "radius_m": RADIUS_M,
                    "listed": [place.name for place in listed],
                    "listed_m": [round(distance_m(resolved, place)) for place in listed],
                    "inside": wanted_inside,
                },
            }
        )
        used.add(anchor.place_id)
    return made


# --------------------------------------------------------------------- poi


# Wrong lengths on either side of the gold, tighter than v4's on both sides because a
# straight-line distance is reproducible to the metre and can carry upstream's spacing.
BELOW_MULTIPLIERS = (0.78, 0.86, 0.92)
ABOVE_MULTIPLIERS = (1.11, 1.20, 1.28)


def poi_straight_distance_tight(
    builder: Builder, pool: Pool, rng: random.Random, count: int
) -> list[dict]:
    """v4's straight-line family with upstream's option spacing, and no rank tell.

    v4 spread its wrong lengths 28% to 75% off the gold, which any correct geocode separates. The
    two mistakes the question is actually about are much closer than that: `1.28` is the road
    reading of a straight line and `0.92` a geocode that landed on the wrong branch of a chain.
    A straight-line distance is a fact about two coordinates and is reproducible to the metre, so
    unlike a duration it can carry options this tight and still be gradeable — provided the gold
    does not announce itself by where it lands in the sorted order.
    """

    landmarks = pool.of("AT4", "CT1", "SW8", "MT1")
    rng.shuffle(landmarks)
    made: list[dict] = []
    used: set[str] = set()
    for a, b in itertools.combinations(landmarks[:260], 2):
        if len(made) >= count:
            break
        if a.place_id in used or b.place_id in used:
            continue
        metres = distance_m(a, b)
        if not 1500 <= metres <= 14000:
            continue
        if not (builder.resolves_to(a) and builder.resolves_to(b)):
            continue
        options = _distance_options(
            metres, straddling_multipliers(rng, BELOW_MULTIPLIERS, ABOVE_MULTIPLIERS)
        )
        if options is None:
            continue
        made.append(
            {
                "question": f"{wa(a.name)} {b.name} 사이의 직선거리는 얼마인가요?",
                "options": options,
                "answer": 0,
                "classification": "distance",
                "mapeval_class": "poi",
                "template_id": "poi_straight_distance_tight",
                "gold_evidence": {"from": a.name, "to": b.name, "distance_m": round(metres)},
            }
        )
        used.update((a.place_id, b.place_id))
    return made



def poi_direction_distance_straddled(
    builder: Builder, pool: Pool, rng: random.Random, count: int
) -> list[dict]:
    """v4's direction-and-distance family, with the wrong length on either side of the gold.

    v4 builds its four options as {gold length, gold x 1.28} x {heading, opposite heading}, so the
    gold length is always the *smaller* of the two numbers and a closed-book model only has to pick
    one of the two smallest and then flip a coin on the direction. Measured: floor 4/8, which is
    exactly a halved search space plus a coin. Here the wrong length is drawn above or below, so
    the gold's rank among the four printed numbers is not a constant, and the direction stays the
    only free coin.

    Defined here rather than patched into `build_mapeval_benchmark.py` so that v4 keeps
    reproducing byte for byte; the difference between the two is what this benchmark is for.
    """

    from src.tools.spatial import SpatialOperatorRegistry

    ops = SpatialOperatorRegistry()
    opposite = {
        "북쪽": "남쪽", "남쪽": "북쪽", "동쪽": "서쪽", "서쪽": "동쪽",
        "북동쪽": "남서쪽", "남서쪽": "북동쪽", "남동쪽": "북서쪽", "북서쪽": "남동쪽",
    }
    landmarks = pool.of("AT4", "CT1", "SW8")
    rng.shuffle(landmarks)
    made: list[dict] = []
    used: set[str] = set()
    for a, b in itertools.combinations(landmarks[:260], 2):
        if len(made) >= count:
            break
        if a.place_id in used or b.place_id in used:
            continue
        if not 1200 <= distance_m(a, b) <= 12000:
            continue
        if not (builder.resolves_to(a) and builder.resolves_to(b)):
            continue
        heading = ops.invoke(
            "bearing_to_direction", {"place_a": a.model_dump(), "place_b": b.model_dump()}
        )["direction_ko"]
        true_km = round(distance_m(a, b) / 1000, 2)
        factor = rng.choice((0.78, 0.86, 1.11, 1.28))
        wrong_km = round(true_km * factor, 2)
        if true_km == wrong_km:
            continue
        options = [
            f"{heading}, {true_km:g}km",
            f"{opposite[heading]}, {true_km:g}km",
            f"{heading}, {wrong_km:g}km",
            f"{opposite[heading]}, {wrong_km:g}km",
        ]
        made.append(
            {
                "question": (
                    f"{eul(a.name)} 기준으로 {eun(b.name)} 어느 방향에 있으며 "
                    "직선거리는 얼마인가요?"
                ),
                "options": options,
                "answer": 0,
                "classification": "direction",
                "mapeval_class": "poi",
                "template_id": "poi_direction_distance_straddled",
                "gold_evidence": {
                    "from": a.name,
                    "to": b.name,
                    "direction_ko": heading,
                    "distance_km": true_km,
                },
            }
        )
        used.update((a.place_id, b.place_id))
    return made


def poi_address_district(
    builder: Builder, pool: Pool, rng: random.Random, count: int
) -> list[dict]:
    """Built, measured, and **not in `FAMILIES`**: its no-tool floor came back 3/5.

    Kept as the record of a family that does not survive its own floor. Rejecting a place whose
    name carries its own district was not enough — the model knows which 구 an ordinary Seoul
    address sits in without consulting anything, which is the whole answer. Anything that
    reinstates it has to beat that floor first.

    Where a place *is*, administratively — upstream's location half of the poi class.

    The districts offered as wrong answers are drawn from the pool's own `district` field, so the
    option set is a list of districts Kakao actually writes in addresses rather than a list of
    districts that ought to exist. A place whose name carries its district ("...강남점") is
    rejected: that row would be answerable from the option text alone, which is the no-tool leak
    v4's build loop was written to catch.
    """

    districts = sorted({record.district for record in _pool_records() if record.district})
    if len(districts) < 4:
        return []
    candidates = [record for record in _pool_records() if record.district]
    rng.shuffle(candidates)
    made: list[dict] = []
    used: set[str] = set()
    for record in candidates:
        if len(made) >= count:
            break
        place, district = record.place, record.district
        if place.place_id in used:
            continue
        if district[:-1] in place.name or district in place.name:
            continue
        others = [value for value in districts if value != district]
        if any(value[:-1] in place.name for value in others):
            continue
        if not builder.resolves_to(place):
            continue
        wrong = rng.sample(others, 3)
        made.append(
            {
                "question": f"{eun(place.name)} 서울특별시 어느 구에 있나요?",
                "options": [district, *wrong],
                "answer": 0,
                "classification": "poi",
                "mapeval_class": "poi",
                "template_id": "poi_address_district",
                "gold_evidence": {"place": place.name, "district": district},
            }
        )
        used.add(place.place_id)
    return made


class _PoolRecord:
    """A pool row with the `district` `to_place` drops, since the question asks about it."""

    __slots__ = ("place", "district")

    def __init__(self, place: Place, district: str) -> None:
        self.place = place
        self.district = district


def _pool_records() -> list[_PoolRecord]:
    from benchmark_core import load_pool, plausible_name, to_place

    seen: set[str] = set()
    records: list[_PoolRecord] = []
    for row in load_pool():
        if not plausible_name(row["name"]) or row["name"] in seen:
            continue
        seen.add(row["name"])
        records.append(_PoolRecord(to_place(row), str(row.get("district") or "")))
    return records


# -------------------------------------------------------------------- trip


def trip_optimal_order(
    builder: Builder, pool: Pool, rng: random.Random, count: int
) -> list[dict]:
    """Upstream's signature trip shape: "Give the most optimized order to visit these places."

    Its own hardest class — Spatial-Agent scores 55.2% on trip over MapEval-API — and the one v4
    has no counterpart for. All four options hold the *same* three stops in different orders, so
    no amount of geocoding separates them: the answer exists only in the route matrix.

    Ordered by **road length at DISTANCE priority**, not by duration. Upstream asks for the fastest
    order and its annotators could, because a Google duration was stable enough to write an option
    against; a Kakao duration is not — the same fixed route came back as 3,243 s and then 4,337 s,
    which is wider than the gap between two orders. Ordering by length keeps the question a fact
    about the road network. The deviation is deliberate and belongs in the write-up.
    """

    bases = pool.of("AD5")
    # Marts and cultural venues alongside the attractions: a closed-book model orders three famous
    # landmarks from memory of the city, and the family's floor came back 5/8 when every stop was
    # one. An ordinary mart carries no such prior.
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
            if len(chosen) == 3:
                break
            if _spread([*chosen, sight], 1500) and builder.resolves_to(sight):
                chosen.append(sight)
        if len(chosen) < 3:
            continue
        legs: dict[tuple[str, str], int] = {}
        broken = False
        for a, b in itertools.permutations([base, *chosen], 2):
            metres = builder.distance_m_driving(a, b)
            if metres is None:
                broken = True
                break
            legs[(a.place_id, b.place_id)] = metres
        if broken:
            continue
        totals: list[tuple[int, tuple[Place, ...]]] = []
        for order in itertools.permutations(chosen):
            path = (base, *order, base)
            totals.append(
                (
                    sum(legs[(path[i].place_id, path[i + 1].place_id)] for i in range(4)),
                    order,
                )
            )
        totals.sort(key=lambda item: item[0])
        if totals[1][0] - totals[0][0] < 300:
            # Under 300 m the two orders are the same drive, and the option is not wrong enough
            # to be a wrong answer.
            continue
        picked = [totals[0], *totals[1:4]]
        rendered = [" → ".join(place.name for place in order) for _, order in picked]
        if len(set(rendered)) < 4:
            continue
        stays = [rng.choice([1.0, 1.5]) for _ in chosen]
        stay_text = ", ".join(
            f"{eul(place.name)} {stay:g}시간" for place, stay in zip(chosen, stays, strict=True)
        )
        made.append(
            {
                "question": (
                    f"{base.name}에서 출발해 {stay_text} 동안 둘러본 뒤 다시 "
                    f"{base.name}로 돌아옵니다. 자동차 총 주행거리가 가장 짧은 방문 순서는 "
                    "다음 중 무엇인가요?"
                ),
                "options": rendered,
                "answer": 0,
                "classification": "trip",
                "mapeval_class": "trip",
                "template_id": "trip_optimal_order",
                "gold_evidence": {
                    "base": base.name,
                    "total_m": [total for total, _ in picked],
                    "margin_m": totals[1][0] - totals[0][0],
                },
            }
        )
        used.add(base.place_id)
    return made


def _spread(places: list[Place], metres: float) -> bool:
    return all(
        distance_m(a, b) >= metres for a, b in itertools.combinations(places, 2)
    )


# ----------------------------------------------------------- unanswerable


# What a map does not rank, asked the way a person asks it. Every one of these is a judgement,
# not a missing column: Kakao could publish every field it has and the question would still have
# no answer, which is upstream's own unanswerable shape.
JUDGEMENTS: list[tuple[str, str]] = [
    ("분위기가 가장 좋은 카페는 다음 중 어디인가요?", "CE7"),
    ("가장 친절한 약국은 다음 중 어디인가요?", "PM9"),
    ("가장 조용해서 오래 앉아 있기 좋은 카페는 다음 중 어디인가요?", "CE7"),
    ("가장 맛있는 음식점은 다음 중 어디인가요?", "FD6"),
    ("주차하기 가장 편한 대형마트는 다음 중 어디인가요?", "MT1"),
]


def unanswerable_subjective(
    builder: Builder, pool: Pool, rng: random.Random, count: int
) -> list[dict]:
    """Upstream's "Which restaurant nearby is considered the best for fresh seafood?".

    Four real neighbours of the anchor and a refusal, where the refusal is gold — not because a
    column is missing, but because the ranking the question asks for is not a property of a place.
    An agent that retrieves correctly gets four names it cannot order, which is the state upstream's
    unanswerable rows put it in.
    """

    anchors = pool.of("SW8", "AT4", "CT1")
    rng.shuffle(anchors)
    made: list[dict] = []
    used: set[str] = set()
    for index, anchor in enumerate(anchors):
        if len(made) >= count:
            break
        if anchor.place_id in used:
            continue
        resolved = builder.as_resolved(anchor)
        if resolved is None:
            continue
        ask, code = JUDGEMENTS[index % len(JUDGEMENTS)]
        try:
            found = [
                place
                for place in builder.provider.nearby_search(
                    resolved, category_code=code, radius_m=1500, limit=15
                )
                if place.place_id != resolved.place_id
            ]
        except Exception:  # noqa: BLE001
            continue
        candidates = take_resolvable(builder, found, 3)
        if len(candidates) < 3:
            continue
        made.append(
            {
                "question": f"{anchor.name} 근처에서 {ask}",
                "options": [UNANSWERABLE_TEXT, *[place.name for place in candidates]],
                "answer": 0,
                "classification": "nearby",
                "mapeval_class": "unanswerable",
                "template_id": "unanswerable_subjective",
                "gold_evidence": {
                    "anchor": anchor.name,
                    "category_code": code,
                    "unrankable": True,
                },
            }
        )
        used.add(anchor.place_id)
    return made


# --------------------------------------------------------------- assembly

# MapEval-API's class mix again, scaled from 300 to 100: nearby 28, trip 22, routing 22, poi 21,
# unanswerable 7. What moved against v4 is *inside* the classes.
FAMILIES: list[tuple[str, Callable[..., list[dict]], int]] = [
    # nearby 28: the two subtype families lose eight rows to the two ordinal/membership ones.
    ("nearby_clinic_subtype", nearby_clinic, 10),
    ("nearby_cuisine_subtype", nearby_cuisine, 8),
    ("nearby_second_nearest", nearby_second_nearest, 6),
    ("nearby_within_radius", nearby_within_radius, 4),
    # poi 21: v4's two measures, both re-cut so the gold does not announce itself by rank.
    # `poi_address_district` was built and dropped: its no-tool floor was 3/5, because the
    # model knows which 구 a Seoul place sits in without consulting anything.
    ("poi_direction_distance_straddled", poi_direction_distance_straddled, 10),
    ("poi_straight_distance_tight", poi_straight_distance_tight, 11),
    # routing 22: unchanged from v4. All three rest on DISTANCE-priority lengths or on the
    # guidance list, both of which the network reproduces.
    ("routing_distance_via", routing_distance_via, 8),
    ("routing_turn_count", routing_turn_count, 7),
    ("routing_next_turn", routing_next_turn, 7),
    # trip 22: `trip_arrival_clock` is gone — a clock gold needs options spaced past the traffic,
    # and those options are what made the family free. Its rows go to the ordering question, which
    # is upstream's own trip shape and the class it scores worst on.
    ("trip_optimal_order", trip_optimal_order, 8),
    ("trip_total_distance", trip_total_distance, 7),
    ("trip_feasible_count", trip_feasible_count, 7),
    # unanswerable 7: four judgements the map cannot make, three columns Kakao does not publish.
    ("unanswerable_subjective", unanswerable_subjective, 4),
    ("unanswerable", unanswerable, 3),
]


def main() -> None:
    run_builder(
        families=FAMILIES,
        open_builder=Builder.open,
        make_pool=Pool,
        finalize=finalize,
        ordered=ORDERED_FAMILIES | {"nearby_within_radius"},
        canonical_seed=SEED,
        canonical_prefix="seoul_kmapeval_v5",
        canonical_out=OUT_PATH,
    )


if __name__ == "__main__":
    main()
