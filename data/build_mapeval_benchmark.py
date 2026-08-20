"""Generate the Korean benchmark MapEval's own way.

`dataset/seoul_kmapeval_v2_mcq_100.jsonl` matched MapEval-API's *class proportions* and missed its
*method*, and the measurement in `docs/REFERENCE_MAPPING.md` says what that cost: 91 of 100 drafted
graphs contained no retrieval operator, because the four MCQ options were the candidate set. This
builder follows the method instead.

What MapQaTor actually does (`mapqator-backend/database/schema.sql`, `mapeval-api/dataset.json`):
an annotator issues map API calls that are cached in `places` / `nearby` / `distance` /
`directions` / `inside`, the calls are assembled into one `context`, and then a person writes the
question and the options by hand against that evidence (`dataset.username`), with `human` holding
the human baseline. Three properties follow, and they are what this file ports:

1. **The options are a value, not the candidate set — except for `nearby`.** Counted over
   `dataset.json`: poi 36 value / 28 names, routing 44 / 22, trip 48 / 19, but nearby 19 / 64. A
   question whose options are `['South, 13.45 kilometers', ...]` or `['61.224 km', ...]` or
   `['10.13', '10.23', ...]` cannot be answered by geocoding the options, so the agent has to
   *find* and *measure*. This is the single change that closes v2's shortcut.
2. **Where the options are names, a constraint decides, not proximity.** Upstream asks for an
   *orthopedic* hospital, a restaurant *rated 4.8+*, the nearest *mosque* — so the nearest place of
   the wrong kind is a wrong answer. Kakao's category paths carry the same subtype vocabulary
   (`의료,건강 > 병원 > 정형외과`, `음식점 > 중식 > 중국요리`).
3. **Some questions have no answer.** 20 of upstream's 300 carry `correct = -1`: the API cannot
   支持 any option. Kakao publishes no rating, no price level and no opening hours, so the
   attribute half of MapEval-API is unanswerable here by nature rather than by choice — and that is
   exactly what those rows should be, instead of being quietly dropped as v2 dropped them.

Two deliberate deviations, both recorded in `docs/REFERENCE_MAPPING.md`:

- **The questions are templated, not hand-written.** There are no annotators here. The *shape* of
  each family is upstream's; the phrasing is generated.
- **The refusal option is not announced.** `Evaluator2.py` prepends "Option0: Unanswerable" only
  when a row is unanswerable, which tells the model the answer before it reads the question. Here
  "주어진 지도 정보로는 알 수 없음" is an ordinary option, gold on the unanswerable rows and a
  distractor on roughly a quarter of the answerable ones, so its presence carries no signal.

And one rule that is *dropped* on purpose. v2 rejected every question whose best and second-best
were close; MapEval's annotators wrote the question and read whatever the evidence said. Margins
here exist only where the provider is not reproducible — a Kakao driving *duration* is a live
estimate (the identical route came back as 3,243 s and then 4,337 s), so clock and duration golds
still have to out-space the traffic. Distances, bearings and DISTANCE-priority route lengths are
facts about the road network and need no engineered margin.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
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
    wa,
)

SEED = 20260820
OUT_PATH = Path(__file__).resolve().parents[1] / "dataset" / "seoul_kmapeval_v4_mcq_100.jsonl"

UNANSWERABLE_TEXT = "주어진 지도 정보로는 알 수 없음"

# MapEval-API's own class mix, scaled from 300 to 100: nearby 83, trip 67, routing 66, poi 64,
# unanswerable 20.
MAPEVAL_CLASS_QUOTA = {"nearby": 28, "trip": 22, "routing": 22, "poi": 21, "unanswerable": 7}

# A complaint and the clinic that treats it, the way upstream asks for an orthopedic hospital.
# The subtype is named in the question, as upstream names it; what the agent must not do is answer
# with the nearer dentist.
COMPLAINTS: list[tuple[str, str]] = [
    ("계단에서 발을 헛디뎌 발목이 심하게 부었습니다", "정형외과"),
    ("어금니가 깨져서 씹을 때마다 아픕니다", "치과"),
    ("귀가 먹먹하고 코가 막혀 잠을 못 잤습니다", "이비인후과"),
    ("눈이 계속 충혈되고 뻑뻑합니다", "안과"),
    ("팔에 두드러기가 번지고 있습니다", "피부과"),
    ("어제부터 배탈이 나서 계속 속이 안 좋습니다", "내과"),
    ("아이가 밤새 열이 내리지 않습니다", "소아청소년과"),
]

# The same idea one category over: a cuisine, not "a restaurant".
CUISINES: list[tuple[str, str]] = [
    ("점심으로 중국요리가 먹고 싶습니다", "중식"),
    ("저녁에 초밥이나 일본식 정식을 먹으려 합니다", "일식"),
    ("파스타나 스테이크 같은 양식을 먹으려 합니다", "양식"),
    ("간단하게 분식으로 때우려 합니다", "분식"),
    ("빵과 커피로 아침을 해결하려 합니다", "제과,베이커리"),
]


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


def _km(metres: float) -> str:
    # Always one decimal: `:g` printed "약 3km" beside "약 2.3km", and the option that looks
    # different is a tell that has nothing to do with the map.
    return f"약 {metres / 1000:.1f}km"


def _distance_options(metres: float, multipliers: tuple[float, ...]) -> list[str] | None:
    """A measured length plus wrong lengths of the same order.

    `0.78` is the straight-line reading of a road distance and `1.28` the road reading of a
    straight line -- the two mistakes the question is actually about, so they belong among the
    options rather than a random spread.
    """

    options = [_km(metres), *[_km(metres * factor) for factor in multipliers]]
    return options if len(set(options)) == len(options) else None


def _clock(moment: datetime) -> str:
    hour, minute = moment.hour, moment.minute
    period = "오전" if hour < 12 else "오후"
    display = hour if hour <= 12 else hour - 12
    return f"{period} {display}시 {minute:02d}분"


def _clock_options(gold: datetime, rng: random.Random) -> list[str]:
    """Clock times spaced past the traffic, for the one measure Kakao does not reproduce."""

    picked = rng.sample([-150, -95, 90, 140, 195], 3)
    return [_clock(gold), *[_clock(gold + timedelta(minutes=value)) for value in picked]]


def _subtype(place: Place, token: str) -> bool:
    """The same test the tools apply, so the builder and the grader cannot disagree.

    Selecting the gold on the category path alone while `filter_places` reads the name too put a
    `미래아이 소아청소년과안과의원` 153 m from the anchor -- filed under 소아청소년과, named for an
    eye clinic -- outside a 안과 question whose gold sat at 497 m. Whatever decides the answer has
    to be what decides the question.
    """

    from src.tools.spatial import matches_required_type

    return matches_required_type(place.model_dump(), token)


# ----------------------------------------------------------------- nearby


def _constrained_nearby(
    builder: Builder,
    pool: Pool,
    rng: random.Random,
    count: int,
    *,
    code: str,
    prompts: list[tuple[str, str]],
    label: Callable[[str], str],
    template_id: str,
) -> list[dict]:
    """The nearest place *of a named subtype*, with nearer places of sibling subtypes offered.

    Upstream's shape: "Suggest me an orthopedic hospital near ICT tower" against a nearby list
    that holds every kind of hospital. Proximity alone answers it wrong, which is the point --
    and unlike v2, the option set is not the candidate set, because the agent has to retrieve the
    neighbourhood to know which of these is nearest at all.
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
        prompt, token = prompts[len(made) % len(prompts)]
        try:
            found = [
                place
                for place in builder.provider.nearby_search(
                    anchor, category_code=code, radius_m=2500, limit=45
                )
                if place.place_id != anchor.place_id and distance_m(anchor, place) > 5
            ]
        except Exception:  # noqa: BLE001 - an empty neighbourhood is simply not usable
            continue
        wanted = [place for place in found if _subtype(place, token)]
        if not wanted:
            continue
        gold = wanted[0]
        # Strictly nearer, with a metre to spare: a decoy that rounds to the same distance as the
        # gold makes the recorded evidence say two places are equally near, and the invariant this
        # family rests on is that proximity alone never selects the gold.
        nearer_siblings = [
            place
            for place in found
            if not _subtype(place, token)
            and distance_m(anchor, place) < distance_m(anchor, gold) - 5
        ]
        decoys = take_resolvable(builder, nearer_siblings, 3)
        if len(decoys) < 3 or not builder.resolves_to(gold):
            continue
        options = [gold, *decoys]
        if len({place.name for place in options}) < 4:
            continue
        made.append(
            {
                "question": (
                    f"지금 {anchor.name}에 있습니다. {prompt}. "
                    f"여기서 가장 가까운 {eun(label(token))} 다음 중 어디인가요?"
                ),
                "options": [place.name for place in options],
                "answer": 0,
                "classification": "nearby",
                "mapeval_class": "nearby",
                "template_id": template_id,
                "gold_evidence": {
                    "anchor": anchor.name,
                    "required_subtype": token,
                    "category_code": code,
                    "gold_distance_m": round(distance_m(anchor, gold)),
                    "nearer_wrong_subtype_m": [
                        round(distance_m(anchor, place)) for place in decoys
                    ],
                },
            }
        )
        used.add(anchor.place_id)
    return made


def nearby_clinic(builder: Builder, pool: Pool, rng: random.Random, count: int) -> list[dict]:
    return _constrained_nearby(
        builder, pool, rng, count,
        code="HP8", prompts=COMPLAINTS, label=lambda token: token,
        template_id="nearby_clinic_subtype",
    )


def nearby_cuisine(builder: Builder, pool: Pool, rng: random.Random, count: int) -> list[dict]:
    return _constrained_nearby(
        builder, pool, rng, count,
        code="FD6", prompts=CUISINES, label=lambda token: f"{token} 음식점",
        template_id="nearby_cuisine_subtype",
    )


# -------------------------------------------------------------------- poi


def poi_direction_distance(
    builder: Builder, pool: Pool, rng: random.Random, count: int
) -> list[dict]:
    """Upstream's `['South, 13.45 kilometers', 'East, 13.56 kilometers', ...]`, in Korean."""

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
        near = round(distance_m(a, b) / 1000, 2)
        far = round(near * 1.28, 2)  # the road reading of a straight line
        if near == far:
            continue
        options = [
            f"{heading}, {near:g}km",
            f"{opposite[heading]}, {near:g}km",
            f"{heading}, {far:g}km",
            f"{opposite[heading]}, {far:g}km",
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
                "template_id": "poi_direction_distance",
                "gold_evidence": {
                    "from": a.name,
                    "to": b.name,
                    "direction_ko": heading,
                    "distance_km": near,
                },
            }
        )
        used.update((a.place_id, b.place_id))
    return made


def poi_which_is_closer(
    builder: Builder, pool: Pool, rng: random.Random, count: int
) -> list[dict]:
    """A two-option question, the shape 16 of upstream's 300 rows use."""

    anchors = pool.of("AD5", "SW8", "AT4")
    rivals = pool.of("CT1", "AT4", "MT1")
    rng.shuffle(anchors)
    rng.shuffle(rivals)
    made: list[dict] = []
    used: set[str] = set()
    for anchor in anchors:
        if len(made) >= count:
            break
        if anchor.place_id in used or not builder.resolves_to(anchor):
            continue
        ranked = sorted(
            (place for place in rivals if 800 <= distance_m(anchor, place) <= 15000),
            key=lambda place: distance_m(anchor, place),
        )
        if len(ranked) < 8:
            continue
        near, far = ranked[1], ranked[6]
        # No engineered margin: whatever the map says is the answer, as upstream has it.
        if not (builder.resolves_to(near) and builder.resolves_to(far)):
            continue
        made.append(
            {
                "question": (
                    f"{anchor.name}에서 {wa(near.name)} {far.name} 중 "
                    "직선거리로 더 가까운 곳은 어디인가요?"
                ),
                "options": [near.name, far.name],
                "answer": 0,
                "classification": "distance",
                "mapeval_class": "poi",
                "template_id": "poi_which_is_closer",
                "gold_evidence": {
                    "anchor": anchor.name,
                    "near": near.name,
                    "far": far.name,
                    "near_m": round(distance_m(anchor, near)),
                    "far_m": round(distance_m(anchor, far)),
                },
            }
        )
        used.add(anchor.place_id)
    return made


def poi_straight_distance(
    builder: Builder, pool: Pool, rng: random.Random, count: int
) -> list[dict]:
    """How far apart, as a number. The 0.78 option is the road-vs-straight-line confusion."""

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
        options = _distance_options(metres, (1.28, 1.75, 0.62))
        if options is None:
            continue
        made.append(
            {
                "question": f"{wa(a.name)} {b.name} 사이의 직선거리는 얼마인가요?",
                "options": options,
                "answer": 0,
                "classification": "distance",
                "mapeval_class": "poi",
                "template_id": "poi_straight_distance",
                "gold_evidence": {"from": a.name, "to": b.name, "distance_m": round(metres)},
            }
        )
        used.update((a.place_id, b.place_id))
    return made


# ---------------------------------------------------------------- routing


def routing_distance_via(
    builder: Builder, pool: Pool, rng: random.Random, count: int
) -> list[dict]:
    """Upstream's "How much distance do I need to cover if I go via A14?" -- a length, in km.

    Routed at DISTANCE, which is the only priority Kakao reproduces, so the gold is a fact about
    the road network rather than a snapshot of the traffic and needs no widened options.
    """

    origins = pool.of("SW8", "AD5")
    waypoints = pool.of("AT4", "CT1", "MT1")
    rng.shuffle(origins)
    rng.shuffle(waypoints)
    made: list[dict] = []
    used: set[str] = set()
    for origin, destination in itertools.combinations(origins[:120], 2):
        if len(made) >= count:
            break
        if origin.place_id in used or destination.place_id in used:
            continue
        if not 4000 <= distance_m(origin, destination) <= 16000:
            continue
        via = next(
            (
                place
                for place in waypoints
                if 1500 < distance_m(origin, place) < distance_m(origin, destination)
                and distance_m(place, destination) > 1500
            ),
            None,
        )
        if via is None:
            continue
        if not all(builder.resolves_to(place) for place in (origin, destination, via)):
            continue
        route = builder.route(origin, destination, waypoints=[via], priority="DISTANCE")
        if route is None or route.distance_m <= 0:
            continue
        options = _distance_options(route.distance_m, (0.78, 1.42, 1.9))
        if options is None:
            continue
        made.append(
            {
                "question": (
                    f"{origin.name}에서 {eul(via.name)} 경유하여 {destination.name}까지 "
                    "자동차로, 거리가 가장 짧은 경로로 이동하려 합니다. 주행 거리는 얼마인가요?"
                ),
                "options": options,
                "answer": 0,
                "classification": "routing",
                "mapeval_class": "routing",
                "template_id": "routing_distance_via",
                "gold_evidence": {
                    "origin": origin.name,
                    "destination": destination.name,
                    "via": via.name,
                    "distance_m": round(route.distance_m),
                },
            }
        )
        used.update((origin.place_id, destination.place_id))
    return made


def routing_turn_count(
    builder: Builder, pool: Pool, rng: random.Random, count: int
) -> list[dict]:
    """Upstream's "How many times do I need to merge onto a toll road?" -- a count."""

    from src.tools.spatial import SpatialOperatorRegistry

    ops = SpatialOperatorRegistry()
    places = pool.of("SW8", "AT4", "CT1")
    rng.shuffle(places)
    made: list[dict] = []
    used: set[str] = set()
    for origin, destination in itertools.combinations(places[:140], 2):
        if len(made) >= count:
            break
        if origin.place_id in used or destination.place_id in used:
            continue
        if not 3000 <= distance_m(origin, destination) <= 13000:
            continue
        if not (builder.resolves_to(origin) and builder.resolves_to(destination)):
            continue
        route = builder.route(origin, destination, priority="DISTANCE")
        if route is None or not route.steps:
            continue
        analysis = ops.invoke("steps_analysis", {"route": route.model_dump()})
        turns = int(analysis.get("left_turn_count") or 0)
        if turns < 2:
            continue
        wrong = [value for value in (turns - 2, turns - 1, turns + 1, turns + 2) if value >= 0]
        options = [f"{turns}번", *[f"{value}번" for value in rng.sample(wrong, 3)]]
        if len(set(options)) < 4:
            continue
        made.append(
            {
                "question": (
                    f"{origin.name}에서 {destination.name}까지 자동차로, 거리가 가장 짧은 "
                    "경로로 이동합니다. 주행 안내에 따르면 좌회전을 몇 번 해야 하나요?"
                ),
                "options": options,
                "answer": 0,
                "classification": "routing",
                "mapeval_class": "routing",
                "template_id": "routing_turn_count",
                "gold_evidence": {
                    "origin": origin.name,
                    "destination": destination.name,
                    "left_turns": turns,
                    "steps": len(route.steps),
                },
            }
        )
        used.update((origin.place_id, destination.place_id))
    return made


def routing_next_turn(
    builder: Builder, pool: Pool, rng: random.Random, count: int
) -> list[dict]:
    """What the guidance says after a named road. The options are instructions, never places."""

    places = pool.of("SW8", "AT4", "CT1")
    rng.shuffle(places)
    made: list[dict] = []
    used: set[str] = set()
    pending: list[tuple[Place, Place, list]] = []
    for origin, destination in itertools.combinations(places[:140], 2):
        if len(pending) >= count * 3:
            break
        if origin.place_id in used or destination.place_id in used:
            continue
        if not 3000 <= distance_m(origin, destination) <= 13000:
            continue
        if not (builder.resolves_to(origin) and builder.resolves_to(destination)):
            continue
        route = builder.route(origin, destination, priority="DISTANCE")
        if route is None or len(route.steps or []) < 8:
            continue
        pending.append((origin, destination, list(route.steps)))
        used.update((origin.place_id, destination.place_id))
    for index, (origin, destination, steps) in enumerate(pending):
        if len(made) >= count:
            break
        named = [
            position
            for position, step in enumerate(steps[:-1])
            if step.road_name and step.road_name.strip()
        ]
        if not named:
            continue
        position = named[len(named) // 2]
        road = steps[position].road_name
        gold = steps[position + 1]
        gold_text = _guidance(gold)
        # Distractors from *this* route first: another instruction on the same drive is the
        # plausible wrong answer, where one lifted from a drive across the city can be excluded
        # by knowing the city rather than by reading the guidance.
        pool_texts: list[str] = []
        for step in steps:
            text = _guidance(step)
            if text and text != gold_text and text not in pool_texts:
                pool_texts.append(text)
        for other_index, (_, _, other_steps) in enumerate(pending):
            if other_index == index or len(pool_texts) >= 6:
                continue
            for step in other_steps:
                text = _guidance(step)
                if text and text != gold_text and text not in pool_texts:
                    pool_texts.append(text)
        if not gold_text or len(pool_texts) < 3:
            continue
        made.append(
            {
                "question": (
                    f"{origin.name}에서 {destination.name}까지 자동차로, 거리가 가장 짧은 "
                    f"경로로 운전하고 있습니다. {road} 구간에 진입한 뒤 이어지는 주행 안내는 "
                    "무엇인가요?"
                ),
                "options": [gold_text, *rng.sample(pool_texts, 3)],
                "answer": 0,
                "classification": "routing",
                "mapeval_class": "routing",
                "template_id": "routing_next_turn",
                "gold_evidence": {
                    "origin": origin.name,
                    "destination": destination.name,
                    "after_road": road,
                    "step_index": position + 1,
                },
            }
        )
    return made


def _guidance(step) -> str:
    text = (step.instruction or "").strip()
    road = (step.road_name or "").strip()
    if not text:
        return ""
    return f"{text} ({road})" if road else text


# ------------------------------------------------------------------- trip


def trip_arrival_clock(
    builder: Builder, pool: Pool, rng: random.Random, count: int
) -> list[dict]:
    """Upstream's "When should I visit X so that ...?" -- a clock time.

    The one family whose gold rides on a duration, so its options out-space the traffic; every
    other family here is measured on something Kakao reproduces.
    """

    bases = pool.of("AD5")
    sights = pool.of("AT4", "CT1")
    rng.shuffle(bases)
    rng.shuffle(sights)
    made: list[dict] = []
    used: set[str] = set()
    for base in bases:
        if len(made) >= count:
            break
        if base.place_id in used or not builder.resolves_to(base):
            continue
        near = [place for place in sights if 1500 < distance_m(base, place) < 9000]
        chosen = take_resolvable(builder, near, 3)
        if len(chosen) < 3:
            continue
        stays = [rng.choice([1.0, 1.5, 2.0]) for _ in chosen]
        chain = [base, *chosen]
        legs = [
            builder.route(a, b, priority="DISTANCE")
            for a, b in zip(chain, chain[1:], strict=False)
        ]
        if any(route is None for route in legs):
            continue
        depart = datetime(2026, 5, 12, 9, 0)
        finish = depart + timedelta(
            seconds=sum(route.duration_s for route in legs) + sum(stays) * 3600
        )
        finish = finish.replace(second=0, microsecond=0)
        visits = ", ".join(
            f"{eul(place.name)} {stay:g}시간" for place, stay in zip(chosen, stays, strict=False)
        )
        made.append(
            {
                "question": (
                    f"오전 9시에 {base.name}에서 자동차로 출발해 {visits} 동안 차례로 "
                    "둘러보려 합니다. 마지막 장소의 관람을 마치는 시각은 몇 시인가요?"
                ),
                "options": _clock_options(finish, rng),
                "answer": 0,
                "classification": "trip",
                "mapeval_class": "trip",
                "template_id": "trip_arrival_clock",
                "gold_evidence": {
                    "base": base.name,
                    "stops": [place.name for place in chosen],
                    "stay_s": [stay * 3600 for stay in stays],
                    "travel_s": [route.duration_s for route in legs],
                },
            }
        )
        used.add(base.place_id)
    return made


def trip_total_distance(
    builder: Builder, pool: Pool, rng: random.Random, count: int
) -> list[dict]:
    """The itinerary's driving length, in km -- a value, and one the network reproduces."""

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
        near = [place for place in sights if 2000 < distance_m(base, place) < 11000]
        chosen = take_resolvable(builder, near, 3)
        if len(chosen) < 3:
            continue
        chain = [base, *chosen]
        legs = [
            builder.route(a, b, priority="DISTANCE")
            for a, b in zip(chain, chain[1:], strict=False)
        ]
        if any(route is None for route in legs):
            continue
        total = sum(route.distance_m for route in legs)
        options = _distance_options(total, (0.78, 1.45, 0.55))
        if options is None:
            continue
        order = " → ".join(place.name for place in chain)
        made.append(
            {
                "question": (
                    f"{order} 순서로 자동차로, 매 구간 거리가 가장 짧은 경로로 이동합니다. "
                    "전체 주행 거리는 얼마인가요?"
                ),
                "options": options,
                "answer": 0,
                "classification": "trip",
                "mapeval_class": "trip",
                "template_id": "trip_total_distance",
                "gold_evidence": {
                    "chain": [place.name for place in chain],
                    "leg_m": [round(route.distance_m) for route in legs],
                    "total_m": round(total),
                },
            }
        )
        used.add(base.place_id)
    return made


def trip_feasible_count(
    builder: Builder, pool: Pool, rng: random.Random, count: int
) -> list[dict]:
    """How many stops fit -- kept only when the travel time is what decides.

    v2's version of this family was answerable 9/10 by assuming a constant 15 minutes a leg and
    never touching the map, because the budget sat more than an hour from the boundary. So a row
    is kept only if the stay times alone give a *different* count, and if the true count still
    holds when every leg is 30% slower or faster -- the traffic spread Kakao actually shows.
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
        chosen = take_resolvable(builder, near, 4)
        if len(chosen) < 4:
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

        # Every budget this itinerary can be asked about, by the count it yields. Taking the
        # first that qualified put all seven golds on "한 곳", and an option set whose position
        # carries meaning cannot be re-shuffled to hide that; demanding one exact count instead
        # threw away four rows in seven. So collect what this base offers and spend it on the
        # count the family has least of.
        by_count: dict[int, tuple[float, int]] = {}
        for candidate in range(int(2.0 * 3600), int(11 * 3600), 900):
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
                "answer": fits(budget, 1.0) - 1,
                "classification": "trip",
                "mapeval_class": "trip",
                "template_id": "trip_feasible_count",
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


# ----------------------------------------------------------- unanswerable


def unanswerable(builder: Builder, pool: Pool, rng: random.Random, count: int) -> list[dict]:
    """The 20/300 of MapEval-API the API cannot answer, which here it cannot by nature.

    Kakao Local publishes no rating, no price level and no opening hours, so upstream's whole
    attribute half has no Korean counterpart. v2 dropped those questions; dropping them is what
    made every remaining row answerable, which is not what a map is like. They belong here as the
    class they are.
    """

    asks = [
        ("평점이 4.5 이상인 카페는 다음 중 어디인가요?", "CE7", "rating"),
        ("일요일 오후 5시에 영업 중인 약국은 다음 중 어디인가요?", "PM9", "opening_hours"),
        ("가격대가 가장 저렴한 음식점은 다음 중 어디인가요?", "FD6", "price_level"),
        ("리뷰가 가장 많은 편의점은 다음 중 어디인가요?", "CS2", "review_count"),
    ]
    anchors = pool.of("SW8", "AT4", "CT1")
    rng.shuffle(anchors)
    made: list[dict] = []
    used: set[str] = set()
    for anchor in anchors:
        if len(made) >= count:
            break
        if anchor.place_id in used or not builder.resolves_to(anchor):
            continue
        ask, code, missing = asks[len(made) % len(asks)]
        try:
            found = [
                place
                for place in builder.provider.nearby_search(
                    anchor, category_code=code, radius_m=1500, limit=15
                )
                if place.place_id != anchor.place_id
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
                "template_id": f"unanswerable_{missing}",
                "gold_evidence": {
                    "anchor": anchor.name,
                    "missing_field": missing,
                    "category_code": code,
                },
            }
        )
        used.add(anchor.place_id)
    return made


# --------------------------------------------------------------- assembly

FAMILIES: list[tuple[str, Callable[..., list[dict]], int]] = [
    ("nearby_clinic_subtype", nearby_clinic, 16),
    ("nearby_cuisine_subtype", nearby_cuisine, 12),
    ("poi_direction_distance", poi_direction_distance, 8),
    ("poi_which_is_closer", poi_which_is_closer, 7),
    ("poi_straight_distance", poi_straight_distance, 6),
    ("routing_distance_via", routing_distance_via, 8),
    ("routing_turn_count", routing_turn_count, 7),
    ("routing_next_turn", routing_next_turn, 7),
    ("trip_arrival_clock", trip_arrival_clock, 8),
    ("trip_total_distance", trip_total_distance, 7),
    ("trip_feasible_count", trip_feasible_count, 7),
    ("unanswerable", unanswerable, 7),
]

# Which answerable rows also carry the refusal option, so that its presence is not the answer.
# Upstream announces "Option0: Unanswerable" only on unanswerable rows, which gives the answer
# away; here roughly a quarter of the answerable rows carry it as a distractor.
DECOY_REFUSAL_EVERY = 4


def finalize(rows: list[dict]) -> list[dict]:
    """Assign gold positions per family, and plant the refusal distractor.

    Ordinal option sets keep their order: "한 곳"…"네 곳" carries meaning in its position.
    """

    ordered_families = {"trip_feasible_count"}
    slots: dict[str, list[int]] = {}
    for row in rows:
        family = row["template_id"]
        if family in slots:
            continue
        size = sum(1 for other in rows if other["template_id"] == family)
        width = len(row["options"])
        # Start each family's cycle at its own offset: a family of 7 over 4 positions leaves a
        # remainder, and every family leaving it at index 0 skewed the whole set toward option 0.
        offset = random.Random(f"offset:{family}:{SEED}").randrange(width)
        positions = [(index + offset) % width for index in range(size)]
        random.Random(f"{family}:{SEED}").shuffle(positions)
        slots[family] = positions

    finished: list[dict] = []
    answerable = 0
    for index, row in enumerate(rows):
        question_id = f"seoul_kmapeval_v4_{index:03d}"
        options = list(row["options"])
        answer = row["answer"]
        if row["mapeval_class"] != "unanswerable":
            answerable += 1
            if answerable % DECOY_REFUSAL_EVERY == 0 and len(options) > 2:
                replaceable = [
                    position for position in range(len(options)) if position != answer
                ]
                options[random.Random(question_id).choice(replaceable)] = UNANSWERABLE_TEXT
        if row["template_id"] in ordered_families:
            order = list(range(len(options)))
        else:
            target = slots[row["template_id"]].pop() % len(options)
            remaining = [
                position for position in range(len(options)) if position != answer
            ]
            random.Random(f"{question_id}:{SEED}").shuffle(remaining)
            order = list(remaining)
            order.insert(target, answer)
        finished.append(
            {
                "id": question_id,
                "question": row["question"],
                "options": [options[position] for position in order],
                "answer": order.index(answer),
                "classification": row["classification"],
                "region": "서울",
                "template_id": row["template_id"],
                "mapeval_class": row["mapeval_class"],
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
