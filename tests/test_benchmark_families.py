"""The drawn parameters a family promises have to be spent across their values, at any build size.

`data/` is offline tooling and runtime code never imports it, so these tests put it on the path
themselves. Kakao is faked: a family's coverage logic is arithmetic over a candidate list, and
none of it needs the city.
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
if str(DATA) not in sys.path:
    sys.path.insert(0, str(DATA))

from build_mapeval_v6_benchmark import (  # noqa: E402
    ORDINAL_SCAN_FLOOR,
    _scan_limit,
    nearby_kth_nearest,
)

from src.models import Place  # noqa: E402

# `audit_dataset.CONCENTRATION`: past this share of one value, a drawn parameter is not drawn.
CONCENTRATION = 0.7
# Metres per degree of latitude, near enough to place a neighbour at a chosen distance.
METRES_PER_DEGREE = 111_320.0

# One anchor in six sits in a neighbourhood whose ranks 1..5 are all separable, so it can supply
# k=3 and k=4; the other five can only ever supply k=2. That is the shape of the real city -- rank
# three through five of a dense block sit inside the 90 m ordinal margin -- and it is what makes
# the scan limit decide the spread.
GENEROUS_EVERY = 6
SPREAD_M = (100.0, 300.0, 500.0, 700.0, 900.0, 1100.0)
TIGHT_M = (100.0, 300.0, 500.0, 520.0, 540.0, 560.0)


def _place(name: str, north_m: float) -> Place:
    return Place(
        place_id=name,
        name=name,
        latitude=37.5 + north_m / METRES_PER_DEGREE,
        longitude=127.0,
        category="테스트",
    )


class _Provider:
    api_call_count = 0

    def nearby_search(self, anchor: Place, **_: object) -> list[Place]:
        index = int(anchor.place_id.split("_")[1])
        offsets = SPREAD_M if index % GENEROUS_EVERY == 0 else TIGHT_M
        base = anchor.latitude * METRES_PER_DEGREE
        return [
            _place(f"{anchor.place_id}_n{rank}", base + metres)
            for rank, metres in enumerate(offsets)
        ]


class _Builder:
    provider = _Provider()

    def as_resolved(self, place: Place) -> Place:
        return place

    def resolves_to(self, _: Place) -> bool:
        return True


class _Pool:
    def of(self, *_: str) -> list[Place]:
        return [_place(f"anchor_{index}", 0.0) for index in range(240)]


def _ordinals(count: int) -> Counter[int]:
    rows = nearby_kth_nearest(_Builder(), _Pool(), random.Random(11), count)
    assert len(rows) == count
    return Counter(row["gold_evidence"]["k"] for row in rows)


def test_a_big_build_still_spends_the_ordinal_across_its_three_values() -> None:
    """The defect the 283-row set shipped: 19 of 24 `nearby_kth_nearest` rows asked for k=2.

    Ranking four options against each other -- which is what an agent does when it does not
    retrieve -- answers a k-th question whenever the k-1 nearer places are all among the decoys:
    60% of the time at k=2, 10% at k=4. A family that lands on k=2 answers itself.
    """

    spread = _ordinals(24)
    assert set(spread) == {2, 3, 4}, spread
    assert spread.most_common(1)[0][1] / 24 <= CONCENTRATION, spread


def test_the_scan_limit_is_what_decides_that(monkeypatch) -> None:
    """Pin the cause, not just the symptom: with the old constant limit the same draw fails.

    Without this the fix reads like a tuning choice, and the next constant that stops growing with
    the build gets written the same way.
    """

    monkeypatch.setattr(
        "build_mapeval_v6_benchmark._scan_limit", lambda floor, count: ORDINAL_SCAN_FLOOR
    )
    spread = _ordinals(24)
    assert spread.most_common(1)[0][1] / 24 > CONCENTRATION, spread


def test_v6_and_v7_draw_exactly_what_they_drew() -> None:
    """Both benchmarks of record draw eight of this family and four of the radius one.

    The floor is three times eight, so the scan limit at those counts is the constant it always
    was and the two sets reproduce as far as live Kakao lets them. A fix to a shared generator
    that moved them would have rewritten two published numbers.
    """

    assert _scan_limit(ORDINAL_SCAN_FLOOR, 8) == ORDINAL_SCAN_FLOOR
    assert _scan_limit(ORDINAL_SCAN_FLOOR, 4) == ORDINAL_SCAN_FLOOR
    # And it grows once the build is bigger than the floor was sized for.
    assert _scan_limit(ORDINAL_SCAN_FLOOR, 24) == 72
