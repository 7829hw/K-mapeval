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


# --------------------------------------------------------------------------------------------
# The dataset label axes.
#
# Every row carries three independent labels: `mapeval_class` (MapEval-API's task category, plus
# this port's `unanswerable`), `classification` (what is measured), and `template_id` (which
# generator wrote it). They were being written by the builders and read by nothing -- `extra=
# "allow"` let the first two through unvalidated, and only `classification` reached a report.
# These pin the relationship between them so a builder cannot drift one axis away from another.
# --------------------------------------------------------------------------------------------

import json  # noqa: E402
from typing import Any  # noqa: E402

import pytest  # noqa: E402

from src.dataset import (  # noqa: E402
    _CLASSIFICATION_TO_MAPEVAL,
    load_dataset,
    resolve_mapeval_class,
)

DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"
DATASETS = sorted(DATASET_DIR.glob("*.jsonl"))
# The rows built before the label existed. Every other file states its own `mapeval_class`.
UNLABELLED = {
    "seoul_mapeval_v1_mcq_100.jsonl",
    "seoul_kmapeval_v2_mcq_100.jsonl",
    "seoul_kmapeval_v3_mcq_100.jsonl",
}


def _rows(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


@pytest.mark.parametrize("path", DATASETS, ids=lambda path: path.name)
def test_the_stored_task_category_agrees_with_what_the_measurement_type_implies(path: Path) -> None:
    """`unanswerable` excepted: it is written as a `nearby` question and only the label says so."""

    for row in _rows(path):
        stored = row.get("mapeval_class")
        if stored is None or stored == "unanswerable":
            continue
        assert stored == _CLASSIFICATION_TO_MAPEVAL[row["classification"]], row["id"]


@pytest.mark.parametrize("path", DATASETS, ids=lambda path: path.name)
def test_every_set_the_labelled_builders_wrote_labels_every_row(path: Path) -> None:
    rows = _rows(path)
    if path.name in UNLABELLED:
        assert all(row.get("mapeval_class") is None for row in rows)
    else:
        assert all(row.get("mapeval_class") for row in rows), path.name
    # `template_id` predates `mapeval_class` and is on every row of every set here.
    assert all(row.get("template_id") for row in rows), path.name


def test_benchmark_labels_are_evaluation_metadata_only() -> None:
    current = {
        row["classification"]
        for path in DATASETS
        if path.name not in UNLABELLED
        for row in _rows(path)
    }
    assert current == {"nearby", "radius", "distance", "direction", "routing", "trip"}


@pytest.mark.parametrize("path", DATASETS, ids=lambda path: path.name)
def test_the_resolver_answers_for_every_row_of_every_set(path: Path) -> None:
    for item in load_dataset(path):
        assert resolve_mapeval_class(item) in {"nearby", "poi", "routing", "trip", "unanswerable"}


def test_the_stored_label_beats_the_derivation_where_they_disagree() -> None:
    """The `unanswerable_*` families are the whole reason the resolver reads the field first."""

    items = load_dataset(DATASET_DIR / "seoul_kmapeval_v7_mcq_300.jsonl")
    resolved = Counter(resolve_mapeval_class(item) for item in items)
    derived = Counter(_CLASSIFICATION_TO_MAPEVAL[item.classification] for item in items)
    assert resolved["unanswerable"] == 21
    assert derived["unanswerable"] == 0
    assert derived["nearby"] - resolved["nearby"] == 21
