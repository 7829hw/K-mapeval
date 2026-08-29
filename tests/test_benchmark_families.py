"""The drawn parameters a family promises have to be spent across their values, at any build size.

`data/` is offline tooling and runtime code never imports it, so these tests put it on the path
themselves. Kakao is faked: a family's coverage logic is arithmetic over a candidate list, and
none of it needs the city.
"""

from __future__ import annotations

import random
import re
import sys
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
if str(DATA) not in sys.path:
    sys.path.insert(0, str(DATA))

from benchmark_core import MAPEVAL_API_CLASS_MIX, candidate_groups  # noqa: E402
from build_mapeval_v5_benchmark import NOUNS  # noqa: E402
from build_mapeval_v6_benchmark import (  # noqa: E402
    ORDINAL_SCAN_FLOOR,
    _scan_limit,
    _scarcest_ordinal,
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


# --------------------------------------------------------------------------------------------
# The scan that hunts for candidates has to grow with the build too.
# --------------------------------------------------------------------------------------------


def test_a_group_scan_offers_enough_disjoint_tuples_for_a_big_build() -> None:
    """`itertools.combinations(items[:55], 4)` could never hand a family a fourteenth row.

    Every family that spends its places retires them into a `used` set, so a constant slice caps
    the family at `slice // size` rows however many are asked for. That is why `--count 300` came
    back 281 to 283 across five draws, every one of them short in `poi_farthest_of_three`, and why
    the `poi` class landed at 16% of those files against the 21% its quota encodes.
    """

    items = list(range(600))
    rng = random.Random(0)
    groups = list(candidate_groups(items, 4, rng, 30))
    # Disjoint within a group, so a row never spends the same place twice.
    assert all(len(set(group)) == 4 for group in groups)
    # And enough distinct ones to fill thirty rows many times over, which the 55-item slice --
    # thirteen disjoint quadruples in total -- could not do at any budget.
    spent: set[int] = set()
    rows = 0
    for group in groups:
        if any(item in spent for item in group):
            continue
        spent.update(group)
        rows += 1
    assert rows >= 30


def test_a_group_scan_stays_linear_in_what_it_offers() -> None:
    """Raising the slice instead would have made the enumeration combinatorial.

    360 landmarks taken four at a time is 1.7 billion tuples, and `used` skips nearly all of them,
    so the fix cannot be a bigger constant.
    """

    rng = random.Random(0)
    offered = list(candidate_groups(list(range(600)), 4, rng, 30))
    assert len(offered) == max(24, 30 * 3) * 8


def test_a_group_scan_spends_the_whole_pool_not_its_first_entries() -> None:
    """The slice also decided *which* places a family could ever draw from."""

    rng = random.Random(0)
    seen = {item for group in candidate_groups(list(range(600)), 2, rng, 40) for item in group}
    assert len(seen) == 600


# --------------------------------------------------------------------------------------------
# The class mix, which is what the family quotas are for.
# --------------------------------------------------------------------------------------------


#: The class of a family no shipped set carries yet, so the mix can be checked the build before
#: its first draw exists. Every other family is read off the rows, and this map is asserted to
#: hold nothing that could be.
UNSHIPPED_FAMILY_CLASSES: dict[str, str] = {}


def _family_classes(rows: list[dict], families: list[str]) -> dict[str, str]:
    """Which MapEval-API class each family writes, read off a set the family actually built.

    Derived rather than declared: a second table naming the class of every family is a table that
    drifts from the rows. The `unanswerable` family writes one template per missing attribute
    (`unanswerable_rating`, `unanswerable_opening_hours`, ...), so a family with no template of
    its own name is matched against the templates it prefixes.
    """

    # v1, v2 and v3 predate both labels, so they carry no evidence about a family's class.
    by_template = {
        row["template_id"]: row["mapeval_class"]
        for row in rows
        if row.get("template_id") and row.get("mapeval_class")
    }
    resolved: dict[str, str] = {}
    for family in families:
        if family in by_template:
            resolved[family] = by_template[family]
            continue
        prefixed = {
            mapeval_class
            for template, mapeval_class in by_template.items()
            if template.startswith(f"{family}_")
        }
        if prefixed:
            assert len(prefixed) == 1, family
            resolved[family] = prefixed.pop()
            continue
        assert family in UNSHIPPED_FAMILY_CLASSES, family
        resolved[family] = UNSHIPPED_FAMILY_CLASSES[family]
    return resolved


def test_the_unshipped_family_table_holds_only_families_no_set_carries() -> None:
    """It exists so a new family can be checked before its first draw, and for nothing else."""

    shipped = {
        row["template_id"]
        for path in DATASETS
        for row in _rows(path)
        if row.get("template_id")
    }
    assert not (set(UNSHIPPED_FAMILY_CLASSES) & shipped), (
        "a family with rows in dataset/ must have its class read off them, not declared"
    )


def test_the_standard_builders_quotas_are_upstreams_class_mix() -> None:
    """MapEval-API is nearby 83 / poi 64 / routing 66 / trip 67 / unanswerable 20 over 300 rows.

    The quotas are that mix at a hundred, and they have to stay it: pooling a class measured at a
    different proportion against upstream's 71.07% compares two different benchmarks. Counted off
    `mapeval-api/dataset.json`; see `docs/REFERENCE_MAPPING.md`.
    """

    import build_kmapeval_dataset

    # Every set, not one of them: a family the builder draws today may have shipped first in the
    # newest file, and pinning one filename makes the check fail on the build that adds a family
    # rather than on the one that misclassifies it.
    rows = [row for path in DATASETS for row in _rows(path)]
    families = [name for name, _, _ in build_kmapeval_dataset.FAMILIES]
    classes = _family_classes(rows, families)
    quotas: Counter[str] = Counter()
    for name, _, quota in build_kmapeval_dataset.FAMILIES:
        quotas[classes[name]] += quota

    assert set(quotas) == set(MAPEVAL_API_CLASS_MIX)
    upstream_total = sum(MAPEVAL_API_CLASS_MIX.values())
    for mapeval_class, upstream in MAPEVAL_API_CLASS_MIX.items():
        share = 100 * upstream / upstream_total
        assert abs(quotas[mapeval_class] - share) <= 1, (mapeval_class, quotas[mapeval_class])


@pytest.mark.parametrize(
    "path",
    [path for path in DATASETS if path.name != "seoul_mapeval_v1_mcq_100.jsonl"],
    ids=lambda path: path.name,
)
def test_no_generated_row_carries_the_annotators_context(path: Path) -> None:
    """MapEval-API is MapEval-Textual with `context` removed, and that is what is measured here.

    Every run answers from live Kakao; a stored evidence block beside the question would be an
    answer key. `dataset/seoul_mapeval_v1_mcq_100.jsonl` keeps its own for provenance and is not
    in this list.
    """

    for row in _rows(path):
        assert "context" not in row, f"{path.name}:{row['id']}"


# --------------------------------------------------------------------------------------------
# The ordinal family: which value an anchor is spent on, and which kinds of place it may ask by.
# --------------------------------------------------------------------------------------------


def test_a_generous_anchor_is_not_spent_on_the_value_every_anchor_can_supply() -> None:
    """`feasible` is a prefix of (2, 3, 4), so k=2 is free and k=4 is what runs out.

    The gap tests are nested: k=4 asks for every gap k=3 asks for and one more. Picking the
    least-produced value with `min(..., key=(produced, k))` broke every early tie toward k=2 and
    burned the scarce anchors on it. On the v8 draw eight of the twenty-four accepted anchors
    could have answered k=3 or k=4; the file shipped 17 / 4 / 3.
    """

    target = {2: 8, 3: 8, 4: 8}
    empty = {2: 0, 3: 0, 4: 0}
    assert _scarcest_ordinal((2, 3, 4), empty, target) == 4
    assert _scarcest_ordinal((2, 3), empty, target) == 3
    assert _scarcest_ordinal((2,), empty, target) == 2
    # And once a value is full the anchor goes to the next one still short.
    assert _scarcest_ordinal((2, 3, 4), {2: 0, 3: 0, 4: 8}, target) == 3
    assert _scarcest_ordinal((2, 3, 4), {2: 0, 3: 8, 4: 8}, target) == 2
    # The remainder belongs to k=2, which is the value the city can always supply.
    assert _scarcest_ordinal((2, 3, 4), empty, {2: 4, 3: 2, 4: 2}) == 2


def test_the_ordinal_family_does_not_ask_by_a_dense_chain_category() -> None:
    """A category Seoul packs four of into one block cannot carry an ordinal question at all.

    Measured over 108 anchors at `ORDINAL_MARGIN_M`: `CE7` and `FD6` yielded no anchor whose
    neighbours are separable, and `CS2`, `BK9` and `PM9` two apiece with *none* past k=2. So the
    family's k was never drawn -- it was dictated by the category, which is why five draws running
    audited as concentrated on k=2 and no amount of fixing the scan or the tie-break moved it. The
    same anchors give `MT1` 43 usable and 23 past k=2, `PO3` 27 and 7, `SC4` 23 and 10, `CT1` 21
    and 9.
    """

    source = (DATA / "build_mapeval_v6_benchmark.py").read_text(encoding="utf-8")
    body = source.split("def nearby_kth_nearest(")[1].split("\ndef ")[0]
    codes = re.search(r"codes = \[([^\]]*)\]", body).group(1)
    asked = {token.strip().strip('"') for token in codes.split(",") if token.strip()}
    assert asked == {"MT1", "SC4", "PO3", "CT1"}, asked
    assert not asked & {"CE7", "FD6", "CS2", "BK9", "PM9"}
    # Every kind the family asks by has to have a Korean noun to print.
    for code in asked:
        assert NOUNS[code]
