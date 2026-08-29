"""The shared dataset-builder command line: apportionment, the clock seed, and the two guards.

`data/` is offline tooling and runtime code never imports it, so these tests put it on the path
themselves rather than making it importable from `src/`.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[1] / "data"
if str(DATA) not in sys.path:
    sys.path.insert(0, str(DATA))

from builder_cli import apportion, run_builder  # noqa: E402


def _family(name: str, *, cap: int | None = None, mapeval_class: str | None = None):
    """A family that draws as many rows as it is asked for, or `cap` of them if it cannot."""

    def draw(_builder, _pool, rng, count: int) -> list[dict]:
        made = min(count, cap) if cap is not None else count
        return [
            {
                "question": f"{name} {rng.random():.12f} {index}",
                "template_id": name,
                "mapeval_class": mapeval_class or name,
            }
            for index in range(made)
        ]

    return draw


FAMILIES = [
    ("nearby", _family("nearby"), 28),
    ("poi", _family("poi"), 21),
    ("routing", _family("routing"), 22),
    ("trip", _family("trip"), 22),
    ("unanswerable", _family("unanswerable"), 7),
]


@pytest.mark.parametrize("count", [100, 60, 50, 30, 14, 7, 5])
def test_a_requested_count_is_the_count_that_is_built(count: int) -> None:
    """`round(quota * scale)` summed to whatever it summed to; a count has to be the count."""

    allocated = apportion(FAMILIES, count)
    assert sum(quota for *_, quota in allocated) == count
    assert all(quota >= 1 for *_, quota in allocated)


def test_shrinking_a_build_keeps_the_class_proportions() -> None:
    """MapEval-API's class proportions are the reason the quotas are what they are."""

    allocated = {name: quota for name, _, quota in apportion(FAMILIES, 50)}
    # Half the rows, so every family should land within a row of half its quota.
    for name, _, quota in FAMILIES:
        assert abs(allocated[name] - quota / 2) <= 1, name


def test_a_count_below_the_family_count_keeps_the_largest_families() -> None:
    """A family cannot produce half a question, so the small ones drop rather than round to zero."""

    allocated = apportion(FAMILIES, 3)
    assert sum(quota for *_, quota in allocated) == 3
    assert {name for name, _, _ in allocated} == {"nearby", "routing", "trip"}


def _fakes(written: list[dict]):
    class _Provider:
        api_call_count = 0

    class _Builder:
        provider = _Provider()

        def close(self) -> None:
            return None

    def finalize(rows, *, seed, prefix, ordered):  # type: ignore[no-untyped-def]
        written.append({"seed": seed, "prefix": prefix, "rows": list(rows)})
        return [
            {"id": f"{prefix}_{index:03d}", "seed": seed, **row}
            for index, row in enumerate(rows)
        ]

    return _Builder, finalize


def test_an_omitted_seed_comes_from_the_clock(tmp_path: Path, monkeypatch) -> None:
    """Two runs of a builder must be two samples, not the same hundred questions twice.

    A builder whose default seed reproduces the set it was tuned against hands back a training set
    under a new filename, and every accuracy read off it is a training-set accuracy.
    """

    written: list[dict] = []
    builder, finalize = _fakes(written)
    monkeypatch.setattr("builder_cli.time.time", lambda: 1_700_000_042.7)
    run_builder(
        families=[("nearby", _family("nearby"), 28)],
        open_builder=builder,
        make_pool=dict,
        finalize=finalize,
        ordered=frozenset(),
        canonical_seed=999,
        canonical_prefix="fake",
        canonical_out=tmp_path / "fake.jsonl",
        argv=[],
    )
    assert written[0]["seed"] == 1_700_000_042
    # The seed is in the filename, because a draw nobody recorded cannot be repeated.
    assert (tmp_path / "fake_28_s1700000042.jsonl").exists()


def test_a_fresh_draw_never_lands_on_a_benchmark_of_record(tmp_path: Path, monkeypatch) -> None:
    """The clock seed makes the default output path dangerous, so it stops being the default."""

    written: list[dict] = []
    builder, finalize = _fakes(written)
    canonical = tmp_path / "canonical.jsonl"
    canonical.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr("builder_cli.time.time", lambda: 1_700_000_000.0)
    run_builder(
        families=[("nearby", _family("nearby"), 28)],
        open_builder=builder,
        make_pool=dict,
        finalize=finalize,
        ordered=frozenset(),
        canonical_seed=999,
        canonical_prefix="fake",
        canonical_out=canonical,
        argv=[],
    )
    assert canonical.read_text(encoding="utf-8") == "{}\n"

    # And an explicit path onto an existing file is refused rather than silently replacing it.
    with pytest.raises(SystemExit, match="already exists"):
        run_builder(
            families=FAMILIES,
            open_builder=builder,
            make_pool=dict,
            finalize=finalize,
            ordered=frozenset(),
            canonical_seed=999,
            canonical_prefix="fake",
            canonical_out=canonical,
            argv=["--out", str(canonical)],
        )


def test_relabelling_the_tuned_set_is_still_refused(tmp_path: Path) -> None:
    """The old guard survives the clock seed: an explicit canonical seed under another prefix."""

    written: list[dict] = []
    builder, finalize = _fakes(written)
    with pytest.raises(SystemExit, match="relabel the tuned set"):
        run_builder(
            families=FAMILIES,
            open_builder=builder,
            make_pool=dict,
            finalize=finalize,
            ordered=frozenset(),
            canonical_seed=999,
            canonical_prefix="fake",
            canonical_out=tmp_path / "fake.jsonl",
            argv=["--seed", "999", "--id-prefix", "held-out"],
        )


def test_the_standard_builder_draws_the_families_the_baseline_can_finish() -> None:
    """v7's method is the standard, and the reason is a step budget, not a preference.

    v6's `trip_optimal_order_four` and `trip_total_distance_four` need about twenty one-leg
    `Directions` calls against langchain's fifteen iterations, so under `--react-tools reference`
    they measure the budget. A three-pass ablation at a budget of 30 took the first from 2/24 to
    21/24. Three stops is what v5 measured and what discriminated. This is one import line away
    from silently reverting, so it is pinned.
    """

    import build_kmapeval_dataset

    names = {name for name, _, _ in build_kmapeval_dataset.FAMILIES}
    assert {"trip_optimal_order", "trip_total_distance"} <= names
    assert not {"trip_optimal_order_four", "trip_total_distance_four"} & names
    # And the four-stop feasibility family: same four live rungs taken down at the bottom, one
    # leg fewer to route. It did *not* fix the planner budget failure it was written for -- 37%
    # against the five-stop version's 39% -- and is kept for costing a leg less, not for that.
    assert "trip_feasible_count_four" in names
    assert "trip_feasible_count_five" not in names
    # A ladder family's options carry meaning in their order, so it must not be shuffled.
    assert "trip_feasible_count_four" in build_kmapeval_dataset.ORDERED_FAMILIES
    assert sum(quota for *_, quota in build_kmapeval_dataset.FAMILIES) == 100


# --------------------------------------------------------------------------------------------
# Apportioning exactly is only half of asking for a count. The other half is that every family
# draws its share, and a family that cannot must not leave the file short.
# --------------------------------------------------------------------------------------------


def _rows(written: list[dict]) -> list[dict]:
    return written[-1]["rows"]


def test_a_family_that_comes_up_short_does_not_shorten_the_file(tmp_path: Path) -> None:
    """`--count 300` shipped 281 to 283 rows across five draws, and always for one family.

    `poi_farthest_of_three` retires four landmarks per row out of a 55-landmark slice, so it could
    never produce a fourteenth however many were asked for. The count the caller gave is the count
    the file has to hold.
    """

    written: list[dict] = []
    builder, finalize = _fakes(written)
    families = [
        ("nearby", _family("nearby"), 28),
        ("poi_wide", _family("poi_wide", mapeval_class="poi"), 11),
        ("poi_narrow", _family("poi_narrow", cap=13, mapeval_class="poi"), 10),
        ("routing", _family("routing"), 22),
        ("trip", _family("trip"), 22),
        ("unanswerable", _family("unanswerable"), 7),
    ]
    run_builder(
        families=families,
        open_builder=builder,
        make_pool=dict,
        finalize=finalize,
        ordered=frozenset(),
        canonical_seed=999,
        canonical_prefix="fake",
        canonical_out=tmp_path / "fake.jsonl",
        argv=["--count", "300", "--seed", "1"],
    )
    assert len(_rows(written)) == 300


def test_the_rows_a_family_could_not_draw_go_to_its_own_class(tmp_path: Path) -> None:
    """The class mix is the reason the quotas are what they are, so the deficit stays inside it.

    Upstream's own `dataset.json` is nearby 83 / poi 64 / routing 66 / trip 67 / unanswerable 20
    over 300 rows. A build that hands `poi`'s missing rows to `nearby` holds the count and reports
    a different benchmark.
    """

    written: list[dict] = []
    builder, finalize = _fakes(written)
    families = [
        ("nearby", _family("nearby"), 28),
        ("poi_wide", _family("poi_wide", mapeval_class="poi"), 11),
        ("poi_narrow", _family("poi_narrow", cap=13, mapeval_class="poi"), 10),
        ("routing", _family("routing"), 22),
        ("trip", _family("trip"), 22),
        ("unanswerable", _family("unanswerable"), 7),
    ]
    run_builder(
        families=families,
        open_builder=builder,
        make_pool=dict,
        finalize=finalize,
        ordered=frozenset(),
        canonical_seed=999,
        canonical_prefix="fake",
        canonical_out=tmp_path / "fake.jsonl",
        argv=["--count", "300", "--seed", "1"],
    )
    drawn = Counter(row["template_id"] for row in _rows(written))
    # 33 + 30 was the apportionment; the narrow family could only manage 13, so the wide one of
    # the same class takes the other 17 and the class still holds 63 of the 300.
    assert drawn["poi_narrow"] == 13
    assert drawn["poi_wide"] + drawn["poi_narrow"] == 63
    # And no other class moved to pay for it.
    assert drawn["nearby"] == 84
    assert drawn["routing"] == 66
    assert drawn["trip"] == 66
    assert drawn["unanswerable"] == 21


def test_a_count_the_pool_cannot_supply_fails_loudly(tmp_path: Path) -> None:
    """Writing fewer rows than were asked for silently is what this replaces."""

    written: list[dict] = []
    builder, finalize = _fakes(written)
    families = [
        ("nearby", _family("nearby", cap=5), 28),
        ("poi", _family("poi", cap=5), 21),
    ]
    with pytest.raises(SystemExit, match="only 10 could be drawn"):
        run_builder(
            families=families,
            open_builder=builder,
            make_pool=dict,
            finalize=finalize,
            ordered=frozenset(),
            canonical_seed=999,
            canonical_prefix="fake",
            canonical_out=tmp_path / "fake.jsonl",
            argv=["--count", "49", "--seed", "1"],
        )


def test_a_redraw_never_repeats_a_question_it_already_has(tmp_path: Path) -> None:
    """A top-up round runs the same family over the same city; most of what it returns is old."""

    written: list[dict] = []
    builder, finalize = _fakes(written)
    run_builder(
        families=[
            ("nearby", _family("nearby"), 28),
            ("poi", _family("poi", cap=13), 21),
        ],
        open_builder=builder,
        make_pool=dict,
        finalize=finalize,
        ordered=frozenset(),
        canonical_seed=999,
        canonical_prefix="fake",
        canonical_out=tmp_path / "fake.jsonl",
        argv=["--count", "49", "--seed", "1"],
    )
    questions = [row["question"] for row in _rows(written)]
    assert len(questions) == len(set(questions)) == 49
