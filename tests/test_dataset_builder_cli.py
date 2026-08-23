"""The shared dataset-builder command line: apportionment, the clock seed, and the two guards.

`data/` is offline tooling and runtime code never imports it, so these tests put it on the path
themselves rather than making it importable from `src/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[1] / "data"
if str(DATA) not in sys.path:
    sys.path.insert(0, str(DATA))

from builder_cli import apportion, run_builder  # noqa: E402

FAMILIES = [
    ("nearby", lambda *_: [], 28),
    ("poi", lambda *_: [], 21),
    ("routing", lambda *_: [], 22),
    ("trip", lambda *_: [], 22),
    ("unanswerable", lambda *_: [], 7),
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
        written.append({"seed": seed, "prefix": prefix})
        return [{"id": f"{prefix}_000", "seed": seed}]

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
        families=[("nearby", lambda *_: [{"template_id": "nearby"}], 28)],
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
        families=[("nearby", lambda *_: [{"template_id": "nearby"}], 28)],
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
