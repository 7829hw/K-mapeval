"""The standard K-MapEval dataset builder: v6's families, any size, a fresh draw every run.

v6's generation method is the one this project settled on, and v7 is that method with its two
four-stop trip families walked back to three so the reference baseline can finish them inside
fifteen iterations. Those two builders exist to reproduce two specific benchmarks of record, and
they default to the seed that reproduces them. This one is for building *new* sets: it takes the
question count directly and draws its seed from the clock, so two runs are two samples rather than
the same hundred questions twice.

    python data/build_kmapeval_dataset.py --count 200
    python data/build_kmapeval_dataset.py --count 50 --out dataset/pilot.jsonl

Both flags matter for what this project measures. A dataset built by a builder whose default seed
reproduces a tuned set is a training set wearing a new filename, and every accuracy read off it is
a training-set accuracy -- see `docs/REFERENCE_MAPPING.md`. The seed is printed and stamped into
the default filename so a draw worth keeping can be drawn again.

Run `python data/audit_dataset.py <out>` afterwards. It exits non-zero on a second answer key, and
every one of the defects it looks for has shipped in a benchmark here before it existed.

**Know what v6's families cost the baseline.** Two of them -- `trip_optimal_order_four` and
`trip_total_distance_four` -- need about twenty one-leg `Directions` calls, against langchain's
fifteen iterations. Every row of both ended on `iteration_limit` with ReAct scoring 0/15, which
measures the step budget rather than the agent, and that is why v7 exists and why
`docs/REFERENCE_MAPPING.md` quotes v7 instead of v6. A set built here inherits them. To draw v6's
method without them, use `data/build_mapeval_v7_benchmark.py`, which is the same families with
those two walked back to three stops; it now takes `--count` and a clock seed as well.
"""

from __future__ import annotations

from pathlib import Path

from benchmark_core import Builder
from build_mapeval_benchmark import Pool, finalize
from build_mapeval_v6_benchmark import FAMILIES, SEED, V6_ORDERED_FAMILIES
from builder_cli import run_builder

OUT_PATH = Path(__file__).resolve().parents[1] / "dataset" / "kmapeval_dataset.jsonl"


def main() -> None:
    run_builder(
        families=FAMILIES,
        open_builder=Builder.open,
        make_pool=Pool,
        finalize=finalize,
        ordered=V6_ORDERED_FAMILIES,
        # v6's own seed, so that asking this builder for a hundred questions on that seed still
        # refuses to relabel the tuned set under another id prefix.
        canonical_seed=SEED,
        canonical_prefix="kmapeval",
        canonical_out=OUT_PATH,
    )


if __name__ == "__main__":
    main()
