"""The standard K-MapEval dataset builder: v7's families, any size, a fresh draw every run.

v7's generation method is the one this project settled on. It is v6's twelve families unchanged
plus v6's two trip families walked back from four stops to three, because at four stops the
reference baseline runs out of iterations before it can finish one. `build_mapeval_v6/v7_...py`
exist to reproduce two specific benchmarks of record and default to the seed that reproduces them.
This one is for building *new* sets: it takes the question count directly and draws its seed from
the clock, so two runs are two samples rather than the same hundred questions twice.

    python data/build_kmapeval_dataset.py --count 200
    python data/build_kmapeval_dataset.py --count 50 --out dataset/pilot.jsonl

Both flags matter for what this project measures. A dataset built by a builder whose default seed
reproduces a tuned set is a training set wearing a new filename, and every accuracy read off it is
a training-set accuracy -- see `docs/REFERENCE_MAPPING.md`. The seed is printed and stamped into
the default filename so a draw worth keeping can be drawn again.

Run `python data/audit_dataset.py <out>` afterwards. It exits non-zero on a second answer key, and
every one of the defects it looks for has shipped in a benchmark here before it existed.

**Why three stops and not four.** Under `--react-tools reference`, `Directions` answers one leg per
call and the loop takes one action per iteration, so a four-stop round trip needs about twenty
calls against langchain's fifteen. Over four passes at that budget ReAct scored 14/60 on v6's two
four-stop families with 24 of those 60 rows stopped by `iteration_limit`. A three-pass ablation at
a budget of 30 clears every `iteration_limit` and takes `trip_optimal_order_four` from 2/24 to
21/24 -- so that family was measuring the budget. It also moved Spatial-Agent, because
`MAX_REASONING_STEPS` bounds planner nodes too, which is why raising the budget is not the repair
and shrinking the family is. (`trip_total_distance_four` did not move at either budget: it fails on
arithmetic, not iterations. Three stops is what v5 measured it at, and there it discriminated.)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_core import Builder  # noqa: E402
from build_mapeval_benchmark import Pool, finalize  # noqa: E402
from build_mapeval_v6_benchmark import V6_ORDERED_FAMILIES  # noqa: E402
from build_mapeval_v7_benchmark import FAMILIES, SEED  # noqa: E402
from builder_cli import run_builder  # noqa: E402

OUT_PATH = Path(__file__).resolve().parents[1] / "dataset" / "kmapeval_dataset.jsonl"


def main() -> None:
    run_builder(
        families=FAMILIES,
        open_builder=Builder.open,
        make_pool=Pool,
        finalize=finalize,
        ordered=V6_ORDERED_FAMILIES,
        # v7's own seed, so that asking this builder for a hundred questions on that seed still
        # refuses to relabel the tuned set under another id prefix.
        canonical_seed=SEED,
        canonical_prefix="kmapeval",
        canonical_out=OUT_PATH,
    )


if __name__ == "__main__":
    main()
