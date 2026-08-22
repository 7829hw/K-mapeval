"""Generate v7: v6, with the two four-stop trip families walked back to three stops.

v6 raised `trip_optimal_order` and `trip_total_distance` from three stops to four. Measured on the
first v6 run, that made both families unanswerable rather than harder — for the *baseline*, and
for a reason that has nothing to do with spatial reasoning:

    trip_optimal_order_four   8 rows   tool_calls median 15 / max 15   ReAct 0/8
    trip_total_distance_four  7 rows   tool_calls median 15 / max 15   ReAct 0/7

Under `--react-tools reference`, `Directions(originId, destinationId, travelMode)` answers one leg
per call and the structured-chat parser takes one action per iteration, inside langchain's
15-iteration budget. A four-stop round trip needs five place ids and, across three candidate
orders, up to fifteen distinct legs — twenty calls against a budget of fifteen. Every row of both
families ended on `iteration_limit`. A family that cannot be finished within the paper's own budget
measures the budget, not the architecture, and no amount of running it produces a comparison.

Three stops fits: four ids plus roughly nine distinct legs, which is what v5 measured (ReAct 3/8 on
ordering, 0/7 on total distance against Spatial-Agent's 7/7 — the aggregation-drift finding this
whole benchmark exists to isolate). So the two families revert to v5's versions and everything else
in v6 stands.

The seed is deliberately v6's, so the twelve untouched families draw from the same
`random.Random(f"{seed}:{name}")` stream they drew from there. That reproduces only 18 of v6's
rows, not the 85 it looks like it should: the draws are made against live Kakao, the cache expires
after 24 hours, and a day later the same query returns a different candidate set -- different
nearby rankings, places that came and went. v7 is therefore a fresh sample that happens to share a
generator with v6, not a patched copy of it, and the two runs cannot be compared row for row.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_core import Builder  # noqa: E402
from build_mapeval_benchmark import Pool, finalize, trip_total_distance  # noqa: E402
from build_mapeval_v5_benchmark import trip_optimal_order  # noqa: E402
from build_mapeval_v6_benchmark import FAMILIES as V6_FAMILIES  # noqa: E402
from build_mapeval_v6_benchmark import SEED as V6_SEED  # noqa: E402
from build_mapeval_v6_benchmark import V6_ORDERED_FAMILIES  # noqa: E402

# v6's seed on purpose; see the module docstring.
SEED = V6_SEED
OUT_PATH = Path(__file__).resolve().parents[1] / "dataset" / "seoul_kmapeval_v7_mcq_100.jsonl"

# The four-stop family -> the three-stop one that fits the reference baseline's budget. Quotas are
# unchanged, so the row count, the class proportions and the question ids all stay where they were.
REPLACEMENTS: dict[str, tuple[str, Callable[..., list[dict]]]] = {
    "trip_optimal_order_four": ("trip_optimal_order", trip_optimal_order),
    "trip_total_distance_four": ("trip_total_distance", trip_total_distance),
}

FAMILIES: list[tuple[str, Callable[..., list[dict]], int]] = [
    (*REPLACEMENTS[name], quota) if name in REPLACEMENTS else (name, function, quota)
    for name, function, quota in V6_FAMILIES
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--id-prefix", default="seoul_kmapeval_v7")
    args = parser.parse_args()
    if args.id_prefix != "seoul_kmapeval_v7" and args.seed == SEED:
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
