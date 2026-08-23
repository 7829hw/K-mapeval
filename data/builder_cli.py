"""The command line every K-MapEval dataset builder shares.

The builders differ only in which families they draw and what they call the result, so the
argument parsing, the seed, the apportionment and the write lived in three near-identical copies
of `main()`. They are here once instead.

Two things this adds over those copies. A build can be asked for a **number of questions** rather
than a multiplier, and the families keep their proportions while summing to exactly that number.
And the **seed comes from the clock** when none is given, so a build is a fresh draw by default
rather than a re-run of whatever the file was last tuned to -- which is the state a held-out set
has to be built in. The seed is printed and stamped into the filename, because a draw nobody
recorded cannot be repeated.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

Family = tuple[str, Callable[..., list[dict]], int]


def apportion(families: Sequence[Family], count: int) -> list[Family]:
    """Split `count` questions across families in their quota proportions, exactly.

    Largest-remainder, with a floor of one row per family: `round(quota * scale)` -- what the
    builders used to do -- neither sums to a total anybody asked for nor keeps a small family
    alive at a small scale. A count below the number of families keeps the largest quotas and
    drops the rest, since a family cannot produce half a question.
    """

    if count < 1:
        raise SystemExit(f"--count must be at least 1, got {count}")
    quotas = [quota for *_, quota in families]
    total = sum(quotas)
    if total <= 0:
        raise SystemExit("families carry no quota to apportion")

    if count < len(families):
        keep = sorted(range(len(families)), key=lambda index: (-quotas[index], index))[:count]
        chosen = set(keep)
        return [
            (name, function, 1)
            for index, (name, function, _) in enumerate(families)
            if index in chosen
        ]

    exact = [count * quota / total for quota in quotas]
    allocation = [max(1, int(value)) for value in exact]
    remainders = [value - int(value) for value in exact]
    # The floor of one can push the total past `count`; the remainder pass can leave it short.
    # Settle both against the fractional parts, largest first.
    while sum(allocation) < count:
        index = max(range(len(allocation)), key=lambda i: (remainders[i], quotas[i]))
        allocation[index] += 1
        remainders[index] -= 1
    while sum(allocation) > count:
        index = max(
            (i for i in range(len(allocation)) if allocation[i] > 1),
            key=lambda i: (allocation[i], -remainders[i]),
            default=None,
        )
        if index is None:
            break
        allocation[index] -= 1
    return [
        (name, function, allocation[index])
        for index, (name, function, _) in enumerate(families)
    ]


def build_parser(
    *, canonical_prefix: str, canonical_out: Path, total: int
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="*", default=None)
    size = parser.add_mutually_exclusive_group()
    size.add_argument(
        "--count",
        type=int,
        default=None,
        help=(
            f"How many questions to build (default {total}). Family quotas keep their "
            "proportions and sum to exactly this number."
        ),
    )
    size.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Multiply every family quota. Superseded by --count; kept for old invocations.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            f"Where to write. Defaults to {canonical_out.name} for a full build on this "
            "builder's own seed, and to a seed-stamped name otherwise so a fresh draw can never "
            "land on a benchmark of record."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Generation seed. Omitted, it is taken from the clock, so every run is a new draw -- "
            "which is what a held-out set has to be built with. The seed used is printed and "
            "stamped into the default filename."
        ),
    )
    parser.add_argument("--id-prefix", default=canonical_prefix)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser


def run_builder(
    *,
    families: Sequence[Family],
    open_builder: Callable[[], Any],
    make_pool: Callable[[], Any],
    finalize: Callable[..., list[dict]],
    ordered: frozenset[str],
    canonical_seed: int,
    canonical_prefix: str,
    canonical_out: Path,
    argv: Sequence[str] | None = None,
) -> None:
    total = sum(quota for *_, quota in families)
    parser = build_parser(
        canonical_prefix=canonical_prefix, canonical_out=canonical_out, total=total
    )
    args = parser.parse_args(argv)

    seed = args.seed if args.seed is not None else int(time.time())
    # A held-out build that silently reproduces the tuned set is worse than no held-out build: it
    # reads as a fresh measurement and is the training set. Caught the hard way -- a seed chosen
    # to look like a date happened to be a builder's default, and all 100 questions came back
    # identical, anchors included.
    if args.id_prefix != canonical_prefix and seed == canonical_seed:
        raise SystemExit(
            f"--seed {canonical_seed} is this builder's default, so --id-prefix "
            f"{args.id_prefix} would relabel the tuned set rather than draw a new sample. "
            "Pick another seed."
        )

    if args.count is not None:
        selected = apportion(families, args.count)
        count = args.count
    elif args.scale is not None:
        selected = [
            (name, function, max(1, round(quota * args.scale)))
            for name, function, quota in families
        ]
        count = sum(quota for *_, quota in selected)
    else:
        selected = list(families)
        count = total

    if args.families:
        selected = [entry for entry in selected if entry[0] in args.families]
        if not selected:
            raise SystemExit(f"no family matched {args.families}")

    canonical_build = (
        seed == canonical_seed and count == total and not args.families
    )
    out = Path(
        args.out
        if args.out is not None
        else (
            canonical_out
            if canonical_build
            else canonical_out.parent / f"{args.id_prefix}_{count}_s{seed}.jsonl"
        )
    )
    if out.exists() and not args.force:
        raise SystemExit(f"{out} already exists; pass --force to replace it")

    print(f"seed={seed} count={count} out={out}", flush=True)
    builder = open_builder()
    pool = make_pool()
    rows: list[dict] = []
    try:
        for name, function, wanted in selected:
            rng = random.Random(f"{seed}:{name}")
            produced = function(builder, pool, rng, wanted)
            print(
                f"{name}: {len(produced)}/{wanted} (api={builder.provider.api_call_count})",
                flush=True,
            )
            rows.extend(produced)
    finally:
        builder.close()

    finished = finalize(rows, seed=seed, prefix=args.id_prefix, ordered=ordered)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in finished) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {out} rows={len(finished)} seed={seed}")
