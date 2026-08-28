"""The command line every K-MapEval dataset builder shares.

The builders differ only in which families they draw and what they call the result, so the
argument parsing, the seed, the apportionment and the write lived in three near-identical copies
of `main()`. They are here once instead.

Three things this adds over those copies. A build can be asked for a **number of questions**
rather than a multiplier, and the families keep their proportions while summing to exactly that
number. The **number asked for is the number written**: apportioning exactly is only half of it,
because a family that cannot draw its share leaves the file short -- `--count 300` shipped 281 to
283 rows across five draws, every one of them missing in `poi_farthest_of_three`, which took the
`poi` class from the 21% its quota encodes to 16%. A family that comes up short is frozen at what
it managed and its rows are handed to families of the same MapEval-API class, so the file holds
the count and keeps the mix. And the **seed comes from the clock** when none is given, so a build
is a fresh draw by default rather than a re-run of whatever the file was last tuned to -- which is
the state a held-out set has to be built in. The seed is printed and stamped into the filename,
because a draw nobody recorded cannot be repeated.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
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


MAX_TOPUP_ROUNDS = 3


def _identity(row: dict) -> str:
    """What makes two drawn rows the same row.

    The question text, which is what a reader would compare. A row without one is compared whole,
    so a family that has not written its question yet still deduplicates rather than collapsing
    every row onto a single `None`.
    """

    question = row.get("question")
    if isinstance(question, str):
        return question
    return json.dumps(row, sort_keys=True, ensure_ascii=False, default=repr)


def _merge(existing: list[dict], produced: list[dict], want: int) -> list[dict]:
    """Add a redraw's new rows to what a family already has, up to `want`.

    A redraw runs the same family over the same city under a different stream, so most of what it
    returns is what it returned the first time. The question text is the identity: two rows that
    print the same question are the same question however they were drawn.
    """

    seen = {_identity(row) for row in existing}
    merged = list(existing)
    for row in produced:
        if len(merged) >= want:
            break
        key = _identity(row)
        if key in seen:
            continue
        seen.add(key)
        merged.append(row)
    return merged


def _reallocate(
    *,
    allocation: dict[str, int],
    produced: dict[str, list[dict]],
    quotas: dict[str, int],
    classes: dict[str, str],
    exhausted: set[str],
) -> bool:
    """Move the rows a family could not draw onto families that can, same class first.

    A family comes up short because the city ran out of anchors it can use, not because it was
    asked for too few, so its own allocation is frozen at what it managed. The count the caller
    asked for is still the count, so the deficit is handed to the families that filled their
    share -- and to ones of the same MapEval-API class before any other, because the class mix is
    the reason the quotas are what they are. Returns whether anything moved.
    """

    deficits: Counter[str] = Counter()
    for name, want in allocation.items():
        missing = want - len(produced[name])
        if missing > 0:
            deficits[classes.get(name, "?")] += missing
            allocation[name] = len(produced[name])
            exhausted.add(name)
    if not deficits:
        return False
    donors = [name for name in allocation if name not in exhausted]
    if not donors:
        return False
    moved = 0
    for family_class, missing in deficits.most_common():
        candidates = [
            name for name in donors if classes.get(name) == family_class
        ] or donors
        candidates.sort(key=lambda name: (-quotas[name], name))
        for offset in range(missing):
            allocation[candidates[offset % len(candidates)]] += 1
            moved += 1
    return moved > 0


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
    elif args.scale is not None:
        selected = [
            (name, function, max(1, round(quota * args.scale)))
            for name, function, quota in families
        ]
    else:
        selected = list(families)

    if args.families:
        selected = [entry for entry in selected if entry[0] in args.families]
        if not selected:
            raise SystemExit(f"no family matched {args.families}")

    # The number of rows the build owes, after `--families` has had its say. It is what the file
    # must hold when the build is done, not a target the draw is free to miss: a set that came
    # back 283 rows short of the 300 it was asked for is short in one family, which moves that
    # family's class away from the proportion the quotas encode.
    count = sum(quota for *_, quota in selected)
    canonical_build = seed == canonical_seed and count == total and not args.families
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
    order = [name for name, *_ in selected]
    functions = {name: function for name, function, _ in selected}
    quotas = {name: quota for name, _, quota in selected}
    allocation = dict(quotas)
    produced: dict[str, list[dict]] = {name: [] for name in order}
    classes: dict[str, str] = {}
    exhausted: set[str] = set()
    try:
        for attempt in range(MAX_TOPUP_ROUNDS + 1):
            for name in order:
                want = allocation[name]
                if len(produced[name]) >= want:
                    continue
                # A redraw needs its own stream, or it draws the rows it already has.
                label = f"{seed}:{name}" if attempt == 0 else f"{seed}:{name}:{attempt}"
                rows = functions[name](builder, pool, random.Random(label), want)
                produced[name] = _merge(produced[name], rows, want)
                print(
                    f"{name}: {len(produced[name])}/{want} "
                    f"(api={builder.provider.api_call_count})",
                    flush=True,
                )
            for name, rows in produced.items():
                if rows and name not in classes:
                    classes[name] = rows[0].get("mapeval_class", "?")
            if sum(len(rows) for rows in produced.values()) >= count:
                break
            if attempt == MAX_TOPUP_ROUNDS:
                break
            if not _reallocate(
                allocation=allocation,
                produced=produced,
                quotas=quotas,
                classes=classes,
                exhausted=exhausted,
            ):
                break
            print(f"top-up round {attempt + 1}: {allocation}", flush=True)
    finally:
        builder.close()

    rows = [row for name in order for row in produced[name][: allocation[name]]]
    if len(rows) != count:
        short = {
            name: f"{len(produced[name])}/{allocation[name]}"
            for name in order
            if len(produced[name]) < allocation[name]
        }
        raise SystemExit(
            f"asked for {count} questions and only {len(rows)} could be drawn; short: {short}. "
            "The pool cannot supply them under this seed -- redraw, or lower --count."
        )

    finished = finalize(rows, seed=seed, prefix=args.id_prefix, ordered=ordered)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in finished) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {out} rows={len(finished)} seed={seed}")
