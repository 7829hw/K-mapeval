"""Audit a built benchmark for answer keys that are not the map.

Every defect this checks for was found by hand first, after the dataset had already been built,
measured and written up:

- **A fixed gold rank.** `_distance_options` was called with one multiplier tuple per family, so
  sorting the four printed numbers put the gold at a constant index. Twenty-eight of v4's hundred
  rows are answerable that way. Fixed by `straddling_multipliers`; found by noticing that the
  no-tool floor on one family was 6/8.
- **A dead option.** `trip_feasible_count` offers 한 곳 through 네 곳 and can never answer 네 곳 —
  the guard that rejects rows answerable from stay times alone always fires on the maximum count,
  because travel plus stays fitting every stop implies stays alone fit every stop. The same held
  for v5's `nearby_within_radius`, which drew its count from three values against a four-rung
  ladder. Both were three-way questions printed as four-way ones.
- **The gold written into its own question**, which the option-listing families risk by design.
- **Duplicate options**, where two rows of the option set are the same string.

None of these is visible in an accuracy, and the no-tool floor only catches the ones a model
happens to exploit. Run this after every build, before the floor:

    python data/audit_dataset.py dataset/seoul_kmapeval_v6_mcq_100.jsonl

Exits non-zero when anything is reported, so it can gate a build loop.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# A family needs at least this many rows before "the gold never lands here" means anything. Below
# it, an unused position is ordinary sampling.
MIN_ROWS_FOR_POSITION = 6

# Unless the family prints the *same option set* on every row -- a ladder like 한 곳…네 곳, where
# the options are the answer space itself rather than four places drawn per question. There an
# unused rung is a property of the generator, not of the draw, and four rows are enough to see it.
# v5's `nearby_within_radius` had exactly four rows and drew its count from three values.
MIN_ROWS_FOR_LADDER = 4

# ... and this many parsed numeric option sets before a constant gold rank is a finding.
MIN_ROWS_FOR_RANK = 5

NUMBER = re.compile(r"(-?\d+(?:\.\d+)?)")

# The refusal `finalize` plants as a distractor; never part of a family's own vocabulary.
REFUSAL = "주어진 지도 정보로는 알 수 없음"


def numeric_options(options: list[str]) -> list[float] | None:
    """The options as numbers, when every one of them carries exactly one.

    A number embedded in a name ("CU 자양뚝섬길점 ATM") is not a measurement, so a set is only
    numeric when every option parses to a single number and no option carries two.
    """

    values: list[float] = []
    for option in options:
        found = NUMBER.findall(option)
        if len(found) != 1:
            return None
        values.append(float(found[0]))
    return values


def audit(path: Path) -> list[str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    findings: list[str] = []

    by_family: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_family[row.get("template_id") or row.get("classification") or "?"].append(row)

    for row in rows:
        options = row["options"]
        if len(set(options)) != len(options):
            findings.append(f"{row['id']}: duplicate options {options}")
        if options[row["answer"]] in row["question"]:
            findings.append(
                f"{row['id']}: the gold option text appears in its own question"
            )

    for family, family_rows in sorted(by_family.items()):
        widths = {len(row["options"]) for row in family_rows}
        positions = Counter(row["answer"] for row in family_rows)
        if len(family_rows) >= MIN_ROWS_FOR_POSITION and len(widths) == 1:
            width = next(iter(widths))
            dead = [index for index in range(width) if positions[index] == 0]
            if dead:
                findings.append(
                    f"{family}: over {len(family_rows)} rows the gold never lands at "
                    f"{dead} of {width} positions -- an option that cannot be the answer is not "
                    "an option"
                )

        # A ladder family draws its options from one small fixed vocabulary rather than from the
        # map, so an unused rung is the generator's, not the draw's, and four rows show it. The
        # refusal is excluded because `finalize` plants it as a distractor on a quarter of the
        # answerable rows, which is what stopped a strict "same options every row" test from
        # recognising v5's `nearby_within_radius` as a ladder at all.
        vocabulary = {
            option
            for row in family_rows
            for option in row["options"]
            if option != REFUSAL
        }
        if len(family_rows) >= MIN_ROWS_FOR_LADDER and len(widths) == 1:
            if len(vocabulary) == next(iter(widths)):
                golds = {row["options"][row["answer"]] for row in family_rows}
                unused = sorted(vocabulary - golds)
                if unused:
                    findings.append(
                        f"{family}: options come from the fixed set {sorted(vocabulary)}, and "
                        f"over {len(family_rows)} rows {unused} is never the answer -- a rung "
                        "that cannot be reached is not a choice"
                    )

        ranks: Counter[int] = Counter()
        for row in family_rows:
            values = numeric_options(row["options"])
            if values is None:
                continue
            order = sorted(range(len(values)), key=lambda index: values[index])
            ranks[order.index(row["answer"])] += 1
        parsed = sum(ranks.values())
        if parsed >= MIN_ROWS_FOR_RANK and len(ranks) == 1:
            rank = next(iter(ranks))
            findings.append(
                f"{family}: on all {parsed} numeric rows the gold is at sorted rank {rank} -- "
                "sorting the options and taking that index answers the family with no map"
            )

    return findings


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: audit_dataset.py <dataset.jsonl> [<dataset.jsonl> ...]")
    total = 0
    for name in sys.argv[1:]:
        path = Path(name)
        findings = audit(path)
        total += len(findings)
        print(f"=== {path}")
        if not findings:
            print("  clean")
        for finding in findings:
            print(f"  {finding}")
    raise SystemExit(1 if total else 0)


if __name__ == "__main__":
    main()
