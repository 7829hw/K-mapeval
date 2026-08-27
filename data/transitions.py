"""Where the offline-changed graphs ended up, and which inputs went unread.

Two diagnostics the footprint alone cannot give. A changed graph is a claim that the run should
differ; whether it differs *for the better* is only visible once both runs exist, so the
transition matrix is reported over exactly the rows the replay said would change. And
`unconsumed_inputs` is reported by operator so a violation is attributed rather than counted.
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "data")
sys.path.insert(0, ".")

from replay_grounding import _pick_log  # noqa: E402

from src.agent.semantics import _PARTIAL_CONSUMERS, unconsumed_inputs  # noqa: E402
from src.dataset import load_dataset  # noqa: E402


def changed_ids(before_path: str, after_path: str) -> set[str]:
    before = json.loads(Path(before_path).read_text(encoding="utf-8"))["questions"]
    after = json.loads(Path(after_path).read_text(encoding="utf-8"))["questions"]
    changed = set()
    for key in set(before) & set(after):
        same = json.dumps(before[key].get("grounded"), sort_keys=True) == json.dumps(
            after[key].get("grounded"), sort_keys=True
        )
        if not same or before[key].get("error") != after[key].get("error"):
            changed.add(key.split(":", 1)[1])
    return changed


def correctness(report_paths: list[str]) -> dict[str, bool]:
    """Correct on a majority of the passes this revision ran, so one draw is not the verdict."""

    tally: Counter = Counter()
    total: Counter = Counter()
    for path in report_paths:
        for row in json.loads(Path(path).read_text(encoding="utf-8"))["results"]:
            tally[row["id"]] += bool(row["answer_correct"])
            total[row["id"]] += 1
    return {key: tally[key] * 2 > total[key] for key in total}


def transitions(base_reports: list[str], new_reports: list[str], ids: set[str]) -> None:
    was, now = correctness(base_reports), correctness(new_reports)
    shared = sorted(ids & set(was) & set(now))
    table = Counter((was[key], now[key]) for key in shared)
    print(f"over the {len(shared)} offline-changed graphs both runs answered:")
    for before, after in ((True, True), (False, True), (True, False), (False, False)):
        label = f"{'correct' if before else 'wrong':7s} -> {'correct' if after else 'wrong'}"
        print(f"  {label:20s} {table[(before, after)]:4d}")
    net = table[(False, True)] - table[(True, False)]
    print(f"  net {net:+d}")


def unconsumed(report_path: str) -> None:
    """Every non-exempt node of every executed graph, checked against the invariant."""

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    items = {
        item.id: item for item in load_dataset(Path(report["metadata"]["dataset_source"]))
    }
    before = Path(report_path).stem.removeprefix("test_")
    offenders: Counter = Counter()
    nodes = 0
    for row in report["results"]:
        item = items.get(row["id"])
        found = _pick_log(item.id, item.question, before) if item else None
        if found is None:
            continue
        for step in _transformed(found["log"]):
            operator = step.get("operator")
            if not isinstance(operator, str):
                continue
            nodes += 1
            if operator in _PARTIAL_CONSUMERS:
                continue
            missed = unconsumed_inputs(step.get("depends_on") or [], step.get("arguments"))
            if missed:
                offenders[operator] += 1
    print(f"unconsumed inputs over {nodes} executed nodes:")
    if not offenders:
        print("  none")
    for operator, count in offenders.most_common():
        print(f"  {operator:24s} {count}")


def _transformed(log_name: str) -> list[dict]:
    marker = "[TRANSFORM] "
    text = Path("logs", log_name).read_text(encoding="utf-8", errors="replace")
    graphs = []
    for line in text.splitlines():
        position = line.find(marker)
        if position < 0:
            continue
        try:
            graphs.append(json.loads(line[position + len(marker):]).get("graph") or [])
        except json.JSONDecodeError:
            continue
    return graphs[-1] if graphs else []


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "unconsumed":
        unconsumed(sys.argv[2])
    else:
        before_file, after_file = sys.argv[2], sys.argv[3]
        split = sys.argv.index("--new")
        transitions(sys.argv[4:split], sys.argv[split + 1:], changed_ids(before_file, after_file))
