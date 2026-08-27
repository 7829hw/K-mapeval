"""Does the reported answer follow the evidence the graph computed, or the generator's own?

Three outcomes per question, from the executed trace and the reported answer:

  agrees              the graph pointed at an option and the generator reported that option
  generator_override  the graph pointed at an option and the generator reported another
  no_graph_evidence   the graph pointed at no option at all

`agrees` is the only one where a correct answer is evidence that the architecture answered it. A
correct `generator_override` is the response stage's lexical knowledge reaching past a
measurement, and a correct `no_graph_evidence` is a guess that happened to land -- neither is
Spatial-Agent reasoning, and pooling them into a family accuracy makes a graph defect invisible.
"""

import json
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, "data")
sys.path.insert(0, ".")

from replay_grounding import _pick_log  # noqa: E402

from src.dataset import load_dataset  # noqa: E402

_EXECUTE = "[EXECUTE] "


def _steps(log_name: str) -> list[dict]:
    text = Path("logs", log_name).read_text(encoding="utf-8", errors="replace")
    position = text.find(_EXECUTE)
    if position < 0:
        return []
    try:
        return json.loads(text[position + len(_EXECUTE):].split("\n")[0]).get("steps") or []
    except json.JSONDecodeError:
        return []


def _index_from(value) -> int | None:
    """An option index a matcher reported, in any of the shapes the operators emit."""

    if isinstance(value, int):
        return value
    if not isinstance(value, dict):
        return None
    for key in ("option_index", "best_option_index", "index"):
        found = value.get(key)
        if isinstance(found, int):
            return found
    return None


def graph_choice(steps: list[dict], options: list[str]) -> int | None:
    """Which option the computed evidence points at, or None when it points at none."""

    for step in reversed([step for step in steps if step.get("status") == "ok"]):
        result = step.get("result")
        if not isinstance(result, dict):
            continue
        for key in ("best_option", "best_distance_option", "best_duration_option"):
            chosen = _index_from(result.get(key))
            if chosen is not None:
                return chosen
        matches = result.get("option_matches") or result.get("comparisons")
        if isinstance(matches, list) and matches:
            scored = [
                row
                for row in matches
                if isinstance(row, dict)
                and (row.get("similarity") or row.get("matched") or row.get("rank") is not None)
            ]
            if scored:
                best = max(scored, key=lambda row: float(row.get("similarity") or 0))
                chosen = _index_from(best)
                if chosen is not None:
                    return chosen
    # Nothing matched options outright. A measure that named a place still points somewhere.
    for step in reversed([step for step in steps if step.get("status") == "ok"]):
        for name in _names(step.get("result")):
            ranked = sorted(
                range(len(options)),
                key=lambda index: SequenceMatcher(None, name, options[index]).ratio(),
                reverse=True,
            )
            if ranked and SequenceMatcher(None, name, options[ranked[0]]).ratio() >= 0.75:
                return ranked[0]
    return None


def _names(value, depth: int = 0) -> list[str]:
    if depth > 4:
        return []
    if isinstance(value, dict):
        found = value.get("name")
        if isinstance(found, str) and found:
            return [found]
        return [name for item in value.values() for name in _names(item, depth + 1)]
    if isinstance(value, list):
        return [name for item in value[:1] for name in _names(item, depth + 1)]
    return []


def report(report_paths: list[str], families: set[str] | None = None) -> None:
    table: Counter = Counter()
    per_family: Counter = Counter()
    for path in report_paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        items = {
            item.id: item
            for item in load_dataset(Path(payload["metadata"]["dataset_source"]))
        }
        before = Path(path).stem.removeprefix("test_")
        since = _window(payload)
        for row in payload["results"]:
            if families and row["template_id"] not in families:
                continue
            item = items.get(row["id"])
            found = _pick_log(item.id, item.question, before, since) if item else None
            if found is None:
                continue
            chosen = graph_choice(_steps(found["log"]), list(item.options))
            reported = row.get("predicted_option")
            if chosen is None:
                outcome = "no_graph_evidence"
            elif reported is not None and int(reported) == chosen:
                outcome = "agrees"
            else:
                outcome = "generator_override"
            table[(outcome, bool(row["answer_correct"]))] += 1
            per_family[(row["template_id"], outcome)] += 1

    print(f"  {'outcome':20s} {'correct':>7s} {'wrong':>7s} {'total':>7s}")
    for outcome in ("agrees", "generator_override", "no_graph_evidence"):
        right, wrong = table[(outcome, True)], table[(outcome, False)]
        print(f"  {outcome:20s} {right:7d} {wrong:7d} {right + wrong:7d}")
    print("\n  by family:")
    for family in sorted({key for key, _ in per_family}):
        counts = {
            outcome: per_family[(family, outcome)]
            for outcome in ("agrees", "generator_override", "no_graph_evidence")
        }
        print(f"    {family:30s} {counts}")


def _window(report: dict) -> str:
    """The earliest log this report's own run could have written.

    A report states when it finished and how long each question took; the run began no earlier
    than its finish minus the total of the slowest concurrent stream, so the start of the slowest
    question is a safe lower bound. Logs older than that belong to some other run, and joining
    one to this report's rows measures a revision that is not the one being reported.
    """

    from datetime import datetime, timedelta

    finished = datetime.strptime(report["metadata"]["timestamp"][:19], "%Y-%m-%dT%H:%M:%S")
    longest = max((row.get("time") or 0) for row in report["results"])
    total = sum((row.get("time") or 0) for row in report["results"])
    concurrency = max(int(report["metadata"].get("concurrency") or 1), 1)
    span = max(total / concurrency, longest) + 120
    return (finished - timedelta(seconds=span)).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    split = sys.argv.index("--families") if "--families" in sys.argv else len(sys.argv)
    report(sys.argv[1:split], set(sys.argv[split + 1:]) or None)
