"""What happened to a stated subtype, per row, read off a run's own executed trace.

Five outcomes, and they do not mean the same thing. One is a reasoning failure, one is an
architectural failure, and the rest are facts about the provider's ontology:

  extraction_failure      the question narrows a kind and the facts did not read the narrowing
  narrowing_not_in_graph  the narrowing was read and the executed graph never applied it
  narrowing_dropped       the filter ran and returned every candidate, none of which qualifies
  attributes_missing      the filter ran and no candidate carried a category to test
  no_provider_match       the filter ran, the candidates carried categories, none qualified
  applied                 the filter ran and what it returned qualifies

`narrowing_not_in_graph` and `narrowing_dropped` are the two that must be zero: both are the
shape answering from the broad category, which is a different question, and the second is the
one that looks like success from outside. `no_provider_match` is a coverage limitation and an
honest empty result.
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "data")
sys.path.insert(0, ".")

from replay_grounding import _pick_log  # noqa: E402

from src.agent.spatial import _extract_target_type, extract_facts  # noqa: E402
from src.dataset import load_dataset  # noqa: E402

_STAGE = '[EXECUTE] '


def _executed(log_name: str) -> list[dict]:
    for line in Path("logs", log_name).read_text(encoding="utf-8", errors="replace").splitlines():
        position = line.find(_STAGE)
        if position < 0:
            continue
        try:
            return json.loads(line[position + len(_STAGE):]).get("steps") or []
        except json.JSONDecodeError:
            return []
    return []


def _carries_category(value) -> bool:
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if item.get("category"):
                return True
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return False


def _candidate_count(step: dict, steps: list[dict]) -> int | None:
    """How many candidates the narrowing step was handed, from the node that produced them."""

    arguments = step.get("arguments") or {}
    for slot in ("candidates", "places", "items", "locations"):
        value = arguments.get(slot)
        # The executor records the *resolved* arguments, so the candidate list is usually here
        # in full. A reference survives only where the step never ran.
        if isinstance(value, list):
            return len(value)
        if isinstance(value, dict):
            for key in ("ranked", "places", "items", "results"):
                nested = value.get(key)
                if isinstance(nested, list):
                    return len(nested)
        if not isinstance(value, str) or not value.startswith("$"):
            continue
        source = value[1:].split(".", 1)[0]
        for other in steps:
            if other.get("id") != source:
                continue
            result = other.get("result")
            values = result if isinstance(result, list) else (result or {}).get("ranked")
            return len(values) if isinstance(values, list) else None
    return None


def _outcome(steps: list[dict]) -> str:
    """Which of the five, from the filter step the run actually executed."""

    narrowing = [
        step
        for step in steps
        if (step.get("arguments") or {}).get("required_types")
        or (step.get("arguments") or {}).get("required_type")
    ]
    if not narrowing:
        return "narrowing_not_in_graph"
    step = narrowing[0]
    if step.get("status") != "ok":
        return "filter_step_failed"
    result = step.get("result")
    kept = result if isinstance(result, list) else (result or {}).get("ranked") or []
    if kept:
        # Did it narrow anything? Measured against the candidate list it was given rather than
        # against a matcher, so the answer does not depend on which revision is asking: the
        # lexicon fallback hands the whole list back, and a filter whose output is its input has
        # not filtered. That reads as success from outside, which is why it is named separately.
        given = _candidate_count(step, steps)
        if given is not None and len(kept) >= given:
            return "narrowing_dropped"
        return "applied"
    inputs = [
        other.get("result")
        for other in steps
        if other.get("id") != step.get("id") and other.get("status") == "ok"
    ]
    return "no_provider_match" if _carries_category(inputs) else "attributes_missing"


def classify(report_path: str, family: str = "nearby_cuisine_subtype") -> None:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    items = {
        item.id: item for item in load_dataset(Path(report["metadata"]["dataset_source"]))
    }
    before = Path(report_path).stem.removeprefix("test_")
    table: Counter = Counter()
    rows = []
    for row in report["results"]:
        if row["template_id"] != family:
            continue
        item = items[row["id"]]
        facts = extract_facts({}, item.question)
        stated = _extract_target_type(item.question)
        if stated and not facts.target_subtype:
            outcome = "extraction_failure"
        elif not facts.target_subtype:
            outcome = "no_subtype_stated"
        else:
            found = _pick_log(item.id, item.question, before)
            outcome = _outcome(_executed(found["log"])) if found else "no_log"
        table[(outcome, bool(row["answer_correct"]))] += 1
        rows.append((row["id"], outcome, bool(row["answer_correct"]), facts.target_subtype))

    print(f"{family} at {report['metadata']['code_revision']}: {len(rows)} rows")
    print(f"  {'outcome':24s} {'correct':>7s} {'wrong':>7s} {'total':>7s}")
    for outcome in sorted({key for key, _ in table}):
        right, wrong = table[(outcome, True)], table[(outcome, False)]
        print(f"  {outcome:24s} {right:7d} {wrong:7d} {right + wrong:7d}")
    print(f"  {'TOTAL':24s} "
          f"{sum(v for (_, c), v in table.items() if c):7d} "
          f"{sum(v for (_, c), v in table.items() if not c):7d} {len(rows):7d}")
    return rows


if __name__ == "__main__":
    classify(*sys.argv[1:])
