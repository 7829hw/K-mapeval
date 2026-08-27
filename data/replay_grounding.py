"""Replay `_ground_graph_literals` over planner graphs a recorded run already produced.

A change to grounding is a change to what every Spatial-Agent question executes, and the cheap
way to measure its footprint is not another benchmark run: the logs already hold, per question,
the Analysis stage's output and the planner's raw graph. Grounding is a pure function of those
plus the question and its options, so it can be re-run offline as many times as there are
revisions to compare, with no LLM calls and no Kakao quota.

    python data/replay_grounding.py reports/test_A.json ... --out before.json
    git checkout <other-revision> && python data/replay_grounding.py ... --out after.json
    python data/replay_grounding.py --diff before.json after.json

The diff is counted by `template_id` and by `mapeval_class`, because "the overall accuracy did
not move" and "no family moved" are different claims and only the second one is evidence.

This is offline dataset tooling: it imports `src/`, and nothing under `src/` imports it.
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"

_STAGE = re.compile(r"\[(ANALYZE|COMPOSE)\] (\{.*\})\s*$")
_QUESTION = re.compile(r"^\d\d:\d\d:\d\d \[INFO \] Question: (.*)$")
_LOG_STAMP = re.compile(r"^(\d{8}T\d{6}Z)_")


def _log_candidates(question_id: str) -> list[Path]:
    return sorted(LOGS.glob(f"*_spatial_agent_id{question_id}_*.log"))


def _read_log(path: Path) -> dict[str, Any] | None:
    """The question text, the Analysis output and the planner's raw graph, or nothing."""

    found: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        asked = _QUESTION.match(line)
        if asked and "question" not in found:
            found["question"] = asked.group(1)
            continue
        staged = _STAGE.search(line)
        if not staged:
            continue
        key = staged.group(1).lower()
        if key in found:
            continue
        try:
            found[key] = json.loads(staged.group(2))
        except json.JSONDecodeError:
            return None
    if {"question", "analyze", "compose"} <= set(found) and isinstance(found["compose"], dict):
        graph = found["compose"].get("graph")
        if isinstance(graph, list) and graph:
            found["graph"] = graph
            return found
    return None


def _pick_log(question_id: str, question: str, before: str) -> dict[str, Any] | None:
    """The latest log for this question written no later than the run that reported it.

    Question ids repeat across draws, so the text has to match too; without that a `kmapeval_042`
    from another 300-question set would be replayed against this one's options.
    """

    for path in reversed(_log_candidates(question_id)):
        stamp = _LOG_STAMP.match(path.name)
        if not stamp or stamp.group(1) > before:
            continue
        parsed = _read_log(path)
        if parsed and _same_question(parsed["question"], question):
            parsed["log"] = path.name
            return parsed
    return None


def _same_question(logged: str, asked: str) -> bool:
    """The log elides a long question with a trailing ellipsis, so compare on the prefix.

    Every `trip_*` question here is long enough to be elided, which is the whole class the
    grounding change touches most -- an exact comparison silently dropped all of them.
    """

    if logged.endswith("..."):
        return asked.startswith(logged[:-3])
    return logged == asked


def _ground(graph: list[dict[str, Any]], question: str, options: list[str], analysis: dict) -> Any:
    """Call whichever signature this revision's grounding has."""

    from src.agent import spatial

    parameters = inspect.signature(spatial._ground_graph_literals).parameters
    if "intent" in parameters:
        return spatial._ground_graph_literals(
            graph,
            question,
            options,
            str(analysis.get("intent") or "poi"),
            inferred_type=analysis.get("target_type"),
        )
    # Grounding takes the question's stated factors now, and reading them is part of what a
    # replay has to reproduce -- `extract_facts` is where the intent branches went.
    # The provider's retrieval vocabulary is part of what the agent grounds with, so a replay
    # that left it at the canonical default would report a footprint the runtime never has.
    from src.tools.kakao import retrieval_specs

    facts = spatial.extract_facts(analysis, question)
    graph = _factorize(graph, analysis, options, facts)
    return spatial._ground_graph_literals(
        graph,
        question,
        options,
        facts,
        retrieval_specs=retrieval_specs,
    )


def _factorize(
    graph: list[dict[str, Any]], analysis: dict[str, Any], options: list[str], facts: Any
) -> list[dict[str, Any]]:
    """G -> G', where this revision has that stage; the recorded graph unchanged where it does not.

    The planner answers in transformations now, and which operator performs each is decided
    deterministically before grounding runs. That whole tail is still a pure function of the
    recorded plan, so it replays the same way -- and a change to the operator choice shows up
    here rather than needing a benchmark pass to see.
    """

    try:
        from src.agent.geoflow import OPERATOR_CONTRACTS
        from src.agent.semantics import factorize_semantic_graph, is_semantic_graph
    except ImportError:
        return graph
    if not is_semantic_graph(graph):
        return graph
    return factorize_semantic_graph(
        graph,
        concepts=analysis.get("concepts") or [],
        options=options,
        facts=facts,
        available=frozenset(OPERATOR_CONTRACTS),
    ).graph


def replay(report_paths: list[Path]) -> dict[str, Any]:
    from src.dataset import load_dataset, resolve_mapeval_class

    grounded: dict[str, Any] = {}
    misses = Counter()
    for report_path in report_paths:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        source = report["metadata"]["dataset_source"]
        items = {item.id: item for item in load_dataset(ROOT / source)}
        before = report_path.stem.removeprefix("test_")
        for row in report["results"]:
            item = items.get(row["id"])
            if item is None:
                misses["not in dataset"] += 1
                continue
            found = _pick_log(item.id, item.question, before)
            if found is None:
                misses["no usable log"] += 1
                continue
            key = f"{report_path.stem}:{item.id}"
            entry: dict[str, Any] = {
                "log": found["log"],
                "template_id": item.template_id,
                "mapeval_class": resolve_mapeval_class(item),
                "classification": item.classification,
                "intent": found["analyze"].get("intent"),
            }
            try:
                entry["grounded"] = _ground(
                    found["graph"], item.question, list(item.options), found["analyze"]
                )
            except Exception as error:  # a graph this revision refuses is itself an outcome
                entry["error"] = f"{type(error).__name__}: {error}"
            grounded[key] = entry
    return {"questions": grounded, "skipped": dict(misses)}


def diff(before: dict[str, Any], after: dict[str, Any]) -> None:
    left, right = before["questions"], after["questions"]
    shared = sorted(set(left) & set(right))
    changed = [
        key
        for key in shared
        if json.dumps(left[key].get("grounded"), sort_keys=True, ensure_ascii=False)
        != json.dumps(right[key].get("grounded"), sort_keys=True, ensure_ascii=False)
        or left[key].get("error") != right[key].get("error")
    ]
    print(f"replayed {len(shared)} graphs, {len(changed)} changed")
    for axis in ("template_id", "mapeval_class"):
        totals = Counter(left[key][axis] for key in shared)
        moved = Counter(left[key][axis] for key in changed)
        print(f"\nby {axis}:")
        for label in sorted(totals):
            print(f"  {str(label):32s} {moved[label]:4d} / {totals[label]:4d}")
    errors_before = sum("error" in left[key] for key in shared)
    errors_after = sum("error" in right[key] for key in shared)
    print(f"\ngrounding raised on {errors_before} graphs before, {errors_after} after")
    for key in changed[:5]:
        print(f"\n--- {key} ({left[key]['template_id']}, intent={left[key]['intent']})")
        print(f"  before error: {left[key].get('error')}")
        print(f"  after  error: {right[key].get('error')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="*", type=Path, help="run reports to replay")
    parser.add_argument("--out", type=Path, help="write the grounded graphs here")
    parser.add_argument("--diff", nargs=2, type=Path, help="compare two --out files")
    arguments = parser.parse_args()
    if arguments.diff:
        first, second = (json.loads(path.read_text(encoding="utf-8")) for path in arguments.diff)
        diff(first, second)
        return
    if not arguments.reports or arguments.out is None:
        parser.error("give one or more reports and --out, or --diff two output files")
    result = replay(arguments.reports)
    arguments.out.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    print(f"replayed {len(result['questions'])} graphs, skipped {result['skipped']}")


if __name__ == "__main__":
    main()
