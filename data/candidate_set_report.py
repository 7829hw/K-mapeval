"""Where a candidate set stops being a set, per row, from a run's own executed trace.

A narrowing that is correctly extracted, correctly attached and correctly enforced still answers
nothing if the set it narrows has one member. Six `nearby_cuisine_subtype` rows showed exactly
that -- the filter or the ranking received one candidate, of four the question offered -- so the
subtype was applied to a set of one and the answer came back unrestricted anyway.

Three signatures, measured and not enforced:

  candidate_set_collapsed          a node produced several candidates and the step that measures
                                   or narrows them received one
  candidate_set_missing            that step received none, or something that is not a set
  candidate_set_identity_mismatch  it received a set, but not the one that node produced

Non-blocking on purpose. The two validators before these were tried as refusals and cost about
three answerable questions per wrong answer prevented, so the bar for making one blocking is that
replay shows it rejects substantially more wrong graphs than correct ones.
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "data")
sys.path.insert(0, ".")

from replay_grounding import _pick_log  # noqa: E402

from src.dataset import load_dataset  # noqa: E402

_EXECUTE = "[EXECUTE] "
#: Steps that consume a candidate set. Their input is where a collapse becomes visible.
_CONSUMES_A_SET = {
    "nearest": "candidates",
    "filter_places": "places",
    "within_radius": "candidates",
    "filter_by_distance": "items",
    "filter_by_direction": "places",
    "sort_by": "items",
    "count_items": "items",
    "pairwise_extremes": "locations",
}
#: Steps that produce one.
_PRODUCES_A_SET = frozenset(
    {"batch_geocode", "nearby_places", "place_search", "merge_places", "recover_option_places"}
)


def _steps(log_name: str) -> list[dict]:
    """The executed steps, each carrying the `depends_on` the executor does not record.

    `[EXECUTE]` holds resolved arguments and results and no edges; `[TRANSFORM]` holds the graph.
    Reading only the first made every step look like it depended on nothing, so a collapse
    between two nodes was invisible -- which is how the first three versions of this diagnostic
    separated nothing.
    """

    text = Path("logs", log_name).read_text(encoding="utf-8", errors="replace")
    position = text.find(_EXECUTE)
    if position < 0:
        return []
    try:
        steps = json.loads(text[position + len(_EXECUTE):].split("\n")[0]).get("steps") or []
    except json.JSONDecodeError:
        return []
    edges: dict[str, list[str]] = {}
    marker = "[TRANSFORM] "
    for line in text.splitlines():
        at = line.find(marker)
        if at < 0:
            continue
        try:
            graph = json.loads(line[at + len(marker):]).get("graph") or []
        except json.JSONDecodeError:
            continue
        edges = {
            str(node.get("id")): [str(value) for value in (node.get("depends_on") or [])]
            for node in graph
        }
    return [{**step, "depends_on": edges.get(str(step.get("id")), [])} for step in steps]


def _members(value) -> list[str] | None:
    """The identities in a collection, or None when it is not one."""

    if isinstance(value, dict):
        for key in ("ranked", "places", "items", "results"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
        else:
            # One located place where a set belongs is a set of one, not an absence. Reading it
            # as absent put the whole failure under `candidate_set_missing`, which is a different
            # claim: the set arrived, with one member, and the narrowing applied to that.
            if value.get("name") or value.get("place_id") or value.get("query"):
                value = [value]
            else:
                return None
    if not isinstance(value, list):
        return None
    names = []
    for item in value:
        if not isinstance(item, dict):
            continue
        place = item.get("place") if isinstance(item.get("place"), dict) else item
        names.append(str(place.get("name") or place.get("query") or ""))
    return names


def diagnose(steps: list[dict]) -> list[dict]:
    """Every place a set the graph produced failed to arrive where it was consumed."""

    produced: dict[str, list[str]] = {}
    for step in steps:
        members = _members(step.get("result")) if step.get("status") == "ok" else None
        if members is not None:
            produced[str(step.get("id"))] = members
    found: list[dict] = []
    for step in steps:
        slot = _CONSUMES_A_SET.get(str(step.get("operator")))
        if slot is None:
            continue
        # The set this step was *meant* to consume is the one the node it depends on produced.
        # Comparing against the largest set anywhere in the graph instead was the first version
        # of this, and it separated nothing: a four-candidate consumer downstream of a
        # forty-five-place retrieval looked collapsed whether or not anything had gone wrong.
        upstream = [
            produced[name]
            for name in _upstream_of(step, steps)
            if name in produced and len(produced[name]) > 1
        ]
        if not upstream:
            continue
        intended = max(upstream, key=len)
        received = _members((step.get("arguments") or {}).get(slot))
        entry = {
            "node": str(step.get("id")),
            "operator": str(step.get("operator")),
            "slot": slot,
            "intended": len(intended),
            "received": None if received is None else len(received),
        }
        if received is None:
            found.append({**entry, "kind": "candidate_set_missing"})
        elif len(received) <= 1:
            found.append({**entry, "kind": "candidate_set_collapsed"})
        elif not (set(received) & set(intended)):
            found.append({**entry, "kind": "candidate_set_identity_mismatch"})
    return found


def _upstream_of(step: dict, steps: list[dict]) -> set[str]:
    """Every node this step transitively reads.

    Transitive on purpose. The collapse that matters happens when a set the graph gathered
    several steps back never reaches the step that measures it -- and the direct dependency then
    looks fine, because what it points at is the single anchor the wiring put in the set's slot.
    """

    by_id = {str(other.get("id")): other for other in steps}
    seen: set[str] = set()
    frontier = list(_dependencies(step))
    while frontier:
        name = frontier.pop()
        if name in seen or name not in by_id:
            continue
        seen.add(name)
        frontier.extend(_dependencies(by_id[name]))
    return seen


def _dependencies(step: dict) -> set[str]:
    """Node ids this step reads, declared or referenced."""

    found = {str(value) for value in (step.get("depends_on") or [])}
    stack = [step.get("arguments")]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, list | tuple):
            stack.extend(value)
        elif isinstance(value, str) and value.startswith("$"):
            found.add(value[1:].split(".", 1)[0])
    return found


def report(report_paths: list[str], families: set[str] | None = None) -> None:
    table: Counter = Counter()
    by_family: Counter = Counter()
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
            kinds = {entry["kind"] for entry in diagnose(_steps(found["log"]))}
            label = "clean" if not kinds else ",".join(sorted(kinds))
            table[(label, bool(row["answer_correct"]))] += 1
            by_family[(row["template_id"], label)] += 1

    print(f"  {'signature':38s} {'correct':>7s} {'wrong':>7s} {'total':>7s} {'acc':>6s}")
    for label in sorted({key for key, _ in table}):
        right, wrong = table[(label, True)], table[(label, False)]
        total = right + wrong
        print(f"  {label:38s} {right:7d} {wrong:7d} {total:7d} {100 * right / total:5.1f}%")
    print("\n  by family:")
    for family in sorted({key for key, _ in by_family}):
        counts = {
            label: by_family[(family, label)]
            for _f, label in by_family
            if _f == family and by_family[(family, label)]
        }
        print(f"    {family:30s} {counts}")


def _window(report: dict) -> str:
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
