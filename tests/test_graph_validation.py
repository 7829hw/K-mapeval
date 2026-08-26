"""Which of G1-G5 refuses always, and which informs the repair and then steps aside.

The split is not "strict versus relaxed"; it is "the paper's constraint versus this port's
heuristic". Upstream Spatial-Agent has no operator input-type table, no reference-shape rule and
no statically-known-argument check. Those three are local additions, and `AGENTS.md` already
records that when the table and an operator's implementation disagree the implementation is
right -- so refusing a plan the executor would have run correctly is a defect, not rigour.

Measured over 2,032 recorded questions at `af51e93`, 122 (7.0%) reached execution only through
the lenient attempt, and they answered 82.0% correctly against 82.7% for everything else. What
they were refused by:

    Concept role ordering violation   62   93.0% correct
    Data availability violation       42
    Type compatibility violation      11
    other                              7
    Role ordering violation (graph)    0

The last line is the point. G2 on the *executable* graph -- the roles factorization assigned to
operator nodes -- was never the last blocker, so it costs nothing to make unconditional, and it
now is. The concept-level check is a different claim: it compares roles the Analysis stage
assigned across edges that stage drew, so a violation reports a disagreement with that stage's
labelling rather than a fault in the graph about to run. It stays a repair signal.
"""

from __future__ import annotations

import pytest

from src.agent.geoflow import normalize_and_validate_graph

_EXTENT = {
    "id": "places",
    "operator": "batch_geocode",
    "arguments": {"place_names": ["가", "나"]},
    "depends_on": [],
    "output_type": "object",
    "role": "extent",
}


def _graph(*steps: dict[str, object]) -> dict[str, object]:
    return {"graph": [_EXTENT, *steps]}


def _fresh(graph: dict[str, object]) -> dict[str, object]:
    """Validation normalizes in place, so a graph checked twice needs two copies."""

    return {"graph": [dict(step) for step in graph["graph"]]}


def test_operator_graph_role_ordering_refuses_on_both_passes() -> None:
    """G2 as the paper states it: a narrower role may not be computed out of a wider one.

    Reachable only among `sub_condition < condition < support`. The measure end of the order is
    already guaranteed by construction -- a `measure` node that something consumes is demoted to
    `support` and the terminals are promoted -- which is the other half of why making this
    unconditional costs nothing.
    """

    inverted = _graph(
        {
            "id": "wide",
            "operator": "identity_measure",
            "arguments": {"value": "$places"},
            "depends_on": ["places"],
            "output_type": "object",
            "role": "support",
        },
        {
            "id": "narrow",
            "operator": "identity_measure",
            "arguments": {"value": "$wide"},
            "depends_on": ["wide"],
            "output_type": "object",
            "role": "condition",
        },
        {
            "id": "answer",
            "operator": "identity_measure",
            "arguments": {"value": "$narrow"},
            "depends_on": ["narrow"],
            "output_type": "object",
            "role": "measure",
        },
    )
    for strict in (True, False):
        with pytest.raises(ValueError, match="Role ordering violation"):
            normalize_and_validate_graph(_fresh(inverted), max_steps=8, strict_types=strict)


def test_a_consumed_measure_is_relabelled_rather_than_refused() -> None:
    """The label is what is wrong there, not the plan, so G2 is met by fixing the label."""

    mislabelled = _graph(
        {
            "id": "answer",
            "operator": "identity_measure",
            "arguments": {"value": "$places"},
            "depends_on": ["places"],
            "output_type": "object",
            "role": "measure",
        },
        {
            "id": "after",
            "operator": "identity_measure",
            "arguments": {"value": "$answer"},
            "depends_on": ["answer"],
            "output_type": "object",
            "role": "support",
        },
    )
    steps, _ = normalize_and_validate_graph(mislabelled, max_steps=8, strict_types=True)

    assert [(step["id"], step["role"]) for step in steps] == [
        ("places", "extent"),
        ("answer", "support"),
        ("after", "measure"),
    ]


def test_the_operator_input_type_table_informs_repair_and_then_steps_aside() -> None:
    """A port-local rule. It refuses strictly so the repair round hears about it, and no more."""

    mistyped = {
        "graph": [
            {
                "id": "legs",
                "operator": "distance_matrix",
                "arguments": {"origins": ["가"], "destinations": ["나"]},
                "depends_on": [],
                "output_type": "field",
                "role": "extent",
            },
            {
                "id": "answer",
                "operator": "haversine_distance",
                "arguments": {"place_a": "$legs", "place_b": "$legs"},
                "depends_on": ["legs"],
                "output_type": "amount",
                "role": "measure",
            },
        ]
    }
    with pytest.raises(ValueError, match="Type compatibility violation"):
        normalize_and_validate_graph(_fresh(mistyped), max_steps=8, strict_types=True)
    steps, _ = normalize_and_validate_graph(_fresh(mistyped), max_steps=8, strict_types=False)

    assert [step["id"] for step in steps] == ["legs", "answer"]


def test_a_graph_with_no_measure_is_given_one_rather_than_refused() -> None:
    """G5 is about reaching a measure; a terminal that nothing consumes is what a measure is."""

    unlabelled = _graph(
        {
            "id": "dangling",
            "operator": "identity_measure",
            "arguments": {"value": "$places"},
            "depends_on": ["places"],
            "output_type": "object",
            "role": "support",
        }
    )
    steps, _ = normalize_and_validate_graph(unlabelled, max_steps=8, strict_types=True)

    assert steps[-1]["role"] == "measure"


def test_an_unreachable_node_refuses_on_both_passes() -> None:
    """G5 proper: a node no dependency reaches cannot contribute to the answer."""

    orphaned = _graph(
        {
            "id": "answer",
            "operator": "identity_measure",
            "arguments": {"value": "$places"},
            "depends_on": ["places"],
            "output_type": "object",
            "role": "measure",
        },
        {
            "id": "stray",
            "operator": "identity_measure",
            "arguments": {"value": "$nowhere"},
            "depends_on": ["nowhere"],
            "output_type": "object",
            "role": "support",
        },
    )
    for strict in (True, False):
        with pytest.raises(ValueError):
            normalize_and_validate_graph(_fresh(orphaned), max_steps=8, strict_types=strict)


def test_an_unbuildable_graph_is_its_own_failure_type() -> None:
    """Not `agent_reasoning_failure`: nothing ran, so nothing reasoned its way anywhere."""

    from src.agent.spatial import GraphValidationError
    from src.evaluator import GRAPH_VALIDATION_FAILURE

    assert GRAPH_VALIDATION_FAILURE == "graph_validation_failure"
    assert issubclass(GraphValidationError, RuntimeError)
