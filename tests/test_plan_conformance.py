"""A retrieved shape's steps are required, and a graph that drops one says so or is refused.

A skeleton is retrieved because the question has that structure. Prose in the planner prompt did
not carry it: eight recorded `nearby_cuisine_subtype` graphs had `Search-Narrow-Rank` in front of
them and no narrowing in them at all -- five copied the rival shape retrieved beside it, two
stopped after the retrieval, one wrote a `FILTER` whose inputs named concepts rather than nodes.

So the requirement is checked before factorization rather than asked for. It is checked *before*
because the alternative is the deterministic stage inserting a step the planner did not reason
its way to, which would be exactly the split this architecture exists to keep.
"""

from __future__ import annotations

from src.agent.geoflow import TEMPLATES, conformance_violations

REQUIRES = {
    "transforms": ("PLACE_SEARCH", "FILTER"),
    "dependencies": (("PLACE_SEARCH", "FILTER"),),
}

CONFORMING = [
    {"id": "anchor", "transform": "RESOLVE_PLACES", "inputs": []},
    {"id": "found", "transform": "PLACE_SEARCH", "inputs": ["anchor"]},
    {"id": "narrowed", "transform": "FILTER", "inputs": ["found"]},
    {"id": "answer", "transform": "MATCH_OPTIONS", "inputs": ["narrowed"]},
]


def test_a_conforming_graph_passes() -> None:
    assert conformance_violations({"graph": CONFORMING}, REQUIRES) == []


def test_a_missing_required_step_is_named() -> None:
    without = [step for step in CONFORMING if step["transform"] != "FILTER"]
    without[-1]["inputs"] = ["found"]

    violations = conformance_violations({"graph": without}, REQUIRES)

    assert len(violations) == 1
    assert "FILTER" in violations[0]


def test_a_required_step_that_consumes_the_wrong_thing_is_named() -> None:
    """Present but disconnected: it narrows something the retrieval never produced."""

    detached = [dict(step) for step in CONFORMING]
    detached[2] = {**detached[2], "inputs": ["anchor"]}

    violations = conformance_violations({"graph": detached}, REQUIRES)

    assert len(violations) == 1
    assert "consume" in violations[0]


def test_a_step_may_be_declined_but_only_out_loud() -> None:
    without = [step for step in CONFORMING if step["transform"] != "FILTER"]

    assert conformance_violations(
        {"graph": without, "not_applicable": ["FILTER"]}, REQUIRES
    ) == []


def test_the_requirement_is_satisfied_through_the_nodes_between() -> None:
    """The steps need not be adjacent; the dependency is what has to survive."""

    indirect = [
        {"id": "anchor", "transform": "RESOLVE_PLACES", "inputs": []},
        {"id": "found", "transform": "PLACE_SEARCH", "inputs": ["anchor"]},
        {"id": "ranked", "transform": "DISTANCE_MEASURE", "inputs": ["anchor", "found"]},
        {"id": "narrowed", "transform": "FILTER", "inputs": ["ranked"]},
    ]

    assert conformance_violations({"graph": indirect}, REQUIRES) == []


def test_a_shape_with_no_declared_structure_requires_nothing() -> None:
    assert conformance_violations({"graph": []}, None) == []


def test_only_shapes_whose_omission_was_measured_declare_structure() -> None:
    """A requirement asserted rather than evidenced turns working graphs into refusals.

    `Search-Narrow-Rank` is the case in point and deliberately declares none: the obvious
    `PLACE_SEARCH -> FILTER` contract is violated by 45 of 47 recorded rows, 37 of which answered
    correctly, because grounding binds the stated kind onto `nearest` as well and a graph that
    measures the anchor against a set enforces the narrowing with no filter in it.
    """

    declaring = {
        template["name"] for template in TEMPLATES.values() if template.get("requires")
    }

    assert declaring == {"Listed-Measure-Filter-Count", "Filter-Aggregate-Measure"}
    assert "Search-Narrow-Rank" not in declaring


def test_the_rival_shape_is_not_offered_beside_the_narrowing_one() -> None:
    """`Geocode-Batch-Compare` resolves the candidates and ranks them, which is the wrong answer.

    Five of the eight graphs that came out with no narrowing had copied it, and it was retrieved
    beside `Search-Narrow-Rank` on every one of them.
    """

    assert "geocode_compare" in TEMPLATES["search_narrow_rank"]["supersedes"]
