"""Macro-template retrieval is a construction prior, never a validity constraint."""

from __future__ import annotations

import pytest

from src.agent.composition import compose_templates
from src.agent.geoflow import TEMPLATES
from src.agent.templates import MACRO_TEMPLATES
from src.agent.validation import validate_geoflow_graph


def test_no_legacy_template_declares_required_structure() -> None:
    declaring = {template["name"] for template in TEMPLATES.values() if template.get("requires")}

    assert declaring == set()


def test_macro_templates_expose_ports_without_required_structure() -> None:
    for template in MACRO_TEMPLATES.values():
        assert template.input_ports
        assert template.output_ports
        assert "requires" not in template.as_dict()


@pytest.mark.parametrize("template_key", sorted(MACRO_TEMPLATES))
def test_each_macro_template_composes_into_a_valid_geoflow_graph(template_key: str) -> None:
    """A prior the validator would refuse is a prior that teaches the planner an invalid graph.

    Composition renames the fragment's placeholders and nothing else, so G1-G5 over the composed
    graph is the closest thing to a unit test a template has.
    """

    result = validate_geoflow_graph(compose_templates([MACRO_TEMPLATES[template_key]]))

    assert all(result.constraints.values())
    assert len(result.topological_order) == len(MACRO_TEMPLATES[template_key].transformation_edges)


def test_declaring_a_port_is_not_a_ranking_advantage() -> None:
    """Ports say what a fragment may be handed; they must not say how well it fits.

    `TIME-WINDOW-REVERSE` declares a second, temporal input port because its clock input is an
    `event`, and ranking on the union of a template's ports turned that into a free point on
    every question carrying an `amount` or an `event`. Over 332 recorded analyses it was offered
    79 times, 22 of them on straight-line distance questions it computes nothing for.
    """

    from src.agent.retrieval import retrieve_macro_templates

    distance_question = [
        {"id": "anchor", "core_concept": "location", "functional_role": "extent"},
        {"id": "candidates", "core_concept": "object", "functional_role": "support"},
        {"id": "farthest", "core_concept": "amount", "functional_role": "measure"},
    ]

    offered = [t.name for t in retrieve_macro_templates(distance_question, [], limit=2)]

    assert "TIME-WINDOW-REVERSE" not in offered


def test_a_templates_own_concepts_decide_its_rank() -> None:
    """The ranking term is the overlap with what the fragment computes over."""

    from src.agent.retrieval import retrieve_macro_templates

    clock_question = [
        {"id": "route", "core_concept": "field", "functional_role": "support"},
        {"id": "window", "core_concept": "event", "functional_role": "temporal_extent"},
        {"id": "departure", "core_concept": "event", "functional_role": "measure"},
    ]

    assert "TIME-WINDOW-REVERSE" in [
        t.name for t in retrieve_macro_templates(clock_question, [], limit=2)
    ]


def test_every_template_can_be_offered_for_the_shape_it_describes() -> None:
    """A fragment no concept set can retrieve is a fragment the planner never sees.

    `ROUTE-STEP-EXTRACT` was exactly that: on a turn-count concept set it tied
    `MULTI-ROUTE-COMPARE` and `ROUTE-OPTIMIZE` and lost both slots to the alphabetical
    tie-break, for the 28 rows of `routing_nth_turn` and `routing_turn_count_via` whose shape it
    is.
    """

    from src.agent.retrieval import retrieve_macro_templates

    reachable = set()
    for template in MACRO_TEMPLATES.values():
        concepts = [
            {
                "id": node.id,
                "core_concept": node.core_concept,
                "functional_role": node.functional_role,
            }
            for node in template.concept_nodes
        ]
        reachable.update(t.name for t in retrieve_macro_templates(concepts, [], limit=2))

    assert {template.name for template in MACRO_TEMPLATES.values()} <= reachable
