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
