"""Macro-template retrieval is a construction prior, never a validity constraint."""

from __future__ import annotations

from src.agent.geoflow import TEMPLATES
from src.agent.templates import MACRO_TEMPLATES


def test_no_legacy_template_declares_required_structure() -> None:
    declaring = {template["name"] for template in TEMPLATES.values() if template.get("requires")}

    assert declaring == set()


def test_macro_templates_expose_ports_without_required_structure() -> None:
    for template in MACRO_TEMPLATES.values():
        assert template.input_ports
        assert template.output_ports
        assert "requires" not in template.as_dict()
