from __future__ import annotations

import pytest

from src.agent.answering import GroundedAnswer
from src.agent.composition import compose_templates
from src.agent.concepts import ConceptNode, FactorNode, GeoFlowGraph, TransformationEdge
from src.agent.factorization import attach_grounding_factors, factorize_plan, plan_to_geoflow
from src.agent.geoflow import OPERATOR_CONTRACTS
from src.agent.retrieval import QuestionGraphExample, QuestionGraphExampleStore
from src.agent.spatial import GroundingFacts
from src.agent.templates import MACRO_TEMPLATES
from src.agent.validation import G1G5ValidationError, validate_geoflow_graph
from src.mcq_adapter import MCQAdapter


def test_concepts_factors_and_transformations_are_distinct_ir_members() -> None:
    graph = GeoFlowGraph(
        (
            ConceptNode("anchor", "location", "extent", {}, "서울역"),
            ConceptNode("places", "object", "support", {}, "약국"),
            ConceptNode("answer", "amount", "measure", {}, "count"),
        ),
        (
            TransformationEdge("search", "PLACE_SEARCH", ("anchor",), ("places",)),
            TransformationEdge("count", "AGGREGATE", ("places",), ("answer",), ("aggregate",)),
        ),
        (FactorNode("aggregate", "aggregate", "count"),),
    )

    encoded = graph.as_dict()

    assert set(encoded) == {
        "concept_nodes",
        "factor_nodes",
        "transformation_edges",
        "metadata",
    }
    assert encoded["concept_nodes"][0]["core_concept"] == "location"
    assert encoded["concept_nodes"][0]["functional_role"] == "extent"
    assert encoded["transformation_edges"][1]["factor_nodes"] == ["aggregate"]


def test_catalogue_contains_only_reusable_port_typed_macros() -> None:
    assert {template.name for template in MACRO_TEMPLATES.values()} == {
        "FILTER-AGGREGATE-MEASURE",
        "OBJECT-FIELD-MEASURE",
        "ROUTE-OPTIMIZE",
        "MULTI-ROUTE-COMPARE",
        "MULTI-SEGMENT-AGGREGATE",
    }
    assert all(
        template.input_ports and template.output_ports for template in MACRO_TEMPLATES.values()
    )
    assert all("radius" not in key and "ordinal" not in key for key in MACRO_TEMPLATES)


def test_template_composition_rejects_type_incompatible_port_binding() -> None:
    first = MACRO_TEMPLATES["filter_aggregate_measure"]
    second = MACRO_TEMPLATES["route_optimize"]
    prior = compose_templates([first])
    measure = next(node for node in prior.concept_nodes if node.functional_role == "measure")

    with pytest.raises(ValueError, match="Type-incompatible"):
        compose_templates(
            [first, second],
            bindings={(second.name, "stops"): measure.id},
        )


def test_strict_g1_g5_validation_has_no_template_requirements() -> None:
    graph = GeoFlowGraph(
        (
            ConceptNode("anchor", "location", "extent"),
            ConceptNode("answer", "object", "measure"),
        ),
        (TransformationEdge("measure", "MEASURE", ("anchor",), ("answer",)),),
    )

    result = validate_geoflow_graph(graph)

    assert all(result.constraints.values())


def test_strict_validation_refuses_a_cycle() -> None:
    graph = GeoFlowGraph(
        (
            ConceptNode("a", "object", "extent"),
            ConceptNode("b", "object", "support"),
            ConceptNode("answer", "object", "measure"),
        ),
        (
            TransformationEdge("one", "FILTER", ("b",), ("a",)),
            TransformationEdge("two", "FILTER", ("a",), ("b",)),
            TransformationEdge("three", "MEASURE", ("b",), ("answer",)),
        ),
    )

    with pytest.raises(G1G5ValidationError, match="G1"):
        validate_geoflow_graph(graph)


def test_strict_validation_refuses_type_incompatible_inputs() -> None:
    graph = GeoFlowGraph(
        (
            ConceptNode("event", "event", "extent"),
            ConceptNode("answer", "field", "measure"),
        ),
        (TransformationEdge("route", "ROUTE_MEASURE", ("event",), ("answer",)),),
    )

    with pytest.raises(G1G5ValidationError, match="G3.*cannot consume event"):
        validate_geoflow_graph(graph)


def test_question_graph_retrieval_is_cosine_and_excludes_the_test_question() -> None:
    graph = compose_templates([MACRO_TEMPLATES["object_field_measure"]])
    store = QuestionGraphExampleStore(
        [
            QuestionGraphExample("leak", "평가 질문", graph),
            QuestionGraphExample("near", "경로 비교 예시", graph),
            QuestionGraphExample("far", "객체 속성 예시", graph),
        ]
    )

    def embed(texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0] if text in {texts[0], "경로 비교 예시"} else [0.0, 1.0] for text in texts
        ]

    found = store.retrieve("평가 질문", embed=embed, limit=2)

    assert [example.example_id for example in found] == ["near", "far"]
    assert all(example.example_id != "leak" for example in found)


def test_example_store_refuses_an_unvalidated_graph() -> None:
    invalid = GeoFlowGraph(
        (ConceptNode("orphan", "object", "support"),),
        (TransformationEdge("orphan", "FILTER", (), ("orphan",)),),
    )

    with pytest.raises(G1G5ValidationError):
        QuestionGraphExampleStore([QuestionGraphExample("bad", "bad", invalid)])


def test_typed_question_facts_become_factor_nodes_on_transformations() -> None:
    analysis = {
        "concepts": [
            {"id": "anchor", "core_concept": "location", "functional_role": "extent"},
            {"id": "places", "core_concept": "object", "functional_role": "support"},
            {"id": "answer", "core_concept": "amount", "functional_role": "measure"},
        ]
    }
    plan = {
        "graph": [
            {
                "id": "resolve",
                "transform": "RESOLVE_PLACES",
                "concept_ids": ["anchor"],
                "inputs": [],
            },
            {
                "id": "search",
                "transform": "PLACE_SEARCH",
                "concept_ids": ["places"],
                "inputs": ["resolve"],
            },
            {"id": "inside", "transform": "FILTER", "inputs": ["resolve", "search"]},
            {
                "id": "count",
                "transform": "AGGREGATE",
                "inputs": ["inside"],
                "concept_ids": ["answer"],
                "factors": {"aggregate": "count"},
            },
        ]
    }

    graph = attach_grounding_factors(plan_to_geoflow(analysis, plan), GroundingFacts(radius_m=600))
    inside = next(edge for edge in graph.transformation_edges if edge.id == "inside")

    assert any(factor.factor_type == "radius_m" for factor in graph.factor_nodes)
    assert any(
        graph.factors_by_id[value].factor_type == "radius_m" for value in inside.factor_nodes
    )

    factorized = factorize_plan(
        analysis,
        graph.as_dict(),
        options=[],
        facts=GroundingFacts(radius_m=600),
        available=frozenset(OPERATOR_CONTRACTS),
    )
    hyperedge = next(
        edge for edge in factorized.operator_hyperedges if edge["operator_id"] == "inside"
    )
    assert hyperedge["input_concepts"]
    assert any(
        graph.factors_by_id[value].factor_type == "radius_m" for value in hyperedge["factor_nodes"]
    )


def test_implicit_network_and_field_concepts_are_completed() -> None:
    graph = GeoFlowGraph(
        (
            ConceptNode("places", "object", "extent"),
            ConceptNode("answer", "field", "measure"),
        ),
        (TransformationEdge("route", "ROUTE_MEASURE", ("places",), ("answer",)),),
    ).with_implicit_concepts()

    assert {node.core_concept for node in graph.concept_nodes if node.implicit} == {
        "network",
        "field",
    }
    route = graph.transformation_edges[0]
    assert "implicit_network" in route.input_concepts
    assert "implicit_route" in route.output_concepts


def test_mcq_adapter_is_outside_grounded_answer_generation() -> None:
    answer = GroundedAnswer(value="경복궁", text="경복궁", confidence=0.9)

    selected = MCQAdapter().select(answer, ["창덕궁", "경복궁"])

    assert selected.index == 1
    assert selected.method == "exact_grounded_text"


def test_planner_catalogue_excludes_mcq_matching() -> None:
    from src.agent.semantics import transform_catalogue

    assert "MATCH_OPTIONS" not in transform_catalogue(include_mcq=False)


def test_spatial_core_apis_accept_neither_intent_nor_mcq_options() -> None:
    import inspect

    from src.agent.geoflow import normalize_analysis
    from src.agent.spatial import ANALYSIS_PROMPT, _factorize_validate_plan

    assert "intent" not in ANALYSIS_PROMPT.casefold()
    assert "options" not in inspect.signature(normalize_analysis).parameters
    assert "fallback_intent" not in inspect.signature(normalize_analysis).parameters
    assert "options" not in inspect.signature(_factorize_validate_plan).parameters
