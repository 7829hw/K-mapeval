"""What the planner may get wrong about its own reply, and what is still its failure.

Measured motivation: on one hundred v7a questions at `876c772`, 18 drafts passed G1-G5 and 82
did not, and the errors were overwhelmingly references -- concepts, factors and edges the graph
named without declaring. The step-shaped wire format never had that failure mode because
`plan_to_geoflow` completed it; the Concept/Edge IR path skipped the completion entirely. These
tests pin the completion, and pin the line it does not cross.
"""

from __future__ import annotations

import pytest

from src.agent.factorization import plan_to_geoflow
from src.agent.validation import validate_geoflow_graph

ANALYSIS = {
    "concepts": [
        {
            "id": "anchor",
            "text": "스테이락호텔",
            "core_concept": "location",
            "functional_role": "extent",
        },
        {
            "id": "candidates",
            "text": "정형외과",
            "core_concept": "object",
            "functional_role": "condition",
            "attributes": {"ordinal": 2},
        },
    ]
}


def _valid(payload: dict) -> object:
    graph = plan_to_geoflow(ANALYSIS, payload)
    validate_geoflow_graph(graph)
    return graph


def test_edges_may_refer_to_the_analysis_stages_concepts_without_restating_them() -> None:
    """The dominant failure. Asking for the concepts twice is asking for them to be copied."""

    graph = _valid(
        {
            "transformation_edges": [
                {
                    "id": "t1",
                    "transformation": "RESOLVE_PLACES",
                    "input_concepts": ["anchor"],
                    "output_concepts": ["anchor"],
                },
                {
                    "id": "t2",
                    "transformation": "DISTANCE_MEASURE",
                    "input_concepts": ["anchor", "candidates"],
                    "output_concepts": ["separations"],
                },
                {
                    "id": "t3",
                    "transformation": "ORDINAL_SELECT",
                    "input_concepts": ["separations"],
                    "output_concepts": ["answer"],
                },
            ]
        }
    )
    concepts = graph.concepts_by_id
    # The Analysis stage's typing survives: completion fills gaps, it does not re-type.
    assert concepts["anchor"].core_concept == "location"
    assert concepts["anchor"].functional_role == "extent"
    assert concepts["candidates"].core_concept == "object"
    # The ordinal the Analysis stage recorded as an attribute is an explicit factor vertex.
    assert any(factor.factor_type == "ordinal" for factor in graph.factor_nodes)


def test_an_undeclared_output_is_typed_by_the_transformation_that_produces_it() -> None:
    graph = _valid(
        {
            "transformation_edges": [
                {
                    "id": "t1",
                    "transformation": "RESOLVE_PLACES",
                    "input_concepts": ["anchor"],
                    "output_concepts": ["L1"],
                },
                {
                    "id": "t2",
                    "transformation": "DISTANCE_MEASURE",
                    "input_concepts": ["L1", "candidates"],
                    "output_concepts": ["M1"],
                },
            ]
        }
    )
    synthesized = graph.concepts_by_id["L1"]
    assert synthesized.core_concept == "object"
    assert synthesized.implicit is True


def test_a_produced_concept_is_retyped_to_what_its_transformation_produces() -> None:
    """`ROUTE_MEASURE` produces a route field. A planner calling the result an AMOUNT is
    guessing at a type, and the vocabulary is the authority on it."""

    graph = _valid(
        {
            "concept_nodes": [
                {"id": "a", "core_concept": "location", "functional_role": "extent"},
                {"id": "b", "core_concept": "location", "functional_role": "condition"},
                {"id": "total", "core_concept": "amount", "functional_role": "measure"},
            ],
            "transformation_edges": [
                {
                    "id": "t1",
                    "transformation": "ROUTE_MEASURE",
                    "input_concepts": ["a", "b"],
                    "output_concepts": ["total"],
                }
            ],
        }
    )
    total = graph.concepts_by_id["total"]
    assert total.core_concept == "field"
    assert total.attributes["type_completed_from"] == "amount"


def test_a_redefined_concept_is_read_by_later_edges_as_the_later_value() -> None:
    """Two producers is a G4 refusal, and neither claim is worth discarding."""

    graph = _valid(
        {
            "transformation_edges": [
                {
                    "id": "t1",
                    "transformation": "RESOLVE_PLACES",
                    "input_concepts": ["anchor"],
                    "output_concepts": ["places"],
                },
                {
                    "id": "t2",
                    "transformation": "PLACE_SEARCH",
                    "input_concepts": ["places"],
                    "output_concepts": ["places"],
                },
                {
                    "id": "t3",
                    "transformation": "DISTANCE_MEASURE",
                    "input_concepts": ["places"],
                    "output_concepts": ["m"],
                },
            ]
        }
    )
    edges = {edge.id: edge for edge in graph.transformation_edges}
    assert edges["t2"].output_concepts == ("places__t2",)
    # The reader written after the redefinition sees the redefinition, so both edges are on the
    # path to the measure rather than one of them dangling.
    assert "places__t2" in edges["t3"].input_concepts


def test_an_edge_may_be_wired_to_another_edge_rather_than_to_its_concept() -> None:
    graph = _valid(
        {
            "transformation_edges": [
                {
                    "id": "t1",
                    "transformation": "RESOLVE_PLACES",
                    "input_concepts": ["anchor"],
                    "output_concepts": ["p"],
                },
                {
                    "id": "t2",
                    "transformation": "DISTANCE_MEASURE",
                    "input_concepts": ["$t1", "candidates"],
                    "output_concepts": ["d"],
                },
            ]
        }
    )
    edges = {edge.id: edge for edge in graph.transformation_edges}
    assert "p" in edges["t2"].input_concepts


def test_a_factor_written_inline_becomes_a_factor_vertex() -> None:
    graph = _valid(
        {
            "transformation_edges": [
                {
                    "id": "t1",
                    "transformation": "RESOLVE_PLACES",
                    "input_concepts": ["anchor"],
                    "output_concepts": ["p"],
                },
                {
                    "id": "t2",
                    "transformation": "SORT",
                    "input_concepts": ["p"],
                    "factor_nodes": [
                        {"id": "sort_metric", "factor_type": "metric", "value": "distance"}
                    ],
                    "output_concepts": ["ranked"],
                },
            ]
        }
    )
    assert graph.factors_by_id["sort_metric"].value == "distance"
    edges = {edge.id: edge for edge in graph.transformation_edges}
    assert "sort_metric" in edges["t2"].factor_nodes


def test_a_graph_with_no_contextual_root_is_rooted_in_what_the_question_gave() -> None:
    """G5 wants the graph rooted in context. A producer-less input is exactly that, whatever
    role the Analysis stage happened to put on it."""

    graph = _valid(
        {
            "concept_nodes": [
                {"id": "a", "core_concept": "location", "functional_role": "condition"},
                {"id": "b", "core_concept": "object", "functional_role": "support"},
            ],
            "transformation_edges": [
                {
                    "id": "t1",
                    "transformation": "RESOLVE_PLACES",
                    "input_concepts": ["a"],
                    "output_concepts": ["b"],
                },
                {
                    "id": "t2",
                    "transformation": "AGGREGATE",
                    "input_concepts": ["b"],
                    "output_concepts": ["n"],
                },
            ],
        }
    )
    assert graph.concepts_by_id["a"].functional_role == "extent"
    assert graph.concepts_by_id["a"].attributes["role_completed_from"] == "condition"


def test_a_graph_that_computes_no_measure_answers_with_its_sink() -> None:
    graph = _valid(
        {
            "concept_nodes": [
                {"id": "a", "core_concept": "location", "functional_role": "extent"},
                {"id": "b", "core_concept": "object", "functional_role": "support"},
            ],
            "transformation_edges": [
                {
                    "id": "t1",
                    "transformation": "RESOLVE_PLACES",
                    "input_concepts": ["a"],
                    "output_concepts": ["b"],
                },
                {
                    "id": "t2",
                    "transformation": "AGGREGATE",
                    "input_concepts": ["b"],
                    "output_concepts": ["n"],
                },
            ],
        }
    )
    assert graph.concepts_by_id["n"].functional_role == "measure"


def test_match_options_is_dropped_and_the_measure_it_named_is_kept() -> None:
    """MCQ matching left the reasoning core with `MCQAdapter`; a planner still writing it is
    naming its measure."""

    graph = _valid(
        {
            "transformation_edges": [
                {
                    "id": "t1",
                    "transformation": "RESOLVE_PLACES",
                    "input_concepts": ["anchor"],
                    "output_concepts": ["p"],
                },
                {
                    "id": "t2",
                    "transformation": "DISTANCE_MEASURE",
                    "input_concepts": ["p", "candidates"],
                    "output_concepts": ["d"],
                },
                {
                    "id": "t3",
                    "transformation": "MATCH_OPTIONS",
                    "input_concepts": ["d"],
                    "output_concepts": ["pick"],
                },
            ]
        }
    )
    assert "MATCH_OPTIONS" not in {
        edge.transformation.upper() for edge in graph.transformation_edges
    }
    assert graph.concepts_by_id["d"].functional_role == "measure"


def test_canonicalization_leaves_a_graph_that_already_passes_alone() -> None:
    """The completion is additive and last-resort, so it cannot move a draft that was already
    valid -- which is what makes it safe to apply to every draft rather than to failures."""

    payload = {
        "concept_nodes": [
            {"id": "a", "core_concept": "location", "functional_role": "extent"},
            {"id": "b", "core_concept": "object", "functional_role": "support"},
            {"id": "m", "core_concept": "amount", "functional_role": "measure"},
        ],
        "transformation_edges": [
            {
                "id": "t1",
                "transformation": "RESOLVE_PLACES",
                "input_concepts": ["a"],
                "output_concepts": ["b"],
            },
            {
                "id": "t2",
                "transformation": "AGGREGATE",
                "input_concepts": ["b"],
                "output_concepts": ["m"],
            },
        ],
    }
    graph = _valid(payload)
    for declared in payload["concept_nodes"]:
        node = graph.concepts_by_id[declared["id"]]
        assert node.core_concept == declared["core_concept"]
        assert node.functional_role == declared["functional_role"]
        assert "role_completed_from" not in node.attributes
        assert "type_completed_from" not in node.attributes
    for declared in payload["transformation_edges"]:
        edge = next(item for item in graph.transformation_edges if item.id == declared["id"])
        assert list(edge.input_concepts) == declared["input_concepts"]
        assert list(edge.output_concepts) == declared["output_concepts"]


def test_a_transformation_the_vocabulary_does_not_have_is_still_the_planners_failure() -> None:
    """Completion fills in references. It never invents a transformation, and a graph whose
    structure is wrong has to fail loudly rather than be laundered into a plausible one."""

    with pytest.raises(ValueError, match="unknown spatial transformation"):
        _valid(
            {
                "transformation_edges": [
                    {
                        "id": "t1",
                        "transformation": "TELEPORT",
                        "input_concepts": ["anchor"],
                        "output_concepts": ["p"],
                    }
                ]
            }
        )


def test_a_reply_with_no_edges_is_refused() -> None:
    with pytest.raises(ValueError, match="does not contain a non-empty graph"):
        plan_to_geoflow(ANALYSIS, {"transformation_edges": []})


def test_a_concept_resolved_in_place_is_still_the_questions_own_data() -> None:
    """`RESOLVE_PLACES` over `target1` into `target1` makes the concept its own producer, so
    nothing in the graph supplies it and G4 reads its whole branch as unavailable -- unless the
    branch happens to be the one the Analysis stage labelled `extent`. 43 refusals in one run
    were that, and it is the shape every multi-place question draws."""

    graph = _valid(
        {
            "concept_nodes": [
                {"id": "anchor", "core_concept": "location", "functional_role": "extent"},
                {"id": "target1", "core_concept": "location", "functional_role": "support"},
                {"id": "d", "core_concept": "amount", "functional_role": "measure"},
            ],
            "transformation_edges": [
                {
                    "id": "t1",
                    "transformation": "RESOLVE_PLACES",
                    "input_concepts": ["anchor"],
                    "output_concepts": ["anchor"],
                },
                {
                    "id": "t2",
                    "transformation": "RESOLVE_PLACES",
                    "input_concepts": ["target1"],
                    "output_concepts": ["target1"],
                },
                {
                    "id": "t3",
                    "transformation": "DISTANCE_MEASURE",
                    "input_concepts": ["anchor", "target1"],
                    "output_concepts": ["d"],
                },
            ],
        }
    )
    # Kept under its own id: the executor's references are built on these, and renaming it
    # broke `$node.place` resolution on 28 graphs when that was tried.
    assert graph.concepts_by_id["target1"].functional_role == "extent"
    edges = {edge.id: edge for edge in graph.transformation_edges}
    assert edges["t2"].output_concepts == ("target1",)


def test_the_implicit_route_is_not_handed_to_transformations_that_produce_one() -> None:
    """`with_implicit_concepts` completing a graph the validator then refuses is this port
    contradicting itself; `ROUTE_MEASURE` takes a place and produces a route, it does not read
    one. 36 questions in one hundred died of exactly that."""

    graph = _valid(
        {
            "concept_nodes": [
                {"id": "a", "core_concept": "location", "functional_role": "extent"},
                {"id": "b", "core_concept": "location", "functional_role": "support"},
                {"id": "total", "core_concept": "amount", "functional_role": "measure"},
            ],
            "transformation_edges": [
                {
                    "id": "t1",
                    "transformation": "ROUTE_MEASURE",
                    "input_concepts": ["a", "b"],
                    "output_concepts": ["leg1"],
                },
                {
                    "id": "t2",
                    "transformation": "ROUTE_MEASURE",
                    "input_concepts": ["b", "a"],
                    "output_concepts": ["leg2"],
                },
                {
                    "id": "t3",
                    "transformation": "ROUTE_EXTRACT",
                    "input_concepts": ["leg1", "leg2"],
                    "output_concepts": ["total"],
                },
            ],
        }
    )
    edges = {edge.id: edge for edge in graph.transformation_edges}
    assert "implicit_route" not in edges["t2"].input_concepts
    # The reader that does consume a route field still gets it.
    assert "implicit_route" in edges["t3"].input_concepts


@pytest.mark.parametrize(
    ("transformation", "core_concept"),
    [
        # Each pair is a type some transformation in this vocabulary actually emits, reaching a
        # consumer whose operator would have run on it. `AGENTS.md`: a plan the executor could
        # have run must never be refused by a declared table.
        ("ROUTE_OPTIMIZE", "location"),
        ("ROUTE_STEPS", "object"),
        ("ROUTE_STEPS", "network"),
        ("FILTER", "amount"),
        ("AGGREGATE", "location"),
    ],
)
def test_the_input_table_accepts_what_the_vocabulary_can_produce(
    transformation: str, core_concept: str
) -> None:
    from src.agent.semantics import accepted_input_types

    accepted = accepted_input_types(transformation)
    assert accepted is not None
    assert core_concept in accepted


def test_the_unconsumed_input_rule_refuses_a_draft_but_not_the_question() -> None:
    """It predicts that one step would read less evidence than the graph handed it. Worth
    refusing a draft for, so the repair round is told; not worth refusing the question for --
    the step is one the executor would have run, and this was the single largest cause of
    `graph_validation_failure` in a run at 28 of 100.
    """

    from src.agent.factorization import factorize_plan
    from src.agent.geoflow import OPERATOR_CONTRACTS
    from src.agent.spatial import extract_facts

    payload = {
        "concept_nodes": [
            {"id": "a", "core_concept": "location", "functional_role": "extent"},
            {"id": "b", "core_concept": "location", "functional_role": "support"},
            {"id": "direct", "core_concept": "field", "functional_role": "support"},
            {"id": "matrix", "core_concept": "field", "functional_role": "support"},
            {"id": "total", "core_concept": "amount", "functional_role": "measure"},
        ],
        "transformation_edges": [
            {
                "id": "t1",
                "transformation": "ROUTE_MEASURE",
                "input_concepts": ["a", "b"],
                "output_concepts": ["direct"],
            },
            {
                "id": "t2",
                "transformation": "ROUTE_MATRIX",
                "input_concepts": ["a", "b"],
                "output_concepts": ["matrix"],
            },
            {
                "id": "t3",
                "transformation": "ROUTE_EXTRACT",
                "input_concepts": ["direct", "matrix"],
                "output_concepts": ["total"],
            },
        ],
    }
    question = "A에서 B까지 가는 경로는 얼마나 더 깁니까?"
    facts = extract_facts({}, question)
    available = frozenset(OPERATOR_CONTRACTS)

    with pytest.raises(ValueError, match="would be gathered and never used"):
        factorize_plan(
            {}, payload, options=[], facts=facts, available=available, strict_types=True
        )

    relaxed = factorize_plan(
        {}, payload, options=[], facts=facts, available=available, strict_types=False
    )
    assert any(row.get("rule") == "unconsumed_inputs" for row in relaxed.semantic.diagnostics)


def test_a_route_says_what_it_passes_through_however_the_planner_spells_it() -> None:
    """`via` is the only thing allowed to say a route has a waypoint -- inferring one from a
    middle input would route "A와 B 중 C에 더 가까운 곳" through the answer. A planner writes it
    beside the transformation as often as inside `attributes`, and reading only the second
    spelling lost every waypoint: `routing_turn_count_via` counted the turns of a route that
    skipped the stop it was asked about.
    """

    payload = {
        "concept_nodes": [
            {"id": "a", "core_concept": "location", "functional_role": "extent"},
            {"id": "b", "core_concept": "location", "functional_role": "support"},
            {"id": "c", "core_concept": "location", "functional_role": "support"},
            {"id": "route", "core_concept": "field", "functional_role": "measure"},
        ],
        "transformation_edges": [
            {
                "id": "t1",
                "transformation": "ROUTE_MEASURE",
                "input_concepts": ["a", "c"],
                "via": ["b"],
                "output_concepts": ["route"],
            }
        ],
    }
    graph = _valid(payload)
    edge = next(item for item in graph.transformation_edges if item.id == "t1")
    assert edge.attributes["via"] == ["b"]


def test_a_measure_whose_places_nothing_resolved_gets_them_geocoded() -> None:
    """A graph that writes ROUTE_MEASURE straight over the Analysis concepts has named its
    places without resolving them, and `directions` then arrives with no origin and no
    destination. On a held-out draw that was 31 of 100 questions, across four operators saying
    it four different ways."""

    graph = _valid(
        {
            "concept_nodes": [
                {"id": "from", "core_concept": "location", "functional_role": "extent"},
                {"id": "to", "core_concept": "location", "functional_role": "support"},
                {"id": "route", "core_concept": "field", "functional_role": "measure"},
            ],
            "transformation_edges": [
                {
                    "id": "t1",
                    "transformation": "ROUTE_MEASURE",
                    "input_concepts": ["from", "to"],
                    "output_concepts": ["route"],
                }
            ],
        }
    )
    resolving = [
        edge
        for edge in graph.transformation_edges
        if edge.transformation.upper() == "RESOLVE_PLACES"
    ]
    assert len(resolving) == 1
    assert set(resolving[0].output_concepts) == {"from", "to"}


def test_a_kind_of_place_is_not_geocoded_because_a_search_reads_it() -> None:
    """`정형외과` is what to look for, not somewhere to look. Only transformations that need a
    located place ask for one."""

    graph = _valid(
        {
            "transformation_edges": [
                {
                    "id": "t1",
                    "transformation": "RESOLVE_PLACES",
                    "input_concepts": ["anchor"],
                    "output_concepts": ["anchor"],
                },
                {
                    "id": "t2",
                    "transformation": "PLACE_SEARCH",
                    "input_concepts": ["anchor", "candidates"],
                    "output_concepts": ["found"],
                },
                {
                    "id": "t3",
                    "transformation": "ORDINAL_SELECT",
                    "input_concepts": ["found"],
                    "output_concepts": ["answer"],
                },
            ]
        }
    )
    assert [edge.id for edge in graph.transformation_edges] == ["t1", "t2", "t3"]


def test_a_radius_narrowing_reads_what_the_graph_measured() -> None:
    """"X에서 반경 300m 이내에 있는 은행은 아래 목록 중 몇 곳인가요" is drawn two ways and both
    strand a branch: one measures each candidate from the anchor and then filters the
    *candidates*, so every distance it computed is read by nothing; the other filters the listed
    places with no anchor in reach. `filter_by_distance` says which is which -- the radius is a
    stated fact grounding binds, and the measured set is what the measure steps produced.
    """

    analysis = {
        "concepts": [
            {
                "id": "anchor",
                "text": "남예종예술실용전문학교 아트홀",
                "core_concept": "location",
                "functional_role": "extent",
                "attributes": {"radius_m": 300},
            },
            {"id": "c1", "text": "A", "core_concept": "location", "functional_role": "support"},
            {"id": "c2", "text": "B", "core_concept": "location", "functional_role": "support"},
        ]
    }
    payload = {
        "transformation_edges": [
            {
                "id": "t1",
                "transformation": "RESOLVE_PLACES",
                "input_concepts": ["anchor"],
                "output_concepts": ["anchor_located"],
            },
            {
                "id": "t2",
                "transformation": "DISTANCE_MEASURE",
                "input_concepts": ["anchor_located", "c1"],
                "output_concepts": ["d1"],
            },
            {
                "id": "t3",
                "transformation": "FILTER",
                "input_concepts": ["c1", "c2"],
                "output_concepts": ["kept"],
            },
            {
                "id": "t4",
                "transformation": "AGGREGATE",
                "input_concepts": ["kept"],
                "output_concepts": ["count"],
            },
        ]
    }
    graph = plan_to_geoflow(analysis, payload)
    validate_geoflow_graph(graph)
    narrowing = next(item for item in graph.transformation_edges if item.id == "t3")
    # The measurement the graph computed and left unread is what the narrowing narrows, so the
    # branch that produced it reaches the measure instead of dangling.
    assert "d1" in narrowing.input_concepts


def test_two_routes_between_the_same_ends_differ_by_what_one_passes_through() -> None:
    """A detour question measures the drive twice and the graph says so: one route reads the two
    places, the other reads those two and one more. The difference between them is the waypoint,
    and reading it that way needs no position -- one planner writes the extra input last and
    another writes it in the middle. The general "middles are waypoints" rule is forbidden and
    stays forbidden: there is no second route beside "A와 B 중 C에 더 가까운 곳".
    """

    payload = {
        "concept_nodes": [
            {"id": "a", "core_concept": "location", "functional_role": "extent"},
            {"id": "b", "core_concept": "location", "functional_role": "support"},
            {"id": "w", "core_concept": "location", "functional_role": "support"},
            {"id": "direct", "core_concept": "field", "functional_role": "support"},
            {"id": "detour", "core_concept": "field", "functional_role": "support"},
            {"id": "cost", "core_concept": "amount", "functional_role": "measure"},
        ],
        "transformation_edges": [
            {
                "id": "t1",
                "transformation": "ROUTE_MEASURE",
                "input_concepts": ["a", "b"],
                "output_concepts": ["direct"],
            },
            {
                # The stop written between the ends, which is where a positional rule got it
                # backwards: it routed to the stop and passed through the destination.
                "id": "t2",
                "transformation": "ROUTE_MEASURE",
                "input_concepts": ["a", "b", "w"],
                "output_concepts": ["detour"],
            },
            {
                "id": "t3",
                "transformation": "AGGREGATE",
                "input_concepts": ["direct", "detour"],
                "output_concepts": ["cost"],
            },
        ],
    }
    graph = _valid(payload)
    edges = {edge.id: edge for edge in graph.transformation_edges}
    assert edges["t2"].attributes["via"] == ["w"]
    assert not edges["t1"].attributes.get("via")
    # Kept among the inputs: taking it out left its resolve node contributing to nothing and G5
    # refused the graph.
    assert "w" in edges["t2"].input_concepts


def test_a_single_route_with_three_places_is_not_given_a_waypoint() -> None:
    """No second route to compare against, so nothing in the graph says which place is the
    stop -- and guessing is the rule `AGENTS.md` forbids."""

    graph = _valid(
        {
            "concept_nodes": [
                {"id": "a", "core_concept": "location", "functional_role": "extent"},
                {"id": "b", "core_concept": "location", "functional_role": "support"},
                {"id": "c", "core_concept": "location", "functional_role": "support"},
                {"id": "route", "core_concept": "field", "functional_role": "measure"},
            ],
            "transformation_edges": [
                {
                    "id": "t1",
                    "transformation": "ROUTE_MEASURE",
                    "input_concepts": ["a", "b", "c"],
                    "output_concepts": ["route"],
                }
            ],
        }
    )
    edge = next(item for item in graph.transformation_edges if item.id == "t1")
    assert not edge.attributes.get("via")


def test_a_lone_route_is_connected_to_the_stated_stop_before_g5_reads_it() -> None:
    """Grounding binds the stop too, but grounding runs after validation and these graphs do not
    survive that long: the planner resolves the stop in its own node and measures a route that
    ignores it, so the resolve node contributes to nothing and G5 refuses the graph.
    """

    from src.agent.factorization import attach_grounding_factors
    from src.agent.spatial import extract_facts

    analysis = {
        "concepts": [
            {"id": "a", "text": "변우석1호숲", "core_concept": "location", "role": "extent"},
            {"id": "w", "text": "모여집", "core_concept": "location", "role": "support"},
            {"id": "b", "text": "충무아트센터", "core_concept": "location", "role": "support"},
        ]
    }
    payload = {
        "transformation_edges": [
            {
                "id": "t1",
                "transformation": "RESOLVE_PLACES",
                "input_concepts": ["a", "w", "b"],
                "output_concepts": ["a", "w", "b"],
            },
            {
                "id": "t2",
                "transformation": "ROUTE_MEASURE",
                "input_concepts": ["a", "b"],
                "output_concepts": ["route"],
            },
        ]
    }
    question = "변우석1호숲에서 모여집을 들러 충무아트센터까지 자동차로 이동합니다."
    graph = attach_grounding_factors(
        plan_to_geoflow(analysis, payload), extract_facts(analysis, question)
    )
    route = next(edge for edge in graph.transformation_edges if edge.id == "t2")
    assert route.attributes["via"] == ["w"]
    # In the inputs as well, because G5 reads connectivity off the concept graph and `via` lives
    # in attributes.
    assert "w" in route.input_concepts


def test_a_detour_graph_is_not_filled_from_the_question() -> None:
    """Two routes and only one goes through the stop; which one is the graph's own structure to
    say."""

    from src.agent.factorization import attach_grounding_factors
    from src.agent.spatial import extract_facts

    analysis = {
        "concepts": [
            {"id": "a", "text": "A모텔", "core_concept": "location", "role": "extent"},
            {"id": "w", "text": "B갤러리", "core_concept": "location", "role": "support"},
            {"id": "b", "text": "C역", "core_concept": "location", "role": "support"},
        ]
    }
    payload = {
        "transformation_edges": [
            {
                "id": "t1",
                "transformation": "ROUTE_MEASURE",
                "input_concepts": ["a", "b"],
                "output_concepts": ["direct"],
            },
            {
                "id": "t2",
                "transformation": "ROUTE_MEASURE",
                "input_concepts": ["a", "b", "w"],
                "output_concepts": ["detour"],
            },
            {
                "id": "t3",
                "transformation": "AGGREGATE",
                "input_concepts": ["direct", "detour"],
                "output_concepts": ["cost"],
            },
        ]
    }
    question = "A모텔에서 C역까지, 곧장 가는 경우와 B갤러리를 경유해서 가는 경우의 차이는?"
    graph = attach_grounding_factors(
        plan_to_geoflow(analysis, payload), extract_facts(analysis, question)
    )
    edges = {edge.id: edge for edge in graph.transformation_edges}
    # The subset rule already said which route has the stop; nothing here overrode it.
    assert edges["t2"].attributes["via"] == ["w"]
    assert not edges["t1"].attributes.get("via")
