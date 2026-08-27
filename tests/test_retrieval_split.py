"""Template retrieval and example retrieval answer different questions.

They had been answering them with one score and returning one payload. *Which macro-template* a
question needs is a structural fact about its concept graph -- a measure over a network, a field
narrowed by a sub-condition -- and is read off the concepts and roles. *Which worked example*
resembles this question is a similarity judgement over prose, which is what an embedding is for.

The embedder is a seam rather than a dependency. The deployment these benchmarks run against
serves one chat model and answers `/v1/embeddings` with HTTP 404, so `None` -- the deterministic
concept-overlap scorer -- is what actually runs, and the split is structural rather than a
behaviour change until an embedding service exists.
"""

from __future__ import annotations

import math

import pytest

from src.agent.geoflow import retrieve_examples, retrieve_templates
from src.agent.retrieval import cosine_similarity

_TRIP = {
    "concepts": [
        {"id": "base", "text": "가예", "concept_type": "location", "role": "extent"},
        {"id": "a", "text": "한옥마을", "concept_type": "location", "role": "condition"},
        {"id": "b", "text": "전망대", "concept_type": "location", "role": "condition"},
        {"id": "order", "text": "방문 순서", "concept_type": "network", "role": "measure"},
    ],
    "measure": "방문 순서",
}
_TRIP_QUESTION = "가예에서 출발해 한옥마을을 1시간, 전망대를 1시간 둘러본 뒤 방문 순서는?"


def test_a_template_is_a_typed_graph_fragment_prior() -> None:
    templates = retrieve_templates(_TRIP, _TRIP_QUESTION)

    assert templates
    for template in templates:
        assert template["input_ports"]
        assert template["output_ports"]
        assert template["concept_nodes"]
        assert template["transformation_edges"]
        assert template["prior_only"] is True


def test_examples_require_an_embedding_backend() -> None:
    examples = retrieve_examples(_TRIP, _TRIP_QUESTION)

    assert examples == []


def test_template_retrieval_ignores_question_literals() -> None:

    names = {t["name"] for t in retrieve_templates(_TRIP, _TRIP_QUESTION, limit=3)}
    paraphrase = {t["name"] for t in retrieve_templates(_TRIP, "unrelated wording", limit=3)}

    assert names == paraphrase


def test_an_embedder_decides_the_example_order_when_one_is_supplied() -> None:
    """A stub that puts Route-Optimize nearest the question proves the seam is wired."""

    wanted = "네트워크 경로를 최적화"

    def embed(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if wanted in text or text is texts[0] else [0.0, 1.0] for text in texts]

    ordered = retrieve_examples(_TRIP, _TRIP_QUESTION, limit=1, embed=embed)

    assert ordered[0]["name"] == "appendix_e_route_optimize"
    assert ordered[0]["example"]["factor_nodes"]
    assert any(edge["factor_nodes"] for edge in ordered[0]["example"]["transformation_edges"])


def test_an_embedder_that_returns_the_wrong_count_is_refused() -> None:
    with pytest.raises(ValueError, match="one vector per input"):
        retrieve_examples(_TRIP, _TRIP_QUESTION, embed=lambda texts: [[1.0]])


def test_similarity_is_cosine_and_ignores_magnitude() -> None:
    assert cosine_similarity([1.0, 0.0], [4.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 3.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 1.0], [1.0, 0.0]) == pytest.approx(1 / math.sqrt(2))
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_the_agent_defaults_to_no_embedder() -> None:
    """Which is what the benchmarks ran with, and what a report should be read against."""

    import inspect

    from src.agent.spatial import SpatialAgent

    assert inspect.signature(SpatialAgent.__init__).parameters["example_embedder"].default is None
