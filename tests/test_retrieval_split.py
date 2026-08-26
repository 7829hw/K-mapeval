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

from src.agent.geoflow import TEMPLATES, retrieve_examples, retrieve_templates

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


def test_a_template_carries_a_pattern_and_no_example() -> None:
    templates = retrieve_templates(_TRIP, _TRIP_QUESTION)

    assert templates
    for template in templates:
        assert set(template) == {"name", "pattern"}


def test_an_example_carries_a_graph_and_no_pattern() -> None:
    examples = retrieve_examples(_TRIP, _TRIP_QUESTION)

    assert examples
    for example in examples:
        assert set(example) == {"name", "example"}
        assert example["example"]["graph"]


def test_the_two_are_ranked_independently() -> None:
    """Same question, two rankings; nothing forces them to agree."""

    names = {t["name"] for t in retrieve_templates(_TRIP, _TRIP_QUESTION, limit=3)}
    examples = {e["name"] for e in retrieve_examples(_TRIP, _TRIP_QUESTION, limit=3)}

    assert names and examples
    assert names <= {t["name"] for t in TEMPLATES.values()}
    assert examples <= {t["name"] for t in TEMPLATES.values()}


def test_an_embedder_decides_the_example_order_when_one_is_supplied() -> None:
    """A stub that puts Route-Optimize nearest the question proves the seam is wired."""

    wanted = "Route-Optimize"

    def embed(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if wanted in text or text is texts[0] else [0.0, 1.0]
                for text in texts]

    ordered = retrieve_examples(_TRIP, _TRIP_QUESTION, limit=1, embed=embed)

    assert ordered[0]["name"] == wanted


def test_an_embedder_that_returns_the_wrong_count_is_refused() -> None:
    with pytest.raises(ValueError, match="one vector per input"):
        retrieve_examples(_TRIP, _TRIP_QUESTION, embed=lambda texts: [[1.0]])


def test_similarity_is_cosine_and_ignores_magnitude() -> None:
    from src.agent.geoflow import _cosine

    assert _cosine([1.0, 0.0], [4.0, 0.0]) == pytest.approx(1.0)
    assert _cosine([1.0, 0.0], [0.0, 3.0]) == pytest.approx(0.0)
    assert _cosine([1.0, 1.0], [1.0, 0.0]) == pytest.approx(1 / math.sqrt(2))
    assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_the_agent_defaults_to_no_embedder() -> None:
    """Which is what the benchmarks ran with, and what a report should be read against."""

    import inspect

    from src.agent.spatial import SpatialAgent

    assert inspect.signature(SpatialAgent.__init__).parameters["example_embedder"].default is None
