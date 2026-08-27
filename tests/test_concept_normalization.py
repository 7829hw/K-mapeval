"""The Analysis boundary, where a concept graph stops describing the question.

Everything downstream of Concept Analysis reads place identity out of the concepts, so a concept
that is not an entity is a `RESOLVE_PLACES` node that geocodes something which is not a place.
Three shapes of that were reaching the graph, and each one failed silently -- the geocode found
nothing, and the operator after it narrowed, ranked or counted one opaque item.

The measured one: eleven `nearby_cuisine_subtype` rows in a three-pass run had the whole
candidate list emitted as a single concept, so `batch_geocode` looked up
"이태원 농담, 코파카바나그릴, 웍바이술0.05, 이태원주식" as one place name, and the subtype filter
that follows had a single item with no category to narrow. The narrowing was correct, enforced
and useless, because it was handed nothing to enforce it on.

Repairs are keyed on evidence the pipeline already has -- the candidate texts and the facts the
deterministic extractors read -- never on a fresh interpretation of the question, and every
concept created or altered says so in `attributes["source"]`.
"""

from __future__ import annotations

from src.agent.geoflow import normalize_analysis
from src.agent.spatial import GroundingFacts, extract_facts

CUISINE_QUESTION = (
    "지금 용산아트홀 소극장가람에 있습니다. 파스타나 스테이크 같은 양식을 먹으려 합니다. "
    "여기서 가장 가까운 양식 음식점은 다음 중 어디인가요?"
)
CUISINE_OPTIONS = ["이태원 농담", "코파카바나그릴", "웍바이술0.05", "이태원주식"]

RADIUS_QUESTION = (
    "왕십리곱창거리에서 반경 300m 이내에 있는 은행은 아래 목록 중 몇 곳인가요? "
    "(하나은행365 왕십리금융센터 ATM, 우리은행 365코너 왕십리역지점(점외), "
    "우리은행 365코너 서울동부노인전문요양센터, KB국민은행ATM 이마트왕십리점)"
)


def _texts(analysis: dict) -> list[str]:
    return [str(concept["text"]) for concept in analysis["concepts"]]


def _by_id(analysis: dict) -> dict[str, dict]:
    return {str(concept["id"]): concept for concept in analysis["concepts"]}


# ---------------------------------------------------------------------------------------------
# A multi-place list emitted as one concept
# ---------------------------------------------------------------------------------------------


def test_a_candidate_list_emitted_as_one_concept_becomes_one_concept_per_candidate() -> None:
    raw = {
        "intent": "nearby",
        "concepts": [
            {"id": "anchor", "text": "용산아트홀 소극장가람", "concept_type": "location",
             "role": "extent"},
            {"id": "candidates", "text": ", ".join(CUISINE_OPTIONS),
             "concept_type": "location", "role": "support"},
        ],
        "measure": "nearest",
        "target_type": "양식 음식점",
    }

    analysis = normalize_analysis(
        raw, CUISINE_QUESTION, "nearby",
        facts=extract_facts(raw, CUISINE_QUESTION), options=CUISINE_OPTIONS,
    )

    assert ", ".join(CUISINE_OPTIONS) not in _texts(analysis)
    for option in CUISINE_OPTIONS:
        assert option in _texts(analysis)
    split = [
        concept
        for concept in analysis["concepts"]
        if str(concept["id"]).startswith("candidates_")
    ]
    assert len(split) == len(CUISINE_OPTIONS)
    assert all(concept["attributes"]["source"] == "fact_completion" for concept in split)
    # The valid concept beside it is untouched, provenance included.
    assert _by_id(analysis)["anchor"]["attributes"] == {}


def test_the_other_two_spellings_of_a_joined_candidate_list_split_the_same_way() -> None:
    """Recorded across three passes: plain, a list repr rendered as a string, and labelled."""

    spellings = [
        ", ".join(CUISINE_OPTIONS),
        str(CUISINE_OPTIONS),
        ", ".join(
            f"Option {index}: {option}" for index, option in enumerate(CUISINE_OPTIONS)
        ),
    ]
    for spelling in spellings:
        raw = {
            "concepts": [
                {"id": "c", "text": spelling, "concept_type": "location", "role": "extent"},
            ],
            "measure": "nearest",
        }

        analysis = normalize_analysis(
            raw, CUISINE_QUESTION, "nearby", options=CUISINE_OPTIONS
        )

        assert [
            concept["text"]
            for concept in analysis["concepts"]
            if str(concept["id"]).startswith("c_")
        ] == CUISINE_OPTIONS, spelling


def test_a_comma_inside_one_name_is_not_a_list() -> None:
    """`제과,베이커리` is one kind of place. Generic comma splitting would make it two."""

    raw = {
        "concepts": [
            {"id": "kind", "text": "제과,베이커리", "concept_type": "object", "role": "support"},
            {"id": "anchor", "text": "서울역", "concept_type": "location", "role": "extent"},
        ],
        "measure": "nearest",
    }

    analysis = normalize_analysis(raw, "서울역 근처 빵집", "nearby", options=["가", "나"])

    assert "제과,베이커리" in _texts(analysis)


def test_a_joined_string_only_splits_where_every_piece_is_a_known_entity() -> None:
    """The split is licensed by the candidates and the facts, not by the punctuation."""

    raw = {
        "concepts": [
            {"id": "c", "text": "이태원 농담, 어딘가 다른 곳", "concept_type": "location",
             "role": "extent"},
        ],
        "measure": "nearest",
    }

    analysis = normalize_analysis(
        raw, CUISINE_QUESTION, "nearby", options=CUISINE_OPTIONS
    )

    assert "이태원 농담, 어딘가 다른 곳" in _texts(analysis)


# ---------------------------------------------------------------------------------------------
# The question itself, offered as a place
# ---------------------------------------------------------------------------------------------


def test_a_clause_of_the_question_is_never_a_geocodable_entity() -> None:
    facts = extract_facts({}, RADIUS_QUESTION)
    raw = {
        "concepts": [
            {"id": "c1", "text": RADIUS_QUESTION[:60], "concept_type": "location",
             "role": "extent"},
        ],
        "measure": "count",
    }

    analysis = normalize_analysis(
        raw, RADIUS_QUESTION, "nearby", facts=facts,
        options=["한 곳", "두 곳", "세 곳", "네 곳"],
    )

    clause = _by_id(analysis)["c1"]
    # Marked rather than deleted: the graph keeps its contextual root, and factorization already
    # excludes a synthetic concept from anything that resolves a place.
    assert clause["attributes"]["synthetic"] is True
    assert clause["attributes"]["source"] == "fact_completion"


def test_a_real_place_name_is_not_mistaken_for_a_clause() -> None:
    raw = {
        "concepts": [
            {"id": "a", "text": "남파김삼준 문화복지기념관", "concept_type": "location",
             "role": "extent"},
        ],
        "measure": "nearest",
    }

    analysis = normalize_analysis(
        raw, "남파김삼준 문화복지기념관에서 가장 가까운 약국은?", "nearby"
    )

    assert not _by_id(analysis)["a"]["attributes"].get("synthetic")


# ---------------------------------------------------------------------------------------------
# Entities the facts found and the concepts do not mention
# ---------------------------------------------------------------------------------------------


def test_listed_candidates_the_facts_read_are_added_when_no_concept_names_them() -> None:
    facts = extract_facts({}, RADIUS_QUESTION)
    raw = {
        "concepts": [
            {"id": "anchor", "text": "왕십리곱창거리", "concept_type": "location",
             "role": "extent"},
        ],
        "measure": "count",
    }

    analysis = normalize_analysis(
        raw, RADIUS_QUESTION, "nearby", facts=facts,
        options=["한 곳", "두 곳", "세 곳", "네 곳"],
    )

    for name in facts.listed_places:
        assert name in _texts(analysis)
    added = [
        concept for concept in analysis["concepts"] if str(concept["id"]).startswith("listed_")
    ]
    assert len(added) == len(facts.listed_places)
    assert all(concept["attributes"]["source"] == "fact_completion" for concept in added)
    assert all(concept["depends_on"] == ["anchor"] for concept in added)


def test_listed_candidates_already_named_are_not_added_twice() -> None:
    facts = GroundingFacts(anchor="기준점", listed_places=("가게 하나", "가게 둘"))
    raw = {
        "concepts": [
            {"id": "a", "text": "기준점", "concept_type": "location", "role": "extent"},
            {"id": "one", "text": "가게 하나", "concept_type": "location", "role": "support"},
            {"id": "two", "text": "가게둘", "concept_type": "location", "role": "support"},
        ],
        "measure": "count",
    }

    analysis = normalize_analysis(raw, "질문", "nearby", facts=facts)

    assert len([text for text in _texts(analysis) if text.startswith("가게")]) == 2


def test_the_fallback_graph_also_carries_the_candidates_the_question_lists() -> None:
    """With no concepts at all, the listed places are still what the question is about."""

    facts = extract_facts({}, RADIUS_QUESTION)

    analysis = normalize_analysis({}, RADIUS_QUESTION, "nearby", facts=facts)

    for name in facts.listed_places:
        assert name in _texts(analysis)
