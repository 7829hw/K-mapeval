"""The floor has to survive the questions it cannot answer.

The no-tool floor is the number that makes every accuracy in this repository readable, and it is
measured over a whole dataset in one thread pool. An exception that escapes one question escapes
the pool and discards every other question's answer with it — which is what happened on the 282-row
v7a set, where one closed-book question spiralled to 65,304 completion tokens and took the run down
after seven minutes of work.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[1] / "data"
if str(DATA) not in sys.path:
    sys.path.insert(0, str(DATA))

import measure_no_tool_floor as floor  # noqa: E402

from src.llm import (  # noqa: E402
    LLMContextOverflowError,
    LLMOutputTruncatedError,
    LLMUnavailableError,
    TokenUsage,
)

ROW = {
    "id": "q_000",
    "template_id": "nearby_kth_nearest",
    "classification": "nearby",
    "question": "질문",
    "options": ["가", "나", "다", "라"],
    "answer": 2,
}


class _Client:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome

    def chat(self, _: list[dict]) -> object:
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return type("Reply", (), {"content": self.outcome})()

    def close(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _no_settings(monkeypatch) -> None:
    monkeypatch.setattr(floor, "Settings", lambda: None)


@pytest.mark.parametrize(
    "exc",
    [
        LLMUnavailableError("gateway gave up"),
        LLMOutputTruncatedError("cut off after 65304 completion tokens", TokenUsage()),
        LLMContextOverflowError("the prompt did not fit"),
    ],
)
def test_a_question_the_endpoint_never_answered_is_recorded_not_raised(monkeypatch, exc) -> None:
    """All three are the endpoint's failure, not the question's, and none may end the run."""

    monkeypatch.setattr(floor, "OpenAIChatClient", lambda _: _Client(exc))
    result = floor.answer_one(dict(ROW))
    assert result["failure"].startswith(type(exc).__name__)
    assert result["predicted"] is None
    assert result["correct"] is False


def test_an_answered_question_still_scores(monkeypatch) -> None:
    monkeypatch.setattr(floor, "OpenAIChatClient", lambda _: _Client("^^2^^"))
    result = floor.answer_one(dict(ROW))
    assert result == {
        "id": "q_000",
        "template_id": "nearby_kth_nearest",
        "classification": "nearby",
        "gold": 2,
        "predicted": 2,
        "correct": True,
        "failure": None,
    }
