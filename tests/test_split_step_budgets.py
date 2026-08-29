"""The two architectures do not count the same thing, so they do not share a budget.

A ReAct step is one loop iteration, which is one tool call. A Spatial-Agent step is one
transformation edge of an authored GeoFlow graph. Only the first has an upstream value --
langchain's `initialize_agent` default of 15, which is what `mapeval-api/Evaluator2.py` runs --
and holding the second to the same number was this repository's own housekeeping. Measured: a
faithful rendering of "how many of N stops fit the budget" costs `3N + 2` transformation edges,
or `4N + 2` when the planner extracts each leg's duration separately, so a five-stop question
costs 17 to 22 edges to say what ReAct says in about ten calls, and 37-39% of that family was
refused before anything executed.
"""

from __future__ import annotations

from src.config import Settings


def _settings(**overrides: object) -> Settings:
    # `_env_file=None` so a developer's own .env cannot decide what these assert.
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def test_each_architecture_falls_back_to_the_shared_budget() -> None:
    """One `.env` line still governs both, which is what the merged setting was for."""

    settings = _settings(max_reasoning_steps=22)
    assert settings.react_steps == 22
    assert settings.spatial_steps == 22


def test_each_architecture_can_be_given_its_own() -> None:
    settings = _settings(max_reasoning_steps=15, react_max_steps=15, spatial_max_steps=30)
    assert settings.react_steps == 15
    assert settings.spatial_steps == 30


def test_one_override_does_not_move_the_other() -> None:
    """Raising Spatial-Agent's budget must not quietly make the baseline stronger too.

    The whole point of the split: `MAX_REASONING_STEPS=30` moved both sides at once, so the
    budget ablation could never say which architecture the change belonged to.
    """

    settings = _settings(max_reasoning_steps=15, spatial_max_steps=30)
    assert settings.react_steps == 15, "ReAct's 15 is langchain's default and upstream's baseline"
    assert settings.spatial_steps == 30


def test_the_default_is_still_upstreams_fifteen() -> None:
    settings = _settings()
    assert settings.max_reasoning_steps == 15
    assert settings.react_steps == 15
    assert settings.spatial_steps == 15
