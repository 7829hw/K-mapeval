"""Topological execution of a validated operator graph."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExecutionState:
    operator_state: dict[str, Any]
    concept_state: dict[str, Any]
    log: tuple[dict[str, Any], ...]


def execute_topologically(
    steps: Sequence[dict[str, Any]],
    *,
    invoke: Callable[[str, dict[str, Any]], Any],
    resolve_arguments: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
) -> ExecutionState:
    """Small execution seam shared by runtime and unit tests.

    Provider/operator dispatch remains injected by the caller, preserving the registry boundary.
    """

    state: dict[str, Any] = {}
    concepts: dict[str, Any] = {}
    log: list[dict[str, Any]] = []
    for step in steps:
        arguments = resolve_arguments(dict(step.get("arguments") or {}), state)
        output = invoke(str(step["operator"]), arguments)
        state[str(step["id"])] = output
        for binding in step.get("output_bindings") or ():
            if str(binding.get("path") or "$") == "$":
                concepts[str(binding["concept_id"])] = output
        log.append(
            {
                "id": step["id"],
                "operator": step["operator"],
                "arguments": arguments,
                "status": "ok",
                "result": output,
            }
        )
    return ExecutionState(state, concepts, tuple(log))
