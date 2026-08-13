from __future__ import annotations

import json

from k_mapeval.agents.base import AgentResult, BenchmarkAgent
from k_mapeval.evaluation.dataset import BenchmarkItem
from k_mapeval.evaluation.evaluator import Evaluator


class FixedAgent(BenchmarkAgent):
    agent_type = "fixed"

    def __init__(self) -> None:
        self.index = 0

    def answer(self, question: str, options: list[str]) -> AgentResult:
        self.index += 1
        return AgentResult(
            agent_type=self.agent_type,
            predicted_answer=self.index,
            response=f"^^{self.index}^^",
            tool_calls=1,
            api_calls=1,
            reasoning_steps=2,
            latency_ms=10,
        )


def test_evaluator_computes_metrics_and_writes_logs(tmp_path) -> None:
    items = [
        BenchmarkItem(id="a", question="q", options=["x", "y"], answer=1, classification="poi"),
        BenchmarkItem(id="b", question="q", options=["x", "y"], answer=1, classification="poi"),
    ]
    report = Evaluator(FixedAgent(), items, output_dir=tmp_path).run()
    assert report.summary["accuracy"] == 0.5
    assert report.summary["tool_calls"] == 2
    assert report.summary["classification_accuracy"]["poi"]["correct"] == 1
    saved = json.loads((tmp_path / "fixed_report.json").read_text(encoding="utf-8"))
    assert saved["summary"]["accuracy"] == 0.5
    assert (tmp_path / "questions" / "fixed" / "a.json").exists()

