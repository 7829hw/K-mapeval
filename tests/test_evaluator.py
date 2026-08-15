from __future__ import annotations

import json

from src.agent.base import AgentResult, BenchmarkAgent
from src.dataset import BenchmarkItem
from src.evaluator import Evaluator


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
    report = Evaluator(
        FixedAgent(), items, output_dir=tmp_path, dataset_path="fixtures/test.jsonl"
    ).run()
    assert report.summary["accuracy"] == 0.5
    assert report.summary["tool_calls"] == 2
    assert report.summary["classification_accuracy"]["poi"]["correct"] == 1
    saved = json.loads((tmp_path / "fixed_report.json").read_text(encoding="utf-8"))
    assert saved["dataset_path"] == "fixtures/test.jsonl"
    assert saved["completed_questions"] == 2
    assert saved["total_questions"] == 2
    assert saved["is_complete"] is True
    assert saved["summary"]["accuracy"] == 0.5
    assert saved["records"][0]["question"] == "q"
    assert saved["records"][0]["options"] == ["x", "y"]
    assert (tmp_path / "questions" / "fixed" / "a.json").exists()


def test_evaluator_checkpoints_report_and_prints_progress(tmp_path, capsys) -> None:
    report_path = tmp_path / "checkpoint_report.json"

    class CheckpointAgent(BenchmarkAgent):
        agent_type = "checkpoint"

        def __init__(self) -> None:
            self.calls = 0

        def answer(self, question: str, options: list[str]) -> AgentResult:
            self.calls += 1
            if self.calls == 2:
                checkpoint = json.loads(report_path.read_text(encoding="utf-8"))
                assert checkpoint["completed_questions"] == 1
                assert checkpoint["total_questions"] == 2
                assert checkpoint["is_complete"] is False
                assert [record["question_id"] for record in checkpoint["records"]] == ["a"]
            return AgentResult(
                agent_type=self.agent_type,
                predicted_answer=1,
                latency_ms=25,
            )

    items = [
        BenchmarkItem(id="a", question="q1", options=["x", "y"], answer=1, classification="poi"),
        BenchmarkItem(id="b", question="q2", options=["x", "y"], answer=1, classification="poi"),
    ]
    Evaluator(CheckpointAgent(), items, output_dir=tmp_path).run()

    output = capsys.readouterr().out
    assert "[checkpoint] QA 1/2 실행 중 | id=a" in output
    assert "[checkpoint] QA 1/2 완료 (50.0%)" in output
    assert "[checkpoint] QA 2/2 완료 (100.0%)" in output
    final_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert final_report["completed_questions"] == 2
    assert final_report["is_complete"] is True
    assert not list(tmp_path.rglob("*.tmp"))
