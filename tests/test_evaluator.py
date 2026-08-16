from __future__ import annotations

import json

from src.agent.base import AgentResult, BenchmarkAgent
from src.dataset import BenchmarkItem
from src.evaluator import Evaluator


class FixedAgent(BenchmarkAgent):
    agent_type = "fixed"

    def __init__(self) -> None:
        self.index = -1

    def answer(self, question: str, options: list[str]) -> AgentResult:
        self.index += 1
        return AgentResult(
            agent_type=self.agent_type,
            predicted_intent="poi",
            predicted_answer=self.index,
            response=f"^^{self.index}^^",
            tool_calls=1,
            api_calls=1,
            reasoning_steps=2,
            latency_ms=10,
            trace=[{"stage": "evaluate", "predicted_option": self.index}],
        )


def test_evaluator_writes_upstream_spatial_agent_report_and_query_logs(tmp_path) -> None:
    report_dir = tmp_path / "reports"
    log_dir = tmp_path / "logs"
    items = [
        BenchmarkItem(id="a", question="q", options=["x", "y"], answer=0, classification="poi"),
        BenchmarkItem(id="b", question="q", options=["x", "y"], answer=0, classification="poi"),
    ]
    evaluator = Evaluator(
        FixedAgent(),
        items,
        output_dir=report_dir,
        dataset_path="fixtures/test.jsonl",
        log_dir=log_dir,
    )
    report = evaluator.run()

    assert report.statistics["overall_answer_accuracy"]["accuracy"] == 0.5
    assert report.statistics["answer_accuracy_by_intent"]["poi"]["correct"] == 1
    assert report.statistics["intent_classification_accuracy"]["accuracy"] == 1.0
    report_files = list(report_dir.glob("test_*.json"))
    assert len(report_files) == 1
    saved = json.loads(report_files[0].read_text(encoding="utf-8"))
    assert set(saved) == {"metadata", "statistics", "results"}
    assert saved["metadata"]["dataset_source"] == "fixtures/test.jsonl"
    assert saved["metadata"]["test_mode"] == "full"
    assert saved["metadata"]["total_samples"] == 2
    assert saved["results"][0]["correct_answer"] == 0
    assert saved["results"][0]["predicted_option"] == 0
    assert saved["results"][0]["predicted_answer"] == "x"
    assert saved["results"][1]["predicted_option"] == 1
    assert not (report_dir / "fixed_report.json").exists()
    assert not (report_dir / "questions").exists()

    query_logs = sorted(log_dir.glob("*_id*.log"))
    assert len(query_logs) == 2
    log_text = query_logs[0].read_text(encoding="utf-8")
    assert "FAST WORKFLOW STARTED" in log_text
    assert "[EVALUATE]" in log_text
    assert "WORKFLOW COMPLETED" in log_text


def test_report_is_created_after_batch_and_terminal_matches_upstream_style(
    tmp_path, capsys
) -> None:
    report_dir = tmp_path / "reports"

    class EndOfBatchAgent(BenchmarkAgent):
        agent_type = "checkpoint"

        def __init__(self) -> None:
            self.calls = 0

        def answer(self, question: str, options: list[str]) -> AgentResult:
            self.calls += 1
            if self.calls == 2:
                assert not list(report_dir.glob("test_*.json"))
            return AgentResult(
                agent_type=self.agent_type,
                predicted_intent="poi",
                predicted_answer=0,
                latency_ms=25,
            )

    items = [
        BenchmarkItem(id="a", question="q1", options=["x", "y"], answer=0, classification="poi"),
        BenchmarkItem(id="b", question="q2", options=["x", "y"], answer=0, classification="poi"),
    ]
    Evaluator(
        EndOfBatchAgent(),
        items,
        output_dir=report_dir,
        log_dir=tmp_path / "logs",
    ).run()

    output = capsys.readouterr().out
    assert "Running evaluation on 2 samples" in output
    assert "[1/2] ID   a | poi" in output
    assert "[2/2] ID   b | poi" in output
    assert "Summary" in output
    assert "Overall answer accuracy: 2/2 (100.0%)" in output
    assert "Report saved to:" in output
    assert len(list(report_dir.glob("test_*.json"))) == 1
