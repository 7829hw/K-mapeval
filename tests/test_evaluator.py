from __future__ import annotations

import json
import time
from contextlib import contextmanager
from threading import Barrier, Lock

import pytest

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
    assert saved["metadata"]["concurrency"] == 1
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


def test_parallel_evaluator_uses_four_isolated_llm_sessions_and_preserves_order(
    tmp_path,
) -> None:
    worker_count = 4
    wave_barrier = Barrier(worker_count)
    state_lock = Lock()
    state = {"created": 0, "closed": 0, "active": 0, "max_active": 0}

    @contextmanager
    def agent_factory():
        with state_lock:
            state["created"] += 1

        class ConcurrentAgent(BenchmarkAgent):
            agent_type = "parallel"

            def answer(self, question: str, options: list[str]) -> AgentResult:
                with state_lock:
                    state["active"] += 1
                    state["max_active"] = max(state["max_active"], state["active"])
                try:
                    wave_barrier.wait(timeout=2)
                    time.sleep(0.01)
                finally:
                    with state_lock:
                        state["active"] -= 1
                return AgentResult(
                    agent_type=self.agent_type,
                    predicted_intent="poi",
                    predicted_answer=0,
                    trace=[{"stage": "evaluate", "marker": question}],
                )

        try:
            yield ConcurrentAgent()
        finally:
            with state_lock:
                state["closed"] += 1

    items = [
        BenchmarkItem(
            id=f"q{index}",
            question=f"marker-{index}",
            options=["x", "y"],
            answer=0,
            classification="poi",
        )
        for index in range(8)
    ]
    report = Evaluator(
        None,
        items,
        agent_factory=agent_factory,
        max_workers=worker_count,
        output_dir=None,
        log_dir=tmp_path / "logs",
    ).run()

    assert state == {"created": 4, "closed": 4, "active": 0, "max_active": 4}
    assert report.metadata["concurrency"] == 4
    assert [row["id"] for row in report.results] == [item.id for item in items]
    for index, item in enumerate(items):
        log_file = next((tmp_path / "logs").glob(f"*_id{item.id}_*.log"))
        log_text = log_file.read_text(encoding="utf-8")
        assert f'"marker": "marker-{index}"' in log_text
        assert all(
            f'"marker": "marker-{other}"' not in log_text
            for other in range(len(items))
            if other != index
        )


def test_parallel_evaluator_rejects_a_shared_agent() -> None:
    with pytest.raises(ValueError, match="isolated agent_factory"):
        Evaluator(FixedAgent(), [], max_workers=4)



def test_an_agent_without_an_intent_stage_is_not_scored_as_wrong(tmp_path) -> None:
    """ReAct has no classifier, so 0.0% was a verdict on a stage the architecture never had."""

    class SilentAgent(BenchmarkAgent):
        agent_type = "react"

        def answer(self, question: str, options: list[str]) -> AgentResult:
            return AgentResult(agent_type=self.agent_type, predicted_answer=0, response="^^0^^")

    items = [
        BenchmarkItem(id="a", question="q", options=["x", "y"], answer=0, classification="poi"),
        BenchmarkItem(id="b", question="q2", options=["x", "y"], answer=0, classification="nearby"),
    ]
    report = Evaluator(
        SilentAgent(), items, output_dir=None, log_dir=tmp_path / "logs"
    ).run()

    intent = report.statistics["intent_classification_accuracy"]
    assert intent["classified"] == 0
    assert intent["accuracy"] is None
    assert intent["total"] == 2
    assert report.statistics["overall_answer_accuracy"]["accuracy"] == 1.0
