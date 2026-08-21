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
    assert report.statistics["answer_accuracy_by_class"]["poi"]["correct"] == 1
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
    # Whose log this is, without reading it: `--agent both` writes two per question into one
    # directory, and the name is what a person greps first.
    assert all("_fixed_id" in path.name for path in query_logs)
    log_text = query_logs[0].read_text(encoding="utf-8")
    assert "FAST WORKFLOW STARTED | AGENT: fixed" in log_text
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



def test_no_intent_is_scored_against_the_datasets_class(tmp_path) -> None:
    """The two are not the same kind of thing, and only one architecture even has an intent.

    `classification` labels the question; a Spatial-Agent intent is a routing decision inside one
    pipeline, and ReAct has no such stage at all. Comparing them scored one architecture on a
    stage the other does not have, so the report keeps the intent as a record of what happened
    and grades nothing against it.
    """

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

    assert not [key for key in report.statistics if "intent" in key]
    assert not [key for key in report.results[0] if key.endswith("intent_correct")]
    assert report.results[0]["predicted_intent"] is None
    assert set(report.statistics["answer_accuracy_by_class"]) == {"poi", "nearby"}
    assert report.statistics["overall_answer_accuracy"]["accuracy"] == 1.0


def test_a_trace_reaches_the_log_while_the_question_is_still_being_answered(tmp_path) -> None:
    """A question can run for minutes; its log should not arrive only once it is over.

    The agent reads its own log file mid-answer, which is the only way to tell streaming apart
    from a well-ordered dump at the end.
    """

    log_dir = tmp_path / "logs"

    class SelfWatchingAgent(BenchmarkAgent):
        agent_type = "watcher"

        def __init__(self) -> None:
            self.saw_itself = False

        def answer(self, question: str, options: list[str]) -> AgentResult:
            trace = self.new_trace()
            trace.append({"stage": "analyze", "note": "first-step"})
            self.saw_itself = any(
                "first-step" in path.read_text(encoding="utf-8")
                for path in log_dir.glob("*.log")
            )
            trace.append({"stage": "evaluate", "note": "last-step"})
            return AgentResult(
                agent_type=self.agent_type,
                predicted_answer=0,
                response="^^0^^",
                trace=list(trace),
            )

    agent = SelfWatchingAgent()
    items = [
        BenchmarkItem(id="a", question="q", options=["x", "y"], answer=0, classification="poi")
    ]
    Evaluator(agent, items, output_dir=None, log_dir=log_dir).run()

    assert agent.saw_itself
    written = next(log_dir.glob("*.log")).read_text(encoding="utf-8")
    # Streamed once, not streamed and then dumped again at the end.
    assert written.count("first-step") == 1
    assert written.count("last-step") == 1


def test_an_agent_that_streams_nothing_still_gets_its_whole_trace_logged(tmp_path) -> None:
    """Test doubles and any future caller that builds a trace by hand keep working."""

    class PlainListAgent(BenchmarkAgent):
        agent_type = "plain"

        def answer(self, question: str, options: list[str]) -> AgentResult:
            return AgentResult(
                agent_type=self.agent_type,
                predicted_answer=0,
                response="^^0^^",
                trace=[{"stage": "compose", "note": "never-streamed"}],
            )

    log_dir = tmp_path / "logs"
    items = [
        BenchmarkItem(id="a", question="q", options=["x", "y"], answer=0, classification="poi")
    ]
    Evaluator(PlainListAgent(), items, output_dir=None, log_dir=log_dir).run()

    assert "never-streamed" in next(log_dir.glob("*.log")).read_text(encoding="utf-8")
