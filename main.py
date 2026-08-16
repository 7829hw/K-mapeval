from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from src.agent import ReactAgent, SpatialAgent
from src.config import Settings
from src.dataset import load_dataset
from src.evaluator import Evaluator
from src.llm import OpenAIChatClient
from src.tools import KakaoMapProvider, ToolRegistry


def build_parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run the K-MapEval benchmark")
    result.add_argument(
        "--agent",
        choices=("react", "spatial", "both"),
        default="both",
        help="Agent architecture to evaluate (default: both)",
    )
    result.add_argument("--dataset", default="dataset/sample.jsonl")
    result.add_argument("--output-dir", default="reports")
    result.add_argument("--ids", nargs="*", help="Optional question IDs")
    result.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Concurrent question/LLM sessions (default: BENCHMARK_CONCURRENCY or 4)",
    )
    return result


@contextmanager
def create_agent_session(
    agent_type: str, settings: Settings
) -> Iterator[ReactAgent | SpatialAgent]:
    """Create resources owned by exactly one benchmark worker thread."""

    provider = KakaoMapProvider(
        settings.kakao_rest_api_key,
        timeout=settings.kakao_timeout_seconds,
        cache_path=settings.kakao_cache_db_path,
        cache_ttl_seconds=settings.kakao_cache_ttl_seconds,
    )
    llm: OpenAIChatClient | None = None
    try:
        llm = OpenAIChatClient(settings)
        tools = ToolRegistry(provider)
        agent = (
            ReactAgent(llm, tools, max_steps=settings.max_reasoning_steps)
            if agent_type == "react"
            else SpatialAgent(llm, tools, max_steps=settings.max_reasoning_steps)
        )
        yield agent
    finally:
        try:
            if llm is not None:
                llm.close()
        finally:
            provider.close()


def run(agent_type: str, args: argparse.Namespace) -> dict:
    settings = Settings()
    settings.require_llm()
    settings.require_kakao()
    dataset = load_dataset(args.dataset)
    if args.ids:
        wanted = set(args.ids)
        dataset = [item for item in dataset if item.id in wanted]
        if not dataset:
            raise ValueError("None of --ids were found in the dataset")
    concurrency = settings.benchmark_concurrency if args.concurrency is None else args.concurrency
    if not 1 <= concurrency <= 32:
        raise ValueError("--concurrency must be between 1 and 32")
    report = Evaluator(
        None,
        dataset,
        agent_factory=lambda: create_agent_session(agent_type, settings),
        max_workers=concurrency,
        output_dir=Path(args.output_dir),
        dataset_path=args.dataset,
        test_mode="ids" if args.ids else "full",
    ).run()
    return report.statistics


def main() -> None:
    args = build_parser().parse_args()
    agent_types = (
        ("react", "spatial_agent")
        if args.agent == "both"
        else ("spatial_agent" if args.agent == "spatial" else "react",)
    )
    summaries = {agent_type: run(agent_type, args) for agent_type in agent_types}
    if len(summaries) == 2:
        print(
            "ReAct accuracy="
            f"{summaries['react']['overall_answer_accuracy']['accuracy']:.3f} | "
            "Spatial-Agent accuracy="
            f"{summaries['spatial_agent']['overall_answer_accuracy']['accuracy']:.3f}"
        )


if __name__ == "__main__":
    main()
