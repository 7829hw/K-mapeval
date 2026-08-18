from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from src.agent import ReactAgent, SpatialAgent
from src.config import Settings
from src.dataset import load_dataset
from src.evaluator import Evaluator
from src.llm import LLMUnavailableError, OpenAIChatClient
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
    result.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Start the benchmark without checking the LLM endpoint first",
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
        search_center=settings.search_center(),
        search_radius_m=settings.kakao_search_radius_m,
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


def preflight_llm(settings: Settings) -> None:
    """Prove the endpoint answers before spending a batch on it.

    `--agent both` runs the two agents back to back, so an endpoint that dies between the legs used
    to score the second agent 0% across the board and write it out as if it were a result.
    """

    llm = OpenAIChatClient(settings)
    try:
        llm.chat([{"role": "user", "content": "ping"}])
    finally:
        llm.close()


def run(agent_type: str, args: argparse.Namespace) -> dict:
    settings = Settings()
    settings.require_llm()
    settings.require_kakao()
    if not args.skip_preflight:
        preflight_llm(settings)
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
        agent_type=agent_type,
        llm_profile={
            "llm_model": settings.llm_model,
            "llm_base_url": settings.llm_base_url,
        },
        abort_after_llm_failures=settings.benchmark_abort_after_llm_failures,
        question_retries=settings.benchmark_question_retries,
        question_retry_backoff_seconds=settings.benchmark_question_retry_backoff_seconds,
    ).run()
    return report.statistics


def main() -> None:
    args = build_parser().parse_args()
    agent_types = (
        ("react", "spatial_agent")
        if args.agent == "both"
        else ("spatial_agent" if args.agent == "spatial" else "react",)
    )
    summaries: dict[str, dict] = {}
    for agent_type in agent_types:
        try:
            summaries[agent_type] = run(agent_type, args)
        except LLMUnavailableError as exc:
            raise SystemExit(
                f"LLM endpoint is unavailable, so the {agent_type} benchmark was not run "
                f"(no report written): {exc}\n"
                "If the endpoint is healthy and this check is wrong, re-run with --skip-preflight."
            ) from exc
    if len(summaries) == 2:
        print(
            "ReAct accuracy="
            f"{summaries['react']['overall_answer_accuracy']['accuracy']:.3f} | "
            "Spatial-Agent accuracy="
            f"{summaries['spatial_agent']['overall_answer_accuracy']['accuracy']:.3f}"
        )
        if not all(
            summary["run_validity"]["valid"] for summary in summaries.values()
        ):
            print(
                "WARNING: at least one leg lost the LLM endpoint mid-run. "
                "The comparison above is not valid."
            )


if __name__ == "__main__":
    main()
