from __future__ import annotations

import argparse
import json
from pathlib import Path

from k_mapeval.agents import ReactAgent, SpatialAgent
from k_mapeval.config import Settings
from k_mapeval.evaluation import Evaluator, load_dataset
from k_mapeval.llm import OpenAIChatClient
from k_mapeval.providers import KakaoMapProvider
from k_mapeval.tools import ToolRegistry


def parser(description: str) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--dataset", default="dataset/sample.jsonl")
    result.add_argument("--output-dir", default="results")
    result.add_argument("--ids", nargs="*", help="Optional question IDs")
    return result


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
    provider = KakaoMapProvider(
        settings.kakao_rest_api_key,
        timeout=settings.kakao_timeout_seconds,
        cache_path=settings.kakao_cache_db_path,
        cache_ttl_seconds=settings.kakao_cache_ttl_seconds,
    )
    try:
        llm = OpenAIChatClient(settings)
        tools = ToolRegistry(provider)
        agent = (
            ReactAgent(llm, tools, max_steps=settings.max_reasoning_steps)
            if agent_type == "react"
            else SpatialAgent(llm, tools, max_steps=settings.max_reasoning_steps)
        )
        report = Evaluator(agent, dataset, output_dir=Path(args.output_dir)).run()
        summary = report.summary
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    finally:
        provider.close()
