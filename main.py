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
from src.tools import KakaoMapProvider, MapProvider, ToolRegistry


def build_parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run the K-MapEval benchmark")
    result.add_argument(
        "--agent",
        choices=("react", "spatial", "both"),
        default="both",
        help="Agent architecture to evaluate (default: both)",
    )
    result.add_argument(
        "--dataset",
        default="dataset/seoul_kmapeval_v4_mcq_100.jsonl",
        help=(
            "Benchmark to evaluate. The default is the MapEval-method reproduction benchmark; "
            "seoul_kmapeval_v3 is the compositional one, seoul_mapeval_v1 the legacy context "
            "dataset, and seoul_kmapeval_v2 the superseded first reproduction benchmark."
        ),
    )
    result.add_argument(
        "--react-tools",
        choices=("reference", "native", "mapeval", "full"),
        default="reference",
        help=(
            "Tool surface for the ReAct baseline. 'reference' (default) is mapeval-api's five "
            "tools with its argument contracts: PlaceSearch(placeName) returns one id, "
            "NearbyPlaces refuses a radius when it ranks by distance, and neither routing tool "
            "takes a waypoint or a priority. 'native' ('mapeval' is the old name for it) keeps "
            "the same five names with this registry's richer arguments, which is a stronger "
            "baseline than the paper's and an ablation. 'full' shares the whole registry, "
            "aggregations included. Recorded in the report metadata; the three are not poolable."
        ),
    )
    result.add_argument("--output-dir", default="reports")
    result.add_argument("--ids", nargs="*", help="Optional question IDs")
    result.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Concurrent question/LLM sessions (default: BENCHMARK_CONCURRENCY or 4)",
    )
    result.add_argument(
        "--repeats",
        type=int,
        default=1,
        help=(
            "How many times to run each agent over the dataset (default: 1). The LLM endpoint is "
            "not reproducible even at temperature 0, so one pass is one draw; repeats print the "
            "spread and every pass writes its own report."
        ),
    )
    return result


def build_provider(settings: Settings) -> MapProvider:
    """Build the Kakao evidence source both architectures share for this run."""

    return KakaoMapProvider(
        settings.kakao_rest_api_key,
        timeout=settings.kakao_timeout_seconds,
        cache_path=settings.kakao_cache_db_path,
        cache_ttl_seconds=settings.kakao_cache_ttl_seconds,
        search_center=settings.search_center(),
        search_radius_m=settings.kakao_search_radius_m,
    )


@contextmanager
def create_agent_session(
    agent_type: str,
    settings: Settings,
    react_tools: str = "reference",
) -> Iterator[ReactAgent | SpatialAgent]:
    """Create resources owned by exactly one benchmark worker thread.

    The two agents get different tool surfaces, because the tool surface is part of the
    architecture under test. ReAct gets the five primitives MapEval's own baseline is given;
    Spatial-Agent gets this registry's aggregations and its local operators, which is the
    arrangement upstream has (`spatial-agent/src/tools/google_maps.py` carries a distance matrix
    that `mapeval-api/Evaluator2.py` never hands its baseline).

    Three ReAct surfaces, and the difference between the first two is *arguments*, not names:

    - `reference` (default) is `mapeval-api`'s contract field for field — `PlaceSearch(placeName)`
      returning one id, `NearbyPlaces` refusing radius when it ranks by distance, `Directions` and
      `TravelTime` taking an origin, a destination and a mode and nothing else.
    - `native` is the same five names with this registry's own arguments: waypoints, a route
      priority, a centre and category and rating filter on the search, both nearby modes at once.
      Restricting the *names* was never enough — an argument is a capability, and on the v5 run
      ReAct issued a waypointed `directions` call on all 8 `routing_distance_via` rows. Keep it as
      the labelled ablation it is.
    - `full` restores the shared surface, aggregations included.

    Report which was used; the three are not poolable.
    """

    provider = build_provider(settings)
    llm: OpenAIChatClient | None = None
    try:
        llm = OpenAIChatClient(settings)
        baseline = agent_type == "react" and react_tools != "full"
        upstream = baseline and react_tools == "reference"
        tools = ToolRegistry(
            provider,
            allowed=ToolRegistry.MAPEVAL_BASELINE_TOOLS if baseline else None,
            contract="reference" if upstream else "native",
        )
        agent = (
            ReactAgent(
                llm,
                tools,
                # The loop travels with the surface, because both halves are the same claim about
                # what MapEval's baseline is. `reference` runs upstream's: one action per
                # iteration and a forced stop that carries no answer, inside `REACT_MAX_STEPS`.
                # `native` keeps what this repository had, which is a stronger agent and a
                # labelled ablation.
                max_steps=settings.react_steps,
                single_action=upstream,
                force_final_answer=not upstream,
            )
            if agent_type == "react"
            else SpatialAgent(llm, tools, max_steps=settings.spatial_steps)
        )
        yield agent
    finally:
        try:
            if llm is not None:
                llm.close()
        finally:
            provider.close()


def run(agent_type: str, args: argparse.Namespace, repeat: int = 1, repeats: int = 1) -> dict:
    settings = Settings()
    settings.require_llm()
    settings.require_kakao()
    dataset = load_dataset(args.dataset)
    if args.ids:
        wanted = set(args.ids)
        dataset = [item for item in dataset if item.id in wanted]
        if not dataset:
            raise ValueError("None of --ids were found in the dataset")
    print("Evidence source: kakao")
    concurrency = settings.benchmark_concurrency if args.concurrency is None else args.concurrency
    if not 1 <= concurrency <= 32:
        raise ValueError("--concurrency must be between 1 and 32")
    report = Evaluator(
        None,
        dataset,
        agent_factory=lambda: create_agent_session(agent_type, settings, args.react_tools),
        max_workers=concurrency,
        output_dir=Path(args.output_dir),
        dataset_path=args.dataset,
        test_mode="ids" if args.ids else "full",
        agent_type=agent_type,
        llm_profile={
            "llm_model": settings.llm_model,
            "llm_base_url": settings.llm_base_url,
            "provider": "kakao",
            # Which tool surface the ReAct baseline had. A run is only comparable to the paper
            # when this says "mapeval", and only comparable across our own agents when "full".
            "react_tools": args.react_tools,
            # A tool surface is only half of what a baseline is; the loop is the other half, and
            # both were stronger here than upstream's. An accuracy that does not record them
            # cannot be compared with the paper's or with this repository's own earlier runs.
            "llm_temperature": settings.llm_temperature,
            # Both budgets, always, whichever agent ran: an accuracy is only comparable to
            # another when the reader can see what each side was allowed. They are separate
            # because a ReAct step is one tool call and a Spatial-Agent step is one edge of an
            # authored graph, and only the first of the two has an upstream default.
            "max_reasoning_steps": settings.max_reasoning_steps,
            "react_max_steps": settings.react_steps,
            "spatial_max_steps": settings.spatial_steps,
            "react_parallel_tool_calls": args.react_tools != "reference",
            "react_forces_final_answer": args.react_tools != "reference",
            # Which code answered. A night of fixes produces a shelf of reports whose accuracies
            # differ for reasons no field records, and "which commit was this?" is not
            # reconstructable from the timestamp once two runs overlap. Read once at import, not
            # per run: the process holds the code it loaded at startup, and a commit landing
            # while `--agent both` is halfway through would otherwise be recorded as the code
            # that answered the second half.
            "code_revision": CODE_REVISION,
            # Which pass of a repeated measurement this is. The endpoint is not reproducible even
            # at temperature 0 -- no sampling parameter reaches it -- so a lone accuracy is one
            # draw from a spread that has measured wider than the differences these runs are
            # asked about. A report that does not say which pass it was cannot be pooled with its
            # siblings afterwards.
            "repeat_index": repeat,
            "repeat_count": repeats,
        },
        question_retries=settings.benchmark_question_retries,
        question_retry_backoff_seconds=settings.benchmark_question_retry_backoff_seconds,
    ).run()
    return report.statistics


def _code_revision() -> str | None:
    """The commit this run's code came from, read from the git directory, never a subprocess."""

    head = Path(__file__).resolve().parent / ".git" / "HEAD"
    try:
        pointer = head.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not pointer.startswith("ref:"):
        return pointer[:12] or None
    reference = head.parent / pointer.split(" ", 1)[1].strip()
    try:
        return reference.read_text(encoding="utf-8").strip()[:12] or None
    except OSError:
        pass
    packed = head.parent / "packed-refs"
    try:
        lines = packed.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    wanted = pointer.split(" ", 1)[1].strip()
    for line in lines:
        commit, _, name = line.partition(" ")
        if name.strip() == wanted:
            return commit[:12]
    return None


# Read once, at import: this is the revision whose code this process is running.
CODE_REVISION = _code_revision()

LABELS = {"react": "ReAct", "spatial_agent": "Spatial-Agent"}


def _report_passes(passes: dict[str, list[float]]) -> None:
    """Print every pass and its spread, never a single number standing in for the measurement.

    The spread is the point. This endpoint answers the same request differently at temperature 0 —
    no sampling parameter reaches it — and two no-tool floor runs over one benchmark came back
    24/100 and 32/100, wider than any architecture difference measured here. A summary line that
    prints one accuracy per agent invites exactly the comparison the spread does not support.
    """

    for agent_type, accuracies in passes.items():
        label = LABELS.get(agent_type, agent_type)
        joined = ", ".join(f"{value:.3f}" for value in accuracies)
        if len(accuracies) == 1:
            print(f"{label} accuracy={joined} (one pass; the spread is unmeasured)")
            continue
        low, high = min(accuracies), max(accuracies)
        mean = sum(accuracies) / len(accuracies)
        print(f"{label} accuracy: {joined} | mean={mean:.3f} spread={high - low:.3f}")
    if len(passes) == 2 and all(len(values) > 1 for values in passes.values()):
        widest = max(max(values) - min(values) for values in passes.values())
        means = [sum(values) / len(values) for values in passes.values()]
        gap = abs(means[0] - means[1])
        verdict = "inside" if gap <= widest else "outside"
        print(f"Mean gap {gap:.3f} is {verdict} the widest single-agent spread {widest:.3f}")


def main() -> None:
    args = build_parser().parse_args()
    agent_types = (
        ("react", "spatial_agent")
        if args.agent == "both"
        else ("spatial_agent" if args.agent == "spatial" else "react",)
    )
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.react_tools == "mapeval":
        # The old name for the native-argument surface, kept working so a saved command still
        # runs. Reports record the resolved name, so nothing quotes "mapeval" going forward.
        args.react_tools = "native"
    passes: dict[str, list[float]] = {agent_type: [] for agent_type in agent_types}
    for repeat in range(1, args.repeats + 1):
        for agent_type in agent_types:
            if args.repeats > 1:
                print(f"\n=== {agent_type}, pass {repeat}/{args.repeats}")
            statistics = run(agent_type, args, repeat=repeat, repeats=args.repeats)
            passes[agent_type].append(statistics["overall_answer_accuracy"]["accuracy"])
    _report_passes(passes)


if __name__ == "__main__":
    main()
