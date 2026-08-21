# AGENTS.md

## Project

K-MapEval is a research MVP comparing a MapEval-style ReAct baseline with a
Spatial-Agent (GeoFlow) port on Korean multiple-choice map questions.

The independent variable is the agent architecture, including its tool surface. ReAct and
Spatial-Agent intentionally have different tools; everything below the tools—provider behavior,
cache, normalized schemas, name resolution, and evaluation—must remain equivalent.

Treat code and tests as the source of truth for current behavior:

- [`main.py`](main.py) and [`src/config.py`](src/config.py): CLI and runtime defaults.
- [`README.md`](README.md): project overview and basic usage.
- [`K-MapEval_PRD.md`](K-MapEval_PRD.md): original research scope and design background.
- [`docs/REFERENCE_MAPPING.md`](docs/REFERENCE_MAPPING.md): upstream mapping, deliberate
  deviations, and benchmark history.

## Architecture boundaries

```text
main.py -> Evaluator -> ReactAgent | SpatialAgent -> ToolRegistry -> MapProvider
                         SpatialAgent also -> SpatialOperatorRegistry
```

- Inject providers through `src/tools/map.py`; never construct them inside an agent.
- Keep Kakao HTTP calls in `src/tools/kakao.py` only.
- Expose only normalized `Place` and `Route` models from `src/models.py`, never raw provider JSON.
- Keep deterministic spatial and temporal computation in `src/tools/spatial.py`; it must not make
  API calls.
- Keep `main.py` as the runtime entry point. Files under `data/` are offline dataset tooling and
  may import `src/`; runtime code under `src/` must not import them.
- Do not introduce shared mutable agent state. Concurrent workers own independent clients,
  providers, registries, and agents.

## Experiment invariants

- ReAct's default surface is `--react-tools reference`: the five names *and* `mapeval-api`'s
  argument contracts — `PlaceSearch(placeName)` returning one id, `NearbyPlaces` refusing a radius
  when it ranks by distance, `Directions`/`TravelTime` taking an origin, a destination and a mode
  and nothing else. Restricting the names alone is not restricting the surface: an argument is a
  capability. `native` (formerly `mapeval`) is the same five names with this registry's richer
  arguments and is a stronger-than-paper ablation. Pin any change to both contracts' argument
  sets in `tests/test_tools_and_agents.py`.
- ReAct's five tool names are exactly `place_search`, `place_details`, `nearby_places`,
  `travel_time`, and `directions`. Adding a `ToolRegistry` tool makes it available to
  Spatial-Agent, not automatically to ReAct. Add a baseline tool only with upstream evidence and
  update the pinned tests.
- The ReAct *loop* travels with the surface: `reference` runs upstream's — 15 iterations
  (langchain's default, which `Evaluator2.py` never overrides), one action per iteration, and a
  forced stop that carries no answer. Do not raise the budget, execute parallel tool calls, or add
  a final "answer now" call under `reference`; each is a capability the paper's baseline lacks.
  `native` keeps all three and is an ablation.
- Report metadata must carry `llm_temperature`, `react_max_iterations`,
  `react_parallel_tool_calls` and `react_forces_final_answer`. An accuracy without them is not
  comparable to anything.
- `--react-tools full` is an ablation. Never pool it with the default `mapeval` surface, and do not
  tune ReAct's prompt, tool contracts, or step budget in response to benchmark misses.
- Select one evidence mode per run: `context`, `hybrid`, or `kakao`. Only `hybrid` may use its
  defined context-to-Kakao fallback; do not create ad hoc agent-specific evidence paths.
- Build context evidence once from the full dataset corpus. Context belongs behind the provider,
  never in the agent prompt; do not scope it per question or replay a pre-ranked context block as
  an answer.
- `BenchmarkItem.agent_input()` exposes only `(question, options)`. Never pass `answer`,
  `classification`, `region`, `difficulty`, `verified_at`, `gold_evidence`, or other eval-only
  metadata to an agent or derive provider configuration from it.
- Use 0-based option indices in datasets, predictions, logs, and reports. The answer wire format is
  `^^N^^`.
- Never special-case a question ID or option string, and never hardcode an answer.
- Missing or weak evidence must fail explicitly. Do not invent measurements, silently choose the
  least-bad match, or collapse distinct `ProviderError` subclasses into agent reasoning failures.
- Route map access through the registry/provider so tool calls, API calls, and cache deltas remain
  measurable.
- Do not pool results across datasets, provider modes, or ReAct tool surfaces. Label reports as
  prompting-only, and include the no-tool floor when interpreting accuracy.
- Decode at `temperature=0`, which is what both upstreams do. Sending no temperature leaves the
  endpoint's default in force, and two floor runs over one benchmark then differed by 11 points.
- The LLM endpoint is not reproducible even greedily — no sampling parameter fixes it, and a
  100-question run carries a spread of about ±8 points. Run each configuration several times and
  read any difference against that spread; a single-run comparison of two architectures is not a
  result. See `docs/REFERENCE_MAPPING.md`.
- Run `python data/audit_dataset.py <dataset>` after every build and before the floor. It exits
  non-zero on a second answer key: a gold sitting at a fixed rank once the options are sorted, an
  option that can never be the answer, a gold whose text appears in its own question, duplicate
  options. Every one of those shipped in a benchmark here before the script existed.
- A numeric option set must not put the gold at a fixed rank once the options are sorted. Use
  `straddling_multipliers`; a fixed multiplier tuple is a second answer key.
- Every rung a question offers has to be reachable *and* reached. A family whose options are a
  fixed ladder must be able to answer each rung, and should spend its rows across them the way
  `trip_feasible_count` does — keying the answer on a loop index spends them wherever the loop
  happened to succeed.
- Every benchmark in `dataset/` has been tuned against, so an accuracy on one is a training-set
  accuracy. Build a held-out set with `--seed`/`--id-prefix` on a builder, change nothing under
  `src/` afterwards, and report that number separately. The reference point is upstream
  Spatial-Agent's own 71.07% on MapEval-API, which `docs/REFERENCE_MAPPING.md` records together
  with the configuration it was measured in.

## Benchmarks

- `dataset/seoul_kmapeval_v6_mcq_100.jsonl`: v5's families each raised one step (composition or
  ordinality) and the radius family's word order fixed. Built by
  `data/build_mapeval_v6_benchmark.py`; passes `data/audit_dataset.py`. **Not yet measured** —
  no agent run and no no-tool floor, so it has no accuracy to quote.
- `dataset/seoul_kmapeval_v5_mcq_100.jsonl`: v4's method at MapEval-API's own difficulty (tight
  options over reproducible measures, ordinal and membership `nearby`, subjective `unanswerable`,
  `trip_optimal_order`). Built by `data/build_mapeval_v5_benchmark.py`.
- `dataset/seoul_kmapeval_v5h_holdout_100.jsonl`: the same builder under seed 613829, 99 rows, no
  question and almost no place in common with v5. **Held out** — nothing under `src/` has been
  tuned against it, and it is the only set here whose accuracy is not also a training-set
  accuracy. Keep it that way: if a run on it exposes an agent bug, fix the bug against v5 and
  rebuild the holdout under another seed before quoting it again.
- `dataset/seoul_kmapeval_v4_mcq_100.jsonl`: MapEval-method reproduction benchmark.
- `dataset/seoul_kmapeval_v3_mcq_100.jsonl`: compositional architecture benchmark.
- `dataset/seoul_mapeval_v1_mcq_100.jsonl`: context/hybrid evidence benchmark.
- `dataset/seoul_kmapeval_v2_mcq_100.jsonl`: superseded benchmark retained for historical runs.

Use `python main.py --help` for current CLI defaults instead of copying them into documentation.

## Setup and checks

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp example.env .env

pytest
ruff check .
```

Tests must mock Kakao and the LLM; ordinary tests require no keys or network. Run relevant focused
tests while developing, then the full test and lint suites before finishing a code change.

Every `main.py` run consumes LLM tokens, and `kakao`/`hybrid` runs may consume Kakao quota.
Dataset builders, verifiers, and the no-tool floor may also use live services. Run any of them only
when the user explicitly asks for live execution.

## Change checklist

- Add regression tests for changed behavior. Keep live-API tests separate and optional.
- When changing `Place` or `Route`, update every provider normalizer, cached payload, and cache
  schema version.
- When adding or renaming a Spatial-Agent operator, update its implementation,
  `OPERATOR_CONTRACTS`/input types, `GRAPH_PROMPT`, argument normalization, and composition tests.
- Preserve explicit provider-versus-agent failure types and per-question metrics/log fields.
- Update `docs/REFERENCE_MAPPING.md` whenever behavior deliberately diverges from MapEval or
  Spatial-Agent upstream.
- Never persist API keys or raw Kakao responses.
