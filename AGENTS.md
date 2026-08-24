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
- The ReAct *loop* travels with the surface: `reference` runs upstream's — one action per
  iteration and a forced stop that carries no answer. Do not execute parallel tool calls or add a
  final "answer now" call under `reference`; each is a capability the paper's baseline lacks.
  `native` keeps both and is an ablation.
- `MAX_REASONING_STEPS` is the one step budget, and every architecture answers under it: ReAct
  loop iterations, Spatial-Agent graph nodes. It defaults to 15 because that is langchain's own
  default and therefore the reference baseline's. Raising it is an ablation and has to be
  reported; do not raise it in response to a benchmark miss.
- Report metadata must carry `llm_temperature`, `max_reasoning_steps`,
  `react_parallel_tool_calls` and `react_forces_final_answer`. An accuracy without them is not
  comparable to anything.
- Every question records what it cost: `llm_calls`, `prompt_tokens`, `completion_tokens`,
  `total_tokens`, `reasoning_tokens`, `reasoning_chars`, in the log line, the report row and the
  run statistics. `reasoning_tokens` is only ever what the server reported — leave it null when it
  reports nothing rather than estimating it, and never sum it unless every row has one.
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
- A loop that used its whole step budget without answering is `iteration_limit`, not
  `answer_parse_failure`: it is still a miss, exactly as upstream counts it, but the report has to
  say the budget ended it so a family that cannot be answered within the budget is visible.
- Send no `max_tokens`: the output ceiling belongs to the vLLM deployment, which is also what both
  upstreams do. It has to sit above the architectures' working range and below a reasoning spiral:
  measured here, a ReAct call writes ~340 completion tokens and a Spatial-Agent planner call ~3,600
  (max 5,854), while a spiral runs past 60,000. A 5,100-token ceiling scored Spatial-Agent 10/100
  by truncating 67 of its questions. Check `llm_output_truncated_count` before reading an accuracy. A completion it cut off is `llm_output_truncated`, never `answer_parse_failure` --
  the question was not answered badly, it was not allowed to finish. Do not retry it, and keep its
  tokens in the question's cost. A prompt that outgrew the window before the model could start is
  `llm_context_overflow`, also not retried; both counts belong beside the accuracy.
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
- Passes are not draws, and a family needs both. Three passes over ~282 rows pin an *overall*
  accuracy to about ±2, and two such draws agreed to within 3.2 points — but between those same
  two draws `trip_total_distance` moved ReAct 58.7 points and the `trip` class changed which
  architecture won it. So a family or class number belongs to the draw it was measured on: quote
  it across draws, or quote it as one draw's. Read it as lift over that draw's own floor, because
  a redraw moves the floor too.
- Run `python data/audit_dataset.py <dataset>` after every build and before the floor. It exits
  non-zero on a second answer key: a gold sitting at a fixed rank once the options are sorted, an
  option that can never be the answer, a gold whose text appears in its own question, duplicate
  options. Every one of those shipped in a benchmark here before the script existed.
- A numeric option set must not put the gold at a fixed rank once the options are sorted. Use
  `straddling_multipliers`; a fixed multiplier tuple is a second answer key.
- A family also has to be answerable *within the step budget*, on the tool contract the baseline
  runs. Count the calls before shipping one: v6's four-stop trip families need about twenty
  one-leg `Directions` calls against langchain's fifteen iterations. Over four passes at that
  budget ReAct scored 14/60 on them with 24 of those 60 rows stopped by `iteration_limit`. A
  three-pass ablation at budget 30 says the budget binds one of the two families and not the
  other: `trip_optimal_order_four` goes 2/24 to 21/24, `trip_total_distance_four` 6/21 to 5/21,
  and ReAct overall 39.0% to 47.3% with no `iteration_limit` left. Raising it is still the wrong
  repair: 15 is langchain's default and therefore the baseline's, and it is *one* budget — the
  same pass moved Spatial-Agent 69.0% to 74.0%, so it is not a controlled comparison of either
  architecture. Shrink the family instead, which is what v7 did.
- Every rung a question offers has to be reachable *and* reached. A family whose options are a
  fixed ladder must be able to answer each rung, and should spend its rows across them the way
  `trip_feasible_count` does — keying the answer on a loop index spends them wherever the loop
  happened to succeed. Drawing the parameter is only half of it: the scan that hunts for the scarce
  values has to be bounded by something that grows with `count`, or the balancing runs out of
  candidates the moment the build gets bigger than the constant was sized for. That is how
  `nearby_kth_nearest` shipped 19 of 24 rows at k=2 in the 283-row set with correct balancing code.
  Re-run `data/audit_dataset.py` at the size you are actually building.
- Every benchmark in `dataset/` has been tuned against, so an accuracy on one is a training-set
  accuracy. Build a held-out set with `--seed`/`--id-prefix` on a builder, change nothing under
  `src/` afterwards, and report that number separately. The reference point is upstream
  Spatial-Agent's own 71.07% on MapEval-API, which `docs/REFERENCE_MAPPING.md` records together
  with the configuration it was measured in.

## Benchmarks

- `dataset/seoul_kmapeval_v7c_mcq_300.jsonl`: the fourth 300-question draw, built and run at
  `a50096a` (both operator fixes in). 282 rows. Floor 25.2 (19.3 excluding `unanswerable_*`), ReAct
  52.0, Spatial-Agent 81.3 over three passes a side — the widest lift of the four draws, gap 29.3.
  **The first of these that stays held out**: nothing under `src/` or `data/` has changed since it
  was drawn, so 52.0/81.3 is a genuine held-out number for `a50096a`. Spatial-Agent had **zero
  `agent_reasoning_failure`** this draw — the two operator fixes measured on a set neither was
  tuned against. Caveat: it **fails `data/audit_dataset.py`** on `nearby_kth_nearest` (20/24 at
  k=2, city-limited draw variance, not a generator bug), so that one family's number is not
  quotable; the overall and every other family stand. See `docs/REFERENCE_MAPPING.md`.
- `dataset/seoul_kmapeval_v7b_mcq_300.jsonl`: the standard builder's third 300-question draw; it
  drew 283. Built and run at `796c683`. Floor 29.5 (24.2 excluding `unanswerable_*`), ReAct 48.5,
  Spatial-Agent 76.8 over three passes a side — a third draw agreeing the gap sits at 27–30. Also
  the first draw run after the argument-spelling fix: "missing arguments" refusals fell from ~31
  per 848 Spatial-Agent runs to 5 per 849, and repair rounds from 76 to 55, with no accuracy
  change — cost fell, not correctness, which is what a vocabulary fix should do. **Spent**: the
  run surfaced a second crash (`dict()` on a bare `"$ref"` argument, fixed at `1cb6bdc`) that
  postdates it, so 48.5/76.8 is what `796c683` scored. See `docs/REFERENCE_MAPPING.md`.
- `dataset/seoul_kmapeval_v7a_mcq_300.jsonl`: the standard builder asked for 300 a second time; it
  drew 282, and it is the first set at this size `data/audit_dataset.py` passes. Built and run at
  `ba92d9c`, six questions in common with the 283-row draw. Floor 26.8 (21.3 excluding the
  `unanswerable_*` families), ReAct 52.1, Spatial-Agent 79.2 over three passes a side. **Spent**:
  the run exposed the argument-spelling refusals and `src/` changed to close them, so those
  numbers belong to `ba92d9c`. Two draws at this size say overall
  accuracy is stable to ~3 points and *family* accuracy is not — `trip_total_distance` moved
  ReAct 58.7 points between them — so quote family and class numbers only across draws.
- `dataset/seoul_kmapeval_v7_mcq_300.jsonl`: the standard builder asked for 300; it drew 283. The
  largest set here and the one whose numbers carry the least sampling noise — three passes a side
  at `6bae55c` spanned 2.1 points for ReAct and 1.8 for Spatial-Agent, against the ±8 a hundred
  rows show. Floor 28.8 (23.5 excluding the `unanswerable_*` families, which are guessable by
  design), ReAct 48.9, Spatial-Agent 78.9, zero `iteration_limit` and zero truncation. **Spent**:
  `data/audit_dataset.py` failed it on `nearby_kth_nearest` — 19 of 24 rows at k=2, because a
  coverage scan limit that did not grow with the build stopped balancing — and `data/` changed to
  fix that, so 48.9/78.9 belongs to `6bae55c`. The 24 rows are not a second answer key and the
  accuracies stand; see `docs/REFERENCE_MAPPING.md`.
- `dataset/seoul_kmapeval_v7_mcq_100.jsonl`: v6 with its two four-stop trip families walked back
  to three stops, because at four the reference baseline runs out of iterations before it can
  finish one. Built by `data/build_mapeval_v7_benchmark.py`; passes `data/audit_dataset.py`. Shares
  a generator with v6 but only 18 rows: the draws are live and the cache had expired.
- `dataset/seoul_kmapeval_v7h3_holdout_100.jsonl`: the v7 builder under seed 750914, `v7h3` ids.
  **Held out** — built and run at `8797217`, the first draw for code carrying the arithmetic
  operators, the ordinal template and a drawn ordinal. Three passes against a floor of 23.5: ReAct
  51.0, Spatial-Agent 72.0. This is the only holdout number that belongs to the current code, and
  it is spent the moment `src/` or `data/` changes again.
- `dataset/seoul_kmapeval_v7h2_holdout_100.jsonl`: seed 481203. Spent — `src/` changed in response
  to what it showed. ReAct 45.7, Spatial-Agent 72.3 at `38566f3`, floor 25.5.
- `dataset/seoul_kmapeval_v7h_holdout_100.jsonl`: the v7 builder under seed 927451, `v7h` ids, one
  question and 30 of 236 place names in common with v7. Held out at `0aabaa9` and measured there
  over three passes against a floor of 29.5 — ReAct 48.0, Spatial-Agent 70.7. **That number is
  now spent**: the run exposed the missing arithmetic operators, `src/` changed to close them,
  and a held-out set stops being held out the moment it is answered against. Rebuild under a new
  seed before quoting a holdout again, and quote 48.0/70.7 only as what the code at `0aabaa9`
  scored.
- `dataset/seoul_kmapeval_v6_mcq_100.jsonl`: v5's families each raised one step (composition or
  ordinality) and the radius family's word order fixed. Built by
  `data/build_mapeval_v6_benchmark.py`; passes `data/audit_dataset.py`. Measured once (ReAct
  54/100, Spatial-Agent 60/100), but its two four-stop trip families spend most of the reference
  baseline's step budget — ReAct 14/60 over four passes, 24 rows stopped by it — so quote v7
  instead. No no-tool floor.
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

`data/build_kmapeval_dataset.py` is the standard builder for *new* sets: **v7's families**,
`--count` for how many questions, and a clock seed so every run is a fresh draw. v7's method is the
standard because it is the one whose every family the reference baseline can finish inside its own
step budget — v6's two four-stop trip families cannot, and a family that cannot be finished within
the paper's budget measures the budget. The versioned builders
(`build_mapeval_v5/v6/v7_benchmark.py`) exist to reproduce their benchmarks of record and default
to the seed that does; v5 and v7 take `--count` and a clock seed too, and refuse to overwrite an
existing file without `--force`. Use v6's builder only to reproduce v6.

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
- The declared-type table in `OPERATOR_INPUT_TYPES` describes what an operator's implementation
  accepts; when the two disagree the implementation is right, and a plan the executor could have
  run must never be refused for it. Output-type compatibility and role ordering are this port's
  own rules — upstream has neither — so they inform the repair round and are skipped
  (`strict_types=False`) on the last attempt before a question is given up on.
- When adding or renaming a Spatial-Agent operator, update its implementation,
  `OPERATOR_CONTRACTS`/input types, `GRAPH_PROMPT`, argument normalization, and composition tests.
- Retrying is bounded by `LLM_RETRY_TIME_BUDGET_SECONDS`, not by the attempt count alone. A
  request the gateway kills for running too long costs its whole timeout every attempt, and
  re-asking does not make it shorter.
- Preserve explicit provider-versus-agent failure types and per-question metrics/log fields.
- Agents build their trace with `self.new_trace()` and only ever append to it: the evaluator
  listens on that append and writes each entry as it happens. Building a trace some other way
  still logs, but only once the question is over.
- Update `docs/REFERENCE_MAPPING.md` whenever behavior deliberately diverges from MapEval or
  Spatial-Agent upstream.
- Never persist API keys or raw Kakao responses.
