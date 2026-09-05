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
  deviations.
- [`docs/EXPERIMENT_HISTORY.md`](docs/EXPERIMENT_HISTORY.md): past measurements and their
  revision-specific interpretation; read when comparing results or investigating past decisions.

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

## Scope, authorization, and completion

- Continue authorized, reversible project work without re-confirming routine implementation
  choices. Resolve questions from the request, code, tests, and available records first. Ask only
  when missing information materially changes correctness, scope, or a required authorization;
  continue independent work while that question is pending.
- An execution that sends requests to the LLM or Kakao requires the user's explicit request for
  live execution. This includes live evaluation, dataset generation/verification, and the no-tool
  floor. That authorization persists within the requested execution scope; do not ask again for
  the same authorized run. A request to fix code alone does not authorize live execution.
- Reading code, `--help`, mocked tests, and offline audit/replay require no live-execution approval.
  If a command's behavior is unclear, inspect its execution path before running it. Never infer
  permission for network calls from the script's name or from its role in verification.
- Complete implementation and applicable offline checks before reporting a live-validation
  blocker. Distinguish completed code/documentation work from unperformed live validation;
  do not claim an unverified research result or a fully validated behavior change.
- Historical advice in EXPERIMENT_HISTORY.md is not an additional approval or completion gate.
  These rules do not expand external-action permissions or override an explicit user constraint.

## Experiment invariants

- ReAct's default surface is `--react-tools reference`: the five names and `mapeval-api`'s
  argument contracts. `PlaceSearch(placeName)` returns one id; `NearbyPlaces` refuses a radius
  when ranking by distance; `Directions`/`TravelTime` take only origin, destination, and mode.
  `native` (historical alias `mapeval`) exposes the same names with richer arguments and is an
  ablation. Pin both contracts' argument sets in `tests/test_tools_and_agents.py`.
- ReAct's five names are exactly `place_search`, `place_details`, `nearby_places`, `travel_time`,
  and `directions`. Adding a ToolRegistry tool exposes it to Spatial-Agent, not automatically
  to ReAct. Add a baseline tool only with upstream evidence and updated pinned tests.
- The `reference` loop executes one action per iteration and stops without an answer when its
  budget ends. Do not add parallel calls or a final "answer now" call. `native` retains those
  capabilities as an ablation. Keep `reference`, `native`, and `full` results separate.
- Do not tune ReAct's prompt, tool contracts, or step budget in response to benchmark misses.
  Shared provider/contract changes need provider or upstream justification, not accuracy alone.
- A ReAct step is one loop iteration; a Spatial-Agent step is one authored GeoFlow transformation
  edge. `REACT_MAX_STEPS` and `SPATIAL_MAX_STEPS` independently fall back to
  `MAX_REASONING_STEPS` when unset. Spatial-Agent has no upstream numeric graph budget.
- The configuration of record is `REACT_MAX_STEPS=15`, `SPATIAL_MAX_STEPS=30`, and
  `--concurrency 32`. ReAct's 15 comes from upstream's langchain default. Label other settings
  as ablations, preserve their actual metadata, and do not pool them with the reference condition.
  Historical runs retain the settings they actually used; do not relabel them with today's values.
- Report metadata must carry `llm_temperature`, `max_reasoning_steps`, `react_max_steps`,
  `spatial_max_steps`, `react_parallel_tool_calls`, `react_forces_final_answer`, and concurrency.
  State both architecture budgets beside accuracy and check metadata before comparing runs.
- Record `llm_calls`, `prompt_tokens`, `completion_tokens`, `total_tokens`, `reasoning_tokens`,
  and `reasoning_chars` in each question's log, report row, and run statistics. `reasoning_tokens`
  is server-reported only: leave it null if absent and sum it only when every row has a value.
- Every runtime evaluation uses KakaoMapProvider. There is no provider selector or context fallback.
  Legacy dataset `context` fields are metadata only, never evidence or agent input.
- `BenchmarkItem.agent_input()` exposes only `(question, options)`. Never pass `answer`,
  `classification`, `region`, `difficulty`, `verified_at`, `gold_evidence`, or other eval-only
  metadata to an agent or derive provider configuration from it.
- Use 0-based option indices throughout datasets, predictions, logs, and reports; the wire format
  is `^^N^^`. Never special-case a question ID or option string or hardcode an answer.
- Report all three independent labels: `mapeval_class`, `classification`, and `template_id`.
  `resolve_mapeval_class` reads the stored field before its legacy classification fallback.
  Report `unanswerable` separately from the paper's four categories; a mean over five is not a
  mean over four. Dataset versions v1–v3 predate the stored `mapeval_class` field.
- Decode at `temperature=0`. Send no `max_tokens`; the output ceiling belongs to vLLM deployment.
  Check `llm_output_truncated_count` before interpreting accuracy. Truncated outputs are
  `llm_output_truncated`, and prompts exceeding the window are `llm_context_overflow`;
  neither is retried or reported as `answer_parse_failure`. Preserve their costs and report counts.
- A loop exhausting its budget without answering is `iteration_limit`, not `answer_parse_failure`.
- Missing or weak runtime evidence must fail explicitly. Do not invent measurements, silently
  choose the least-bad match, or collapse distinct ProviderError subclasses into reasoning failures.
- Route map access through the registry/provider so tool/API calls and cache deltas are measurable.
- Label reports prompting-only. Do not pool datasets or ReAct surfaces. Include each draw's own
  no-tool floor and keep `unanswerable` inclusion explicit when interpreting accuracy.
- Research comparisons require repeated passes per configuration and interpretation against their
  observed spread; one pass is not a comparative result. Family/class claims need independent
  draws or explicit qualification as one draw's result. A diagnostic run does not establish a claim.
- Held-out status belongs to the dataset, measured revision, and use of results in development.
  For held-out evaluation, freeze the evaluated source before building/running the new draw with
  `--seed`/`--id-prefix`. If results guide implementation or configuration changes, use a fresh
  draw for subsequent held-out validation. Preserve earlier results as measurements of their
  original revision; use unknown status when provenance is insufficient. Do not infer status
  from a filename or treat all files in `dataset/` alike.

## Verification by change scope

- For changes to answer generation, grounding, execution semantics, provider behavior, or dataset
  distribution, measure the family footprint before claiming the change is validated. A rule can
  affect one family disproportionately without naming that family in code. Record the footprint
  beside affected family results and justify capabilities added to only one architecture as part
  of that architecture, including abstention, name repair, or retries.
- For deterministic factorization/grounding changes, use `data/replay_grounding.py` on the same
  recorded planner graphs, Analysis output, questions, and options at both revisions. Diff by
  family/class without LLM calls or Kakao quota. Check the imported source/revision provenance
  (including PYTHONPATH for isolated checkouts) so both sides do not accidentally use one revision.
- Replay cannot measure adoption of new planner vocabulary or other unrecorded LLM behavior.
  Use appropriate tests and state that limitation. Any necessary live validation still requires
  the explicit authorization above; finish independent work and report pending validation if absent.
- Documentation/formatting changes that do not affect evaluation behavior need no family replay
  or live benchmark. Verify their content, references, and diff instead.

## Dataset construction

- `data/build_kmapeval_dataset.py` is the standard builder for new sets, using v7's families,
  `--count`, and a clock seed by default. Versioned builders reproduce their historical sets;
  use v6's builder only to reproduce v6. Do not overwrite existing datasets without an intentional
  replacement request; builders require `--force` for an existing destination.
- `--count N` must produce exactly N questions and preserve the upstream class mix: nearby 83,
  poi 64, routing 66, trip 67, unanswerable 20 per 300, subject to integer apportionment.
  Candidate scans use `candidate_groups`/`_scan_limit`, not fixed bounds. Freeze a short family's
  achieved count and redistribute within its class. Fail non-zero rather than write a short set.
- Generated rows have no `context` field. Only legacy `dataset/seoul_mapeval_v1_mcq_100.jsonl`
  retains its context for provenance; runtime ignores it.
- Run `python data/audit_dataset.py <dataset>` after every build, at the actual build size,
  and before the floor. Check for fixed gold rank, unreachable/unused options, gold text in the
  question, duplicate options, context fields, and ordinal imbalance. Historical audit claims
  do not replace the current audit; identify affected families when interpreting older results.
- Numeric options use `straddling_multipliers`; a fixed multiplier tuple must not reveal gold rank.
- Every offered ladder rung must be reachable and represented as required by the audit. Ensure the
  candidate pool supplies each parameter value and the search grows with count. Inspect recorded
  `gold_evidence.ranked_m` supply before blaming the balancer. Preserve `_scarcest_ordinal`'s
  tie-break toward the scarcer higher ordinal for nested feasibility constraints.
- Check that families can be answered on each architecture's contract within its own recorded
  budget. Count calls/edges before shipping a family; infeasibility measures the budget. Do not
  raise ReAct's reference budget to repair a benchmark miss. Document dataset-design changes
  and distinguish them from architecture ablations.

## Spatial semantics and maintenance

- A route through waypoints states `via`; itinerary totals state `SELECT_LEGS` over a matrix.
  Never infer waypoints or driven legs from input position. Wiring must account for every input.
- Grounding binds stated order (`fixed_order`), objective (`metric`), and homecoming
  (`return_to_start`), as it binds stays and budgets. Check returned values, including distance
  versus duration matrices, not merely successful operator execution.
- Grounding takes GroundingFacts and tests for the facts present in the question. Do not
  reintroduce intent gates or the removed intent paths in retrieval/evaluation. Historical intent
  labels are not runtime requirements.
- The OPERATOR_INPUT_TYPES table must reflect the operator implementation. Keep output-type,
  role-ordering, and statically knowable argument-value checks in repair, but skip these local
  checks on the last `strict_types=False` attempt. Formal constraints, including data availability,
  remain mandatory on lenient attempts. Preserve partial execution and explicit step failures.
- When adding or renaming an operator, update its implementation, OPERATOR_CONTRACTS/input types,
  GRAPH_PROMPT, argument normalization, and composition tests.
- When changing Place or Route, update every provider normalizer, cached payload, and cache schema
  version. Preserve explicit provider-versus-agent failures and per-question metrics/log fields.
- Retry limits include LLM_RETRY_TIME_BUDGET_SECONDS, not attempt count alone. This does not
  permit retries for truncation or context overflow.
- Agents create traces with `self.new_trace()` and append entries so the evaluator logs them live.
- Keep `lint.dummy-variable-rgx` at `^_$` so ruff detects redefined underscore-prefixed helpers;
  intentionally discarded values must be named `_`, not `_something`.
- Update docs/REFERENCE_MAPPING.md when behavior deliberately diverges from upstream. Record
  measurements and past failure narratives in docs/EXPERIMENT_HISTORY.md, linking report evidence;
  do not accumulate experimental results in this instruction file.
- Never persist API keys or raw Kakao responses.

## Setup and checks

Python 3.11 or newer is required. Use `python main.py --help` for current CLI defaults.
For initial setup (preserve an existing environment and .env):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
# Only when .env does not already exist:
cp .env.example .env
```

Tests must mock Kakao and the LLM; ordinary tests require no keys or network. Add regression tests
for changed runtime behavior. Run relevant focused tests while developing, then the full test and
lint suites before finishing a code change. Keep live-API tests separate and optional.

```bash
pytest
ruff check .
```
