# AGENT.md

Canonical working guidance for coding agents in this repository. `CLAUDE.md` imports this file;
edit here, not there.

## What this is

A research MVP that compares a MapEval-style **ReAct** agent against a **Spatial-Agent** (GeoFlow)
port on Korean multiple-choice map questions, over one shared evidence source. The research
question is whether Spatial-Agent's reported gains reproduce on Korean geography and POI data.

**The independent variable is the architecture, and an agent's tool surface is part of its
architecture.** The two agents therefore get *different* tools: ReAct gets the five primitives
`mapeval-api/Evaluator2.py` gives its baseline, Spatial-Agent gets this registry's aggregations
plus `SpatialOperatorRegistry`. That is the arrangement upstream has — `spatial-agent/src/tools/
google_maps.py` carries `get_distance_matrix`, and MapEval's baseline is never handed anything
like it. What must stay identical is everything *below* the tools: provider, cache, normalized
schemas, region prior, name resolution, evaluator. An agent may reach fewer tools than the other;
it must never see different evidence through the ones it reaches.

The tool layer has three interchangeable evidence sources, chosen per run and never mixed:

- **kakao** (`KakaoMapProvider`) — live Kakao Local / Kakao Mobility, with the SQLite cache and the
  region prior. **This is the reproduction setting**, the analogue of the paper's live Google Maps,
  and the one `dataset/seoul_kmapeval_v2_mcq_100.jsonl` is graded against.
- **context** (`ContextMapProvider`) — one corpus built from every context the dataset carries, in
  MapEval's context format, serving the tools instead of any API. This is the port of upstream
  Spatial-Agent's local context cache, and it needs a dataset whose rows carry a `context`.
- **hybrid** — that corpus with `KakaoMapProvider` behind it for what it does not hold. This is
  upstream's own arrangement (cache first, Google Maps on a miss) with Kakao in Google's place.
`K-MapEval_PRD.md` is the full spec. `docs/REFERENCE_MAPPING.md` records every deliberate deviation
from the upstream MapEval / Spatial-Agent implementations — update it when you add another one.

Upstream references, when porting or checking behavior:

- `MapEval/MapEval-API` — the ReAct baseline. `Evaluator2.py` for the agent loop and MCQ
  evaluation, `FormattedTools.py` / `Tools.py` for tool interfaces and formatting.
- `ecerybao/Spatial-Agent` — the Spatial-Agent side. `src/agent/spatial_agent.py`,
  `src/agent/operators.py`, `src/tools/google_maps.py`, `test_agent.py`.
- `MapQaTor/mapqator-backend` — architectural reference only; do not reproduce its HTTP backend.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
cp example.env .env          # .env.example has the same content; either works

pytest                       # full suite; Kakao + LLM are stubbed, no keys/network needed
pytest tests/test_tools_and_agents.py::test_react_executes_common_tool_then_parses_answer
pytest -k geoflow
ruff check .                 # line-length 100, rules E,F,I,UP,B

# Does the architecture separate the two agents? The compositional benchmark answers that.
python main.py --agent both --dataset dataset/seoul_kmapeval_v3_mcq_100.jsonl

# ReAct runs on MapEval's five primitives by default. `full` is the ablation, not the benchmark:
# it shares this registry's aggregations with the baseline and answers a different question.
python main.py --agent react --react-tools full

# The reproduction run: the Kakao-grounded benchmark against live Kakao, both architectures.
python main.py --agent both                   # dataset/seoul_kmapeval_v2_mcq_100.jsonl, kakao evidence
python main.py --agent react
python main.py --agent both --ids seoul_kmapeval_v2_000 seoul_kmapeval_v2_024

# The context-cache benchmark, which needs the dataset whose rows ship a context.
python main.py --agent both --dataset dataset/seoul_mapeval_v1_mcq_100.jsonl --provider context
python main.py --agent both --dataset dataset/seoul_mapeval_v1_mcq_100.jsonl --provider hybrid

# Rebuild and re-verify the Kakao benchmark (offline tooling, costs Kakao quota).
PYTHONPATH=data python data/build_pool.py
PYTHONPATH=data python data/build_benchmark.py
PYTHONPATH=data python data/verify_benchmark.py
PYTHONPATH=data python data/build_hard_benchmark.py     # the compositional benchmark
PYTHONPATH=data python data/verify_hard_benchmark.py

python main.py --agent spatial --concurrency 4
```

Running `main.py` costs real LLM tokens, and `--provider kakao` costs Kakao API quota on top. Every
test in `tests/` fakes both, so verify with `pytest` first and only run the benchmark when the user
asks for live numbers. Keep unit tests mocked (`httpx.MockTransport` for Kakao, a queued fake for
the LLM); any live-API test stays separate and optional.

## Layering (do not shortcut it)

```
main.py → Evaluator → ReactAgent | SpatialAgent → ToolRegistry → MapProvider
                                                    ├→ ContextMapProvider → the row's own context
                                                    └→ KakaoMapProvider → SQLiteMapCache → Kakao
Spatial-Agent additionally → SpatialOperatorRegistry (pure local computation, zero API calls)
```

- Kakao HTTP calls exist **only** in `src/tools/kakao.py` — Kakao Local for place/category/address/
  coordinate queries, Kakao Mobility for driving directions. Agents, evaluator, dataset, and tests
  talk to the abstract `MapProvider` in `src/tools/map.py`, and providers are injected, never
  constructed inside an agent.
- Raw Kakao JSON never reaches an agent. Everything is normalized to the frozen `Place` / `Route`
  models in `src/models.py`; adding a field there means adding it to the provider normalizers and
  the cache payload.
- Deterministic math (haversine, bearing, sorting, min/max, route comparison, TSP-TW, time
  arithmetic) belongs in `src/tools/spatial.py` and must never spend an API call.
- **The two agents get different tool surfaces, and that is the experiment, not a violation of
  it.** ReAct is constructed with `allowed=ToolRegistry.MAPEVAL_BASELINE_TOOLS` — the five tools
  `mapeval-api/Evaluator2.py` (35d481a, line 33) instantiates: PlaceSearch, PlaceDetails,
  NearbyPlaces, TravelTime, Directions. Read `Evaluator2.py`, not `Tools.py`, when you need that
  list — `Tools.py` also defines a `PlaceIdTool` the evaluator never constructs, and
  `FormattedTools.PlaceSearchTool` is itself documented as "Get place ID for a given location", so
  the two are one primitive under two names rather than a sixth tool. Everything this registry
  adds beyond those five — `batch_geocode`, `batch_place_details`, `distance_matrix`,
  `calculate_finish_time`, `recover_option_places` — is an *aggregation* over them, which is
  exactly what GeoFlow's operator graph exists to express, and every one of them was added by a
  commit whose subject names Spatial-Agent. `geocode` / `reverse_geocode` are withheld too:
  upstream reaches every place through a place id and converts between an address and coordinates
  nowhere, and in a measured run `geocode` was resolving bare place names — PlaceSearch again,
  through a second index.
- **Adding a tool to `ToolRegistry` gives it to Spatial-Agent only.** Putting one in
  `MAPEVAL_BASELINE_TOOLS` is a claim that `Evaluator2.py` constructs its counterpart; make that
  claim only with the upstream line in front of you. `tests/test_tools_and_agents.py` pins the
  set.
- **`--react-tools full` is an ablation, not the benchmark.** It restores the shared surface to
  ask whether the graph adds anything on top of strong aggregation tools. That is a different
  question from the paper's, `metadata.react_tools` records which was asked, and runs from the two
  surfaces must never be pooled. Under `full`, ReAct reached an aggregation tool on 79 of 100
  compositional questions and answered `trip_finish_time` 16/16 and `trip_latest_departure` 14/14
  — with `batch_geocode` doing the joint, anchor-relative name resolution that PlaceSearch has no
  equivalent of. Every report in `reports/` predating this default is a `full` run.
- **Place disambiguation is anchor-relative, and it lives below the tool surface.** It is how
  the provider resolves a name at all, so it reaches whichever agent calls `place_search` and
  is not capability handed to one of them. Korean POI names repeat
  across branches and cities, so `_best_place_match` (`src/tools/registry.py`) scores proximity to a
  known anchor *below* the name-evidence terms (exact / branch / category / containment) and *above*
  string similarity — without that term a bare brand name resolves to whichever branch has the
  shortest name, anywhere in the country. `_batch_geocode` then reconciles the batch against itself:
  when the anchor ends up farther than `radius_m` from every other resolved name, it re-searches the
  anchor in the peers' neighbourhood, because a nationwide keyword search for an ambiguous short
  name never surfaces the intended place as a candidate at all. An argument in
  `PLACE_ARGUMENT_NAMES` that arrives as `None` raises `PlaceNotFoundError` before pydantic can
  report it as a validation error.
- **The region prior is deployment configuration, not evidence.** `KAKAO_SEARCH_CENTER` /
  `KAKAO_SEARCH_RADIUS_M` bias the first Kakao keyword query toward the benchmark's region inside
  `KakaoMapProvider`, because Kakao searches nationwide and Korean POI names repeat across cities. A
  name with no match in the region still falls back to the unbiased nationwide search, so the prior
  can never hide a place; the cache key carries it so biased and unbiased runs cannot share entries.
  It is evidence-layer configuration, so it applies to whichever agent queries and reads nothing
  from `BenchmarkItem` — deriving it from
  `region` would leak eval-only metadata. Blank disables it, and reports must say which it was.
- **Reconciliation is for pairs only, and the anchor is authoritative.** `_reconcile_batch` runs
  when exactly two names resolved. With three or more the batch is an anchor plus option texts, the
  anchored search has already placed each option, and "tightest span" lets scattered option brands
  out-vote a correct anchor and drag the batch to another district. A wrongly distant option is
  harmless — `nearest` never picks it — but a moved anchor invalidates every operator after it.
- **Option recovery answers the question's category, stays inside its radius, and never returns
  the anchor.** `recover_option_places` takes the `category_code` the retrieval used (bound by
  `_ground_graph_literals` from `_nearby_retrieval_specs`) and skips the nationwide fallback when
  it is set, so an option is satisfied by the kind of place being asked for rather than any
  namesake — `목동` in a station question otherwise matched 교보문고 목동점, the anchor itself. The
  uncategorised fallback that remains is a nationwide search, so `_within_anchor_radius` keeps only
  what is actually within the radius asked about: `꽃담공방` came back from 순천, 129 km away, and
  entered the candidate set as if it were a neighbour.
- **A name matches one place, and a place answers one option.** `_assign_unique_matches`
  (`src/tools/spatial.py`) pairs option texts with retrieved POIs so neither side is used twice —
  scoring each option independently let one 서울공릉초등학교 clear the floor for both
  서울오륜초등학교 and 서울평화초등학교, and the tie-break then handed the answer to whichever came
  first in the list. `distinguishing_similarity` supplies the other half: Korean POI names of one
  kind share long generic affixes, so similarity is capped by how well the *residue* between the
  shared prefix and suffix matches (오륜 vs 공릉, not 서울…초등학교). A residue too short to
  distinguish anything (`CU 가락센트럴점` against Kakao's `CU 가락센타점`) is a spelling variant and
  is left alone. `_names_the_same_place` applies the same test, which is what stops the
  brand-only retry in `_query_variants` — `CU` for `CU 구로소담점` — from resolving to whichever
  branch of the brand sits nearest the region prior's centre.
- **A reference the provider handed out is a reference it must take back.**
  `parse_coordinate_literal` (`src/tools/spatial.py`, shared by every provider) lets a `"lat,lng"`
  string stand where a place is expected, and `ContextMapProvider._dereference` accepts a
  `place_id` it minted as well. An agent that already holds a POI's coordinates or id is asking
  what is near *them*; sending either through the keyword search raised `PlaceNotFoundError`, and
  a ReAct run then burned its remaining steps re-searching a name that was never a name. Adding a
  provider means implementing both.
- **A ranking never invents evidence.** `max` always yields a candidate, so `_best_place_match`
  applies `NAME_EVIDENCE_FLOOR` and returns `None` when the winner shares no containment and too
  little similarity with the query — a name Kakao does not have must fail as `PlaceNotFoundError`,
  not resolve to whatever scored least badly (`마천1치안센터` → `웅동파출소`, 100 km away, zero
  characters in common). Every candidate must clear it, wherever it came from.
  `match_distance_options` reports the same idea as `fits` / `error_ratio`:
  the nearest option is always *some* option, and a kilometre-scale error means the places were
  resolved wrong, not that the answer is the least-bad number. Do not restore a "closest wins"
  fallback in either place.
- **Proximity is not identity, and containment is not always evidence.** Kakao's keyword search
  is tolerant, so asking an anchor's neighbourhood for a name Kakao does not carry answers with
  places of the same *kind*: `신사정육점` came back as `한아름축산`, `쌍문1치안센터` as
  `수유6치안센터`. Both resolved, both were a different POI, and every operator downstream then
  computed correctly over the wrong place. So the anchored path in `_resolve_batch` requires name
  evidence too; being near the anchor buys exactly one licence, `allow_cross_script`, for a brand
  whose Kakao entry is written in the other script (`A TWOSOME PLACE` / 투썸플레이스, `S-OIL` /
  에쓰오일), where characters cannot testify either way — and `strict_names` withdraws even that.
  When the neighbourhood holds nothing by that name, widen to the nationwide search before giving
  up; that recovers more places than the old exemption ever fabricated. `_containment_is_evidence`
  guards the other end: a short name is a substring of a great many long ones, so containment
  counts only when the shorter key leads the longer (올리브영 / 올리브영 거여역점) or makes up half
  of it — `압구정` inside `해피냠냠라면가게한강버스압구정선착장점` resolved a distance question to a
  POI 12 km from the one asked about.
- **A name is a name, not a name plus an address.** A dataset that has to separate two same-named
  options appends the address to the option text (`버거킹 - 서울특별시 용산구 한강로2가 한강대로
  92`). Kakao indexes names only, so `strip_location_qualifier` (`src/tools/spatial.py`) drops that
  tail inside `_search_key`, `_name_key`, and `_query_variants` — without it the tail drags every
  similarity below the floor and the option cannot resolve at all. `_search_key` folds 파출소 into
  치안센터 for the same reason: which of the two names an institution goes by is editorial, so the
  distinguishing part (연남) has to decide, not the institution word.
- **A place is not near itself.** `_excluding_self` (`src/tools/spatial.py`) drops the anchor from
  its own `nearest` ranking and its own `filter_by_direction` sector, by id or by standing on the
  same spot. The anchor is a place of the type being asked about often enough to appear among the
  candidates — a nearest-convenience-store question lists the store it starts from, and a stored
  retrieval heads its own block — and ranked by distance it wins with 0.0 m every time, which the
  generation stage then reports faithfully. It is kept only when it is the sole candidate, because
  an empty ranking answers nothing.
- **Leniency about shape is a property of the tools, not only the operators.** It is about
  argument shapes a *planner* emits, so in practice it serves Spatial-Agent; it grants no
  evidence and no capability, which is why it is not withheld from anyone.
  `_as_place_argument` / `_as_place_list_argument` (`src/tools/registry.py`) normalize every
  `Place`-typed tool argument the way `_as_place` normalizes an operator's: a one-element list is
  the geocode result the planner forgot to index into, a wrapper carries the place under `place` or
  `location`, an enriched place still is that place (`Place` forbids the `distance_m` /
  `candidate_index` keys an operator staples on), and an anchor written as a name is that place
  named. Without this the artifact `_as_place` shrugs off failed as a `ValidationError` before any
  tool ran, and the cascade emptied the rest of the graph.
- **A place written as a name is the place the plan already resolved.** The local operators spend
  no API call by design, so a name is not something they can look up: `filter_by_direction` handed
  the four option texts instead of the four places the plan had just geocoded dropped every one of
  them, and the empty sector read downstream as "nothing lies north". `_bind_named_places`
  (`src/agent/spatial.py`) substitutes, for the place-valued operator arguments only, the place the
  plan itself resolved under that name — by the query text or by the name Kakao stores. It grants
  no evidence the run had not already gathered, and a name the plan never resolved is left alone so
  it still fails as a missing place. The tools keep resolving names through the provider.
- **A place is no distance from itself, and both architectures are told so identically.** Kakao
  refuses that leg — "출발지와 도착지가 5 m 이내로 설정된 경우 경로를 탐색할 수 없음" — and a trip
  matrix asks for its own diagonal, so one run spent 750 matrix calls plus 64 baseline
  `travel_time` calls collecting the refusal, which the generation stage then read as legs that
  had failed. `_self_route` answers it at zero cost in `directions`, `travel_time` and
  `distance_matrix` alike: the tool surfaces differ between the agents, the evidence below them
  does not. It is the only leg that may be filled — `build_duration_matrix` still reports an
  absent off-diagonal leg as missing evidence rather than a free hop.
- **An empty ranking is a claim, and a list that resolves nothing cannot make it.**
  `_as_place_list` used to return `[]` when every candidate was a bare name or the `{"error": …}`
  marker of a failed step, which downstream is indistinguishable from "no candidate qualifies" —
  and the generation stage then answered from coordinates it read off the trace. A non-empty input
  that resolves nothing raises `PlaceNotFoundError`. An input that is genuinely empty still ranks
  empty: nothing to rank is not the same as candidates that could not be read.
- **ReAct's step budget is generous on purpose.** `MAX_REASONING_STEPS` is 30 here against
  langchain's `initialize_agent` default of 15, and on five primitives a four-stop itinerary needs
  four PlaceSearch turns plus four TravelTime turns before any arithmetic. Being generous to the
  baseline is the conservative direction for the paper's claim; cutting it would handicap ReAct
  beyond what MapEval does. Do not "fix" it downward to make a gap appear.
- No separate HTTP backend server, web UI, or extra datastore beyond the SQLite cache. Keep
  `src/agent/` and `src/tools/` as the only source subpackages and `main.py` as the only *runtime*
  entry point. `data/build_*.py` and `data/verify_benchmark.py` are offline dataset tooling,
  mirroring upstream's `data/build_cache.py`: they import `src/` but nothing in `src/` may import
  them, and no benchmark run may depend on one having been executed.

## Invariants that break silently if violated

- **0-based option indices everywhere** — dataset `answer`, agent `predicted_answer`, logs,
  reports. The final-answer wire format is `^^N^^` (`src/parsing.py::parse_answer`).
- **Gold answer and eval-only metadata never reach an agent.** `BenchmarkItem.agent_input()`
  returns only `(question, options)`; `answer`, `classification`, `region`, `difficulty`, and
  `verified_at` stay in the evaluator.
- **No per-question special-casing.** Heuristics must be generic over a classification, never
  keyed to a specific question id or option string, and answers are never hardcoded.
- Provider failures and agent-reasoning failures are distinct `failure_type`s; `find_provider_failure`
  in `src/agent/base.py` recognizes provider errors by their serialized `ProviderError` class-name
  prefix, so keep those names when adding exception types. Place-not-found, timeout, auth failure,
  rate limit, route-not-found, and unsupported travel mode must each fail as their own explicit
  `ProviderError` subclass rather than degrade silently.
- **Intent accuracy is scored over the questions an intent was predicted for.** ReAct has no
  classification stage, so every row carries `predicted_intent=None`; the old denominator reported
  0.0% for a classifier the architecture does not have. `statistics.intent_classification_accuracy`
  now carries `classified` alongside `total`, and `accuracy` is `None` when nothing was classified.
- Counters (`api_calls`, `cache_hits`, `cache_misses`, `tool_calls`) are recorded as deltas around
  each question, read off the shared provider/registry — new tool paths must go through the
  registry or their calls vanish from the metrics.
- Per-question logs carry `question_id`, `classification`, `agent_type`, `predicted_answer`,
  correctness, and every tool name with normalized arguments, status, and API-call counts.
- **An operator pair the planner is told to use must actually compose.** `distance_matrix` returns
  `{"routes": [...]}` and `tsp_tw` reads a square matrix, and while the two did not meet the only
  matrix a planner could pass was one it invented — so the paper's flagship trip path was
  unreachable no matter how well the prompt described it. `build_duration_matrix`
  (`src/tools/spatial.py`) is the seam: `distance_matrix` emits `nodes`/`matrix`/`matrix_complete`
  alongside its routes, and `tsp_tw` accepts a matrix node, a route list, or a literal. A matrix
  missing an off-diagonal leg is reported incomplete, never filled — an absent leg is missing
  evidence, not a zero-cost hop. Tests in `tests/test_spatial_ops.py` pin the composition; if you
  add another operator the prompt tells a planner to chain, pin that chain too.
- **Every `Place`-typed tool argument is normalized, including the plural ones.**
  `DistanceMatrixArgs.origins`/`destinations` and `RoutePair.origin`/`destination` carry
  `_as_place_list_argument` / `_as_place_argument` like every other place argument. A planner
  writes `origins: "$places"` and gets back `batch_geocode`'s `{query, place, candidates}` records;
  rejecting that shape failed the matrix before one route was requested, and every `tsp_tw`
  downstream failed with it. A missing validator on one argument is invisible until a whole family
  of questions quietly scores zero.
- **Trip stays and the time budget are question literals.** `_extract_trip_schedule` binds
  `tsp_tw.service_times` and `time_budget` in `_ground_graph_literals`, positionally against the
  node list the plan geocoded. The plan chooses the order; how long each visit takes and how much
  time there is are given, not inferred.
- **A question literal is written the way the question writes it.** `_extract_radius_m` knew only
  the word 반경 and silently used its 2000 m default for `직선거리 600m 이내`; `_extract_anchor`
  knew only `에서 가장 가까운` and returned nothing for `지금 X에 있습니다`; the temporal operators
  accepted only ISO 8601 and raised on `오전 10시 00분`, which is exactly what a planner copies out
  of a Korean question. Each of those made a whole family unanswerable while every other stage
  worked. When you add an extractor, cover the phrasings a Korean question actually uses, and pin
  them with a parametrized test.
- **When the question does not name the kind of place, the Analysis stage infers it.**
  `normalize_analysis` carries `target_type` and `_ground_graph_literals` binds it when
  `_extract_target_type` finds nothing in the text — a question phrased as a need ("우산을 사야
  합니다") never states 편의점, and without the inferred type the retrieval loses its category and
  the ranking answers "nearest of anything", which is a *closer* place of the wrong kind.
- **The kind of place asked for is a question literal, bound like the radius and the direction.**
  `nearest` takes `required_type` and `_ground_graph_literals` binds it from the question or from
  the Analysis stage's inference. Telling the planner in prose to retrieve by category moved this
  family from 1/14 to 3/14; binding it moved the same family to 11/14. A planner that ranks the
  option texts directly builds no retrieval node for a category to live on, so prose has nothing
  to attach to — that is the general reason literal binding beats prompt guidance here.
  The filter is a preference, not a requirement: when nothing matches it is dropped, because an
  empty result is a gap in the category vocabulary rather than evidence that nothing qualifies.
- **A requested kind is matched by what Kakao calls it, and only at the type level.**
  `CATEGORY_ALIASES` maps 지하철역 onto 지하철,전철 and 대형마트 onto 슈퍼마켓 / 대형슈퍼, because
  the question's noun does not appear in those paths at all. Keep the terms to words the taxonomy
  uses for a *type*: a bare 마트 also matches `가정,생활 > 편의점 > 이마트24`, which let a
  convenience-store brand answer a 대형마트 question. Add terms from category strings you have
  actually observed, never from what a category ought to be called. `CATEGORY_CODE_NOUNS` reads
  the other vocabulary the same lexicon has to speak: a planner copies `CS2` out of `GRAPH_PROMPT`
  and writes it where a kind of place belongs, and looking for those three letters inside a Korean
  category path matches nothing. `filter_places` normalizes its input like every other coordinate
  operator (it was reading `category` off `batch_geocode`'s `{query, place, candidates}` wrapper,
  where every field is `None`), treats several required types as alternatives rather than demanding
  one path contain them all, and — as `nearest` does — drops a kind filter that matches nothing.
  Each of those emptied the candidate list, and an empty list is what the generation stage guesses
  over: the inferred-category family answered with a cafe 16 m from the anchor.
- **An operator that only reports totals cannot answer a bounded question.** `steps_analysis`
  returned whole-route turn counts, so "how many left turns *before* 왕십리로" had no number
  available except the route's total — and the answer came back confidently over-counted rather
  than failing. With a landmark it now also reports `landmark_index` and
  `*_before_landmark` / `*_after_landmark` counts. Before adding a question shape, check the
  operator can express its scope, not just its measure.
- **Every grounding branch edits the same `arguments` copy, and the fall-through must append it.**
  `_ground_graph_literals` ends with a generic branch for operators it has nothing special to do
  with; appending the original step there threw away whatever earlier branches had bound. A
  routing priority bound for `directions` never reached it, and the family scored 6/14 against
  13/14 once the copy was carried through. Any new branch must keep this shape: edit `arguments`,
  then append `{**step, "arguments": arguments}`.
- **Stated stays are bound, not left to the planner.** `calculate_finish_time.stay_durations_s`
  is grounded from the question like `tsp_tw.service_times`, because dropping one visit or
  inventing one for the return leg moves a clock answer by a whole stay — further than the gap
  between two options, which is what `trip_finish_time` was losing on. Look the stay up by the
  name the plan already holds (`_stay_stated_for`): reading names out of the sentence instead
  swallowed the clause before the first one, so the starting point inherited a visit it never
  makes.
- **G3 types a reference by the field it names, not by the node it starts at.** `tsp_tw` outputs a
  `network`, and `$tsp.total_cost` is the tour's duration: typing that reference by the node
  refused eleven correctly-composed plans in one run, all of them the chain a "what time must I
  leave" question needs. `OUTPUT_FIELD_TYPES` types the projections planners actually take, an
  unknown path is left unconstrained rather than refused, and a *bare* reference of the wrong type
  still fails — path-awareness must not disarm the constraint.
- **A node nothing consumes is pruned, not refused.** An unused `batch_geocode` left in a draft is
  a planner leftover; the rest of the plan answers the question. `normalize_and_validate_graph`
  drops such nodes and re-sorts, which yields a graph that satisfies G5 rather than one that fails
  it. Likewise a `depends_on` written as arithmetic ("drive_time + 3600") still names a real node,
  and `_normalize_dependency` reads it out when exactly one known id appears in the text.
- **Only `DISTANCE` routing is reproducible; a duration never is.** RECOMMEND and TIME both
  optimize against live speeds, so the same pair answered 17,879 m and then 23,041 m. Even at a
  fixed priority the *duration* is a live estimate: the identical DISTANCE route came back as
  3,243 s and then 4,337 s. So a distance gold is a fact about the road network and a duration
  gold is a snapshot of the traffic. Benchmark builders route with `DISTANCE`
  (`Builder.route` in `data/benchmark_core.py` defaults to it), and any question whose answer
  rides on a duration must space its options wider than that spread — the time-window families
  keep at least 85 minutes between options for exactly this reason. This is a property of the
  provider, not a bug to fix: do not "stabilize" it by inventing a speed.
- **A route-shaped gold names its route, in the builder, in the question, and in the verifier.**
  Which turns come in which order, and how many of them are left, are properties of *one* route,
  and Kakao serves a different one per priority. `dataset/seoul_kmapeval_v2_mcq_100.jsonl` was
  first built before `Builder.route` defaulted to DISTANCE, so its two guidance families were
  built on RECOMMEND — and the day traffic moved, eleven rows stopped re-deriving, none of them
  because an answer had changed. They are now built on DISTANCE, their questions say
  `거리가 가장 짧은 경로로` the way v3's do, and `data/verify_benchmark.py` asks for the same
  priority the builder used. A duration family is the mirror image: `routing_via_compare` asks
  which detour is *fastest*, so its gold is built with TIME and verified with TIME — reading a
  duration off the DISTANCE route measures a route nobody drives. When you add a routing family,
  make the builder, the question text and the verifier agree on the priority, and check the
  verifier still re-derives after the next traffic change, not just today. Both agents can ask for
  the route the question names — `directions.priority` is in the schema ReAct sees, documented as
  "RECOMMEND, TIME, or DISTANCE" — but only Spatial-Agent binds it, in `_extract_route_priority`.
  Two of ReAct's turn-count misses on the reproduction benchmark, and all three of its
  multi-segment misses (each answer ~1.4x its gold, the RECOMMEND detour), are exactly the
  RECOMMEND reading of a question that says 거리가 가장 짧은 경로로. That asymmetry is grounding,
  an architectural stage; it is not a tool the baseline was denied, and a write-up should say so
  rather than let it read as one. The parameter's *description* is a different matter and now
  names what each value optimizes — documenting a parameter is not strategy for a question, and
  being generous to the baseline is the conservative direction, as with its step budget.
- **A travelled distance comes from a route, a straight line from haversine, and they are not
  interchangeable.** Road distance runs roughly a quarter longer than the straight line between
  the same points, which is near enough to land on a plausible wrong option: every miss in the
  multi-segment family was the 0.78x distractor, the signature of summing haversine legs for a
  주행 거리 question. `GRAPH_PROMPT` says which operator each phrasing means.
- **A question with two anchors is not a neighbourhood retrieval.** `poi_between` and
  `poi_common_nearby` were the two families where ReAct beat Spatial-Agent on the reproduction
  benchmark, and the traces say why: the planner retrieved around *one* anchor and matched the
  options against what came back, which cannot see the other anchor at all. Both shapes compose
  out of operators that already exist — the corridor question is a detour sum over
  `haversine_distance` (or a route through the option as a waypoint), and "within R of both" is
  `within_radius` chained over its own output — so `GRAPH_PROMPT` maps the phrasings onto them.
  When a family loses to the baseline, check first whether the planner is composing the wrong
  operators before concluding the evidence is unreachable.
- **`classification` is the intent the agent routes on, not the paper family it exercises.** It is
  `SUPPORTED_INTENTS`, so it decides template retrieval and grounding; the Appendix E family is
  recorded in `template_id`. Labelling a radius-scoped share question `poi` because it is
  Object-Field-Measure made the intent metric measure the label, not the router.
- Do not create persistent dumps of raw Kakao responses; usage rights are not established for them.

## Spatial-Agent / GeoFlow

Pipeline in `src/agent/spatial.py::SpatialAgent.answer`, preserving the upstream
`Route → Plan → Execute → Evaluate → Generate` shape:
`analyze → retrieve_templates → compose (LLM graph) → factorize + validate (→ one LLM repair →
pre-validated template fallback) → topological execute → evaluate → generate`.

The formalism lives in `src/agent/geoflow.py`: `OPERATOR_CONTRACTS` (output core-concept type per
operator), `OPERATOR_INPUT_TYPES`, `TEMPLATES` (Appendix E macro families, all executable and
G1–G5-valid), `factorize_geoflow`, and `normalize_and_validate_graph` (G1 acyclicity, G2 role
ordering over `sub_condition < condition < support < measure` with contextual roles excluded,
G3 type compatibility, G4 executability, G5 both-direction reachability
`EXTENT/TEXTENT → node → MEASURE`).

Reuse upstream operator semantics wherever they are still valid; change the planner, state, or an
operator only where Kakao compatibility forces it.

**Adding or renaming an operator touches four places, and tests enforce the consistency:**

1. an implementation — `ToolRegistry` (`src/tools/registry.py`, external/API-backed) *or*
   `SpatialOperatorRegistry` (`src/tools/spatial.py`, local/deterministic);
2. `OPERATOR_CONTRACTS` + `OPERATOR_INPUT_TYPES` in `src/agent/geoflow.py`;
3. the operator-contract list inside `GRAPH_PROMPT` in `src/agent/spatial.py` — the planner LLM
   only knows what this prompt spells out;
4. `_normalize_arguments` in `src/tools/spatial.py` if planners emit argument aliases for it.

`SpatialAgent.__init__` raises at construction time if any `OPERATOR_CONTRACTS` entry is not
executable, so a missing step 1 or 2 fails fast; a missing step 3 only shows up as degraded
benchmark accuracy.

**A contract's required arguments must be ones the tool itself demands.** Adding an
`arrival_time` mode to `calculate_finish_time` while its contract still required `start_time`
made G4 refuse every plan that used the new mode — five questions in one run, each with its
evidence already gathered, and the prompt had just told planners to compose exactly that. A
one-of relationship between two arguments belongs in the args model, which can say so; the
contract lists only what is unconditionally required. `tests/test_spatial_ops.py` pins
`OPERATOR_CONTRACTS` against every tool schema, and it caught a second instance the moment it was
written (`recover_option_places` required `candidates`, which defaults to empty).

Executed steps are recorded twice: `results` keyed by step id (operator state) and `concept_state`
materialized through each step's `output_bindings` (concept state). Both go into the trace the
generation stage conditions on. Step failures are isolated into `{"error": ...}` rather than
aborting the run — keep that behavior.

**Execution is lenient about shape, strict about evidence.** Planners routinely reference the object
that *contains* a place instead of the place. Mirroring upstream's concept-reference resolution:

- `_resolve_references` / `_descend_reference` degrade an over-specified `$node.path` to the closest
  resolvable object (and only raise for an unknown node id); `_resolve_output_binding` does the same
  so concept-state materialization can never abort a run.
- Every coordinate operator normalizes its inputs through `_as_place` / `_as_place_list` in
  `src/tools/spatial.py`, which unwrap `{query, place, candidates}`, `{"location": …}`, `nearest`
  results, single-element branches, and `lat`/`lng`/`x`/`y` spellings. A genuinely unresolved place
  raises `PlaceNotFoundError`; never let it surface as a `TypeError` or `KeyError`.
- The operator's contract decides its output type. A planner's `output_type` is a guess it has no
  authority over, so `normalize_and_validate_graph` corrects it instead of raising — a `tsp_tw`
  node declaring `object` used to lose a plan whose every leg had already been looked up.
- Route `priority` accepts the words a planner reaches for (`fastest` → TIME, `shortest` →
  DISTANCE) through `_as_priority`, but only where the meaning is unambiguous. An unrecognized
  word passes through and still fails: this is leniency about wording, never about meaning, and a
  silent default would quietly answer a different question.
- Do not tighten these back into hard failures. A shape mismatch is a planner artifact; only missing
  evidence is a real failure.

Question literals are bound after drafting and before validation in `_ground_graph_literals`: the
anchor name, the requested direction, the exact radius, the candidate option texts for all three
`match_*` operators, the two compared POI names of a `distance` question
(`_extract_compared_places`, which the template path shares), and the retrieval spec. `_nearby_retrieval_specs` fans a place type out over
every Kakao keyword/category that covers it, and `_retrieval_steps` merges the branches back under
the planner's original node id so downstream references stay valid. The pre-validated template path
already emits one node per spec, so it grounds with `expand_retrieval=False`.

The generation stage asks for `predicted_answer` *and* `predicted_option`; `_select_option`
reconciles them text-first (exact candidate text → declared index → single containment match) and
records which path fired in the trace as `selection_method`. A clock the graph *computed* outranks
all of them — the generation stage kept revising one, once "for real-world traffic" and once for an
"unrecorded return trip", each adjustment moving the answer exactly one option — but only when it
picks an option decisively, twice as close to it as to the runner-up. The nearest option is always
*some* option: a plan whose stays failed to bind computed a travel-only 12:30 against options at
14:23 and 15:23, and taking the nearer one overruled a generation stage that had added the four
stated hours itself and written the right answer. `derived_clock` names which end the operator
computed, because run backwards it echoes the deadline the question supplied.

## Concurrency, cache, and outputs

`Evaluator` runs `BENCHMARK_CONCURRENCY` (default 4) worker threads, each entering its own
`create_agent_session` context in `main.py` with a private `OpenAIChatClient`, `KakaoMapProvider`,
and agent. Never share an agent across workers (`Evaluator` rejects a shared `agent` when
`max_workers > 1`), and never introduce module-level mutable agent state. Result order is restored
by index, not completion. `src/logging.py` builds a fresh `logging.Logger` per question for the
same reason — a shared logger with temporary handlers would cross-write concurrent traces.

The LLM endpoint is treated as slow, not as dead. It is a self-hosted deployment behind a reverse
proxy: it answers 502/503 while it reloads, reports 404 for a model name it serves again a minute
later, and takes minutes to answer a ReAct call carrying a long trace. **Do not add code that judges
the endpoint's health and changes control flow on the verdict** — no preflight ping, no circuit
breaker, no run-level "invalid" stamp. Every one of those turns a slow endpoint into lost questions,
and none of them makes an answer arrive sooner. Wait instead.

Waiting happens at two scales. `OpenAIChatClient` (`src/llm.py`) drives its own retries (the SDK's
are disabled) with exponential backoff, jitter, and a `MAX_RETRY_DELAY_SECONDS` ceiling, for every
failure except `REQUEST_STATUS_CODES` (400/413/422) — those describe the request we sent, so
repeating it only repeats the mistake, and they propagate unchanged as the agent's problem.
`LLM_TIMEOUT_SECONDS` is deliberately generous for the same reason. When the attempts really do run
out, `LLMUnavailableError` is raised and both agents record `failure_type="llm_unavailable"` — never
`agent_reasoning_failure`, since an outage says nothing about an architecture.

`Evaluator._run_single` then retries the whole *question*: `BENCHMARK_QUESTION_RETRIES` extra
attempts with their own backoff and jitter (workers must not retry in lockstep), because an endpoint
can stay down for the entire minute one question takes. Only what `is_transient_failure` accepts
qualifies: `llm_unavailable`, plus a `provider_failure` whose message starts with
`ProviderTimeoutError` / `ProviderRateLimitError`. Never retry a wrong answer, an
`agent_reasoning_failure`, an `answer_parse_failure`, or a `PlaceNotFoundError` — those are the
result the architecture earned, and re-rolling them measures luck. Retries are counted, not hidden:
each row carries `attempts`, and `performance` carries `retried_question_count`,
`retry_recovered_ids`, and `llm_unavailable_count`. That last count is a fact for the write-up to
report next to the accuracy, not a verdict the code acts on. Report `metadata` carries `agent_type`,
`llm_model`, `llm_base_url` and `code_revision` so a report is attributable after the fact — a
session of fixes leaves a shelf of reports whose accuracies differ for reasons no other field
records, and once two runs overlap the timestamp no longer says which code answered.

`SQLiteMapCache` (`src/tools/cache.py`) is keyed by operation + canonicalized arguments, stores
normalized `Place`/`Route` payloads only — never raw responses or keys — with a TTL (`0` = never
expire) and a `SCHEMA_VERSION` that must be bumped when the stored payload shape changes.

`logs/`, `reports/`, and `data/*.db` are generated and gitignored: per-question traces at
`logs/<UTC>_id<id>_<slug>.log`, one `reports/test_<UTC>.json` per batch with `metadata` /
`statistics` / `results`. Primary metric is overall MCQ accuracy; per-classification accuracy, tool
calls, API calls, cache hits/misses, latency, and failures are reported alongside it.

## The context provider

`src/tools/context.py` ports upstream Spatial-Agent's local context cache. `docs/REFERENCE_MAPPING.md`
records the file-by-file comparison against `ecerybao/Spatial-Agent@6876bba`; the invariants are:

- **The context reaches the provider, never the agent.** `BenchmarkItem.context` is provider
  evidence, not agent input — `agent_input()` still returns only `(question, options)`. Upstream is
  the same: `test_agent.py` evaluates on MapEval-API, which has no context field, and no agent
  module reads the text outside the cache. Handing it to the agent instead would delete the tool
  layer from the experiment and measure prompt reading, not agent architecture.
- **One corpus, shared by every question.** `main.py` collects `item.context` across the dataset
  and builds it once, the way `data/build_cache.py` builds one `context_cache.db` from the whole
  MapEval-Textual corpus. Do not scope it per question: an earlier revision did, and it made the
  mere existence of a name an answer signal — "which option exists at all" answered 14 of 100
  questions under per-question scoping against 9 under the corpus. A real map holds places that
  are not the answer, and so must this.
- **The corpus is the cache.** No lookup is an API call; a lookup it answers is a cache hit and one
  it cannot answer is a cache miss. With a `fallback` provider the miss goes there, which is
  upstream's Google Maps fallback; without one the miss is the answer and the caller raises the
  `PlaceNotFoundError` / `RouteNotFoundError` a missing POI deserves.
- **The corpus is a place database, not an answer sheet.** A MapEval context stores the *result*
  of the query its question asks: a nearby list already filtered by type and already sorted by
  distance. Replaying that block — which is what upstream's `get_nearby_places` does — hands the
  agent the answer for the price of one tool call, and the benchmark stops being distinguishable
  from MapEval-Textual, which is exactly what happened: ReAct scored 100/100. What a stored block
  legitimately contributes is its *places*; the ranking is computed in `nearby_search` from
  coordinates, over every place the corpus holds, including the ones belonging to other questions.
  Do not restore block replay.
- **A retrieval filters by type, in whichever vocabulary the caller speaks.** `TYPE_SYNONYMS` maps
  the context's own token, the Kakao category code a planner emits, and the Korean noun a question
  asks by onto one place type. It is generic over place types, never over questions — the same
  lexicon a geocoder keeps. A filter that matches nothing is not evidence of absence in a sparse
  corpus, so the unfiltered neighbourhood answers instead: the source tags a butcher, a stationer
  and an electronics dealer alike as `store`, and a question asking for one of those has to be
  answered from what is actually there. Never conflate two types to make a filter hit — lumping
  `store` with `supermarket` put 정육점 above the supermarkets in a supermarket question.
- **A place is not among its own neighbours,** in the provider as well as in `nearest`: the anchor
  stands at zero metres from itself and would head every ranking it appears in.
- **Containment is evidence in one direction only.** A brand may lead the branch that extends it
  (`CU` → `CU 삼청점`), because the registry's own query variants shorten names that way. The
  reverse must not match: a corpus entry for a bare `GS25` recorded whichever GS25 the retrieval
  found, not the `GS25 합정프리미엄점` an option names. Allowing it also lets a place-type question
  answer itself, since `편의점` sits inside `다모아편의점`. Below containment, `NAME_MATCH_FLOOR`
  and `distinguishing_similarity` apply exactly as on the Kakao path.
- **The category is served as the context wrote it** (`convenience_store`, `amenity=bank`), not
  translated into Korean. A cache serves what it stored; inventing a Korean label for a place-type
  question would be supplying part of the answer.

## Datasets

JSONL, one `BenchmarkItem` per line, unique ids, 2–4 options, `answer` a 0-based index, and
`classification` from `nearby | poi | routing | trip | type | direction | distance | radius`
(the same eight values are `SUPPORTED_INTENTS` in `src/agent/spatial.py` — extending the set means
touching both, plus the intent heuristics and evaluation rules). Extra fields are allowed
(`context`, `template_id`, `gold_evidence`, …) and every one of them is evaluation-only:
`agent_input()` returns `(question, options)` and nothing else, so `gold_evidence` — which records
*why* an answer is the answer — can never reach an agent.

Three benchmarks ship. They answer different questions and their numbers must never be pooled.

### `dataset/seoul_kmapeval_v2_mcq_100.jsonl` — the reproduction benchmark

The one to run when the question is whether Spatial-Agent's reported gains reproduce. Built by
`data/build_pool.py` + `data/build_benchmark.py` from Kakao Local and Kakao Mobility, seed
`20260818`, and graded against the same provider the agents query, so a wrong answer is the agent's
and never a disagreement between the evidence and the grader.

**The class mix mirrors MapEval-API's answerable half, because that is where the paper's gains
live.** MapEval-API is nearby 83 / trip 67 / routing 66 / poi 64 / unanswerable 20: roughly 45% of
it is trip and routing, families no single retrieval answers. The first Korean benchmark here had
none of that — every row was one lookup — and both architectures saturated, which is a property of
the questions, not a finding about the architectures. Quotas, out of 100:

- `trip` 24 — `trip_optimal_order` 14 (which visiting order is fastest; requires the full driving
  duration matrix over base + 3 sights, then a TSP), `trip_feasible_count` 10 (how many of 4 places
  fit in a budget; requires the matrix plus stay arithmetic over every subset).
- `routing` 23 — `routing_via_compare` 8 (which of 4 detours is fastest; four waypoint routes),
  `routing_next_turn` 8 (what the guidance says after a named road; turn-by-turn guides),
  `routing_turn_count` 7 (how many left turns on a two-leg drive).
- `poi` 23 — `poi_farthest_pair` 8 (which of 4 pairs is farthest; eight lookups, four comparisons),
  `poi_between` 8 (which candidate is on the corridor between two anchors),
  `poi_common_nearby` 7 (which place is within the radius of *both* anchors).
- `nearby` 12, `direction` 9, `radius` 9 — the single-hop families, kept so the benchmark still
  reports the shapes the first one measured.

`unanswerable` is deliberately absent. MapEval encodes it as `answer = 0` against 1-based option
indices — a sentinel meaning *no option is right*, not option 0. That is a refusal channel this
MCQ format does not have, and mapping it onto a real index would make "always answer 0" score 20/20.

Construction rules, all enforced in the generator:

- **Every name round-trips.** A place is only used as an anchor, gold or distractor if searching
  its bare name through the provider lands back within 200 m of it (`Builder.resolves_to`). A name
  the tools cannot look back up is not a question, it is a scoring accident.
- **Every question has a decisive answer.** Ties are rejected at build time: ≥180 s between the
  best and second-best visiting order, ≥120 s between the best and second-best detour, ≥120 m
  between the nearest and runner-up, ≥1500 m between the farthest and second-farthest pair, and
  exactly one candidate inside the radius for a radius question.
- **The runner-up is a distractor.** A nearest-by-type question offers the second-nearest place, so
  rough position cannot answer it; a direction question offers places that are *nearer* but in the
  wrong sector, so the direction constraint is what decides.
- **Distinct endpoints per family.** `itertools.combinations` holds its first element fixed while
  the second sweeps, which put one origin at the head of an entire family; the generator records
  used endpoints and skips a pair that reuses one.
- **Options are shuffled per row**, seeded by question id, except the ordinal option sets
  (`한 곳`…`네 곳`, `1번`…`4번`) where position carries meaning.
- **No leaked totals.** Travel times never appear in a trip option, so the answer cannot be read
  off the option text. (MapEval leaks one; 66 of its 67 trip rows do not.)

### `dataset/seoul_kmapeval_v3_mcq_100.jsonl` — the compositional benchmark

The one to run when the question is whether the *architecture* separates the two agents. v2 mixed
MapEval's class proportions but kept its questions explicit and closed — the anchor named exactly
as Kakao stores it, the category named, the radius stated, the orderings enumerated — so one
`batch_geocode` plus one `distance_matrix` finished a trip question. That leaves a ReAct loop one
decision to get right where Spatial-Agent has four, and structure cannot pay for its overhead.

v3 is selected against the coverage gap. Of the ten Appendix E macro families in `TEMPLATES`, v2
exercised six; **Time-Window-Reverse, Multi-Segment-Aggregate and Object-Field-Measure had no
question at all**, and those are the compositional ones. (Place-Attribute-Query stays unported:
it asks for ratings, price levels and opening hours, and Kakao Local exposes none of them.
Inventing those values would be fabricating evidence.) Built by `data/build_hard_benchmark.py`,
seed `20260819`, graded through the same provider the agents query:

- `trip_finish_time` 16 and `trip_latest_departure` 14 — **Time-Window-Reverse**. A fixed
  itinerary's return time, and the latest departure that still meets a deadline with errands on
  the way. Four legs plus stays plus clock arithmetic, forwards and backwards.
- `multisegment_total` 14 — **Multi-Segment-Aggregate**. Total driving distance over a stated
  four-stop chain; every leg has to be looked up and summed.
- `poi_brand_share` 14 — **Object-Field-Measure**. What share of a neighbourhood's convenience
  stores belong to one brand: retrieve, filter, divide. Wrong options are the shares the *other*
  brands give, so each is a real number about the same neighbourhood.
- `routing_turns_before_road` 14 — **Route-Step-Extract, bounded**. Left turns *before* a named
  road, so the boundary must be located before the count means anything. A bound that changes
  nothing is rejected at build time.
- `poi_bearing_and_distance` 14 — direction **and** straight-line distance in one answer, as a
  2×2 of (right/opposite heading) × (right/wrong distance). One correct half is not enough.
- `nearby_from_need` 14 — the category is **inferred, not stated**: a need ("갑자기 비가 쏟아져서
  우산을 사야 합니다") instead of 편의점. Every option set holds a *closer* place of a different
  kind, so retrieving the wrong category is punished rather than merely unhelpful.

Construction rules beyond v2's (round-trip resolvable names, decisive margins, distinct
endpoints):

- **Gold positions are assigned, not drawn.** A per-row shuffle is uniform in expectation and
  lumpy in practice — one family drew index 0 eight times in fourteen. Per-family accuracy is a
  reported number, so each family's gold positions are balanced by construction.
- **Numeric options are matched by nearest value, not by string.** Routing priority moves a
  duration by minutes (`calculate_finish_time` defaults to TIME, a bare `directions` call to
  RECOMMEND), so options sit tens of minutes apart and `data/verify_hard_benchmark.py` scores the
  closest one — what a solver does. Percentage options are kept at least 6 points apart for the
  same reason: 2/14 and 2/13 both read "약 14%".

`data/verify_hard_benchmark.py` re-derives all 100 through `ToolRegistry` +
`SpatialOperatorRegistry`; it currently reports 100/100.

### `dataset/seoul_mapeval_v1_mcq_100.jsonl` — the context-cache benchmark

100 rows sampled from `dataset/seoul_mapeval_v1.json` (an OSM-derived Seoul pool, 1530 complete
records — the file was transferred truncated mid-record, so a reader must decode the complete
objects and stop at the break). Each row ships a MapEval-format `context`, and it is the dataset to
use with `--provider context` / `--provider hybrid`. The recipe, seed `20260818`:

- One quota per source template, so no family is spent on one question shape: `nearest_by_type` 20,
  `direction_by_type` 20, `distance_between` 20, `within_radius_by_type` 15, `type` 15,
  `routing_duration_value` 4, `routing_distance_value` 3, `routing_shortest_duration` 3.
- Distinct anchor place per row (100/100), and no repeated question text.
- `classification` comes from the source `template_id`, not its coarse `nearby|poi|routing` label —
  the eight-way vocabulary is what this benchmark reports by, and the template already encodes it.
- **Options are shuffled per row**, seeded by question id. The source generator emits distance
  options in ascending order and never shuffles, so its entire `distance_between` family carries
  gold at index 2; shuffling every row alike removes option position as evidence in every family
  without special-casing one.

All 100 gold answers are derivable from the shipped context by deterministic computation through
the provider. Its questions are single-hop by construction, so it measures evidence handling, not
composition — do not read a tie on it as a finding about architecture.

### Verifying a benchmark before trusting a run

`data/verify_benchmark.py` re-derives every gold answer through `ToolRegistry` +
`SpatialOperatorRegistry` — the same tools the agents call — and reports the rows where they
disagree. Run it after regenerating a dataset and after any change to matching, ranking, or
retrieval: a drop in accuracy means nothing until you know the answers are still reachable.

Because this repo runs the prompting-only path (no SFT/DPO and no embedding retrieval), reports
must be labeled prompting-only and must not be presented as reproducing the paper's headline
numbers. A report's `metadata.provider` records which evidence source produced it, and runs from
different sources must never be pooled.
