# Reference implementation mapping

| Upstream concept | K-MapEval implementation | Deliberate deviation |
|---|---|---|
| MapEval `Evaluator2.py` structured ReAct loop | `src/agent/react.py` | Removes the localhost backend, sleeps, and remote writes. Uses user-requested 0-based `^^N^^` answers. |
| MapEval tools/backend | `src/tools/registry.py`, `src/tools/map.py`, `src/tools/kakao.py` | Provider injection replaces the separate HTTP backend; tools expose normalized JSON. |
| Spatial information theory analysis | `normalize_analysis`, `ConceptGraph` | Preserves all seven core-concept labels and six roles. Missing EXTENT/MEASURE concepts are made explicit and marked synthetic. Dataset classification metadata is not passed in. |
| Concept transformation drafting | `retrieve_templates`, Spatial-Agent `compose` | All ten Appendix E examples are G1–G5-valid and executable. Retrieval remains deterministic intent/keyword scoring rather than embedding cosine similarity. |
| Concept graph `G` | `ConceptGraph`, `ConceptNode` | Stores IDs, types, roles, attributes, and explicit Analysis dependencies. Role adjacency never fabricates edges; executable G' edges come from actual operator references. |
| Operator-concept hypergraph `G'` | `factorize_geoflow`, `OperatorHyperedge` | Factorization is deterministic rather than learned. Each hyperedge records input concepts, supplementary literal factor parameters, and one or more output-path bindings. Radius, direction, category, and similar constants remain factors instead of being fabricated as operator outputs. Derived intermediate concepts are explicitly marked. |
| Five GeoFlow constraints | `normalize_and_validate_graph` | Enforces G1, G3, G4, and both halves of G5 on operator and concept graphs. G2 applies only to SUBCOND→COND→SUPPORT→MEASURE; contextual roles are excluded as required by Appendix B. |
| Core-concept execution types | `OPERATOR_CONTRACTS`, `SpatialOperatorRegistry` | LOCATION, OBJECT, FIELD, EVENT, NETWORK, AMOUNT, and PROPORTION all have executable producers. The current benchmark directly exercises only a subset. |
| Contextual/functional roles | `factorize_geoflow`, `ROLE_PRIORITY` | EXTENT/TEXTENT are scheduled as context but do not participate in procedural precedence. |
| Topological executor | Spatial-Agent `execute` stage | Records operator state and separately materializes concept state through output bindings, including multiple bindings from one operator. |
| Lenient concept-reference resolution (`_resolve_concept_reference`, `_extract_coordinates_from_concept`, `resolve_place_name`) | `_resolve_references` / `_descend_reference` in `src/agent/spatial.py`, `_as_place` / `_as_place_list` in `src/tools/spatial.py` | An over-specified `$node.path` degrades to the closest resolvable object instead of failing the operator, and every coordinate operator unwraps the place-shaped record it was handed. Only a genuinely unresolved place raises, as an explicit `PlaceNotFoundError`. |
| Executor evidence preprocessing (auto option distances) | `match_options`, `match_distance_options`, `match_type_options` | Candidate option texts are bound verbatim from the question at grounding time, so a planner cannot paraphrase or numerically re-type them before the Measure comparison. |
| Evaluator answer selection | `_select_option` in `src/agent/spatial.py` | Keeps upstream's text-first reconciliation (exact candidate text, then declared index, then a single containment match) on top of the repository's 0-based `^^N^^` contract. |
| Appendix C operators | `ToolRegistry`, `SpatialOperatorRegistry` | Adds reverse/batch details, waypoint directions, instruction route filtering, extractors, pairwise extremes, place filtering, travel-time nearest, temporal operators, and step analysis with paper-level semantics. |
| TSP-TW | `SpatialOperatorRegistry.tsp_tw` | Exhaustive optimization up to nine nodes with service times, windows, and budget; infeasible instances return the paper's nearest-unvisited partial feasible fallback. OR-Tools is not bundled. |
| Temporal operators | `timezone`, `open_at_time`, `calculate_finish_time`, `calculate_start_time` | Handles cross-midnight/24-hour periods. Multi-stop finish time queries cached/live route durations and adds stays. Latest-departure calculation is an explicit template helper. |
| Spatial-Agent evaluator/generator | Spatial-Agent `evaluate` and `generate` | The LLM conditions on question, final state, and full trace. Deterministic match evidence no longer overwrites the generated selection. |
| Google Maps client | `KakaoMapProvider` | Kakao Local handles search/geocode/reverse-geocode/nearby; Kakao Mobility handles driving and verified multi-waypoint routes with optional guides. |
| Spatial-Agent context cache | `SQLiteMapCache` | Both agents share one normalized Kakao cache. The upstream MapEval-Textual/Google context snapshot is not bundled. |
| MapEval-API benchmark (300 rows, live Google Maps) | `dataset/seoul_kmapeval_v2_mcq_100.jsonl` (100 rows, live Kakao) | Class mix mirrors MapEval-API's answerable half (nearby 30 / trip 24 / routing 23 / poi 23) so the multi-hop families the paper's gains come from are actually present. Gold answers are computed from Kakao Local and Kakao Mobility, the same provider the agents query. `unanswerable` is excluded — see below. |
| SFT and DPO | Not implemented | This repository evaluates the off-the-shelf prompting path and does not claim fine-tuned Qwen results. |

## Remaining non-equivalences

- Template retrieval is keyword based, not embedding based.
- Factorization and concept binding are deterministic, not SFT/DPO learned.
- Evidence comes from one of two interchangeable sources, recorded per run in
  `metadata.provider`: a per-question context shipped with the dataset (the port of upstream's
  MapEval-Textual setting, see below), or live Kakao POIs.
- Kakao Mobility support is driving-only.
- The benchmark router uses the LLM analysis intent; Korean heuristics are only a fallback when the
  returned intent is missing or unsupported.
- The MCQ trip format evaluates option-order comparison, as MapEval's does. `trip_optimal_order`
  offers four orderings whose travel totals differ by at least 180 s, and `trip_feasible_count`
  asks how many stops fit a budget; both are re-derived through `distance_matrix` + `tsp_tw` in
  `data/verify_benchmark.py`. `tsp_tw` is still never silently substituted for option semantics.
- Kakao Local exposes phone and Kakao URL but not rating, price level, or structured opening hours.
  These optional `Place` fields can only be used when an authoritative enriched cache supplies them;
  missing values are never invented.
- Offline coordinate-to-timezone resolution deliberately covers the Korean benchmark extent only.
- Grounded nearby retrieval fans a requested place type out over the Kakao keywords and category
  codes that actually cover it (`_nearby_retrieval_specs`), because Kakao POI names diverge from
  the requested word (경찰서 also appears as 파출소/지구대/치안센터, 도서관 as 문고/도서실). This is
  planner knowledge inside Spatial-Agent, not a tool or evidence source the ReAct baseline cannot
  reach — ReAct issues the same `nearby_places` tool with whatever keyword its own LLM chooses.
- The planner's authored graph is still bounded by `MAX_REASONING_STEPS`; the deterministic
  retrieval fan-out added during grounding gets its own allowance on top of that bound.

These differences mean the project implements the paper's explicit graph formalism and execution
constraints, but it does not claim numerical reproduction of the paper's trained model or original
map-data environment.

Reports from this repository must therefore be labeled **prompting-only**. They are not directly
comparable to the paper's SFT+DPO headline results, because the learned policy, original weights,
embedding example store, and original map evidence snapshot are not included.

## Context evidence (upstream's local context cache)

Verified against `ecerybao/Spatial-Agent` at `6876bba`: `src/tools/context_parser.py`,
`src/tools/local_context_db.py`, `data/build_cache.py`, `src/agent/operators.py`, `README.md`.

What upstream does:

- Evaluation runs on **MapEval-API** — the multiple-choice rows *without* context
  (`test_agent.py`); the cache is built separately from **MapEval-Textual**, whose rows carry the
  `context` field (`data/build_cache.py` → `data/context_cache.db`).
- The cache is **one SQLite database over the whole corpus** — tables `places`, `travel_times`,
  `routes`, `nearby_places` — and every question queries all of it, with a four-level fuzzy
  cascade (exact → LIKE → normalized → Levenshtein).
- Operators query the cache first and **fall back to the Google Maps API on a miss**
  (`query_local_place` returns `None`, the geocode call runs). The README states this plainly:
  "Google Maps API fallback for cache misses."
- **The cache is optional and is not the paper's evaluation setting.** `SpatialAgent.__init__`
  always constructs a `GoogleMapsClient` and only initializes the cache `if db_path.exists()`,
  logging a fall-through to the live API otherwise; `test_agent.py` has no flag for it at all; and
  the README calls it "an optional local context cache" for "faster and more reliable evaluation",
  shipping no prebuilt database. It is a cost and latency shortcut in a reimplementation, not the
  arrangement the paper's MapEval-API numbers come from — which is live API calls.
- **MapEval-API and MapEval-Textual are the same 300 questions**, ids included; API is Textual
  with the `context` field removed. So a cache built from Textual and used on API is, question for
  question, the retrieval that answers it — which is what "the strongest results depend on a local
  SQLite context cache" amounts to.
- `get_nearby_places(reference_place, category)` returns **the stored block for that reference
  place and nothing else**, re-ranked by haversine distance. It never scans the place table by
  radius.
- **The agent never sees the context text.** `test_agent.py` does not mention it, and no agent
  module reads it outside the database.

What this repo does, and why:

- Same shape: the corpus is built from every context in the dataset and shared by all questions,
  loaded *behind* the tool layer so both architectures still choose tools and still read
  normalized `Place` / `Route` objects. `BenchmarkItem.agent_input()` is unchanged.
- The context travels in the benchmark row rather than in a second file. `main.py` collects
  `item.context` across the dataset and builds one corpus, which is upstream's arrangement without
  the extra artifact. Nothing per-question is bound: an earlier revision scoped the corpus to the
  running question, and that made the mere existence of a name an answer signal — "which option
  exists at all" answered 14 of 100 questions under per-question scoping and 9 under the shared
  corpus.
- `--provider hybrid` is upstream's cache-then-live arrangement with Kakao in Google's place, and
  it is what `--provider auto` now resolves to for a context-carrying dataset. `--provider context`
  runs the corpus alone, so a run needs no Kakao key and a miss stays a miss — a closed world
  stricter than anything upstream measures, which makes it an ablation to ask for by name rather
  than the default a bare run lands on. `resolve_provider_kind` and its test pin the choice.
  One asymmetry the mode carries and upstream does not: upstream's cache and its fallback are both
  Google, while `seoul_mapeval_v1`'s contexts are OSM-derived and the fallback is Kakao, so a
  hybrid run here mixes two gazetteers where upstream mixes none.
- **Retrievals are computed, not replayed.** This is the one place the port deliberately does not
  follow upstream, and the reason is an evaluation-validity flaw in upstream that this repo
  reproduced and then measured. MapEval-API is MapEval-Textual with the `context` field removed —
  the same 300 questions, the same ids — so a cache built from Textual holds, for every API
  question, the retrieval result that answers it. `get_nearby_places` returns that block already
  filtered by type and already sorted by distance, which makes one tool call sufficient and
  collapses the API setting into the Textual one. Ported faithfully, it produced ReAct 100/100.
  Here the block contributes its *places* to the corpus and `nearby_search` computes the ranking
  from coordinates over the whole corpus, filtered by type through `TYPE_SYNONYMS`. That is also
  what a live map API does: Kakao and Google compute, only a frozen context is pre-computed.
- Counters follow the upstream framing: the corpus *is* the cache, so it costs no API call; an
  answered lookup is a hit, an unanswerable one a miss.

Deliberate deviations from upstream, and the reason for each:

- **Normalized objects, not text blocks.** Upstream's `get_place_by_name` returns the context's
  `information` text. This repo's standing invariant is that raw provider text never reaches an
  agent, so the corpus is parsed into `Place` / `Route`.
- **Stricter name matching.** Upstream's cascade ends in a Levenshtein search over every place
  name in the corpus. Korean POI names of one kind share long generic affixes, so the Kakao path's
  `NAME_MATCH_FLOOR` and `distinguishing_similarity` apply here too, with containment narrowed to
  one direction — undirected, `편의점` inside `다모아편의점` let a place-type option resolve to the
  place the question was asking about.
- **A place is not near itself.** Upstream never excludes the reference place from its own nearby
  block, and never had to: Google's "Nearby Restaurants of St. Lawrence Market" does not contain
  the market. Our OSM-derived questions ask a 편의점 for its nearest 편의점, so the anchor heads
  its own block at 0 m and won every ranking.
- **No intent gate.** Upstream's `ContextManager.should_use_local_db` restricts the cache to
  `{routing, trip, poi, nearby}`, which is all four answerable MapEval classes — the gate excludes
  nothing. Porting it to this repo's eight-way vocabulary would exclude four classes for no reason,
  so it is not ported.
- Place types are served in the context's own vocabulary (`convenience_store`, `amenity=bank`).
  Translating them into the Korean nouns a place-type question offers as options would be supplying
  part of the answer.

### What upstream's own number was measured with

`~/spatial-agent/reports/test_20260808T210842Z.json`, 280 of MapEval-API's 300 rows (the 20
unanswerable ones excluded), is the reference point for every accuracy this repository quotes:

```
overall 71.07%   trip 55.2%   poi 53.1%   nearby 80.7%   routing 92.4%   failed 41
```

`data/context_cache.db` (589 KB) was present and newer than that report when it was written, and
`SpatialAgent.__init__` initializes the cache whenever the file exists. So that 71.07% is a
**context-assisted** number, on the arrangement described above — the corpus built from the same
300 questions' curated evidence, with the live API behind it. The configuration here that
corresponds to it is `hybrid`, not `kakao`; a `kakao` run has no curated corpus at all and is a
harder setting than the one the reference number comes from. Any comparison that puts our number
beside 71.07% has to say which of the two it is.

The per-class split is the more useful half of it: upstream's losses are concentrated in `poi`
(53.1%) and `trip` (55.2%), and `poi` there is dominated by rating, opening-hours and
reservability questions that Kakao cannot be asked. A benchmark built on Kakao that reports a high
number has usually just left out the classes upstream fails.

## Kakao-specific constraints

- Raw Kakao JSON never reaches an agent.
- Kakao Local has no standalone place-details REST method; `place_details` and
  `batch_place_details` read normalized cached records from earlier retrieval/enrichment.
- `reverse_geocode` uses `/v2/local/geo/coord2address.json`; waypoint routes use the official
  `POST /v1/waypoints/directions` and verify returned waypoint names and coordinates — within
  `WAYPOINT_TOLERANCE_M`, because Kakao echoes a waypoint about a metre from the coordinate it was
  sent and snaps a POI to the nearest road. The check is for a route sent through somewhere else,
  which is hundreds of metres away, not for rounding.
- `geocode` falls back to the place-name index when the address index has no entry. Google's
  geocoder answers a place name; Kakao keeps addresses and place names apart, and a planner writes
  the question's place name into `geocode` as readily as an address — `대림동 우리 골목형상점가`
  has one exact place entry and no address entry. The fallback is the lookup `place_search`
  performs, so it reaches nothing the tool surface does not already carry.
- Unsupported travel modes fail explicitly. Kakao Mobility routes cars only, where Google Directions
  answers a walking query, so every "걸어가기에 가장 가까운" question was costing a planner a failed
  call: `GRAPH_PROMPT` says that phrasing means `haversine_distance` / `nearest(metric="haversine")`
  here. The benchmark's own golds for those families are straight-line distances, so nothing is
  approximated by saying so.
- **A leg from a place to itself is answered, not requested.** Google returns a zero-length route
  for identical endpoints; Kakao Mobility refuses with "출발지와 도착지가 5 m 이내로 설정된 경우
  경로를 탐색할 수 없음". A trip matrix asks for its own diagonal, so one run collected 750 of
  those refusals through `distance_matrix` and 64 more through the baseline's `travel_time`, and
  the generation stage read them as legs that had failed. `_self_route` answers the leg locally in
  `directions`, `travel_time` and `distance_matrix` alike — the same evidence for both
  architectures — while an absent *off-diagonal* leg is still reported as missing.
- SQLite stores normalized `Place` and `Route` payloads, not raw Kakao responses or API keys.
- **Region prior on name lookups (`KAKAO_SEARCH_CENTER` / `KAKAO_SEARCH_RADIUS_M`).** Neither
  upstream implementation needs one: Google Places disambiguates from a session location, and the
  paper's MapEval-Textual snapshot is a closed evidence set. Kakao Local searches nationwide, and
  Korean POI names repeat across cities, so an ambiguous name resolved to whichever city ranked
  first — 18 of the 100 benchmark anchors landed outside 서울, including 제주, 경남 and 경북, and
  every operator downstream then computed correctly over a POI in the wrong province. The prior
  biases the *first* keyword query to the benchmark's region; a name with no match there still
  resolves nationwide, so it can never hide evidence. It is a deployment setting read from `.env`
  and applied inside `KakaoMapProvider`, identically for ReAct and Spatial-Agent: it says where to
  look, never which option is correct, and no `BenchmarkItem` field reaches it. Blank disables it.
  Reports must state whether it was enabled, since it changes which places the tool layer can see.
- **Coordinate literals accepted where a place name is expected.** Google's Places API takes a
  `location` bias directly, so upstream never has to name a point. Kakao Local resolves a place
  reference by keyword search, so an agent passing back coordinates it already retrieved got
  `PlaceNotFoundError`. `_parse_coordinate_literal` resolves a `"lat,lng"` string into a `Place`
  without a network call. It adds no evidence — the coordinates came from an earlier tool result.
- **Name matching is capped by the distinguishing residue.** Upstream compares English POI names,
  where a brand and its branch share little. Korean POI names of one kind share long generic
  affixes (서울…초등학교, …주유소, CU …점), so plain string similarity clears any sane floor for
  two different places. `distinguishing_similarity` compares only what is left between the shared
  prefix and suffix, and option-to-POI matching (`_assign_unique_matches`) is one-to-one so a
  single retrieved POI cannot answer several options at once.
- **Neighbourhood membership is not name evidence.** Google Places answers a keyword query with
  places bearing that name, so upstream can trust a location-biased search. Kakao Local answers a
  name it does not carry with places of the same *kind* near the bias point: `신사정육점` returned
  `한아름축산`, `쌍문1치안센터` returned `수유6치안센터`. Both looked resolved, and both replaced
  the question's POI with a different one. The anchored branch of `_resolve_batch` therefore applies
  the same name-evidence floor as the nationwide branch, and falls back to the nationwide search
  when the neighbourhood has nothing by that name. The one licence proximity buys is
  `allow_cross_script`, for a brand whose Kakao entry is transliterated (`A TWOSOME PLACE` /
  투썸플레이스); `strict_names`, which the distance path sets, withdraws it.
- **Option texts may carry an address, and institutions may carry two names.** The Korean source
  datasets disambiguate namesakes by appending the address to the option text and follow OSM's
  choice between 치안센터 and 파출소, neither of which appears in a Kakao place name.
  `strip_location_qualifier` drops the appended address before any name comparison or keyword
  query, and `_search_key` folds the two institution words together. Upstream needs neither: its
  option texts are bare names over a single naming authority.

## What MapEval's own answer encoding turned out to be

Measured directly on `MapEval-API.jsonl` (300 rows), because it decides what can be ported:

- **`answer` is 1-based.** The distribution over all 300 rows is `{1: 74, 2: 73, 3: 71, 4: 62,
  0: 20}`, and the twenty `0`s are exactly the twenty `unanswerable` rows. `0` is a sentinel
  meaning *no option is right*, not an index. This repository's contract is 0-based throughout
  (`^^N^^`), so any port of a MapEval row must shift the index and drop the sentinel.
- **`unanswerable` is therefore a refusal channel, not a class of question.** It is excluded from
  `dataset/seoul_kmapeval_v2_mcq_100.jsonl`: mapping the sentinel onto a real option index would
  make "always answer the first option" score 20/20 on it, and the MCQ format has nowhere to put
  a refusal. Adding it would mean adding an answer channel to `parse_answer`, the evaluator, and
  both agents — a separate change, not a dataset one.
- **MapEval-API is MapEval-Textual minus `context`.** Same 300 ids, 297 identical question
  strings, and `MapEval-Textual.jsonl`'s key set is API's plus `context`. A cache built from
  Textual and queried by API is therefore question-for-question an answer key, which is why
  `ContextMapProvider` computes retrievals over the whole corpus instead of replaying the stored
  block (see above).
- **The ReAct baseline's budget is 15 iterations**, langchain's default, which is what
  `mapeval-api/Evaluator2.py` runs. An earlier version of this file argued 30 from
  `max_tool_rounds: int = 30` in `mapeval-api/mapeval_api_evaluator.py`; that file is untracked
  upstream, so it is a local adapter and justifies nothing about the paper's baseline. The
  concern behind the 30 was real -- a four-stop ordering question needs twelve route legs, so a
  budget can turn an architectural result into an artifact -- which is why the budget is now one
  `.env` line, recorded in every report, and read beside the accuracy rather than assumed.

## Why the first Kakao run inverted the paper's result

The first run of `dataset/seoul_kmapeval_v2_mcq_100.jsonl` against live Kakao gave ReAct 90/100 and
Spatial-Agent 77/100 — the paper's ordering reversed, and worst on `trip` (ReAct 23/24 against
Spatial-Agent 15/24), the family Spatial-Agent exists for. Four causes, three of them defects in
this port rather than findings about the architecture:

1. **`distance_matrix` and `tsp_tw` could not compose** (fixed). The tool returned
   `{"routes": [...]}`; the operator read `distance_matrix["matrix"]`. No planner could bridge
   that, so `tsp_tw` never ran on a real matrix and the trip questions were answered by summing
   `directions` calls instead. See the invariant in `AGENT.md`.
2. **`DistanceMatrixArgs.origins`/`destinations` rejected `batch_geocode` output** (fixed). A
   planner writes `origins: "$places"`; the records are `{query, place, candidates}`, and every
   other place argument in the registry already normalized that shape. Matrices failed with a
   64-error `ValidationError` before a route was requested.
3. **The `Route-Optimize` template taught an unexecutable pattern** (fixed). Its example carried a
   hardcoded literal matrix and never showed where a matrix comes from, and `GRAPH_PROMPT`
   separately told planners to answer trip questions with route pairs and `aggregate_route_groups`.
   Together they steered the planner away from `tsp_tw` even when the right template was retrieved.
   The prompt now describes the matrix-to-`tsp_tw` chain and the two trip question shapes.
4. **Intent classification had no rule separating trip from routing, or poi from anything**
   (fixed): 12 of 24 trip rows were classified `routing`, and `poi` was never predicted at all for
   23 poi rows. After the prompt change, trip intent went from 11/24 to 23/23 on a re-run and trip
   answers from 15/24 to 18/23.

What is *not* a defect, and matters for reading any number from this benchmark: **its questions are
explicit and closed.** They name the anchor exactly as Kakao stores it, name the category, state
the radius, and enumerate the orderings to compare. One `batch_geocode` plus one `distance_matrix`
answers a trip question, so a ReAct loop has one decision to get right where Spatial-Agent has
four (intent, graph, bindings, generation). MapEval-API's questions are implicit and open — "I am
at X and hungry, where can I eat quickly?" — and that is where decomposition earns its overhead.
A benchmark of fully specified questions measures pipeline reliability, not composition, and should
not be reported as evidence for or against the paper's claim on its own.

## Why harder questions widened the gap instead of closing it

`dataset/seoul_kmapeval_v3_mcq_100.jsonl` was built to test the three Appendix E families v2 never
exercised — Time-Window-Reverse, Multi-Segment-Aggregate, Object-Field-Measure — on the assumption
that compositional questions would let the operator graph earn its overhead. The first run went the
other way: ReAct 92/100, Spatial-Agent 67/100, against 90/77 on the shallower v2. Spatial-Agent
also issued **177 tool calls to ReAct's 400**, which is the tell: it was not failing to reason over
the evidence, it was failing to fetch it.

The losses were not spread across the compositional families. Five of seven were fine or close
(`poi_bearing_and_distance` 14/14 both, `poi_brand_share` 14/14 both, `trip_latest_departure` 13/14
against 14/14). Two collapsed, and each for a single mechanical reason:

- **`trip_finish_time` 8/16.** The authored graph was already correct — `batch_geocode` then
  `calculate_finish_time` with `stay_durations_s` of the right length. It failed on
  `ValueError: Invalid isoformat string: '오전 10시 00분'`: the planner copied the time out of the
  Korean question, and the temporal operators accepted only ISO 8601. `parse_clock_text`
  (`src/tools/spatial.py`) now reads 오전/오후/아침/저녁/밤 with the 12-o'clock boundaries, and
  `_parse_datetime` falls back to it, which fixes `calculate_start_time` on the same path.
- **`nearby_from_need` 1/14.** Every wrong answer was the nearer decoy of the wrong kind the
  family plants deliberately, so the category constraint was never applied. `normalize_analysis`
  returned only `{intent, concepts, measure}`, so a place type the Analysis stage inferred from a
  need was discarded before grounding could bind it; the retrieval then ran with no category and
  the ranking answered "nearest of anything". Analysis now carries `target_type` and
  `_ground_graph_literals` binds it when the question text names no type.

A third defect was in the dataset rather than the port: two families were labelled by their
Appendix E family instead of the intent the agent routes on, and `classification` *is*
`SUPPORTED_INTENTS`. A question that searches a stated radius around one anchor is `radius`
whatever macro family it exercises; the family stays in `template_id`. `Object-Field-Measure`
gained `radius`/`nearby` intents so a neighbourhood-scoped share question can still retrieve it.

The general lesson, and the reason these are recorded as invariants in `AGENT.md`: **a
compositional question only gives structure a chance to pay off — any one stage that cannot read
the question throws that chance away entirely.** Three separate extractors had been written
against a single Korean phrasing each (`반경` only, `에서 가장 가까운` only, ISO 8601 only), and
each silently disabled a whole family while every other stage worked.

## Kakao routing is only reproducible by distance

Measured on one pair (은보갤러리 → 롯데슈퍼프레시 광진구의점, 13,990 m apart in a straight line):

| priority | distance | duration |
|---|---|---|
| RECOMMEND | 17,879 m, then 23,041 m | 2,603 s / 2,457 s |
| TIME | 23,041 m | 3,285 s |
| DISTANCE | 16,992 m | 3,243 s, then 4,337 s |

RECOMMEND and TIME both optimize against live speeds, so the route itself moves between calls —
a 29% spread on distance, wider than the gap between two answer options. DISTANCE is a
shortest-path over the road graph and returned the same distance every time. But the *duration*
moved by a third even on that fixed route, because a duration is always an estimate of current
traffic.

The consequence for this benchmark: a distance gold is a fact about the road network and can be
graded exactly; a duration gold is a snapshot and can only be graded coarsely. `Builder.route`
therefore routes with DISTANCE, and the time-window families space their options at least 85
minutes apart so a traffic estimate cannot reach the neighbouring option. The first v3 build did
neither — it took durations from RECOMMEND and spaced options 25 minutes apart — which is why its
`multisegment_total` golds disagreed with the tools by up to 15% on a whole chain.

## The ReAct baseline was given the composition it was supposed to lack

The paper's claim is that a GeoFlow operator graph beats a ReAct agent working over map API
primitives, and it reports that gain on models as weak as GPT-3.5-Turbo. That rules out "our model
is too small" as an explanation for the gain not reproducing here: a 35B MoE is not worse at
tool-use loops than GPT-3.5-Turbo. The difference is on the other side of the comparison.

MapEval's own baseline (`mapeval-api/Tools.py`) exposes six tools: PlaceSearch, PlaceId,
PlaceDetails, NearbyPlaces, TravelTime, Directions. `src/tools/registry.py` exposes twelve, and
the five extra ones are all aggregations over those primitives:

| tool here | what the paper's baseline must do instead |
|---|---|
| `batch_geocode` | PlaceSearch once per name, across as many turns |
| `batch_place_details` | PlaceDetails once per id |
| `distance_matrix` | TravelTime once per ordered pair, all held in context |
| `calculate_finish_time` | routes, stays and clock arithmetic by hand over many turns |
| `recover_option_places` | ground each option separately |

Aggregating primitives into a typed, executable plan is what GeoFlow is *for*. An earlier revision
of this repository put those aggregations in a tool layer both agents shared, on a rule that the
two must differ only in architecture. **That rule was a design error**, and it is retired. A tool
surface *is* part of an architecture: upstream carries `get_distance_matrix` in
`spatial-agent/src/tools/google_maps.py` and `mapeval-api/Evaluator2.py` hands its baseline nothing
of the kind, so making the surfaces identical did not remove a confound — it deleted the very
difference the paper measures, and handed the baseline the work the architecture was meant to do.

The measurements under the shared surface line up with that reading exactly. In the ReAct run
behind `reports/test_20260819T045330Z.json` (92/100), 79 of 100 questions saw ReAct call at least
one aggregation tool; `trip_finish_time` was 16/16 and `trip_latest_departure` 14/14, every one of
them using them. A four-stop itinerary took five tool calls — one `batch_geocode` plus four
`travel_time` — where the paper's baseline needs four PlaceSearch calls, four TravelTime calls, and
the place ids carried across turns. `batch_geocode` also reconciles the four names against each
other and against an anchor, which PlaceSearch has no equivalent of at all.

So ReAct is now constructed with `allowed=ToolRegistry.MAPEVAL_BASELINE_TOOLS` and `--react-tools`
defaults to `mapeval`. `--react-tools full` restores the shared surface as an explicit ablation —
"does the graph add anything on top of strong aggregation tools" rather than the paper's "does the
graph beat primitives" — and `metadata.react_tools` records which question a run asked. Reports
from the two surfaces must never be pooled; every report predating this default is a `full` run.

The same correction applies to the ReAct prompt. MapEval's baseline is a stock langchain
`STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION` agent whose prompt is the question, the options and
the answer format, with no strategy. `REACT_SYSTEM_PROMPT` used to name the benchmark's question
taxonomy and say which tool each shape wants ("coordinates for direction and straight-line
distance", "directions only for road-route questions"): planning handed to the baseline in prose,
the same error in another currency. It now carries role, evidence discipline and the wire format
only. Tool contracts stay in the tool descriptions, which is where MapEval keeps them too.

`MAPEVAL_BASELINE_TOOLS` is exactly the five tools `Evaluator2.py` constructs at line 33 of
`mapeval-api@35d481a`, mapped onto their Kakao counterparts:

| upstream (`FormattedTools.py`) | here | note |
| --- | --- | --- |
| `PlaceSearch` | `place_search` | upstream returns a place id; Kakao's keyword search returns the place itself, so there is no id round-trip |
| `PlaceDetails` | `place_details` | |
| `NearbyPlaces` | `nearby_places` | |
| `TravelTime` | `travel_time` | |
| `Directions` | `directions` | |

Two of this registry's primitives are deliberately **not** in that set. `geocode` (address →
coordinates) and `reverse_geocode` (coordinates → address) have no upstream counterpart at all:
upstream reaches every place through a place id and never converts between an address and
coordinates. They are primitives rather than aggregations, so they do not hand ReAct the
composition GeoFlow provides, but they are still capability the paper's baseline was measured
without — and in the ReAct run behind `reports/test_20260819T045330Z.json`, 40 of the 479 tool
calls were `geocode`, most of them on bare place names (`플라이아트센터`, `노량진컵밥거리`), which
is PlaceSearch reached through a second index rather than address handling.

`Tools.py` additionally defines a `PlaceIdTool` that `Evaluator2.py` never instantiates; an
earlier revision of this document and of `AGENT.md` counted it and called the baseline six tools.
It is five. `FormattedTools.PlaceSearchTool` is documented as "Get place ID for a given
location", which is the same primitive.

### The port is finished, and that is a rule rather than a status

`src/agent/react.py` carries the element-by-element comparison in its header. What matters beyond
the table is the direction of maintenance: **the baseline is not edited in response to a benchmark
result.** An accuracy gap ReAct shows *is* the finding, so closing one by rewriting its prompt,
raising its budget, or glossing a parameter it reaches would make the baseline a function of the
test set — the same overfitting the shared-tool-surface correction above was about, one layer
down. Only two surfaces stay live, and both are shared with the other architecture: the provider
below the tools, and the tool contracts both agents read. A change to either has to be argued from
the provider or from upstream, never from a question ReAct got wrong.

Two consequences, applied together:

- `directions` returns its turn-by-turn guidance by default, and `travel_time` does not
  (`TravelTimeArgs`). Upstream's `DirectionsTool` prints every step's instructions on every call
  while `TravelTimeTool` reports one duration and one distance — the two tools differ by what they
  *report*, not by what an agent has to know to ask for. Our port had `include_steps=False` on
  both, which made the guidance a capability the port withheld from the baseline rather than one
  MapEval's design withholds; `steps_analysis` on the Spatial-Agent side had to bind the flag
  during grounding to get a route it could count turns on, and that binding is now unnecessary.
- `DirectionsArgs.priority` is documented as `"RECOMMEND, TIME, or DISTANCE"` — the accepted
  values, the way upstream documents `travelMode` as "(driving, walking, bicycling, transit)". An
  earlier revision glossed each value with what it optimizes, written after watching ReAct read a
  question that says 거리가 가장 짧은 경로로 as RECOMMEND. Kakao requires the parameter, so the
  baseline has it; *which* priority a question asks for is grounding, and grounding is a
  Spatial-Agent stage under measurement (`_extract_route_priority`). A write-up should report that
  asymmetry as architectural, which it is, rather than close it in the baseline's prose.


## Why the v2 scores are high, measured rather than guessed

ReAct 91 and Spatial-Agent 96 on `dataset/seoul_kmapeval_v2_mcq_100.jsonl` sit far above what
MapEval reports for its API split, and a *baseline* at 91 from a 35B-A3B model is the part that
cannot be right. Five candidate explanations were measured on the run behind
`reports/test_20260819T224214Z.json` (ReAct) and `reports/test_20260819T225908Z.json`
(Spatial-Agent). Four are refuted:

| candidate | measurement | verdict |
| --- | --- | --- |
| the questions are answerable from the model's own knowledge | no-tool floor **31/100**, chosen-option histogram 29/26/23/22 against a 25% chance rate | refuted |
| our place-matching stack carries the baseline | for the 247 option names Kakao returns candidates for, `_best_place_match` agrees with `candidates[0]` — upstream's `results[0]` — in **208**; the 7 overrides are all non-place option strings ("좌회전 (아차산로)", "7번") | refuted |
| ReAct's 30-step budget against langchain's 15 | ReAct used at most 10 steps, and only **5/100** questions took more than 15 tool calls | refuted |
| Spatial-Agent's score is our question-literal grounding | `_ground_graph_literals` replaced by a pass-through: **96 → 92**, and the loss is almost all `routing` (22 → 18, the priority binding) while `trip` rose 22 → 24 | refuted |
| **the questions do not require the map to find anything** | **91/100** of Spatial-Agent's drafted graphs contain no retrieval operator at all; ReAct called `nearby_places` on **16/100** questions and `place_search` 480 times | **confirmed** |

The shape of a v2 answer, from the trace of a radius question:

```
batch_geocode(["호스텔온기"])                     # the anchor, verbatim in the question
batch_geocode([option0, option1, option2, option3])  # the four options *are* the candidate set
within_radius(center, candidates, 800)               # the radius, verbatim in the question
identity_measure -> ^^2^^
```

**The MCQ options are the candidate set, so retrieval never happens.** "Which 대형마트 is within
800 m" becomes "geocode four names and test a predicate" — which is also exactly what
`data/verify_benchmark.py` does, and why it re-derives 100/100. The hard part of a map agent,
finding the places, is answered by the question paper.

Three construction rules remove what is left:

- `Builder.resolves_to` admits a place only if its bare name searches back to within 200 m, so
  place resolution — the failure mode MapEval's baseline spends its budget on — cannot fail. That
  is also why the naive `results[0]` and our matcher agree.
- Ties are rejected with wide margins, and the recorded evidence shows how wide: a
  `routing_via_compare` row offers detours of 1639 / 3406 / 3568 / 3868 s, a factor of two; a
  `trip_feasible_count` row has a 4 h budget against best finishes of 1.3 / 3.1 / 5.2 h. Only
  `trip_optimal_order` is close (194 s over ~2900 s). A constant "15 minutes per leg" and **no map
  at all** answers `trip_feasible_count` 9/10.
- No `unanswerable` class, where MapEval-API puts 20 of its 300 rows and where models lose most.

And one thing no construction rule can fix: Kakao Local publishes no rating, no price level and no
opening hours, so the whole attribute-reasoning half of MapEval-API — Place-Attribute-Query, and
every question that turns on whether somewhere is open or well reviewed — has no Korean
counterpart here. What remains is geometry and routing, which are exact.

So v2 measures evidence handling and arithmetic over exact lookups, not map reasoning, and its
numbers must never be presented as reproducing the paper's. `dataset/seoul_kmapeval_v3_mcq_100.jsonl`
already carries the fix for the biggest of these: **six of its seven families answer with a value**
(`오후 3시 07분`, `약 49.8km`, `약 86%`, `3번`, `동쪽, 약 6.9km`), which no amount of geocoding the
options can shortcut. Only `nearby_from_need` still offers POI names, and it is the family whose
plans do retrieve.

Report the floor next to every accuracy (`data/measure_no_tool_floor.py`). Without it, 91 against
96 is uninterpretable.

## How MapEval built its QA set, and what v4 ports

`mapqator-backend/database/schema.sql` shows the pipeline behind `mapeval-api/dataset.json`: an
annotator issues map API calls that are cached in `places` / `nearby` / `distance` / `directions` /
`inside`, the calls are assembled into one `context`, and a person then writes the question and the
options against that evidence (`dataset.username` records who). `human` holds the human baseline
that the paper reports models against. There is no generator — the questions are hand-written over
collected evidence.

Counting `dataset.json` says what the annotators actually wrote, per class:

| class | value options | name options |
| --- | --- | --- |
| nearby | 19 | 64 |
| poi | 36 | 28 |
| routing | 44 | 22 |
| trip | 48 | 19 |
| unanswerable | 5 | 15 |

`['South, 13.45 kilometers', …]`, `['61.224 km', …]`, `['10.13', '10.23', …]`, `['1','2','3','0']`.
Every one of v2's eleven families offers place names, orderings or guidance strings instead, which
is where its retrieval bypass comes from — geocode the four options and the question is over.

`data/build_mapeval_benchmark.py` ports the method rather than the proportions:

- **Values where upstream uses values.** 45 of 100 rows. `poi_straight_distance`,
  `routing_distance_via`, `trip_total_distance`, `trip_arrival_clock`, `routing_turn_count`,
  `poi_direction_distance`.
- **A constraint where upstream uses names.** Upstream asks for an *orthopedic* hospital; Kakao's
  paths carry the same subtypes, so `nearby_clinic_subtype` / `nearby_cuisine_subtype` offer three
  *nearer* siblings and the subtype is what decides. The gold is selected with
  `matches_required_type` — the tools' own test — because choosing it on the category path while
  `filter_places` also reads the name left one question whose true answer was not among its
  options.
- **The unanswerable class, which here is not a choice.** Kakao publishes no rating, price level or
  opening hours, so upstream's attribute half cannot be asked. Those 7 rows ask it anyway and the
  gold is the refusal; `data/verify_mapeval_benchmark.py` verifies them in the negative, by
  checking every candidate carries `None` for the field.
- **No engineered margins.** v2 rejected close calls; MapEval's annotators read whatever the
  evidence said. Margins survive only where Kakao is not reproducible: `trip_arrival_clock`
  out-spaces the traffic, `trip_feasible_count` keeps only budgets whose answer holds at ±30% per
  leg. `trip_feasible_count` additionally rejects any row a constant travel assumption answers —
  the flaw measured on v2, where "15 minutes a leg" and no map scored 9/10.

### The floor is part of the build loop, not a report afterwards

The first cut of v4 measured **52/100** with no tools, which is not a benchmark. The leaks were in
the option sets and the fix came from upstream's own data:

| family | first floor | leak | fix | floor now |
| --- | --- | --- | --- | --- |
| `nearby_clinic_subtype` | 15/16 | the gold was the only option carrying 정형외과 in its name | three options of the requested subtype plus one *nearer* place of another — upstream's "nearest Mosque among four mosques" beside its "orthopedic hospital among several specialties" | 4/16 |
| `nearby_cuisine_subtype` | 8/12 | same, weaker | same | 5/12 |
| `poi_which_is_closer` | 6/7 | two options over a city the model knows | dropped; its rows went to `poi_straight_distance` | — |

**35/100 overall**, against a 25% chance rate, with every measured family at or near chance:
`routing_turn_count` 0/7, `trip_feasible_count` 1/7, `trip_total_distance` 1/7,
`routing_distance_via` 2/8. The 7 `unanswerable_*` rows score 5/7 without tools, and that is
correct rather than a leak — recognising that a question has no answer is a knowledge task, which
is why MapEval includes the class at all.

Building v4 also surfaced a construction bug the other benchmarks may share: a question was asked
about the *pool's* copy of its anchor while an agent reaches the anchor through `place_search`, and
the two sit up to `ROUND_TRIP_TOLERANCE_M` apart. One 피부과 question put its gold at 153 m from
the pool's 신사동가로수길 while the resolved anchor had a different clinic at 93 m — a gold no
agent could reach. `Builder.as_resolved` returns the resolved place, and the same tolerance is why
a same-subtype option must sit at least 60 m farther than the gold.

Two deviations, both because we have no annotators and one because upstream leaks:

- The questions are templated. The *shape* of every family is upstream's; the phrasing is
  generated, so v4 does not carry the colloquial naming ("the hospital", "the lake") that makes
  upstream's place resolution hard.
- `Evaluator2.py` prepends "Option0: Unanswerable" only when a row is unanswerable, which tells the
  model the answer before it reads the question. Here "주어진 지도 정보로는 알 수 없음" is an
  ordinary option: gold on the 7 unanswerable rows and a distractor on 22 answerable ones.

## What v4 measured, including the part that refutes the diagnosis

First paired run, `reports/test_20260820T011219Z.json` (ReAct) and `test_20260820T014038Z.json`
(Spatial-Agent), prompting-only, `--react-tools mapeval`, provider kakao:

```
no-tool floor  35/100
ReAct          89/100
Spatial-Agent  91/100
```

**The option-set bypass was not what made v2 easy.** Closing it — 45 of 100 rows now answer with a
value, and the constrained families punish both name-reading and proximity — moved the floor from
31 to 35 and the accuracies from 91/96 to 89/91. The families with the lowest floors are the ones
the agents score highest on: `routing_turn_count` floor 0/7 against 7/7 and 7/7,
`trip_total_distance` floor 1/7 against 7/7 and 7/7. So the questions do require the map, the
agents use it, and they are simply good at it. That correction stands against the hypothesis in the
section above, which predicted a substantial drop.

Two things it did establish:

- **The architectures are indistinguishable here.** 83 questions both answer, 3 neither, 6 ReAct
  only, 8 Spatial-Agent only — exact McNemar p = 0.79.
- **A graph that must end in a Measure cannot refuse.** All three of Spatial-Agent's losses to
  ReAct outside the noise are `unanswerable_*` rows: ReAct reads the options, finds the map silent
  and picks the refusal; the GeoFlow pipeline composes a graph, executes it and reports a number.
  That is an architectural property worth a line in a write-up, and it is the opposite of the
  paper's direction.

What is left to explain the gap with MapEval's reported numbers, in the order the evidence supports:

1. **The evidence layer never fails at plumbing.** Upstream's agent threads `place_id` *strings*
   parsed out of langchain's text protocol through a REST backend: `PlaceSearch` returns an id,
   every other tool consumes one, and dropping or garbling it is a whole error class. Ours passes
   normalized `Place` objects, accepts names and coordinates interchangeably, and normalizes every
   place-typed argument. Measured on this run, only 5.9% of ReAct's tool calls and 1.1% of
   Spatial-Agent's returned an error at all. That difference sits *below* both agents, which is
   exactly why both land near 90 — and porting it faithfully (ids only, text returns) is the one
   remaining change that would move these numbers.
2. **The attribute half of MapEval-API cannot be ported.** "Can I visit at 5 PM Saturday?", "a
   restaurant rated 4.8+", "does it serve dinner" — Kakao Local publishes no rating, price level or
   opening hours. Those are where MapEval's models actually fail, and here they exist only as the
   7 unanswerable rows.
3. **The questions are templated.** No colloquial referring expressions ("the hospital", "the
   lake"), no typos, and every place named as Kakao stores it, so the question text never makes
   resolution hard.
4. **The model is current.** `nvidia/Qwen3.6-35B-A3B-NVFP4` against the paper's 2024 frontier
   models, on a benchmark whose measure is exact arithmetic over exact lookups.

## Porting the id-threading, and what it cost

The section above named the plumbing as the last untested explanation for the gap with MapEval's
reported numbers, so it was ported: a place argument on the five baseline tools is now a reference
the provider issued, and a name is refused with an error saying to call `place_search` first
(`ToolRegistry._reference`, `MapProvider.dereference`, and the same refusal inside both providers).
The aggregations keep resolving names, because resolving the names a plan holds is what makes them
aggregations over PlaceSearch.

Same benchmark, same model, before and after:

| | tool calls | error rate | "not a reference" errors | accuracy |
| --- | --- | --- | --- | --- |
| ReAct, names accepted | 579 | 5.9% | — | 89/100 |
| ReAct, references only | 784 | **25.5%** | 159 | 87/100 |
| Spatial-Agent, names accepted | 1080 | 1.1% | — | 91/100 |
| Spatial-Agent, references only | 1082 | 0.7% | 0 | 91/100 |

The burden is real and it is asymmetric exactly as upstream's design implies: a quarter of ReAct's
tool calls now fail, 159 of them on the id-threading MapEval's baseline does by hand, while
Spatial-Agent never hits it — its planner resolves names in one `batch_geocode` and carries
references through the graph, which is the architectural claim the paper makes.

**It moved the accuracy by two points.** ReAct 89 → 87, Spatial-Agent 91 → 91, exact McNemar
p = 0.50 — still indistinguishable. The reason is visible in the same table: ReAct *recovers*. It
reads the error, calls `place_search`, and retries, spending 784 calls and 550 reasoning steps over
100 questions — an average of 5.5 against a budget of 30. The plumbing costs it turns, not answers,
because the budget absorbs them and the error message says what to do. Upstream's 15-iteration
default and its silent "Incorrect place name" would absorb fewer.

So the ledger on this benchmark now reads: floor 35, ReAct 87, Spatial-Agent 91, and the
architectures separated by 4 points that a paired test does not distinguish. What remains
unported is not plumbing but evidence — Kakao publishes no rating, price level or opening hours,
so the half of MapEval-API that turns on reading an attribute table cannot be asked here at all,
and the questions are templated rather than written by a person. A write-up should say that the
comparison reproduces MapEval's *method* and not its *difficulty*.

One family keeps pointing the other way, and it is the one worth reporting: Spatial-Agent scores
3/7 on the `unanswerable_*` rows against ReAct's 7/7. A graph that must terminate in a Measure
composes, executes and reports a number; ReAct reads the options, finds the map silent and picks
the refusal.

## The v3 run, and why 89/98 is not this repository's number

`reports/test_20260820T114441Z.json` and `test_20260820T120113Z.json` read ReAct 89/100 and
Spatial-Agent 98/100. Both were run against `dataset/seoul_kmapeval_v3_mcq_100.jsonl`, which is the
*compositional* benchmark and not the MapEval-method one; `main.py --dataset` defaults to v4 and
those two runs overrode it. Four properties of v3 make that pair unreportable as an architecture
result, and they are worth writing down because three of them survive into v4 in weaker form.

1. **Seven families, not a hundred questions.** v3 draws from seven templates and both agents
   saturate five of them: `trip_finish_time` 16/16 and 16/16, `poi_bearing_and_distance` 14/14 and
   14/14, `trip_latest_departure` 14/14 and 13/14. The effective sample is the family count, so
   the interval around 98/100 is nothing like the interval around 98 independent trials.
2. **The wrong options sit outside the measurement.** `_time_options` places them 85 to 200
   minutes from the gold and `_survives_traffic` *discards* any row whose answer could flip under a
   second traffic sample. Thirty of the hundred rows are that family. Any agent that computes the
   legs at all answers them; the question grades whether the arithmetic happened, not whether it
   was right.
3. **The gold and the run read the same bytes.** `data/benchmark_core.py` builds every gold through
   `src.tools.kakao.KakaoMapProvider` against `data/kakao_cache.db`, which is the same cache file
   and the same TTL the evaluation uses. `seoul_kmapeval_v3_000` was answered with **0 API calls
   and 8 cache hits** — the very responses its `leg_s` was computed from. This is deliberate ("a
   wrong answer means the agent, never a source mismatch"), and the cost of it is that the whole
   provider-disagreement error class, which is where a human-curated gold like MapEval's puts real
   agents, cannot occur here.
4. **v3 has no `poi` class and no unanswerable rows.** Those are upstream's two weakest classes
   (53.1% and the 20 rows it excludes). v4 fixes the composition — its `mapeval_class` mix is
   28/22/22/21/7 against upstream's 27.7/22.3/22.0/21.3/6.7 — and v3 does not have it.

`_select_option` also carries a deterministic override, `_computed_clock_option`, that outranks the
generation stage whenever every option is a wall clock and the graph computed exactly one. It
decided **27 of the 100** answers in that Spatial-Agent run. On this run it agreed with the
generation stage's own answer text on all 27, so it did not move the score, but it is answer
selection performed by the harness rather than by the architecture and it should be reported
whenever the trace shows `"selection_method": "computed_clock"`.

## No held-out split, and what v5 is for

The deeper problem is not any one family. `src/agent/spatial.py`'s grounding stage, its
`ANALYSIS_PROMPT` intent rules and its operator contracts were each edited in response to specific
misses on these same hundred rows — the source comments name the questions, and the commit log
reads as a sequence of them. That is legitimate development, and `src/agent/react.py` is explicitly
protected from it ("do not tune it against benchmark results"), which means the *gap* between the
two agents is partly a gap between a tuned system and an untuned one. Nothing in this repository
has ever been measured on a sample it had not already been fixed against.

Two things follow, and both are now runnable:

- `data/build_mapeval_benchmark.py` and `data/build_mapeval_v5_benchmark.py` take `--seed` and
  `--id-prefix`. A build under a seed nothing under `src/` has seen draws a different sample from
  the same 12,138-place pool and is a **held-out** set. Report its number separately; it is the
  first accuracy here that is not also a training-set accuracy.
- `data/build_mapeval_v5_benchmark.py` keeps v4's method and class proportions and replaces the
  three things that made v4 easier than MapEval-API, each read off `MapEval-API.jsonl` itself:

  | property | upstream | v4 | v5 |
  | --- | --- | --- | --- |
  | option spacing | `['18 mins','19 mins','20 mins','21 mins']` | clocks 90–195 min apart, lengths 28–75% apart | tight options, built only from reproducible measures (straight line, DISTANCE-priority length, turn counts, orderings by length) |
  | nearby shape | "the **second** nearest park", "within **500 meters** of" | nearest-of-a-subtype, answered by taking row 0 | `nearby_second_nearest`, `nearby_within_radius` — the nearest right-kind place is a distractor |
  | unanswerable | "the most beautiful route", "best for fresh seafood" | "Kakao has no rating column", one fact learned once | `unanswerable_subjective` — a ranking no map publishes, over four real neighbours |

  It also restores `trip_optimal_order`, which is upstream's signature trip shape and the class it
  scores 55.2% on, and drops `trip_arrival_clock`, whose options cannot be tightened: a Kakao
  duration is a live estimate (3,243 s and then 4,337 s for the same route), so a clock question is
  only gradeable with options spaced past that spread, and options spaced past that spread are what
  made the family free. Ordering in v5 is by **road length at DISTANCE priority** rather than by
  duration for the same reason, which is a deviation from upstream's "most optimized order" and
  belongs in the write-up.

### What building v5 cost, and the two defects it exposed

Built against live Kakao on 2026-08-20: **100/100 rows, 4,887 API calls**, class mix 28/22/22/21/7
against upstream's 27.7/22.3/22.0/21.3/6.7, gold positions 28/23/28/21. Two things only a real
build could show, both now fixed in the builder:

- **A category rotation keyed on `len(made)` deadlocks.** `nearby_second_nearest` picked its Kakao
  category by how many rows it had already produced, so the first category with no usable
  neighbourhood anywhere pinned every remaining anchor to itself: 2 rows of 6, after 3,389 Kakao
  calls. Rotating by the anchor being tried costs a barren category one anchor instead of the
  family. `nearby_within_radius` and `unanswerable_subjective` carried the same latent bug.
- **"Which of these is within 500 m" is the nearest-of-a-type question wearing a hat.** With
  exactly one option inside the radius, the one inside is necessarily the nearest of the four, so
  an agent that ranks and reports row 0 answers it without ever reading the radius — precisely the
  shortcut the family was added to punish. It is asked as a *count* instead ("다음 네 약국 중 …
  몇 곳인가요"), with one, two or three of the four inside, so every option has to be measured and
  the options are a value rather than a name. The count ladder must not be shuffled, which is why
  `finalize` now takes its `ordered` families as an argument.

The first defect is worth generalizing: every family in `data/build_*.py` that rotates a category,
a complaint or a cuisine by `len(made)` has it, and it shows up as a family that quietly returns
fewer rows than its quota rather than as an error.

### The floor found a second answer key, and v4 has it too

v5's first floor came back **36/100**, and three families sat well above the 25% chance rate:
`poi_straight_distance_tight` 6/8, `trip_optimal_order` 5/8, `poi_address_district` 3/5. The first
was not the model knowing Seoul. `_distance_options` was called with a fixed multiplier set on
every row, so once the four printed lengths are sorted **the gold sits at a constant index**, and
"take the second smallest" answers the family with no map at all.

The shipped v4 file has the same tell in four families, three of them deterministic:

| family | multipliers | gold's rank among the sorted options |
| --- | --- | --- |
| `poi_straight_distance` | `(1.28, 1.75, 0.62)` | second-smallest, 10 of 10 parsed rows |
| `routing_distance_via` | `(0.78, 1.42, 1.9)` | second-smallest, 6 of 6 |
| `trip_total_distance` | `(0.78, 1.45, 0.55)` | third-smallest, 5 of 5 |
| `poi_direction_distance` | `1.28` against the true length | one of the two smallest, 6 of 6 |

That is **28 of v4's 100 rows answerable by sorting the options and taking a constant index**. The
closed-book model did not find the rule — v4's measured floor of 35/100 stands as a floor, and the
87/91 pair was not produced this way — but a benchmark with a second answer key is unsound
regardless of whether a particular model used it, and any accuracy quoted from that file has to
carry this paragraph. `straddling_multipliers` draws how many wrong lengths fall below the gold,
per row, and is now used at every call site in both builders; the shipped v4 file predates it and
should be rebuilt before it is quoted again.

`poi_address_district` was dropped outright: floor 3/5, because the model knows which 구 an
ordinary Seoul address sits in. Its function stays in the file as the record of a family that did
not survive its own floor. `trip_optimal_order` drew its stops from attractions and cultural venues
only, which a model can order from memory of the city; marts are in the pool now.

### Every number this repository has published is one sample from a temperature-1 decoder

Re-measuring the floor after the fixes gave **27/100**, then **38/100**. Families the rebuild did
not touch moved as much as the ones it did: `routing_next_turn` 2/7 then 6/7, `trip_feasible_count`
1/7 then 5/7. That is not a benchmark property.

`OpenAIChatClient.chat` sent no `temperature`, so the endpoint's own default decided it — 1.0 on a
vLLM deployment. **Both upstreams decode greedily**: `mapeval-api/GPT_4o_mini.py:10` and
`spatial-agent/src/agent/spatial_agent.py:215` each construct their client with `temperature=0`.
So every accuracy in the sections above — the v2, v3 and v4 runs, both floors, the id-threading
before-and-after table, the McNemar tests — is a single draw from a sampler the reference
implementations do not use, and the two-point and four-point differences those sections reason
about are inside the spread just measured.

`Settings.llm_temperature` now exists and defaults to `0.0`, which is the upstream setting;
`LLM_TEMPERATURE` overrides it. Anything quoted from before this change should be re-run, and a
report that compares two architectures should say which temperature produced it.

### And temperature was not the whole of it: this endpoint is not reproducible at all

Setting it to 0 did not settle the floor either — two more runs over the identical file gave
**24/100** and **32/100**. Probing the endpoint directly, six identical requests at
`temperature=0` returned four distinct answers, and nothing fixed it:

| request | distinct answers in 5 calls |
| --- | --- |
| `temperature=0` | 5 |
| `temperature=0, seed=12345` | 3 |
| `temperature=0, top_p=1, top_k=1` | 3 |
| `temperature=0, seed=12345, top_p=1, top_k=1` | 4 |

No sampling parameter reaches it, so the variation is not sampling. `nvidia/Qwen3.6-35B-A3B-NVFP4`
is a mixture-of-experts model served behind a reverse proxy: expert routing and reduction order
shift with whatever else is in the batch, and a proxy may be spreading requests over replicas. The
practical consequence is the same either way.

**A single run of 100 questions on this deployment carries a spread of roughly ±8 points.** That is
wider than every architecture difference this repository has reported — ReAct 89 against
Spatial-Agent 98 on v3 is 9 points, 87 against 91 on v4 is 4 — and the McNemar tests in the
sections above were computed on one draw each, so they measure disagreement between two agents on a
single sample and not the instability of either. No comparison here is safe until each
configuration is run several times and the difference is read against the repeat-to-repeat spread.
Report *k* runs, not one.

`data/build_mapeval_benchmark.py` has not been re-run under a held-out seed yet. The build loop
that v4 established applies unchanged — build, measure the no-tool floor with `data/measure_no_tool_floor.py`, drop any family
whose floor is high, and only then run the agents. `poi_address_district` is the family to watch
there, since a model may know a district without a map; it already rejects any place whose name
carries its own district, and the floor is what decides whether that is enough.

## What v5 measured

`reports/test_20260820T135925Z.json` (ReAct) and `test_20260820T162733Z.json` (Spatial-Agent),
`--provider kakao`, `--react-tools mapeval`, `temperature=0`, prompting-only:

```
no-tool floor  24/100 and 32/100   (two runs, same file)
ReAct          84/100
Spatial-Agent  88/100
```

75 questions both answer, 3 neither, 9 ReAct only, 13 Spatial-Agent only — exact McNemar p = 0.52.
**The architectures are still indistinguishable**, and the four-point gap is well inside the
repeat-to-repeat spread the floor just showed. That is the same conclusion v4 reached, now on a
benchmark whose floor sits at the 25% chance rate instead of ten points above it.

| family | ReAct | Spatial-Agent | floor (2 runs) |
| --- | --- | --- | --- |
| `nearby_clinic_subtype` | 9/10 | 10/10 | 0, 1 |
| `nearby_cuisine_subtype` | 5/8 | 6/8 | 2, 4 |
| `nearby_second_nearest` | 5/6 | 6/6 | 3, 0 |
| `nearby_within_radius` | **4/4** | **0/4** | 0, 1 |
| `poi_direction_distance_straddled` | 10/10 | 10/10 | 2, 3 |
| `poi_straight_distance_tight` | 11/11 | 11/11 | 4, 2 |
| `routing_distance_via` | 6/8 | 8/8 | 3, 4 |
| `routing_next_turn` | 7/7 | 7/7 | 2, 2 |
| `routing_turn_count` | 6/7 | 7/7 | 2, 3 |
| `trip_feasible_count` | 7/7 | 7/7 | 2, 2 |
| `trip_optimal_order` | 6/8 | 6/8 | 2, 4 |
| `trip_total_distance` | **2/7** | **7/7** | 0, 2 |
| `unanswerable_*` (4 families) | 6/7 | 3/7 | 2–3 of 4 on the subjective rows |

Three results the earlier benchmarks could not isolate:

- **Aggregating a length is where the architectures actually differ.** `trip_total_distance` runs
  ReAct 2/7 against Spatial-Agent 7/7 — a complete inversion, on the family v4 had both agents
  scoring 7/7 because its options were 22% to 45% apart. Tightened, ReAct accumulates four legs by
  hand across four tool calls and drifts; the GeoFlow graph puts them through one `distance_matrix`
  and sums exactly. This is the paper's own claim about composed retrieval, and it took options
  narrow enough to punish drift before it showed up.
- **Counting membership against a stated radius is where Spatial-Agent breaks.** `nearby_within_radius`
  runs 4/4 against 0/4. None of the four is a parse failure — every one returned a valid count, just
  the wrong one, on rows whose insides sit at 421/264/390 m against a 500 m boundary. A pipeline
  that composes a graph per question gets a question whose answer is a *count over a predicate*
  wrong in a way that reading four distances one at a time does not.
- **A graph that must terminate in a Measure still cannot refuse.** ReAct 6/7 on the unanswerable
  rows against Spatial-Agent's 3/7, reproducing v4's finding on new questions — including the
  subjective ones, where the map holds the places and simply does not rank them.

Two families stay saturated for both, and it is worth saying why rather than tightening them
further: `poi_straight_distance_tight` and `poi_direction_distance_straddled` rest on a haversine
over two resolved coordinates, which is exact arithmetic. No option spacing defeats it, so those 21
rows measure name resolution and nothing else. MapEval's distance questions are hard because
Google's *road* distances and durations are estimates; that half cannot be asked here.

## v6: every family raised, and the radius family's word order fixed

Seven of v5's fourteen families were saturated by both agents, and a saturated family cannot show a
difference. Tightening their options does not help where the measure is exact — a haversine over
two resolved coordinates is arithmetic, not an estimate — so v6 raises each family along one of two
axes instead: **composition**, where the answer needs two measurements and an operation between
them, and **ordinality**, which denies the agent the first row of a ranking.

| v5 family | v6 family | what changed |
| --- | --- | --- |
| `poi_straight_distance_tight` | `poi_distance_difference` | two haversines and a subtraction |
| `poi_direction_distance_straddled` | `poi_farthest_of_three` | three haversines and a maximum |
| `nearby_second_nearest` | `nearby_kth_nearest` | k drawn from 2..4, options from ranks 1..6 |
| `nearby_clinic_subtype` | `nearby_subtype_kth` | the k-th of a named subtype, not the nearest |
| `routing_next_turn` | `routing_nth_turn` | count into the guidance list, not match a road |
| `routing_turn_count` | `routing_turn_count_via` | counted on a route through a waypoint |
| `routing_distance_via` | `routing_detour_cost` | via length minus direct length |
| `trip_optimal_order` | `trip_optimal_order_four` | four stops, 24 orders instead of 6 |
| `trip_total_distance` | `trip_total_distance_four` | four stops, five legs |
| `nearby_within_radius` | `nearby_within_radius_count` | word order fixed, radius 300/500/800 |

`nearby_cuisine_subtype`, `trip_feasible_count`, `unanswerable` and `unanswerable_subjective` carry
over unchanged, because a family that is already discriminating only spends rows when raised.
Class proportions stay MapEval-API's: nearby 28, poi 21, routing 22, trip 22, unanswerable 7.

**The radius family's 0/4 was the question's fault, not the agent's.** v5 asked "다음 네 약국 중
<anchor>에서 반경 500m …", and `_extract_anchor` splits a radius question on `" 반경"` and takes
everything before it — so the anchor handed to `batch_geocode` was the whole clause
"다음 네 약국 중 신이문역 1호선에서", which resolves to nothing and fails every step downstream. All
four traces are that one failure. No Korean speaker puts the list before the landmark, so v6 asks
"<anchor>에서 반경 500m 이내에 있는 약국은 아래 목록 중 몇 곳인가요?" — the ordinary word order, and
the one `_extract_anchor` was written for. Fixing the phrasing was right and tuning the splitter
would not have been: the agent's reader is part of the architecture under test.

The counts are spread across the ladder deliberately: keying the answer on a loop index spends the
rungs wherever the loop happened to succeed, which is how v5's four-rung radius family shipped
drawing from three values. `dataset/seoul_kmapeval_v6_mcq_100.jsonl` passes
`data/audit_dataset.py`. **No agent has been run against it yet**, and its no-tool floor is not
measured either; both have to happen before any v6 accuracy means anything.

## The baseline's tool *names* were upstream's; its arguments were not

`MAPEVAL_BASELINE_TOOLS` restricts ReAct to the five names `mapeval-api/Evaluator2.py` line 33
constructs, and a test pinned that roster. The roster was never the surface. Field for field
against `mapeval-api/Tools.py` and `FormattedTools.py` at 35d481a:

| upstream | ours, before `contract="reference"` |
| --- | --- |
| `PlaceSearch(placeName)` → `data['results'][0]['place_id']`, one id | `query`, `center`, `category_code`, `radius_m`, `min_rating`, `open_now`, `limit`, returning ranked candidates |
| `PlaceDetails(placeId)` | same |
| `NearbyPlaces(placeId, type, rankby, radius)`, and `rankby=distance` **refuses** a radius | `center`, `query`, `category_code`, `radius_m`, `limit` — bounded *and* distance-ordered in one call |
| `TravelTime(originId, destinationId, travelMode)` | + `priority`, + `waypoints` (up to 30), + `include_steps` |
| `Directions(originId, destinationId, travelMode)` | the same three additions |

An argument is a capability, so this was a stronger baseline than the paper's, and the two places
it shows are measurable:

- **Waypoints.** A detour is two routes and an addition for upstream's agent. On the v5 run ReAct
  issued a waypointed `directions` call on all 8 `routing_distance_via` rows and took 6 of them.
- **Radius with distance ordering.** "The nearest pharmacy within 500 m" is two calls and a
  comparison upstream; ours answered it in one, which is the whole of `nearby_within_radius`.

`ToolRegistry(provider, contract="reference")` replaces the five with upstream's contracts,
including its refusal text word for word, and `--react-tools reference` is now the default.
`native` is the old surface under its true name — a stronger-than-paper ablation — with `mapeval`
kept as an accepted alias so saved commands still run. `full` is unchanged. The three are recorded
in report metadata and are not poolable. `tests/test_tools_and_agents.py` pins the argument sets of
both contracts, not just their names.

Verified live against Kakao: `place_search("서울역")` returns `'9113903'` and nothing else, an
unresolvable name returns upstream's "Incorrect place name. Please use the same name as in the
question.", `nearby_places(rankby="distance", radius=500)` returns upstream's refusal,
`travel_time` reports a duration and distance with no steps while `directions` reports 13, and a
`waypoints` argument is rejected by the schema.

**Every accuracy in the sections above was measured on the `native` surface.** They are ReAct with
arguments MapEval's baseline does not have, and should be relabelled rather than re-read.

### Two related claims, checked

- **`MAX_REASONING_STEPS = 30` had no upstream basis, and the loop had two more deviations.**
  Fixed; see the section below.
- **"The paper's ReAct" is the wrong name for this port.** Spatial-Agent's Table 1 reports
  `ReAct (GPT-4o-mini)` at 32.98% and `MapEval API (GPT-4o-mini)` at 23.00% as separate rows, and
  what is ported here is the second one's public code. The paper ships no implementation of the
  first, so equivalence with it cannot be checked. Call this a MapEval-API baseline port.

## The loop was the other half of the baseline

A tool surface is what an agent *can* reach; the loop is how many times and how widely it may
reach. Ours was stronger than upstream's in three ways at once, and all three are now tied to
`--react-tools reference`.

| `mapeval-api/Evaluator2.py` @ 35d481a | ours, before this change |
| --- | --- |
| `initialize_agent(...)` with no `max_iterations` → langchain's default **15** | `MAX_REASONING_STEPS`, **30** in the run environment |
| `STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION` parses **one** JSON action per response | every tool call in the assistant message is executed |
| `early_stopping_method="force"` (the default) returns `"Agent stopped due to iteration limit or time limit."` and makes **no further call** | an extra LLM call asking for the best-supported option |

The 30 was argued from `mapeval-api/mapeval_api_evaluator.py`. That file is **untracked** in the
upstream checkout — `git ls-files` lists `Evaluator2.py` and not it — so it is a local adapter and
cannot justify anything about the paper's baseline.

The other two compound: parallel calls turn one of upstream's iterations into many, and the forced
final answer converts an exhausted budget into a scored guess. On the v5 run 17 of 100 questions
issued more than 15 tool calls and 13 of those were correct; one question executed 24 tool calls in
6 LLM rounds, which upstream's parser could not have produced at any budget.

`ReactAgent` now takes `max_steps`, `single_action` and `force_final_answer`, and `main.py` sets
them from the surface — `reference` gets 15 / one action / forced stop, `native` and `full` keep
what this repository had. `ITERATION_LIMIT_OUTPUT` is langchain's own string, so an exhausted run
ends as an `answer_parse_failure` exactly as upstream's regex leaves it. Three tests pin it: one
action per iteration with no orphan `tool_call_id`, an exhausted budget answering nothing and
costing no extra call, and the native loop still executing several.

Report metadata now carries `llm_temperature`, `react_max_iterations`, `react_parallel_tool_calls`
and `react_forces_final_answer` beside `react_tools`. Every accuracy recorded above predates all
of them.

## What a question costs, and how much of it is thinking

Per-question logs, report rows and run statistics now carry `llm_calls`, `prompt_tokens`,
`completion_tokens`, `total_tokens`, `reasoning_tokens` and `reasoning_chars`, summed over every
completion the question asked for. Two agents that score the same are not the same result if one
spent three times the tokens getting there, and on a reasoning model most of a completion is text
the parser never sees — the Spatial-Agent pipeline makes three to four calls per question against
ReAct's one per iteration, and neither cost was recorded before.

**`reasoning_tokens` is whatever the server reported, and nothing else.** This deployment returns
`completion_tokens_details: null` while returning a populated `message.reasoning`: 441 completion
tokens for a one-sentence answer, with the thinking in a field the usage block does not count.
There is no `/tokenize` route either — both `…/tokenize` and `…/v1/tokenize` are 404 — so the split
cannot be recovered without a tokenizer that would have to agree with the server's. So the field
stays `null` rather than being estimated, `reasoning_chars` records the thinking text that did come
back, and the summary line says which of the two it is printing. On a server that does report
`completion_tokens_details.reasoning_tokens`, the field fills in by itself and the aggregate sums
it — but only when every row reported one, because summing a column the server filled in sometimes
would read as a total.

Measured on two v6 questions with `--react-tools reference`:

```
Tokens: 37846 total (32167 prompt + 5679 completion) over 16 LLM calls,
        18923 per question | 9288 reasoning chars (server reports no reasoning-token split)
```

The prompt-to-completion ratio is the thing to watch in a write-up: a ReAct question re-sends its
whole growing trace on every iteration, so 85% of what this run spent was prompt, and a budget
change moves that number quadratically rather than linearly.

### A completion the server cut off is its own failure

Nothing here sends `max_tokens`. The output ceiling is the vLLM deployment's to set, which is also
the arrangement both upstreams run under — `mapeval-api/GPT_4o_mini.py` and
`spatial-agent/src/agent/spatial_agent.py` construct their clients without one.

A ceiling still lands sometimes, and when it does the question is not answered badly, it is not
allowed to finish. A completion that comes back with `finish_reason="length"` raises
`LLMOutputTruncatedError`, both agents record it as `llm_output_truncated`, the run statistics
carry `llm_output_truncated_count`, and the summary says so. Without that the row read
`answer_parse_failure` — the same label a genuinely confused agent earns, and indistinguishable
from one in a report.

It matters more here than it would elsewhere. This endpoint bills the chain of thought to the
completion and emits it first, so a ceiling tight enough to bite is spent thinking and the answer
is what gets cut. A truncated question is never retried, because the serving limit is the same on
the second ask, and the tokens the cut-off call burned still count toward what the question cost,
since they were spent.

### One step budget, not two

`REACT_MAX_ITERATIONS` and `MAX_REASONING_STEPS` were two names for the same quantity — how many
reasoning steps a question may take — and which one applied depended on the agent and the tool
surface: 15 for `reference` ReAct, 8 for `native` ReAct, 8 for Spatial-Agent's graph. No single
`.env` line said what the budget was, and a reader of a report had to reconstruct it from
`react_tools`. They are now one setting, `MAX_REASONING_STEPS`, defaulting to 15.

Two effects worth stating rather than burying. Spatial-Agent's authored graph may now hold 15
nodes instead of 8 — that bound is a local guard, not something upstream has, so loosening it
takes nothing away from the comparison, but a plan that used to be rejected as too long can now
run. And `native` ReAct's loop went from 8 iterations to 15; it was always the stronger-than-paper
ablation, and this makes it stronger still. Reports carry `max_reasoning_steps`, so neither
change is invisible after the fact.

## What the v5 and v6 runs measured

One pass per agent, `--react-tools reference`, `--provider kakao`, temperature 0, 15 reasoning
steps, code revision `bb40d56`. Read every difference below against the ±8-point spread this
endpoint carries: these are single draws.

| | ReAct (reference) | Spatial-Agent |
| --- | --- | --- |
| **v5** | **57/100** | **84/100** |
| **v6** | 54/100 | 60/100 |

v5 by class — ReAct against Spatial-Agent: direction 10/10 vs 10/10, distance 7/11 vs 11/11,
nearby 21/31 vs 22/31, radius 2/4 vs 2/4, routing 8/22 vs **22/22**, trip 9/22 vs **17/22**. The
routing and trip columns are the paper's own claim about composed retrieval, on questions tight
enough to punish accumulating legs by hand.

**v6's ReAct number is not a measurement of ReAct.** Eighteen of its hundred questions ended on the
step budget without answering, against zero on v5, and twelve of those are the two four-stop trip
families:

| family | rows | tool calls (median/max) | ReAct |
| --- | --- | --- | --- |
| `trip_optimal_order_four` | 8 | 15 / 15 | 0/8 |
| `trip_total_distance_four` | 7 | 15 / 15 | 0/7 |

Every row hit the cap exactly. Under the reference contract `Directions` answers one leg per call
and the structured-chat parser takes one action per iteration, so a four-stop round trip needs five
place ids and up to fifteen distinct legs across three candidate orders — twenty calls against
langchain's budget of fifteen. The family measures the budget. `dataset/seoul_kmapeval_v7_mcq_100.jsonl`
walks both back to three stops; nothing else about v6 changed.

Both runs predate the failure types that name these, so the counts here are recomputed from the
rows — a budget stop by `llm_calls == 15` with no answer, a window overflow by the endpoint's own
400 text:

| | v5 ReAct | v5 Spatial | v6 ReAct | v6 Spatial |
| --- | --- | --- | --- | --- |
| `iteration_limit` | 0 | – | **18** | – |
| `answer_parse_failure` | 4 | 0 | 6 | 0 |
| `provider_failure` | 7 | 0 | 3 | 0 |
| `llm_output_truncated` | 3 | 2 | 3 | 6 |
| `llm_context_overflow` | 0 | 3 | 0 | 5 |
| `agent_reasoning_failure` | 0 | 0 | 0 | 7 |

### The 64k window bit in two different ways, and neither is "the question needs 64k of thought"

- **Truncation (14 rows across the four runs).** One completion ran 60–62k *completion* tokens with
  240–280k characters of thinking behind it, and the window ran out while the answer was being
  written. That is a reasoning spiral, not a large prompt: four of Spatial-Agent's six v6
  truncations are `routing_detour_cost`, one family.
- **Prompt overflow (8 rows, all Spatial-Agent).** `your prompt contains at least 65537 input
  tokens` — the execution trace outgrew the window before the evaluation stage could read it. These
  arrive as a 400 and were being recorded as `agent_reasoning_failure`, which read as the
  architecture being confused by a question it had never been shown.

### Four of Spatial-Agent's v6 failures are our validator, not its planner

Of the seven `agent_reasoning_failure` rows, four are the same rejected plan:

```
dist1..dist3  haversine_distance  -> amount
max_dist      select_max          -> object      # the farthest one
answer        match_distance_options(distance=$max_dist)   # rejected: wants an amount
```

The plan is right; `select_max` returns the chosen item and the matcher wants its measure. **The
type check that rejects it has no upstream counterpart** — there is no output-type compatibility
check anywhere in `~/spatial-agent`, so upstream would have executed these graphs. One more row
(`Concept role ordering violation`) is likewise a rule only this port has. That made four of ten
`poi_farthest_of_three` losses ours rather than the architecture's.

Closed in two steps, neither of which touches how a plan is authored:

- **The declared-type table must not be stricter than the operator it describes.**
  `match_distance_options` reads metres off a number *or* any record carrying
  `distance_m`/`distance`/`value`/`amount`/`meters`/`distance_km` (`_distance_value`), and said so
  before this benchmark existed — the entry claiming it accepts only `amount` and `proportion` was
  a second, wrong description of the same operator. Widened to everything that can carry a
  measurement. This alone makes the four `select_max` graphs valid on the first pass.
- **Neither local rule may lose a graph that would have run.** `normalize_and_validate_graph`
  takes `strict_types=False`, which keeps every structural rule — unknown operator, a dependency
  that is not a node, a cycle, no Measure node, the step budget — and skips output-type
  compatibility and role ordering. The agent uses it as the last thing it tries: plan, repair,
  prevalidated template, and then the planner's own graph under the rules upstream actually has.
  Whatever is really wrong with such a graph now surfaces as the step that could not execute,
  which is a finding about the architecture instead of about our validator.

Both rules stay on by default, because their message is what the repair round is handed to work
with — a diagnostic upstream does without, and worth keeping as long as it cannot cost a question.

The remaining two are a real limit of the formalism. Asked for a difference of two distances, the
planner emitted `identity_measure` with `value: "$dist1.distance_m - $dist2.distance_m"` — an
arithmetic expression as a *string*, because no subtraction operator exists here or upstream
(`pairwise_extremes` and `compare_routes` are the closest). `identity_measure` passes its value
through, so even with the type check gone that string is never evaluated. `poi_distance_difference`
can only be answered by the evaluation-stage LLM doing the arithmetic, which is presumably how the
seven it did get right were answered.

## What v6 and v7 measured, once the validator stopped costing rows

All at revision `c3f8914`, `--react-tools reference`, `--provider kakao`, temperature 0, 15
reasoning steps, one pass each. The ±8-point spread applies to every cell.

| | ReAct (reference) | Spatial-Agent |
| --- | --- | --- |
| **v6** | 53/100 | 73/100 |
| **v7** | 62/100 | *see below* |

**The v7 rebuild did what it was for.** ReAct's budget stops fell from 18 to 11 and its `trip`
class went 5/22 to 13/22: the two families that scored 0/8 and 1/7 with every row pinned at fifteen
tool calls now score 4/8 and 5/7. Those rows measure the architecture again.

**What the validator fix bought, and what is noise.** v6 Spatial-Agent moved 60 → 73 on the same
dataset, so the rows can be compared one by one: six of the seven the fix targeted came back (three
of four `select_max`, both arithmetic-as-a-string rows, the role-ordering row), and
`agent_reasoning_failure` went to zero. But twenty-two rows flipped wrong-to-right and **nine
flipped right-to-wrong**, so the honest reading is "about six to eight rows from the fix, the rest
inside the spread". A single pass cannot separate them further.

### The tail that stopped a run finishing, and what it cost

The first v7 Spatial-Agent attempt was killed after two and a half hours with 65 questions
answered, 12 in flight and 23 never started. Every one of the twelve was blocked at the same place
— the planner call — and four had been there over 70 minutes. The logs end the same way:

```
LLMUnavailableError: LLM endpoint failed after 9 attempts: InternalServerError: Error code: 504
```

The reverse proxy kills a long generation with 504. Nine attempts at a request whose *length* is
the objection cost 95 minutes per question and answered none of them; `llm_unavailable` is
retryable, so each was then asked up to twice more. `LLM_RETRY_TIME_BUDGET_SECONDS` now bounds one
completion by the clock rather than by the attempt count.

### A serving-side output cap has to sit above the work, and 5k does not

With the proxy timeouts gone the next two v7 Spatial-Agent runs finished in forty minutes — and
scored **10/100 and 11/100**, with 67 and 68 questions ending in `llm_output_truncated`. The
deployment had been given an output ceiling of about 5,100 tokens per completion, and that is not
above this architecture's working range. It is inside it:

| | working call (correct rows) | a spiral |
| --- | --- | --- |
| ReAct | median 340, p90 567, max 1,164 | 66,730 / 74,705 / 67,782 |
| Spatial-Agent | median 3,640, p90 4,647, max 5,854 | 60,000+ |

ReAct writes a tool call; Spatial-Agent writes a whole graph and the reasoning behind it, an order
of magnitude more per call. A 5,100-token ceiling barely touches the first and cuts off most of the
second — which is exactly what the two runs show, and why they measure the ceiling rather than the
architecture. Neither number is quotable.

The gap between honest work (≤6k) and a spiral (≥60k) is wide enough that any ceiling in the
12k–32k range separates them. **16,384** is roughly three times Spatial-Agent's largest observed
working call and still ends a spiral at a quarter of its cost. Whatever is chosen, it belongs in
the write-up beside the accuracy: a run whose `llm_output_truncated_count` is not near zero is
partly a measurement of the ceiling.

### Third configuration: no ceiling, and the clock takes the rows instead

`reports/test_20260822T140526Z.json` — v7 Spatial-Agent, ceiling removed again (the cut-offs it
does record sit at 61.9k–62.6k, i.e. the context window). It scored **56/100 with 28 questions lost
before they could be answered**: 20 `llm_unavailable`, every one of them a 504 after
`LLM_RETRY_TIME_BUDGET_SECONDS` expired on the first attempt, tried three times; 5
`llm_output_truncated`; 3 `llm_context_overflow`.

Where the 20 landed is the finding:

| family | lost to 504 | scored | same families in the v6 run |
| --- | --- | --- | --- |
| `poi_distance_difference` | 10 of 11 | 1/11 | 7/11 |
| `routing_detour_cost` | 8 of 8 | 0/8 | 6/8 |
| `nearby_subtype_kth` | 2 of 10 | 4/10 | 8/10 |

The first two are the families that ask for a **difference** — via length minus direct length, one
distance minus another — which is the composition the operator vocabulary cannot express and where
the planner has been observed writing arithmetic into a string. Those are the calls that spiral,
and with no ceiling a spiral runs until either the window (62k) or the proxy (504) stops it. The
third row is the caution against reading this as purely architectural: `nearby_subtype_kth` does
not spiral and still lost two questions, so healthy calls were being killed as well.

So 56/100 is a floor rather than a score: the architecture's v7 number is somewhere between 56 and
84 depending on 28 questions that were never answered. And the retry budget cuts both ways — it is
what let the run finish at all, and it is also what converts a slow-but-answerable call into a lost
row. It needs the output ceiling in place to stop firing.

The configuration these three runs argue for, all of it measured rather than guessed: **an output
ceiling near 16,384** (working calls top out at 5,854, spirals start above 60,000), the retry
budget left where it is, and Spatial-Agent's concurrency reduced if 504s persist — its calls are an
order of magnitude longer than ReAct's, so twelve of them at once is not the same load as twelve
ReAct workers.

## A second model, and the first sweep where nothing was lost to the endpoint

`google/gemma-4-E4B-it-qat-w4a16-ct` in place of `nvidia/Qwen3.6-35B-A3B-NVFP4`, revision
`a5742a3`, `--react-tools reference`, `--provider kakao`, temperature 0, 15 steps, concurrency 32,
one pass each. **These numbers may not be pooled with the Qwen ones**; report metadata carries
`llm_model` so the two families stay separable.

| | ReAct (reference) | Spatial-Agent |
| --- | --- | --- |
| **v5** | 49/100 | 75/100 |
| **v6** | 34/100 | 62/100 |
| **v7** | 46/100 | 74/100 |

The serving problems that dominated the last three days are simply absent: no truncation, no 504,
one context overflow across six runs, and a hundred questions in three to seven minutes instead of
hours. Gemma does not emit a chain of thought, so there is nothing to spiral — the 60k-token
completions, the gateway timeouts and the retry storms were all one model's thinking budget meeting
a proxy. What is left is 7 `iteration_limit` on v6 and 1 on v7, which are the benchmark working as
intended, and 7 Spatial-Agent graph failures across the three runs, three of which were still this
port's own concept-level role rule (fixed in `16e73c1`, after these runs).

Spatial-Agent leads ReAct by 26, 28 and 28 points. The per-family split says where:

| v5 family | ReAct | Spatial-Agent |
| --- | --- | --- |
| `poi_straight_distance_tight` | **0/11** | **11/11** |
| `routing_turn_count` | 1/7 | 6/7 |
| `routing_distance_via` | 3/8 | 8/8 |
| `nearby_second_nearest` | 6/6 | 2/6 |
| `trip_feasible_count` | 7/7 | 4/7 |

`poi_straight_distance_tight` is the sharpest result in this repository so far: eleven questions
answerable by a haversine over two resolved coordinates, which the graph gets right every time and
the tool-calling loop gets wrong every time. The same shape repeats on v7 —
`poi_distance_difference` 3/11 against 10/11, `routing_turn_count_via` 1/7 against 7/7. The two
families running the other way are worth as much: an ordinal over neighbours and a stay-time count
are questions where reading results one at a time beats composing a graph.

**What is missing before any of this is quotable.** There is no no-tool floor for this model, and
without one an accuracy of 49 has no scale — the Qwen floor was 24–32/100 on v5 and says nothing
about Gemma. The spread is also unmeasured here; a hundred questions now costs minutes, so
`--repeats 3` is affordable and a single pass is not a result.

### The no-tool floor for this model

Same model, same options, no tools, closed book, two passes each
(`data/measure_no_tool_floor.py`, 32 workers). Nothing unparsed and nothing failed in six passes.

| | floor (2 passes) | ReAct | Spatial-Agent | what the map explains |
| --- | --- | --- | --- | --- |
| **v5** | 22, 26 | 49 | 75 | +25 / +51 |
| **v6** | 28, 31 | 34 | 62 | **+4** / +32 |
| **v7** | 29, 31 | 46 | 74 | +16 / +44 |

The floors sit at or just above the chance rate of 25, and the chosen-option histograms are flat
(`{0: 14, 1: 27, 2: 35, 3: 24}` is typical), so there is no position prior and no family answerable
from the model's own knowledge. Family by family the pattern is what the design intended: 1/11 on
`poi_distance_difference`, 0–2/8 on `routing_detour_cost`, 0/6 on `nearby_second_nearest`, 1/10 on
`nearby_clinic_subtype`. The one high cell, `unanswerable_subjective` 4/4, is correct by
construction — the gold is a refusal, and refusing is exactly what a closed book should do.

**The v6 line is the one to read twice.** ReAct scores 34 against a floor of 28–31: the map
explains about four questions out of a hundred, which is inside the floor's own two-pass spread.
On v6 the reference baseline is, within noise, a model answering from its own knowledge with the
tools adding nothing. That is a statement about the benchmark as much as about the agent — v6 is
hard enough that the tool-calling loop cannot convert its evidence into answers — and it is why v7
exists. On v7 the same agent recovers to +16 while Spatial-Agent holds +44.

## Three passes each, and the first comparison this repository can defend

`google/gemma-4-E4B-it-qat-w4a16-ct`, revision `49721ca`, `--react-tools reference`,
`--provider kakao`, temperature 0, `MAX_REASONING_STEPS=15`, `--repeats 3`. Floors are the two
closed-book passes from the section above.

| | floor | ReAct (reference) | Spatial-Agent | gap |
| --- | --- | --- | --- | --- |
| **v5** | 24 (22, 26) | **53.0** (50, 54, 55) | **77.3** (76, 76, 80) | 24.3 |
| **v6** | 29.5 (28, 31) | **39.3** (37, 39, 42) | **70.0** (67, 70, 73) | 30.7 |
| **v7** | 30 (29, 31) | **43.3** (38, 45, 47) | **68.3** (65, 69, 71) | 25.0 |

Over the floor: ReAct +29.0, +9.8, +13.3; Spatial-Agent +53.3, +40.5, +38.3.

**Every single-agent spread is 4 to 9 points and every gap is 24 to 31.** That is the first result
here that a repeated measurement supports: the difference between the architectures is three to
seven times the run-to-run noise of either one, on three benchmarks, against a measured floor. Up
to now every comparison in this file was a single draw from a distribution wide enough to contain
the thing being compared.

The v6 line still says what it said with one pass: ReAct at 39.3 against a floor of 29.5 gains
about ten questions from having a map, where Spatial-Agent gains forty. A tool-calling loop with
fifteen iterations and one action per iteration cannot convert v6's evidence into answers.

Nine Spatial-Agent runs — nine hundred questions — produced eight failures in total: two plans
larger than the step budget (18 and 21 operators), two invented operator names (`select_by_index`,
`sum_amounts` — the arithmetic gap again), one node missing an argument, one node whose operator
was a *list*, one context overflow and one 62k-token truncation. No role-ordering violations at
all, which is `16e73c1` holding.

**What these numbers still are not.** All three benchmarks have been tuned against — `src/` changed
in response to what v5 and v6 showed — so these are training-set accuracies. `dataset/seoul_kmapeval_v5h_holdout_100.jsonl`
predates most of that code and has never been run on this model. A held-out number, built from the
v7 builder under a fresh seed and run once with nothing changed afterwards, is the one thing
missing before any of this is quotable as a general claim rather than a result on these hundred
questions.

## The held-out hundred

`dataset/seoul_kmapeval_v7h_holdout_100.jsonl`, the v7 builder under seed 927451 with `v7h` ids:
one question and 30 of 236 place names in common with v7. Built at `0aabaa9`, audited clean, and
run with nothing under `src/` changed afterwards. Same configuration as the table above —
`google/gemma-4-E4B-it-qat-w4a16-ct`, `--react-tools reference`, `--provider kakao`, temperature 0,
`MAX_REASONING_STEPS=15`, `--repeats 3` — at revision `0aabaa9` rather than `49721ca`. The one code
change between them is `235e51e`, which turns a planner node whose `operator` is a list from a
crash into a named failure; it affected one question in nine hundred.

| | floor | ReAct (reference) | Spatial-Agent | gap |
| --- | --- | --- | --- | --- |
| **v7 (tuned)** | 30 (29, 31) | 43.3 (38, 45, 47) | 68.3 (65, 69, 71) | 25.0 |
| **v7h (held out)** | 29.5 (29, 30) | **48.0** (52, 45, 47) | **70.7** (66, 73, 73) | 22.7 |

**The holdout reproduces the result, and neither agent was worse on it.** ReAct gains 4.7 points
and Spatial-Agent 2.4 going from the tuned set to the one nothing was tuned against — both inside
the seven-point spread each agent shows across its own three passes, so the honest reading is that
the difference is noise and there is *no measurable training-set advantage* in either direction.
The gap of 22.7 is again more than three times the widest single-agent spread. The two floors agree
to within a point, so the holdout is the same difficulty closed-book as the set it was drawn to
match.

Where the gap lives, summed over three passes each:

| class | rows | ReAct | Spatial-Agent |
| --- | --- | --- | --- |
| distance | 63 | 36.5% | **82.5%** |
| routing | 66 | 40.9% | **86.4%** |
| radius | 12 | 41.7% | 58.3% |
| nearby | 93 | 53.8% | 58.1% |
| trip | 66 | 59.1% | 63.6% |

The whole difference is in the two measurement-heavy classes. On `nearby` and `trip` the two
architectures are four to five points apart, which is inside the noise; on `distance` and `routing`
they are forty-five points apart. That is a narrower claim than "Spatial-Agent is better" and a
more useful one: what the graph buys is *getting a measurement right and carrying it*, not
reasoning about places in general.

Cost went the other way from what the accuracies suggest. Over three passes ReAct made 2,384 LLM
calls for 4.05M tokens and averaged 33.9s a question; Spatial-Agent made 924 calls for 5.39M
tokens and averaged 61.2s. Spatial-Agent asks a third as often, in much larger requests, and takes
about twice as long.

Failures, over six hundred questions: one ReAct `iteration_limit`, seven Spatial-Agent
`agent_reasoning_failure`. No truncations and no context overflows on either side. All seven of
Spatial-Agent's are the planner writing a graph that cannot run — three nodes missing a required
argument, two invented operator names (`select_by_index`, `select_second_closest`), one 17-operator
plan against a budget of 15, one incomplete factorization. As on the tuned sets, no role-ordering
violation survived the lenient pass to end a question.

**Building the holdout found a defect in the builder, not in the seed.** The first draw failed
`data/audit_dataset.py`: `nearby_within_radius_count` offers a four-rung ladder over four rows and
picks the least-used rung each time, but it stopped scanning anchors at `count`, so it only spread
the rungs the first four anchors happened to offer. Three of those anchors sat where `outside` came
back empty, which leaves "네 곳" the only feasible rung — 1 through 3 each need a place beyond the
radius and 4 needs none — and the family shipped a ladder whose third rung could never be the
answer. The generator now keeps scanning while a rung is uncovered. This is the invariant working:
the audit caught, before any agent ran, exactly the class of defect that used to ship.

## The arithmetic gap, and closing it

Across every run in `logs/`, Spatial-Agent's planner named an operator that does not exist 21
times. The names scatter but the intents do not:

| intent | names written | events |
| --- | --- | --- |
| take the k-th of a ranking | `select_by_index` ×6, `select_second_closest`, `select_second_nearest`, `select_second_min`, `select_subset`, `select` | 11 |
| add measurements | `sum_amounts` ×4, `calculate_path_distance` ×2, `calculate_total_distance`, `sum_distances` | 8 |
| subtract measurements | `subtraction`, `calculate_difference`, `subtract` | 3 |

The operator set really did have those holes. `sort_by` orders a list and `select_min`/`select_max`
take an end off it, so an ordinal question — "the second closest" — had no operator to finish on.
`sum_route_metrics` totals a route list and `aggregate_route_groups` totals route indexes per
option, but a graph that had already reduced each leg to an amount had nothing that could add two
numbers. And nothing anywhere subtracted, though a detour cost and "how much farther" are both
subtractions and both are families in these benchmarks.

Three operators close them:

- `select_by_index(items, index, key?, descending?) -> object`. The index is **0-based**, which is
  what the planners themselves wrote — `index: 1` for the second closest, `2` for the third. An
  index past the end fails rather than clamping: the nearest place is not the second nearest.
- `sum_amounts(amounts, key?) -> amount`. Route-shaped records total `distance_m` and `duration_s`
  together. A sum of durations reports `duration_s` and no `value`, so a plan that pipes seconds
  into `match_distance_options` fails where it stands instead of answering in the wrong unit.
  Elements that are text — `["dist_A_C", "dist_C_B"]`, node ids a planner forgot to mark with `$` —
  fail rather than adding up as zero.
- `difference(minuend, subtrahend) -> amount`. Keeps `difference` signed and reports `value` as the
  magnitude, because a numeric option states the ordering in words and leaves the number positive.

Three smaller things came out of fixing it, and two of them were pre-existing:

**The declared table was stricter than the implementation, again.** `_normalize_arguments` accepts
several spellings per slot, but the required-argument check only looked for the canonical one, so
`sum_amounts(items=[$leg1, $leg2])` — which the executor would have run — was refused as "missing
arguments: amounts". `REQUIRED_ARGUMENT_ALIASES` now lists the same spellings the normalizer takes.
The same defect was already sitting under `select_min`/`select_max`, whose contract required a
`key` the implementation defaults.

**A ranking with no key named ranked by nothing.** Forty-five calls in `logs/` write
`select_max(items=[$d1, $d2, $d3])` with no key. The normalizer defaulted to `"value"`, which a
`haversine_distance` record does not carry, so the operator raised "No item contains comparable
key: value". It now infers the measurement the records actually hold. This had to be fixed *with*
the contract relaxation rather than after it: without it, relaxing the contract would have moved
the same loss from validation, where a repair round could still save it, to execution, where
nothing can.

**`select_min(items, index=1)` is an ordinal, not a minimum.** Both `seoul_kmapeval_v7h_001` and
`_010` wrote exactly that. Accepting the plan while dropping `index` would have returned the
*nearest* candidate to a question asking which is second nearest — a confident wrong answer, which
is worse than the refusal it replaced. An explicit `index` on either selector now routes to
`select_by_index`.

`OPERATOR_SYNONYMS` maps `subtraction`, `subtract`, `calculate_difference` and `sum_distances` onto
the operators that do the work, rewritten once in `normalize_and_validate_graph` so the executor is
handed the canonical name and there is no second table to keep in step. `select_second_closest` and
its kin are deliberately **not** there: turning that name into `select_by_index(index=1)` means
reading an ordinal out of an identifier, and a question answered one rung off looks exactly like one
answered wrongly. The prompt now says the name does not exist and points at the operator that does.

Replaying every compose stage in `logs/` that named an invented operator — 17 distinct plans — five
now validate and execute where none of them could. Six distinct questions across all recorded runs
ended on a cause this removes. What is left is deliberate: `calculate_path_distance` and
`calculate_total_distance` (3 plans) want a trip total keyed by stop *names*, which
`aggregate_route_groups` already computes from route indexes — a second operator for the same job
would compete with the one the trip families currently answer through, and the prompt now points at
it instead. `select_second_min` and `select_second_nearest` (2 plans) are the name-parsing cases
above. `select` and `select_routes` (2 plans) name no operation precisely enough to implement.

**This spends the holdout.** `src/` changed in response to what `seoul_kmapeval_v7h` showed, so
48.0/70.7 are now what the code at `0aabaa9` scored on it, not a held-out number for the code that
exists today. Quoting a holdout again means drawing one under a new seed and leaving `src/` alone
afterwards.

## The second holdout, run on the code that has the arithmetic

`dataset/seoul_kmapeval_v7h2_holdout_100.jsonl`, the v7 builder under seed 481203: one question in
common with v7, one with v7h, 36 of 231 place names shared with v7 and 31 with v7h. Clean on the
first `audit_dataset.py` draw, radius ladder included. Built and run at `38566f3` with nothing
under `src/` changed in between. Same configuration throughout: `--react-tools reference`,
`--provider kakao`, temperature 0, `MAX_REASONING_STEPS=15`, `--repeats 3`.

| | floor | ReAct (reference) | Spatial-Agent | gap |
| --- | --- | --- | --- | --- |
| v7 (tuned, `49721ca`) | 30 (29, 31) | 43.3 (38, 45, 47) | 68.3 (65, 69, 71) | 25.0 |
| v7h (holdout, `0aabaa9`) | 29.5 (29, 30) | 48.0 (52, 45, 47) | 70.7 (66, 73, 73) | 22.7 |
| **v7h2 (holdout, `38566f3`)** | 25.5 (25, 26) | **45.7** (43, 48, 46) | **72.3** (77, 71, 69) | **26.7** |

Over the floor: ReAct +20.2, Spatial-Agent +46.8. The gap is 26.7 against a widest single-agent
spread of 8 — the same three-to-one relationship the tuned sets and the first holdout showed, now
on a third independent draw.

**What the operator fix demonstrably changed is the failure count, not the accuracy.**
Spatial-Agent's `agent_reasoning_failure` count over three passes went from 7 in 300 questions on
v7h to 2 in 300 here, and that drop is attributable because the causes were code-level: an operator
that did not exist, or a required argument the executor would have accepted. The accuracy moved
70.7 → 72.3, which is inside the eight-point spread, so at a hundred questions a run this size
cannot separate the fix from noise. Two recovered failures are worth at most two points, which is
exactly what "inside the spread" means. The honest claim is the narrow one: the planner stopped
losing questions to a missing operator; whether that is worth measurable accuracy needs more rows
than this.

The operators are not decorative. Across the run's 600 logs `select_by_index` appears in 68,
`difference` in 57 and `sum_amounts` in 43, and they execute — `difference` completed 48 times,
`sum_amounts` 32, `select_by_index` 14. Both remaining Spatial-Agent failures are genuine planner
errors, not gaps: one graph whose concept factorization never bound its anchor, and one that
emitted `sum_amounts` with `"arguments": {}` — no arguments at all, correctly refused.

| class | rows | ReAct | Spatial-Agent |
| --- | --- | --- | --- |
| distance | 63 | 25.4% | **71.4%** |
| routing | 66 | 33.3% | **87.9%** |
| radius | 12 | 41.7% | 75.0% |
| trip | 66 | 59.1% | **75.8%** |
| nearby | 93 | 59.1% | 59.1% |

`nearby` lands on exactly the same number for both architectures, and it is Spatial-Agent's worst
class on both holdouts (58.1%, 59.1%). Whatever the graph buys, it does not buy this: on questions
that rank retrieved POIs the two architectures are indistinguishable, and both sit about thirty
points over the floor. That is where the next real gain is, and it is not an operator gap — the
ordinal operator now exists and gets used.

Comparisons *between* v7h and v7h2 by class are not available: they are different draws, so a class
moving twelve points between them says as much about which questions were drawn as about the code.
Only the failure counts, whose causes are code-level, carry across.

Cost, three passes each: ReAct 2,441 LLM calls for 4.18M tokens at 35.9s a question;
Spatial-Agent 925 calls for 5.55M tokens at 62.5s — a third the calls, larger ones, about twice
the wall clock, the same shape as v7h.

## The template catalogue had a shape missing, and it cost the ordinal families

On the second holdout Spatial-Agent scored 25.0% on `nearby_kth_nearest` — six of twenty-four,
which on four options is chance, and *below* ReAct's 45.8%. The class totals hid it: `nearby` came
out 55/93 for both architectures, and the profiles underneath are nothing alike.

| nearby template | rows | ReAct | Spatial-Agent |
| --- | --- | --- | --- |
| `nearby_kth_nearest` | 24 | 45.8% | **25.0%** |
| `nearby_subtype_kth` | 30 | 50.0% | 56.7% |
| `nearby_cuisine_subtype` | 18 | 44.4% | **77.8%** |
| `unanswerable_*` | 21 | 100% | 85.7% |

The first hypothesis was an off-by-one in the ordinal operator: 0-based `select_by_index` against
a Korean ordinal that counts from one. It is wrong. Every plan writes the index the project's
convention wants — `두 번째` → 1, `세 번째` → 2, in all eight questions.

The actual cause is upstream of the operator. `nearby_kth_nearest` draws its gold as rank k of
everything within 1800 m and its three decoys from ranks 1 through 6, so **the options are a
subset of the ranking and the nearest place is only sometimes among them**. Ranking the four
options against each other answers a different question and hits the gold only by luck. Across the
v7 runs, 130 composed plans for the two ordinal families retrieved a neighbourhood **four times**.
On the holdout it was zero out of fifty-four.

That is not the planner being careless — it is the template retrieval stage handing it the wrong
worked example. `retrieve_templates` scores `Geocode-Batch-Compare` at 4 for intent plus 1 for
"가까운", and its example is literally *geocode the anchor and the option names, then take the
nearest*. There was no template for an ordinal, so the planner copied the one it was given.

`Retrieve-Rank-Ordinal` fills the shape: `nearby_places(center, category/keyword) -> nearest ->
select_by_index(k-1) -> match_options`. It outranks `Geocode-Batch-Compare` on an ordinal
question (intent 4 + "번째" + "가까운" = 6 against 5) and stays behind it on a superlative, where
the options really are the candidates. Radius questions are untouched. The chain runs end to end
on `seoul_kmapeval_v7h2_011`'s real coordinates and returns option 3, which is its gold; ranking
the three named options would have returned it first instead.

**This is a deliberate deviation.** Upstream's Appendix E has ten macro families and this is an
eleventh, so `test_template_catalog_covers_appendix_e_macro_families` now asserts Appendix E as a
subset and names the port's own addition separately, rather than letting the catalogue drift
silently. A template is what the retrieval stage hands the planner as a worked example, so adding
one changes what every question of that shape gets composed from — it deserves to be visible.

`FakeProvider.nearby_search` returned a single place, which meant no test using it could exercise
a ranking at all — `select_by_index(index=1)` had nothing to select from. It returns three
distinct spots now.

**What this does not yet show.** No run has been made against the new template; the evidence is
that 126 of 130 tuned-set plans composed a graph that cannot answer the question, and that the
replacement chain returns the right option on real coordinates. Whether it moves the family is a
measurement, not a claim. And the diagnosis began in the v7h2 class breakdown, so **v7h2 is spent
too** — 45.7/72.3 belong to `38566f3`. A number for the code with this template needs a third
draw.

### Measured: the ordinal families, before and after

Eighteen questions (`nearby_kth_nearest` ×8, `nearby_subtype_kth` ×10) from v7, Spatial-Agent
only, three passes at `c393db5` against the same eighteen pooled over every earlier v7 run.

| | plans | take a k-th | end on `nearest` | accuracy |
| --- | --- | --- | --- | --- |
| before (≤ `49721ca`) | 130 | **0** | 26 | **31.7%** (40/126) |
| after (`c393db5`) | 53 | **52** | 1 | **79.6%** (43/54) |

Per family: `nearby_kth_nearest` 23.2% → 83.3%, `nearby_subtype_kth` 38.6% → 76.7%. ReAct's
standing number on the same eighteen is 60.0%, so the family flips from Spatial-Agent scoring half
of ReAct to scoring well above it.

**The template did this, not the operator.** The two landed in separate commits and there is a run
between them: v7h2 at `38566f3` had `select_by_index` and no ordinal template, and scored 25.0% on
`nearby_kth_nearest` — indistinguishable from the 23.2% before the operator existed. An operator
the retrieval stage never suggests using is not reachable. The mechanism is visible in the shapes:
no plan in 130 took a k-th of anything, and 52 of 53 do now.

Two things this measurement does *not* support:

**Retrieval is still not the norm.** Only 12 of 53 plans call `nearby_places`; the rest copy the
template's shape — `nearest -> select_by_index -> match_options` — while still geocoding the four
option names as their candidate set. The shape is what moved the number, not the retrieval the
template was written to teach.

**And it did not have to be.** Ranking only the options answers `nearby_kth_nearest` correctly
whenever all k−1 nearer places happen to be among the three decoys, which is
`C(m−k, 4−k) / C(m−1, 3)` for a pool of m. Over v7's eight rows that is **56.2%** — because k is 2
in seven of the eight. `kth = 2 + (index % 3)` keys the ordinal on the anchor *loop index*, so k is
spent wherever the loop happened to succeed rather than across the three values, which is the
defect `AGENTS.md` already names for rung ladders: "keying the answer on a loop index spends them
wherever the loop happened to succeed". The family is weaker than it was designed to be, and a
build that draws k the way `trip_feasible_count` spends its rungs would make the difference between
ranking the options and ranking the neighbourhood matter as much as it should. Recorded here, not
fixed: changing it means rebuilding v7.

One new failure appeared, `llm_context_overflow` on one question of fifty-four. Retrieving a
neighbourhood puts more evidence in the graph and therefore in later prompts. One in fifty-four is
worth watching rather than acting on, but it is the first overflow any of these runs has produced.
