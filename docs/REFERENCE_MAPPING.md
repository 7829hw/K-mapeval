# Reference implementation mapping

| Upstream concept | K-MapEval implementation | Deliberate deviation |
|---|---|---|
| MapEval `Evaluator2.py` structured ReAct loop | `src/agent/react.py` | Removes the localhost backend, sleeps, and remote writes. Uses user-requested 0-based `^^N^^` answers. |
| MapEval tools/backend | `src/tools/registry.py`, `src/tools/map.py`, `src/tools/kakao.py` | Provider injection replaces the separate HTTP backend; tools expose normalized JSON. |
| Spatial information theory analysis | `normalize_analysis`, `src/agent/concepts.py` | Preserves all seven core-concept labels and six roles. Runtime intent classification and dataset labels are absent; missing concepts are completed from typed facts and marked implicit/synthetic. |
| Concept transformation drafting | `src/agent/templates.py`, `src/agent/composition.py`, Spatial-Agent `compose` | Appendix E's ten reusable macro fragments expose typed I/O ports. Retrieval reads typed concepts and factors, and templates are priors rather than hard constraints. The planner authors transformation edges and refers to the Analysis stage's concepts by id; it declares only concepts that stage does not carry. |
| Concept graph `G` | `GeoFlowGraph`, `ConceptNode`, `TransformationEdge` | Vertices are concepts with `core_concept`, `functional_role`, and `attributes`; directed hyperedges are transformations. The executable operator graph is a later representation. |
| Factors and operator-concept hypergraph `G'` | `FactorNode`, `attach_grounding_factors`, `factorize_semantic_graph` | Radius, ordinal, direction, time budget, stays, route objective, fixed order, and return-to-start are explicit factor vertices connected to transformation/operator hyperedges. Tool selection remains deterministic from types, factors, and contracts. |
| Five GeoFlow constraints | `src/agent/canonicalization.py`, `src/agent/validation.py`, `normalize_and_validate_graph` | Deterministic canonicalization completes the drafted graph's references before G1–G5 read it: the Analysis stage's concepts and their derived factors are in scope, an undeclared output is typed by the transformation that produces it, a redefined concept is renamed and later readers follow it, and a produced concept's type comes from the vocabulary rather than the planner. Every completion is additive and last-resort, so a draft that already passes canonicalizes to itself. G1–G5 then refuse strictly on both the draft and the repair. This port's own output-type, role-ordering and argument-value checks are skipped (`strict_types=False`) on a last attempt over the repaired graph and then the original; a graph still invalid there becomes `graph_validation_failure`. |
| Core-concept execution types | `OPERATOR_CONTRACTS`, `SpatialOperatorRegistry` | LOCATION, OBJECT, FIELD, EVENT, NETWORK, AMOUNT, and PROPORTION all have executable producers. The current benchmark directly exercises only a subset. |
| Contextual/functional roles | `factorize_geoflow`, `ROLE_PRIORITY` | EXTENT/TEXTENT are scheduled as context but do not participate in procedural precedence. |
| Topological executor | `src/agent/execution.py`, Spatial-Agent `execute` stage | Executes the validated order through injected registries, records operator state, and separately materializes concept state through output bindings. |
| Lenient concept-reference resolution (`_resolve_concept_reference`, `_extract_coordinates_from_concept`, `resolve_place_name`) | `_resolve_references` / `_descend_reference` in `src/agent/spatial.py`, `_as_place` / `_as_place_list` in `src/tools/spatial.py` | An over-specified `$node.path` degrades to the closest resolvable object instead of failing the operator, and every coordinate operator unwraps the place-shaped record it was handed. Only a genuinely unresolved place raises, as an explicit `PlaceNotFoundError`. |
| Grounded answer generation | `src/agent/answering.py` | The spatial core produces a `GroundedAnswer` from execution evidence without seeing MCQ options. |
| Evaluator answer selection | `MCQAdapter` in `src/mcq_adapter.py` | MCQ reconciliation is outside GeoFlow (exact grounded text, then unambiguous containment/value match) and preserves the 0-based `^^N^^` contract. An index emitted by the grounded-answer stage is ignored; `MATCH_OPTIONS` is not a core transformation. |
| Appendix C operators | `ToolRegistry`, `SpatialOperatorRegistry` | Adds reverse/batch details, waypoint directions, instruction route filtering, extractors, pairwise extremes, place filtering, travel-time nearest, temporal operators, and step analysis with paper-level semantics. |
| TSP-TW | `SpatialOperatorRegistry.tsp_tw` | Exhaustive optimization up to nine nodes with service times, windows, and budget; infeasible instances return the paper's nearest-unvisited partial feasible fallback. OR-Tools is not bundled. |
| Temporal operators | `timezone`, `open_at_time`, `calculate_finish_time`, `calculate_start_time` | Handles cross-midnight/24-hour periods. Multi-stop finish time queries cached/live route durations and adds stays. Latest-departure calculation is an explicit template helper. |
| Spatial-Agent evaluator/generator | Spatial-Agent `evaluate` and `generate` | The LLM conditions on question, final state, and full trace. Deterministic match evidence no longer overwrites the generated selection. |
| Google Maps client | `KakaoMapProvider` | Kakao Local handles search/geocode/reverse-geocode/nearby; Kakao Mobility handles driving and verified multi-waypoint routes with optional guides. |
| Spatial-Agent context cache | Not implemented | Legacy dataset context fields are ignored. `SQLiteMapCache` caches normalized Kakao responses only; it is not a MapEval context corpus. |
| MapEval-API benchmark (300 rows, live Google Maps) | `dataset/seoul_kmapeval_v2_mcq_100.jsonl` (100 rows, live Kakao) | Class mix mirrors MapEval-API's answerable half (nearby 30 / trip 24 / routing 23 / poi 23) so the multi-hop families the paper's gains come from are actually present. Gold answers are computed from Kakao Local and Kakao Mobility, the same provider the agents query. `unanswerable` is excluded — see below. |
| SFT and DPO | Not implemented | This repository evaluates the off-the-shelf prompting path and does not claim fine-tuned Qwen results. |

## Remaining non-equivalences

- Macro-template retrieval is typed and deterministic. Question–validated-graph demonstration
  retrieval uses embedding cosine similarity when an embedding backend is configured; exact test
  questions and explicitly excluded example IDs cannot be returned.
- Factorization and concept binding are deterministic, not SFT/DPO learned.
- Evidence always comes from Kakao and is recorded as `kakao` in `metadata.provider`.
- Kakao Mobility support is driving-only.
- Spatial-Agent has no benchmark-label or intent router. `classification`, `mapeval_class`, and
  `template_id` remain evaluator/reporting metadata only.
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

Current K-MapEval behavior:

- The context-cache port has been removed. Every run constructs `KakaoMapProvider`; there is no
  `--provider` selector, corpus parsing, or context-to-Kakao fallback.
- Legacy MapEval-Textual `context` fields remain in their dataset rows for provenance, but they are
  metadata only: runtime code never collects them, parses them, or derives provider configuration
  from them.
- `SQLiteMapCache` remains because it caches normalized Kakao requests and responses for both
  architectures. It is independent of the removed MapEval context corpus.

Historical context-cache port (retained here only to interpret reports from older revisions):

The following bullets describe removed behavior, not the current runtime.

- Same shape: the corpus was built from every context in the dataset and shared by all questions,
  loaded *behind* the tool layer so both architectures still chose tools and still read
  normalized `Place` / `Route` objects. `BenchmarkItem.agent_input()` is unchanged.
- The context travelled in the benchmark row rather than in a second file. `main.py` collected
  `item.context` across the dataset and built one corpus, which was upstream's arrangement without
  the extra artifact. Nothing per-question is bound: an earlier revision scoped the corpus to the
  running question, and that made the mere existence of a name an answer signal — "which option
  exists at all" answered 14 of 100 questions under per-question scoping and 9 under the shared
  corpus.
- `--provider hybrid` was upstream's cache-then-live arrangement with Kakao in Google's place, and
  `--provider auto` resolved to it for a context-carrying dataset. `--provider context` ran the
  corpus alone, so a run needed no Kakao key and a miss stayed a miss — a closed world
  stricter than anything upstream measures, which makes it an ablation to ask for by name rather
  than the default a bare run landed on. `resolve_provider_kind` and its test pinned the choice.
  One asymmetry the mode carried and upstream did not: upstream's cache and its fallback are both
  Google, while `seoul_mapeval_v1`'s contexts are OSM-derived and the fallback is Kakao, so a
  hybrid run here mixes two gazetteers where upstream mixes none.
- **Retrievals were computed, not replayed.** This was the one place the port deliberately did not
  follow upstream, and the reason is an evaluation-validity flaw in upstream that this repo
  reproduced and then measured. MapEval-API is MapEval-Textual with the `context` field removed —
  the same 300 questions, the same ids — so a cache built from Textual holds, for every API
  question, the retrieval result that answers it. `get_nearby_places` returns that block already
  filtered by type and already sorted by distance, which makes one tool call sufficient and
  collapses the API setting into the Textual one. Ported faithfully, it produced ReAct 100/100.
  Here the block contributed its *places* to the corpus and `nearby_search` computed the ranking
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
300 questions' curated evidence, with the live API behind it. The removed configuration that
corresponded to it was `hybrid`, not `kakao`; a current run has no curated corpus at all and is a
harder setting than the one the reference number comes from. Any comparison that puts a current
number beside 71.07% has to state that evidence-setting difference.

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
  the removed `ContextMapProvider` computed retrievals over the whole corpus instead of replaying
  the stored block (see above).
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

`_select_option` also carried a deterministic override, `_computed_clock_option`, that outranked the
generation stage whenever every option is a wall clock and the graph computed exactly one. It
decided **27 of the 100** answers in that Spatial-Agent run. On this run it agreed with the
generation stage's own answer text on all 27, so it did not move the score, but it is answer
selection performed by the harness rather than by the architecture and it should be reported
whenever a historical trace shows `"selection_method": "computed_clock"`. The current revision
removes that override; see "Removing family-specific recovery and answer selection" below.

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

### (b) Raising retrieval adoption changed the mechanism and not the number

Three passes of the same eighteen v7 ordinal questions, three times, one change each:

| | offered the template | retrieved | accuracy | spread | failures |
| --- | --- | --- | --- | --- | --- |
| `c393db5` template only | 40/54 | 13/54 | 79.6% | 11.1 | 1 overflow |
| + both intents, + supersedes | **54/54** | **33/54** | 74.1% | 11.1 | **6** (5 overflow) |
| + retrieval depth 15 | 54/54 | 30/54 | 75.9% | 5.6 | **0** |

Every one of those accuracies is inside the others' spread. **Raising retrieval adoption from 13
to 30 of 54 produced no measurable accuracy change.**

What it did produce, at first, was a new failure mode. Forty-five place records travel from
`nearby_places` through `nearest` into every later prompt: the median question reached 27k prompt
tokens against a 65,536 window and five of fifty-four died of `llm_context_overflow` — not
answered badly, not allowed to finish. The template's example asked for 45 because that is the
tool's maximum, and the planner then wrote 100 in fourteen plans. Asking for 15 — k plus margin,
which is what an ordinal needs — took the median to 23k and the failures to zero, and the planner
copies the number: 27 of 30 retrievals now ask for 15. The worst single prompt is still 68k, so
the risk is reduced rather than removed.

Two mechanisms were worth fixing on their own evidence, independent of the accuracy:

- The Analysis stage labelled 14 of these 54 questions `poi` and the rest `nearby`, and the
  template was gated on `nearby` alone, so those 14 were handed `Geocode-Batch-Compare` — which
  spans four intents and shows the option-ranking shape. A template gated on one intent label is
  gated on that stage guessing right.
- Of the 40 that *were* offered it, 27 still built `Geocode-Batch-Compare`'s shape, because it sat
  beside it as the second example. A worked example that answers a different question is worse
  than no second example. Templates now name what they supersede for a question shape, and
  suppression runs only downward from the winner, so a superlative question keeps
  `Geocode-Batch-Compare` in front where it belongs.

Prose was not the lever, and it is worth recording why. `GRAPH_PROMPT` already said, in as many
words, "four named options are not a candidate set, they are answer texts, and geocoding them and
taking the nearest answers 'which of these is closest' instead of the question asked" — and 41 of
53 plans geocoded the options anyway. What moved behaviour both times was the worked example the
retrieval stage hands over, not the paragraph above it.

**Why the number did not move is the finding.** The family is answerable from the option list
alone often enough — 56.2% by construction on v7's eight `nearby_kth_nearest` rows — that
retrieving cannot show an advantage over not retrieving. That is a defect in the benchmark, not in
the agent, and it is what (c) addresses.

### (c) The ordinal was indexed, not drawn — and the gate could not see it

Both ordinal families keyed k on the anchor loop index: `kth = 2 + (index % 3)` for
`nearby_kth_nearest` and `2 + (index % 2)` for `nearby_subtype_kth`. Over v5, v6, v7, v7h and v7h2
that produced **k=2 on seven of eight rows** every time, which is the same defect `AGENTS.md`
already names for rung ladders — "keying the answer on a loop index spends them wherever the loop
happened to succeed" — except that no option ever prints k, so nothing that reads the options
could see it.

It matters because ranking the four options against each other, which is what the agent does when
it does not retrieve, answers a k-th question whenever all k−1 nearer places are among the three
decoys: `C(m−k, 4−k) / C(m−1, 3)`, or 60% at k=2, 30% at k=3, 10% at k=4. A family that is 56%
answerable without a map cannot show whether an agent used one — which is exactly why (b)'s
adoption gains had nowhere to appear.

Least-used-first alone did not fix it, and measuring said why: under the 90 m ordinal margin,
**six of one draw's eight anchors were separable at k=2 only**, because ranks three through five of
a dense neighbourhood sit within 90 m of each other. There was nothing to choose. The family now
keeps scanning anchors while a value is short, bounded by `_scan_limit`, and fills equal
quotas at the end with the remainder going to k=2 — the value the city can always supply.

Rebuilt under the same seed: `nearby_kth_nearest` goes from `{2: 7, 3: 1}` to `{2: 4, 3: 2, 4: 2}`
and its option-only ceiling from **56.2% to 40.0%**; `nearby_subtype_kth` from `{2: 10}` to
`{2: 5, 3: 5}`. `subtype_counts` was incremented at the point k was chosen rather than the point a
row was made, so rows that failed their trap or resolvability checks still spent a value; it now
increments on the success path.

`data/audit_dataset.py` grew the check that would have caught this: for a family whose
`gold_evidence` carries `k`, more than 70% of rows on one value while the family varies it at all
is a drawn parameter that is not being drawn. A fair draw over three values lands 7 of 8 on one of
them about once in three hundred times, so the threshold is not tight. It fires on v5, v6, v7, v7h
and v7h2 as built, and on nothing else.

The datasets on disk are unchanged — this changes what the builder draws next, which is the third
holdout.

### (a) The third holdout

`dataset/seoul_kmapeval_v7h3_holdout_100.jsonl`, the v7 builder under seed 750914: three questions
in common with v7, one with v7h, none with v7h2. Clean on the first audit including the new k
check. Built and run at `8797217` — the first set drawn for code that has the arithmetic
operators, the ordinal template, the retrieval depth, and a drawn ordinal.

| | floor | ReAct (reference) | Spatial-Agent | gap |
| --- | --- | --- | --- | --- |
| v7h (`0aabaa9`) | 29.5 | 48.0 | 70.7 | 22.7 |
| v7h2 (`38566f3`) | 25.5 | 45.7 | 72.3 | 26.7 |
| **v7h3 (`8797217`)** | 23.5 (25, 22) | **51.0** (49, 50, 54) | **72.0** (72, 74, 70) | **21.0** |

Three independent draws, three architectures' worth of code changes between them, and the gap sits
at 21–27 against single-agent spreads of 4–8. The floor drifted down as the families got harder;
`nearby_kth_nearest` is now 0/8 closed-book on both passes.

**The ordinal families moved, and are still not solved.**

| | v7h2 (`38566f3`) | v7h3 (`8797217`) | ReAct on v7h3 |
| --- | --- | --- | --- |
| `nearby_kth_nearest` | 25.0% | **50.0%** | 62.5% |
| `nearby_subtype_kth` | 56.7% | **63.3%** | 43.3% |

The comparison understates it if anything: v7h3's `nearby_kth_nearest` is the harder family, with k
drawn `{2:5, 3:2, 4:1}` instead of `{2:7, 3:1}` and an option-only ceiling of 46.2% rather than
56.2%. Spatial-Agent doubled on a family that got harder. It is still **below ReAct** there, which
is worth saying plainly — the five tools and fifteen iterations answer a k-th nearest question
better than the graph does, and the graph's advantage on this class comes entirely from the subtype
variant.

`nearby` remains Spatial-Agent's weakest class (62.4%, against 74.6% distance and 90.9% routing),
but it is no longer tied with ReAct's: 62.4% against 60.2%.

**Retrieval's tax is now the leading failure mode.** Eight of Spatial-Agent's 300 questions ended
in `llm_context_overflow` and six of those eight are the two ordinal families — the ones that
retrieve. The median question is 13,581 prompt tokens and only one call in 300 crossed 65,536, so
this is a tail, not a norm; but it is a tail the ordinal template created, and at eight questions
it costs about 2.7 points. Lowering retrieval depth from 45 to 15 reduced this and did not remove
it. The next move on it is architectural — what the graph carries forward into later prompts —
rather than another number in a template.

Cost is unchanged in shape: ReAct 2,510 calls for 4.24M tokens at 36.5s a question, Spatial-Agent
919 calls for 6.42M tokens at 71.1s.

**v7h3 is spent the moment `src/` or `data/` changes again.** As of `8797217` it is the only
holdout number that belongs to the current code.

## Would a bigger step budget fix it? Measured as an ablation, three passes a side

`MAX_REASONING_STEPS` is configurable, so the obvious question about v6's four-stop trip families
is whether raising it repairs them. It was first answered from one existing pass at 30; that pass
was one draw and it read wrong. Both architectures were then run over v6 three times at each
budget, at revision `ece22ef`, `--provider kakao`, `--react-tools reference`, temperature 0, on
`google/gemma-4-E4B-it-qat-w4a16-ct`. **Every number below is an ablation and none of it may be
pooled with a default-budget run.**

| | overall mean | passes | four-stop (45 rows) | other 85 rows | `iteration_limit` | LLM calls/pass |
| --- | --- | --- | --- | --- | --- | --- |
| ReAct @ 15 | **39.0%** | 40, 39, 38 | 8/45 = 17.8% | 109/255 | 19 | 843 |
| ReAct @ 30 | **47.3%** | 52, 48, 42 | 26/45 = **57.8%** | 116/255 | **0** | 883 |
| Spatial-Agent @ 15 | **69.0%** | 67, 71, 69 | 22/45 = 48.9% | 185/255 | 0 | 306 |
| Spatial-Agent @ 30 | **74.0%** | 73, 78, 71 | 28/45 = 62.2% | 194/255 | 0 | 309 |

**The budget is binding, and the one-pass reading of it was wrong.** The earlier entry here said
the whole benchmark landed on 38.0% either way. Over three passes it does not: ReAct moves 39.0 to
47.3, and the two budgets' run ranges (38–40 against 42–52) do not overlap. `iteration_limit` goes
to zero, as the single pass had already shown.

**But the gain lives in one family, not two.** Per family, ReAct's `trip_optimal_order_four` goes
2/24 to **21/24** while `trip_total_distance_four` does not move at all — 6/21 to 5/21. Nothing
else on the benchmark shifts by more than three rows in ninety. So the budget was the whole story
for ordering four stops, and none of the story for summing four legs: that family fails on the
arithmetic ReAct has to do in prose, and twice the iterations buys it nothing. A family that ends
on `iteration_limit` is not thereby a family the budget explains.

**Raising it moves both architectures, which is the reason it is not a repair.** Spatial-Agent
gains too — 69.0 to 74.0, and its own `trip_optimal_order_four` 9/24 to 15/24 — because
`MAX_REASONING_STEPS` bounds the nodes its planner may author as well as ReAct's loop iterations.
The architecture gap is 30.0 points at budget 15 and 26.7 at budget 30: it is not an artefact of
the budget, and the budget does not close it.

Three reasons the default stays at 15 anyway:

- **It is the baseline's configuration, not a tuning knob.** 15 is langchain's own default, which
  is what the reference baseline runs. A larger budget is a stronger-than-paper baseline — a
  labelled ablation whose number cannot be pooled with the default runs or set beside upstream
  Spatial-Agent's 71.07%.
- **It is one budget, not ReAct's.** The same setting bounds the nodes Spatial-Agent's planner may
  author, and the table above shows it moving both architectures at once. A run at 30 is not a
  controlled comparison of either.
- **Longer loops crowd the context window.** Each iteration carries another observation into the
  prompt, and the worst prompt in the v7h3 run already reached 68,430 tokens against a 65,536
  window with eight questions lost to `llm_context_overflow`.

Shrinking the family, which is what v7 did, keeps the baseline faithful *and* leaves a family that
discriminates — and it is the better repair for the second reason above: v7 changes what the
question asks, so it moves neither architecture's budget.

**Two corrections to this file's own record.** The step-budget invariant said "every row of both
ended on `iteration_limit` with ReAct scoring 0/15"; pooled over four passes it is 24 of 60 rows
and 14/60 correct. And the one-pass ablation entry said a budget of 30 left the benchmark at 38.0%;
three passes a side say 39.0 to 47.3, concentrated in one of the two families. `AGENTS.md` and
`data/build_kmapeval_dataset.py` are corrected to match both.

## The first build at scale: 283 questions, three passes, one revision

Every number above rests on a hundred questions, where this endpoint's run-to-run spread is about
±8 points and a single-run comparison of two architectures is not a result. So the standard builder
was asked for 300 — it drew 283, the rest lost to live Kakao — and both architectures ran it three
times at `6bae55c`, `--provider kakao`, `--react-tools reference`, temperature 0,
`MAX_REASONING_STEPS=15`, no output ceiling.

| | passes | mean | pooled |
| --- | --- | --- | --- |
| no-tool floor | 84, 79 | **28.8** | 163/566 |
| ReAct (reference) | 49.8, 47.7, 49.1 | **48.9** | 415/849 |
| Spatial-Agent | 79.5, 77.7, 79.5 | **78.9** | 670/849 |

**The spread collapses with the sample.** ReAct's three passes span 2.1 points and Spatial-Agent's
1.8, against the 4–8 the hundred-row sets showed. Nothing about the endpoint changed; the sample
got 2.8× bigger. This is the first set here on which one pass a side would have been worth reading.

**The gap is 30.0 points**, the widest measured, and it holds in every class:

| class | rows | floor | ReAct | Spatial-Agent |
| --- | --- | --- | --- | --- |
| routing | 66 | 20.5 | 27.3 | **91.4** |
| distance | 46 | 17.4 | 35.5 | **82.6** |
| nearby | 93 | 41.4 | 65.2 | **78.5** |
| trip | 66 | 26.5 | 55.1 | 67.2 |
| radius | 12 | 33.3 | 58.3 | 63.9 |

Routing is where the architectures diverge most (64 points) and where the floor is lowest — the
turn-level questions cannot be guessed and ReAct cannot hold a step list across one-leg
`Directions` calls. `trip` and `radius` are the two classes where Spatial-Agent's lead is under 15
points, and `nearby_within_radius_count` (58.3 / 63.9) and `poi_farthest_of_three` (20.5 / 53.8)
are the two families both architectures do worst on. Those are family findings, not architecture
ones.

**The floor's own composition matters.** 28.8 overall, but the five `unanswerable_*` families
score 40/42 closed-book: refusing is guessable without a map, by design. Over the other 262 rows
the floor is **23.5**, just under four-way chance, and the chosen-option histogram is flat
(53/76/81/73 and 59/71/79/74). No family is answerable from the option set.

**v7's three-stop trip families fit the budget.** Zero `iteration_limit` in 849 ReAct rows, against
24 of 60 on v6's four-stop pair. They are not easy — `trip_optimal_order` 41.7% and
`trip_total_distance` 34.9% — they are *lost*, which is what a family is supposed to measure. No
`llm_output_truncated` either, on either side.

**Failures are Spatial-Agent's alone, and there are twelve in 849 rows (1.4%).** Six
`llm_context_overflow` and six `agent_reasoning_failure`; ReAct had none of either. The overflows
are still retrieval's tail — one row, `kmapeval_278`, overflowed on all three passes — and they
explain the whole of Spatial-Agent's one loss to ReAct on `unanswerable_rating` (5/9 against 9/9):
four losses, all of them two rows that never finished. Four of the six reasoning failures are
`trip_total_distance`.

Cost keeps its shape: ReAct 2,292 LLM calls for 4.0M tokens at 37.7s a question; Spatial-Agent 870
calls for 6.3M tokens at 79.9s. Fewer, larger calls.

### The audit failed this set, and the cause was a constant that stopped growing

`data/audit_dataset.py` exits non-zero on it:

    nearby_kth_nearest: k varies across [2, 3, 4] but 19 of 24 rows ask for k=2

This is the defect (c) above was supposed to have closed. Drawing the ordinal was not enough: the
family's coverage scan was bounded by `ORDINAL_SCAN_LIMIT = 24`, which at v6's quota of **eight**
rows is three times the count and leaves room to keep hunting the scarce ordinals, and at the
standard builder's **twenty-four** *is* the count. The loop stopped the instant it had enough rows
and shipped whatever the first anchors offered — and a dense city offers k=2, because ranks three
through five of a block sit inside the 90 m ordinal margin. The balancing code was present and
correct the whole time; it never got a candidate to balance.

The constant is now a floor, `_scan_limit(floor, count) = max(floor, count * 3)`. At every quota in
v6 and v7 — eight rows for this family, four for the radius one — that is the constant it always
was, so **both benchmarks of record still draw what they drew**; a fix to a shared generator that
moved them would have silently rewritten two published numbers. `tests/test_benchmark_families.py`
pins the cause rather than the symptom: the same fake draw fails the audit's concentration
threshold when `_scan_limit` is pinned back to the floor.

Two things follow. The 24 affected rows are 8.5% of the set and they are not a second answer key —
the options are place names and the gold position is shuffled — so the accuracies above stand; what
the family measured is narrower than advertised, mostly "second nearest" rather than k ∈ {2,3,4}.
And **the set is spent**: `data/` changed in response to what it showed, so 48.9/78.9 is what the
code at `6bae55c` scored, not a holdout number for anything built afterwards.

## v7a: the same size a second time, and what two draws say that one could not

`dataset/seoul_kmapeval_v7a_mcq_300.jsonl` is the standard builder asked for 300 again; it drew
282, and it is the **first set at this size that `data/audit_dataset.py` passes**. Built at
`ba92d9c` fifteen minutes after the scan-limit fix and run there, with nothing under `src/` or
`data/` changed since and six of its 282 questions in common with the previous draw. A holdout.

| | passes | mean | pooled |
| --- | --- | --- | --- |
| no-tool floor | 76, 75 | **26.8** | 151/564 |
| ReAct (reference) | 50.7, 51.8, 53.9 | **52.1** | 441/846 |
| Spatial-Agent | 78.0, 81.2, 78.4 | **79.2** | 670/846 |

The gap is 27.1. Set beside the previous draw — same generator, same revision family, 283 rows
against 282 — the overall numbers are stable and the floor is too:

| | floor | ReAct | Spatial-Agent | gap |
| --- | --- | --- | --- | --- |
| v7 300 (`6bae55c`) | 28.8 | 48.9 | 78.9 | 30.0 |
| v7a 300 (`ba92d9c`) | 26.8 | 52.1 | 79.2 | 27.1 |

**The ordinal fix worked, and did not solve the family.** `nearby_kth_nearest` goes from
`{2: 19, 3: 3, 4: 2}` to `{2: 16, 3: 6, 4: 2}`, which clears the audit's 70% concentration
threshold at 66.7% — cleared, not comfortably. The remaining lopsidedness is the city rather than
the code: an anchor whose ranks one through five are all separable by 90 m is rare, and a scan that
finds none cannot balance what it does not have. Raising the limit further buys nothing; the next
move on this family, if it needs one, is the margin or the rank pool.

### Family-level accuracy does not survive a redraw, and overall accuracy does

This is the finding two draws bought that one could not. Both sets are ~282 rows at three passes a
side, so per-pass noise is 1.6–1.8 points. Between draws:

| family | floor | ReAct | Spatial-Agent |
| --- | --- | --- | --- |
| `trip_total_distance` | 23.8 → **40.5** | 34.9 → **93.7** | 68.3 → 82.5 |
| `nearby_subtype_kth` | 26.7 → 13.3 | 71.1 → **47.8** | 86.7 → 90.0 |
| `routing_nth_turn` | 16.7 → 11.9 | 38.1 → **60.3** | 96.8 → 93.7 |
| `routing_turn_count_via` | 9.5 → 2.4 | 22.2 → **41.3** | 82.5 → 93.7 |
| `nearby_within_radius_count` | 33.3 → 33.3 | 58.3 → **38.9** | 63.9 → 66.7 |

ReAct's `trip_total_distance` moved 58.7 points on 63 rows. The option structure is identical
across the two draws — gold sorted-rank spread flat in both, closest-distractor relative gap
median 0.20 in both — and the visible difference is trip length: total driving distance median
22.2 km, then 30.9 km. **The floor moved with it, 23.8 to 40.5**, which says a third of that swing
is the family becoming guessable rather than ReAct becoming able: a three-leg drive across Seoul
that totals ~31 km sits closer to whatever prior the model has for such a trip than one that totals
~22 km, so the right bucket gets picked without measuring anything. Lift over the floor is the
honest read, and it still moved: 11.1 points, then 53.2.

The consequence for this repository is a rule, not a caveat. **A single draw supports an overall
accuracy and does not support a family or class claim.** In this draw ReAct wins the whole `trip`
class, 73.7 against Spatial-Agent's 69.2; in the previous one it lost it, 55.1 against 67.2. Every
family-level diagnosis published here before this section rests on one draw.

### What the floor says about `distance`

| class | rows | floor | ReAct | Spatial-Agent |
| --- | --- | --- | --- | --- |
| routing | 66 | 12.1 | 44.9 | **92.4** |
| distance | 45 | 25.6 | **25.9** | 83.7 |
| nearby | 93 | 36.6 | 56.3 | 76.3 |
| trip | 66 | 27.3 | 73.7 | 69.2 |
| radius | 12 | 33.3 | 38.9 | 66.7 |

**ReAct scores 25.9 on `distance` against a floor of 25.6.** Five tools and fifteen iterations
bought it three tenths of a point over answering closed-book; on `poi_farthest_of_three` it scores
25.0 against a floor of 29.2, which is below it. Spatial-Agent gets 83.7 on the same rows, so the
questions are answerable and the measurements are there to be made. This is the sharpest statement
this repository has of what the baseline cannot do: it is not that ReAct measures distances badly,
it is that on this class it is not measuring at all.

Routing is the mirror image — the lowest floor in the set (12.1, and `routing_turn_count_via` is
1/42 closed-book) and the class ReAct climbs furthest above its own floor, 32.8 points. The widest
architecture gap is `distance`'s 57.8, for the reason above: one side is measuring and the other
is not.

`unanswerable_*` remains free for everyone: 40 of 42 rows closed-book. Over the other 522 rows the
floor is **21.3**, below four-way chance, and the chosen-option histogram is flat in both passes.

### Failures, and one that repeats

ReAct: three `iteration_limit` in 846 rows (0.35%) — two `trip_optimal_order`, one
`nearby_within_radius_count`. Not the 24-of-60 that v6's four-stop families produced, but not zero
either; v7's three-stop trips fit the budget with little to spare. No truncation, no overflow.

Spatial-Agent: eleven in 846 (1.3%) — six `llm_context_overflow`, all but one in the two ordinal
families that retrieve, and five `agent_reasoning_failure`. **`kmapeval_223` failed on all three
passes**, as `trip_total_distance` did four times on the previous draw. A failure that repeats
across passes is a defect rather than a draw, and it is the one worth opening next.

Cost is unchanged in shape: ReAct 2,333 calls for 4.0M tokens at 36.7s a question, Spatial-Agent
868 calls for 6.5M tokens at 79.6s.

**One tooling defect, found by running this floor.** `data/measure_no_tool_floor.py` measured a
dataset in one thread pool and caught only `LLMUnavailableError`. A closed-book question spiralled
to 65,304 completion tokens, `LLMOutputTruncatedError` escaped the pool, and every other answer in
the run went with it. It now catches the same three the evaluator separates — unavailable,
truncated, overflow — records each as a per-question failure and names the rows, so a floor over
fewer rows than it claims says so. Fixed at `218bcce`; both passes above ran clean afterwards.

### The repeated failure was a spelling, and it was costing far more than the rows it lost

`kmapeval_223` failed on all three v7a passes, and across the two 300-row draws seven of the ten
`agent_reasoning_failure` rows read `GeoFlow node ... is missing arguments: ...`. Reading the logs
rather than the counts showed the counts were the small part: **31 of the 848 Spatial-Agent
question-runs in the v7a passes hit that refusal at least once**, and most of them survived only by
spending a repair round on it — 76 repair calls in those three passes.

Three shapes, all of them the port's vocabulary rather than the planner's reasoning.

**A node whose whole input is its one dependency.** The planner writes `arguments: {}` and
`depends_on: ["route_legs"]`, because it has already said where the value comes from:

    {"id": "leg_distances",  "operator": "extract_distance", "arguments": {}, "depends_on": ["route_legs"]}
    {"id": "total_distance", "operator": "sum_amounts",      "arguments": {}, "depends_on": ["leg_distances"]}

One missing slot and one named source is one binding, not a guess, so the validator now makes it:
`arguments[missing] = "$<dependency>"`. Deliberately narrow — two missing arguments, or two
dependencies, and which value belongs in which slot *is* a guess, so nothing is filled and the plan
is refused exactly as before. This is also why the repair round was not the answer: on
`kmapeval_223` it filled `extract_distance` and left `sum_amounts` empty, then the lenient pass
failed on the node the repair had skipped.

**`nearest(center=…)`.** `nearby_places` calls that point `center` and `nearest` calls it `anchor`,
so a planner that retrieves with one node and ranks with the next writes one vocabulary across
both. The implementation accepted only `anchor`, and two `nearby_subtype_kth` questions died on the
spelling. `center`, `origin`, `from_place` and `reference` now normalize onto `anchor` — spellings
for the same point, and nothing else: no ordinal, no candidate list.

**`extract_distance(routes=[…])`.** Measuring every leg before adding them is one unambiguous
thing, and `$segments.routes` is a list either way. The plural now maps onto the slot and a list is
measured element-wise. It still refuses to invent: a `$node` reference that never resolved raises
where it stands rather than reporting zero, which is the rule that caught `sum_amounts` reading
`"extract_distance(route_A1_A2)"` as text.

Replayed against the graphs the planner actually wrote — pulled out of `logs/`, not reconstructed —
29 of the 31 refusals now validate at the point they were refused, which removes 29 of the 76
repair rounds as well as the questions that were lost outright. The two that still fail are a
hallucinated operator (`"operator": "pairs"`) and a genuinely under-specified node, and both should
fail: refusing a graph the executor could not have run is the validator working.

The risk this takes on, stated plainly: a plan that used to be refused and repaired into something
correct can now execute on the filled binding and answer wrongly, where before it failed loudly. It
is bounded by the operators still raising on a shape they cannot read, and by the fill being the
only binding the plan allows — but it is a real trade, and it is why the fill is one slot and one
source rather than a best match.

**This spends `seoul_kmapeval_v7a_mcq_300.jsonl`.** `src/` changed in response to what that set
showed, so 52.1/79.2 is what the code at `ba92d9c` scored. The next holdout has to be a new draw.

## v7b: a third draw, and the pattern across three

`dataset/seoul_kmapeval_v7b_mcq_300.jsonl` is the standard builder's third 300-question draw; it
drew 283. Clean on `data/audit_dataset.py`, four questions in common with v7a and seven with v7.
Built and run at `796c683`, the argument-spelling fix.

| | passes | mean | pooled |
| --- | --- | --- | --- |
| no-tool floor | 84, 83 | **29.5** | 167/566 |
| ReAct (reference) | 48.4, 49.8, 47.3 | **48.5** | 411/849 |
| Spatial-Agent | 79.5, 72.4, 78.5 | **76.8** | 652/849 |

| | floor | ReAct | Spatial-Agent | gap |
| --- | --- | --- | --- | --- |
| v7 (`6bae55c`) | 28.8 | 48.9 | 78.9 | 30.0 |
| v7a (`ba92d9c`) | 26.8 | 52.1 | 79.2 | 27.1 |
| v7b (`796c683`) | 29.5 | 48.5 | 76.8 | 28.3 |

Three draws now agree the gap sits at 27–30 and neither side's mean overall accuracy has moved
more than 3.6 points off its three-draw average. Spatial-Agent's spread on this draw is wider than
the other two (sd 3.81 against 1.02 and 1.75) — one pass, the middle one, lost 4 questions to
`llm_context_overflow` and 2 to `agent_reasoning_failure` that its other two passes did not. That
is endpoint variance the way the rest of this document already treats it, not a regression: nothing
that touches Spatial-Agent's prompting changed between v7a and v7b.

**`trip_total_distance` supports the length hypothesis a third time.** v7a's draw showed the
family's floor and ReAct's score both moving with the median trip length and flagged it as a
prediction to check. Three draws now line up monotonically with length:

| draw | median total | floor | ReAct | lift |
| --- | --- | --- | --- | --- |
| v7 | 22.2 km | 23.8 | 34.9 | 11.1 |
| v7b | 26.5 km | 21.4 | 52.4 | 31.0 |
| v7a | 30.9 km | 40.5 | 93.7 | 53.2 |

Lift over the draw's own floor rose with trip length across these three; the floor itself did not
(v7b's is the lowest of the three despite a longer median trip than v7's). **The fourth draw (v7c)
broke the monotonic reading — see the v7c correction below.** Length is a positive but noisy
predictor of ReAct's score on this family, not a ladder.

### The argument-spelling fix, measured

`796c683` was written against v7a's logs and had not been measured on an independent draw. It has
one now:

| | runs | repair rounds | hit "missing arguments" |
| --- | --- | --- | --- |
| v7 (`6bae55c`, pre-fix) | 849 | 76 (9.0%) | 33 (3.9%) |
| v7a (`ba92d9c`, pre-fix) | 848 | 76 (9.0%) | 31 (3.7%) |
| **v7b (`796c683`, post-fix)** | 849 | **55 (6.5%)** | **5 (0.6%)** |

The refusal it targeted dropped by 84%, and total repair rounds by a third, on a draw the fix was
never tuned against. It did not move overall accuracy — v7b's 76.8 sits inside the other two draws'
range — which is what "vocabulary, not reasoning" predicts: the fix lets more plans *execute*
where they used to be silently discarded and retried, and a repaired plan mostly already answered
correctly, so removing the repair step removes cost, not right answers. `agent_reasoning_failure`
carrying "missing arguments" fell from most of that category to one row (`kmapeval_117`, four
independently computed distances handed to `pairwise_extremes` under its wrong argument name *and*
as three bare `dist_1`/`dist_2`/`dist_3` — two missing arguments and a shape that would not resolve
even relabelled, so refusing it is correct: the planner used the wrong operator, not the wrong
spelling).

### A second refusal, and it was not new

`kmapeval_269` failed all three v7b passes on the same message: `dictionary update sequence element
#0 has length 1; 2 is required`. It was `identity_measure` handed `"arguments": "$nearest_result"`
— a bare reference, not an object — and `_ground_graph_literals` called `dict()` on it
unconditionally; `dict()` on a string iterates its characters, none of them length-2 pairs, and the
repair round was handed a Python internals leak with nothing about the graph in it. The same
message appears twice more in `logs/`, dated back to the v6 and first v7 runs — not new, just never
traced before now that a question repeated it three times in one draw and made it worth opening.

Fixed the same way as the missing-arguments case: a non-dict `arguments` is wrapped under the
operator's one required slot when it has exactly one — `identity_measure`'s is `value`, the same
shape its own auto-generated closing step already writes — and left alone otherwise, so a
multi-argument operator is still refused, now with the graph's own "arguments must be an object"
rather than a string that names no defect. Fixed at `1cb6bdc`, after the v7b run, so this draw's
numbers do not reflect it — three rows out of 849, one per Spatial-Agent pass.

**This spends `seoul_kmapeval_v7b_mcq_300.jsonl`.** `src/` changed twice in response to what its
run showed — once measuring the earlier fix, once from a defect the run itself surfaced — so
48.5/76.8 is what `796c683` scored, not a number to hold the next draw against. The next holdout
needs a fresh draw under `1cb6bdc` or later.

## v7c: the fourth draw, the first that stays held out, and a correction

`dataset/seoul_kmapeval_v7c_mcq_300.jsonl` is the standard builder's fourth 300-question draw; it
drew 282. Built and run at `a50096a` — which carries both the argument-spelling fix (`796c683`) and
the bare-reference crash fix (`1cb6bdc`). One question in common with v7, six with v7a, three with
v7b.

| | passes | mean | pooled |
| --- | --- | --- | --- |
| no-tool floor | 75, 67 | **25.2** | 142/564 |
| ReAct (reference) | 52.1, 52.5, 51.4 | **52.0** | 441/846 |
| Spatial-Agent | 83.0, 81.9, 79.1 | **81.3** | 688/846 |

| | floor | ReAct | Spatial-Agent | gap |
| --- | --- | --- | --- | --- |
| v7 (`6bae55c`) | 28.8 | 48.9 | 78.9 | 30.0 |
| v7a (`ba92d9c`) | 26.8 | 52.1 | 79.2 | 27.1 |
| v7b (`796c683`) | 29.5 | 48.5 | 76.8 | 28.3 |
| v7c (`a50096a`) | 25.2 | 52.0 | 81.3 | 29.3 |

Four draws hold the gap at 27–30. Spatial-Agent's 81.3 is the highest of the four and ReAct's 52.0
ties its highest; the floor is the lowest of the four, so the lift is the widest measured.

**v7c is the first of these that stays held out.** v7a and v7b were each spent by an `src/` change
made in response to what they showed. Nothing under `src/` or `data/` had been changed in response
to v7c, so **52.0/81.3 is a genuine held-out number for `a50096a`** — the fixes were already in
when it was drawn, not written against it. (`src/` has changed twice since: `c7d49cb` deleted the
context cache provider, a path these runs never took, and the v7d entry's crash fix. Neither was
written against v7c, so the number stands as `a50096a`'s; it is no longer *current* code's.)

### Both operator fixes, measured on an independent draw

| | runs | repair rounds | "missing arguments" | bare-ref crash | `agent_reasoning_failure` |
| --- | --- | --- | --- | --- | --- |
| v7 (pre-both) | 849 | 76 | 33 | — | 5 |
| v7a (pre-both) | 848 | 76 | 31 | 1 | 6 |
| v7b (spelling only) | 849 | 55 | 5 | 1 | ~6 |
| **v7c (both)** | 846 | **49** | **2** | **0** | **0** |

Spatial-Agent recorded **zero `agent_reasoning_failure` across all three v7c passes** — every one of
its failures this draw was `llm_context_overflow` (5 in 846, the retrieval tail). The category that
was the leading fixable failure across the first three draws is gone on the fourth, on a draw
neither fix was tuned against. Repair rounds are down to 49 from the pre-fix 76. What the fixes did
not do is move overall accuracy: 81.3 sits at the top of the four-draw range but inside the ±8 this
endpoint carries, which remains the reading — the fixes remove cost and silent discards, not wrong
answers.

### Correction: the trip-length hypothesis is positive but not monotonic

The v7b entry above said lift over the draw's own floor is "monotonic in trip length across all
three draws." The fourth draw breaks that:

| draw | median total | floor | ReAct | lift |
| --- | --- | --- | --- | --- |
| v7 | 22.2 km | 23.8 | 34.9 | 11.1 |
| v7b | 26.5 km | 21.4 | 52.4 | 31.0 |
| v7c | 30.5 km | 35.7 | 63.5 | 27.8 |
| v7a | 30.9 km | 40.5 | 93.7 | 53.2 |

v7c is longer than v7b and lifts *less* (27.8 vs 31.0), and the two draws at essentially the same
median length — v7c at 30.5 km and v7a at 30.9 km — differ by 25 points in ReAct's raw score and by
25 in lift. Longer trips are easier for ReAct on average, but length is a noisy predictor, not a
ladder, and the clean monotonic reading was three points lining up by luck. `trip_total_distance`
is a family whose per-draw score swings widely — which is the same lesson the `trip` class taught at
n=2 draws, now confirmed at n=4: quote it across draws or not at all.

### The audit failed, worst of the four, and this one is the city not the code

`data/audit_dataset.py` exits non-zero on v7c:

    nearby_kth_nearest: k varies across [2, 3, 4] but 20 of 24 rows ask for k=2   (83.3%)

worse than v7's 79% and past v7a/v7b, which cleared the 70% threshold at 66.7%. This is the defect
`ba92d9c` addressed by growing the coverage scan — and it is addressed: the scan now runs far
enough. What it cannot manufacture is anchors the city does not have. An anchor whose ranks one
through five are all separable by the 90 m ordinal margin is rare, and a draw that happens to find
few of them lands most of its rows on k=2 no matter how long the scan runs.

> **Corrected by v7d.** This paragraph originally continued: "v7a and v7b found enough; v7 and v7c
> did not. This is draw variance in what Seoul offers, not a generator bug." It is not variance.
> A fifth draw made it five for five in this family against zero for five in `nearby_subtype_kth`,
> which draws k identically, and the mechanism is measurable in the two families' rank gaps: this
> one anchors on Seoul's four densest chain categories. It is a generator defect in the category
> pool. See the v7d entry.

The consequence is bounded and the same as v7's: the 24 `nearby_kth_nearest` rows are not a second
answer key (place-name options, shuffled gold), so **the overall 52.0/81.3 and every other family
stand**; what does not stand is a `nearby_kth_nearest` family number for this draw, which measured
mostly "second nearest." If a future draw needs that family to discriminate at k∈{3,4}, the lever
is the ordinal margin or the anchor category pool, not the scan limit — and changing either would
spend whatever set exposed the need.

## v7d: the fifth draw, and two claims the fifth draw retires

`dataset/seoul_kmapeval_v7d_mcq_300.jsonl` is the standard builder's fifth 300-question draw; it
drew 281. Built and run at `c7d49cb`, which is `a50096a` plus the removal of the context cache
provider — a deletion of a path these runs never took (`--provider` resolved to `kakao` for every
v7 dataset, none of which carries a `context`), so v7d and v7c are measured on the same runtime.

| | passes | mean | pooled |
| --- | --- | --- | --- |
| no-tool floor | 79, 75 | **27.4** | 154/562 |
| ReAct (reference) | 47.0, 48.0, 47.7 | **47.6** | 401/843 |
| Spatial-Agent | 81.5, 80.1, 79.7 | **80.4** | 678/843 |

| | floor | ReAct | Spatial-Agent | gap |
| --- | --- | --- | --- | --- |
| v7 (`6bae55c`) | 28.8 | 48.9 | 78.9 | 30.0 |
| v7a (`ba92d9c`) | 26.8 | 52.1 | 79.2 | 27.1 |
| v7b (`796c683`) | 29.5 | 48.5 | 76.8 | 28.3 |
| v7c (`a50096a`) | 25.2 | 52.0 | 81.3 | 29.3 |
| v7d (`c7d49cb`) | 27.4 | 47.6 | 80.4 | **32.8** |

Five independent draws, three passes a side each, hold the gap at 27–33. Excluding the
`unanswerable_*` families the floor is **21.9**, below four-way chance, and the chosen-option
histogram is flat on both passes.

### Where each architecture's accuracy actually comes from

| class | rows | floor | ReAct | lift | Spatial-Agent | lift |
| --- | --- | --- | --- | --- | --- | --- |
| `distance` | 46 | 16.3 | 18.8 | **+2.5** | 89.1 | +72.8 |
| `routing` | 66 | 18.2 | 26.3 | **+8.1** | 96.0 | +77.8 |
| `nearby` | 91 | 42.3 | 59.7 | +17.4 | 79.5 | +37.2 |
| `radius` | 12 | 25.0 | 58.3 | +33.3 | 61.1 | +36.1 |
| `trip` | 66 | 24.2 | 70.2 | **+46.0** | 63.6 | +39.4 |
| overall | 281 | 27.4 | 47.6 | +20.2 | 80.4 | +53.0 |

Two readings the pooled number hides. First, **ReAct's 47.6 is almost entirely `trip` and
`radius`**: on `distance` it clears its own closed-book floor by 2.5 points and on `routing` by
8.1, which is the v7a finding (25.9 against a floor of 25.6 on `distance`) reproduced on an
independent draw and extended to routing. Five tools and fifteen iterations buy nearly nothing on
109 of 281 rows. Spatial-Agent gains 72.8 and 77.8 on those same rows, so the rows are answerable
and it is the baseline's surface that cannot reach them.

Second, **ReAct out-lifts Spatial-Agent on `trip` for the first time** (+46.0 against +39.4), and
wins the class outright 70.2 to 63.6. Spatial-Agent's `trip_optimal_order` fell to 52.8 — its worst
of the five draws — and `trip_feasible_count_five` to 58.7, also its worst, while ReAct's
`trip_total_distance` came in at 90.5, its second best. This is a per-family swing, not a finding
about the architectures; see the next section for why that is now the expected shape.

### Per-family accuracy is the *baseline's* instability, not the family's

With five draws each measured over three passes, the spread of a family's accuracy across draws can
be separated from pass noise. Mean cross-draw range per family, excluding the `unanswerable_*`
families:

| | mean cross-draw range | worst family |
| --- | --- | --- |
| ReAct | **23.6 pts** | `trip_total_distance`, 57.9 (34.9 → 92.9) |
| Spatial-Agent | **13.6 pts** | `poi_farthest_of_three`, 30.8 (43.6 → 74.4) |

Same rows, same three-pass averaging, same endpoint: ReAct's family score moves nearly twice as far
between draws as Spatial-Agent's. Ten of the twelve measured families swing more for ReAct than for
Spatial-Agent. So "quote a family number across draws or as one draw's" is not symmetric advice —
it binds hardest on the baseline, and a single-draw family comparison flatters or damns ReAct by up
to 30 points for reasons that have nothing to do with the architecture.

### The trip-length hypothesis is retired

The v7b entry proposed that ReAct's lift on `trip_total_distance` rises with median trip length;
the v7c entry downgraded it to "positive but not monotonic." At five draws it is nothing:

| draw | median total | floor | ReAct | lift |
| --- | --- | --- | --- | --- |
| v7 | 22.2 km | 23.8 | 34.9 | 11.1 |
| **v7d** | **25.1 km** | **26.2** | **90.5** | **64.3** |
| v7b | 26.5 km | 21.4 | 52.4 | 31.0 |
| v7c | 30.5 km | 35.7 | 63.5 | 27.8 |
| v7a | 30.9 km | 40.5 | 92.9 | 52.4 |

v7d is the second *shortest* draw and carries the *largest* lift of the five. Across the five
points the correlation between median length and lift is r = 0.34, which at n = 5 is indistinguishable
from none. The hypothesis was three points in a row and should not have been written as a mechanism.
What survives is the reason it was noticed at all — this family's per-draw score is unstable for
ReAct by 58 points, which the table above now explains as the general case rather than a special one.

### The audit failed again, and the correction: this is the code, not the city

    nearby_kth_nearest: k varies across [2, 3] but 20 of 24 rows ask for k=2

The v7c entry called this "draw variance in what Seoul offers, not a generator bug, and it does not
warrant another scan-limit change." **The first half of that is wrong.** Five draws for five:

| draw | k distribution, `nearby_kth_nearest` | `nearby_subtype_kth` |
| --- | --- | --- |
| v7 | {2: 19, 3: 3, 4: 2} | {2: 15, 3: 15} |
| v7a | {2: 16, 3: 6, 4: 2} | {2: 15, 3: 15} |
| v7b | {2: 16, 3: 5, 4: 3} | {2: 15, 3: 15} |
| v7c | {2: 20, 3: 3, 4: 1} | {2: 15, 3: 15} |
| v7d | {2: 20, 3: 4, 4: 0} | {2: 14, 3: 14} |

`nearby_kth_nearest` has never once balanced; `nearby_subtype_kth`, which draws k the same way in
the same builder, has balanced perfectly every time. Five for five in one family and zero for five
in the other is not variance, and the difference between them is measurable in the datasets
themselves — the consecutive rank gaps each family's gold evidence records:

| draw | `kth_nearest` median rank gap | under 90 m | `subtype_kth` median | under 90 m |
| --- | --- | --- | --- | --- |
| v7 | 70 m | 50% | 139 m | 25% |
| v7a | 84 m | 50% | 154 m | 14% |
| v7b | 96 m | 45% | 148 m | 20% |
| v7c | 77 m | 50% | 137 m | 18% |
| v7d | 83 m | 51% | 131 m | 21% |

Stable to a few metres across five independent draws. `nearby_kth_nearest` anchors its ordinal in
`CE7`/`BK9`/`PM9`/`CS2` — cafés, banks, pharmacies, convenience stores, the four densest chain
categories in Seoul — where about half of all consecutive rank gaps fall under the 90 m
`ORDINAL_MARGIN_M`. k = 3 needs three consecutive gaps to clear it and k = 4 needs four, so even
treating the gaps as independent that is ~12% and ~6% of anchors, and they are not independent: a
dense neighbourhood is dense at every rank. `nearby_subtype_kth` searches a *subtype* keyword
(내과, 정형외과) where only ~20% of gaps are under the margin, k = 3 clears at ~51%, and the
balancing code that is identical in both families has candidates to work with.

So the cause is the family's category pool, and `ba92d9c`'s scan-limit fix could not have addressed
it — the scan ceiling is `max(24, count * 3)` = 72 *successful* rows at this quota, and among 72
anchors that produced a row v7d found four that supported k = 3 and none that supported k = 4,
which is exactly what a 12%/6% rate predicts. Two levers remain, and they trade off:

- **Sparser anchor categories** (what `nearby_subtype_kth` already does): balances k without extra
  quota, but changes what the family asks — "the k-th nearest hospital" is not "the k-th nearest
  café", and the family's benchmarks of record were all drawn on the dense pool.
- **Scan for the scarce value rather than for a multiple of `count`**: keeps the family identical,
  but reaching eight k = 4 rows at a ~6% rate needs on the order of 150–250 successful rows against
  today's 72, i.e. roughly 2,000 extra Kakao calls in one family of one build.

**What the skew does and does not cost.** The docstring's guessability argument — ranking the four
options alone answers a k-th question 60% of the time at k = 2, 30% at k = 3, 10% at k = 4 — assumes
a guesser who can order the options by distance, which a closed-book model cannot. Measured, the
family's floor on v7d is 10/48 = **20.8%, below four-way chance**, and 2/24 on v7. The skew is a
real generator defect, five for five, but it has not made the family guessable in practice: the
overall 47.6/80.4 and every other family stand, and what does not stand is a `nearby_kth_nearest`
family number for this draw, which measured mostly "second nearest."

### One crash the run surfaced

Two of 843 Spatial-Agent runs died on `AttributeError: 'dict' object has no attribute 'split'`.
`kmapeval_196` composed `batch_geocode(place_names=[{"cost": "$cost_0", "index": 0}, ...])` — using
the geocoder to build a list of records rather than to resolve names. `ToolRegistry` refuses that
cleanly with a pydantic validation error the repair round can read, but grounding runs first, and
`_is_shortened_name` called `.split()` on the dict, so the question was thrown away with a Python
internals leak instead of being refused. Same class as the bare-`$ref` crash at `1cb6bdc`: a
planner mistake surfacing as a harness crash. `_is_shortened_name` is now total — whatever is not a
name is not a shortened one — and the operator does the refusing.
`tests/test_spatial_nonstring_place_names.py` pins it, including that the fix is causal (the
grounding test fails on the unpatched predicate) and that the operator's own refusal still fires.

**v7d is spent by that fix**, as v7a and v7b were before it: 47.6/80.4 is what `c7d49cb` scored,
not a number to hold the next draw against.

## The trip families: three defects in one operator

Spatial-Agent's weakest class across all five 300-row draws is `trip` — 65.2%, 65.4% and 76.5%
pooled over 1,056 runs for `trip_feasible_count_five`, `trip_optimal_order` and
`trip_total_distance`, against 90+ on `routing` and `distance`. The cause is not the step budget
and not the planner. It is `tsp_tw`, which answered a different question than the one it was asked,
three different ways.

### It is not `MAX_REASONING_STEPS`

The obvious suspect was the step budget, since `MAX_REASONING_STEPS` bounds the nodes the planner
may author. Over 1,363 Spatial-Agent runs on v7d it is not:

| family | runs | hit the 15-node cap | p90 graph size |
| --- | --- | --- | --- |
| `trip_feasible_count_five` | 84 | **0** | 6 |
| `trip_total_distance` | 105 | **0** | 7 |
| `trip_optimal_order` | 120 | **5** | 8 |
| every other family | 1,054 | **0** | ≤10 |

Five refusals in 1,363 runs, all in one family, and zero `iteration_limit` in every pass. The five
graphs that hit it authored 18, 23, 25, 25 and 40 operators — a planner enumerating all six
permutations by hand instead of calling `tsp_tw` once. Raising the cap would let that brute-force
plan through; it would not have fixed the objective, and the three defects below all show up in
graphs of six to eight nodes. The budget also stays at 15 for the reasons already recorded above:
it is langchain's default and therefore the reference baseline's, it is *one* budget that moves
both architectures at once, and the v6 ablation measured it moving Spatial-Agent 69.0 → 74.0.

### Defect 1: a stated itinerary was answered by reordering it

`trip_feasible_count_five` lists its stops 적힌 순서대로 and asks how many fit a time budget.
`tsp_tw` permutes, and when no full tour fits it fell back to a nearest-first greedy walk that
reaches stops the stated order cannot. On v7d, **15 of Spatial-Agent's 26 misses in that family
were exactly the count you get by ignoring travel time**, and ReAct made the same mistake 6 times
in 7. Replayed offline on real cached Kakao legs over 155 recorded itineraries, the permuting path
overcounts **24 of them by exactly +1**; the stated-order walk gets 153, and both remaining misses
are legs whose cached duration has drifted since the set was built. Against the durations the
datasets themselves recorded it is 161 of 161.

`fixed_order` walks the nodes as listed. Grounding binds it from the question, as it already binds
the stays and the budget — the order is a question literal too. It fires on every
`trip_feasible_count_five`, `trip_total_distance` and `multisegment_total` row in every dataset
here and on no `trip_optimal_order` row.

Two further repairs came with it: the greedy fallback no longer appends an `end_index` it cannot
reach (it was appended unconditionally, so a tour that had spent its budget came back naming a stop
it never gets to and a `total_cost` above the `time_budget` it was handed), and every branch now
reports `visited_count` and which objective it optimised for.

### Defect 2: every tour was ranked by seconds, and left open

"자동차 총 주행거리가 가장 짧은 방문 순서" is a question about metres, and the four orders it
chooses between sit a median **2.07% of the tour** apart — so seconds are not an approximation of
metres, they select a different option. The tour was also left open where the question closes it
("…둘러본 뒤 다시 제일모텔로 돌아옵니다"), and the cheapest way out is not the cheapest loop.
Over 114 `trip_optimal_order` rows with complete cached legs:

| | picks the gold option |
| --- | --- |
| distance + closed tour | **93/114 = 81.6%** ← what the question asks |
| distance, open | 53/114 = 46.5% |
| duration + closed | 41/114 = 36.0% |
| duration, open | 35/114 = 30.7% ← what the operator did |

The old behaviour was the worst of the four. The 21 rows the right reading still misses are legs
whose cached length has drifted, not a disagreement about the objective. `metric` and
`return_to_start` are both bound from the question by grounding. Seconds beside a matrix of metres
are refused rather than allowed to cancel out — a stay added to a distance is an invented
measurement — and a distance question states its stays as decoys, so grounding drops them.

### Defect 3: the metric argument was a no-op on the only shape that matters

`distance_matrix` returns `{routes: [...], matrix: [...]}` — routes that carry both numbers, and a
pre-built matrix that is always durations — and `_matrix_argument` preferred the pre-built one. So
`tsp_tw(metric="distance")` over `$legs`, which is what every trip graph passes, returned seconds
and reported them as metres. Caught by asserting the value rather than the call: 600 came back
where 9,000 was asked for. The routes are now re-read in the metric requested, and a matrix with no
routes behind it is refused for a non-default metric rather than reinterpreted.

### Measured: three passes a side on v7d

Spatial-Agent only; ReAct's `reference` surface has no `tsp_tw` and is untouched. All at
`--concurrency 32` except the middle column, which is noted because run conditions are not poolable.

| | `c7d49cb` (before) | `01f7f64` (defect 1) | `e114f4b` (all three) |
| --- | --- | --- | --- |
| concurrency | 32 | **4** | 32 |
| passes | 81.5, 80.1, 79.7 | 79.0, 79.0, 78.3 | 83.3, 79.0, 82.2 |
| overall | 80.4 | 78.8 | **81.5** |
| `trip` class | 63.6 | 68.7 | **80.8** |
| `trip_optimal_order` | 52.8 | 52.8 | **86.1** |
| `trip_feasible_count_five` | 58.7 | 88.9 | 73.0 |
| `trip_total_distance` | 81.0 | 66.7 | 82.5 |

Per pass, `trip_optimal_order` goes 11, 13, 14 of 24 to **22, 19, 21** — the two ranges do not
overlap, and it is the largest single-family move any change here has produced. `trip_total_distance`
goes 19, 14, 18 to 17, 17, 18: unchanged in the mean and tighter in spread.

**One regression, found and repaired inside the same change.** Defect 1's prompt edit told the
planner to set `fixed_order` on "A → B → C 순서로" questions, which is exactly how
`trip_total_distance` is phrased. That pushed the family onto `tsp_tw` — 8% of its runs to 27% —
where `total_cost` was seconds and the options are km, and within that run the rows that reached
`tsp_tw` scored 55% against 71% for those that did not. The family fell 81.0 → 66.7 and came back
to 82.5 once `metric` worked. This is why defect 3 mattered: without it the metric argument was
present, documented, bound by grounding, and doing nothing.

**What did not move, and one thing that did without a mechanism.** `routing` (96.0 → 95.5) and
`nearby` (79.5 → 76.2) are flat within the endpoint's spread. `poi_farthest_of_three` fell 74.4 →
46.2 across the three revisions (11, 10, 8 → 9, 5, 7 → 6, 5, 7 of 13), and **no run of that family
touched `tsp_tw` at all** — there is no code path from these changes to it. It is also the family
with the largest Spatial-Agent cross-draw range already on record (30.8 points over five draws), on
13 rows where one row is 7.7 points. Recorded as unexplained rather than attributed.

**Every dataset here is spent by this.** These changes strengthen Spatial-Agent's operator surface,
so no accuracy measured before `e114f4b` is comparable to one after it, and v7d's 80.4 is what
`c7d49cb` scored. The next held-out number needs a fresh draw.

## Removing family-specific recovery and answer selection

An overfitting audit separated two kinds of changes that benchmark-driven development had mixed
together:

- Operator corrections remain when they follow from the question or the operator contract without
  reference to a gold answer. A stated itinerary must not be reordered, a distance objective must
  read metres, a closed tour must include its return leg, and a stop outside a time budget was not
  visited. These are semantic and unit invariants, not family knobs.
- Recovery or answer selection is removed when only selected intents receive it or when its
  threshold comes from the spacing of benchmark options.

Three code changes implement that boundary:

1. `_computed_clock_option` is deleted. The evaluator LLM still receives the executed operators'
   answer-bearing evidence and is instructed to prefer computed evidence, but `_select_option`
   now reconciles only the answer text and index that generation returned. Clock questions no
   longer have a second, harness-owned answer channel or the option-relative `nearest_gap * 2`
   threshold.
2. The handwritten post-repair solver for `distance`, `nearby`, `direction` and `radius` is
   deleted. After the normal repair round, every intent gets the same last attempts: execute the
   repaired graph with this port's type/role restrictions relaxed, then the original graph under
   the same rules only if the repair is structurally invalid. Structural graph rules remain in
   force throughout. A validation miss is no longer forgiven according to its family.
3. Question literals are parsed by semantic components rather than only by the generator's full
   sentence shapes. Direction constraints now preserve all eight compass sectors; anchors accept
   ordinary `에서`, `으로부터`, `기준` and proximity constructions; straight-line pairs
   accept `사이`, `간` and `에서 …까지`; trip durations accept hours, minutes and mixed
   forms such as `1시간 30분`. Metamorphic tests cover variants not emitted by the datasets.

The ordinal-nearby operator and template remain. "The k-th nearest place in the neighbourhood"
and "the nearest among these options" are mathematically different queries, and retrieval before
ranking follows from that distinction without a gold answer. It remains a deliberate capability
addition beyond upstream's ten Appendix E macro families and must be reported as such.

This revision changes Spatial-Agent behaviour, so every existing dataset is spent for it. No
accuracy is claimed here: a result for this code requires a fresh seed and an untouched holdout,
and neither dataset construction nor a live benchmark was run as part of this change.

## Structural reference validation and bounded evaluation evidence

The first three-pass run after removing the family-specific fallback exposed two general failures
on `seoul_kmapeval_v7_mcq_100.jsonl`. They are recorded as defects found on a spent set, not as a
new accuracy claim.

First, `batch_geocode` has a statically known top-level shape: it returns one ordered record for
each `place_names` entry. Plans nevertheless referenced `$places.3` after supplying three names,
or projected `$places.anchor_place` directly from the list. The executor's deliberately lenient
path handling degraded those references to the closest resolvable object, so downstream distance
operators received the whole list or treated the first destination as the origin. When a plan
supplies an explicit anchor and its own references prove that it expects one additional leading
record, grounding prepends that anchor. Otherwise validation rejects only what is knowable before
execution: a non-numeric first projection from a `batch_geocode` list, or a numeric index outside
its literal input length. Unknown fields below a valid record remain lenient, and this structural
data-availability rule also applies on the final `strict_types=False` attempt.

Second, the report trace remains complete, but the final generation prompt no longer duplicates
it verbatim. A 45-place neighbourhood was repeated in the retrieval result, ranking arguments,
ranking result, later arguments and concept bindings; six of 300 Spatial-Agent attempts crossed
the model's 65,536-token context window at the final call. The generation stage now receives a
deterministic projection that:

- retains every operator's identity, role, status, error, scalar arguments and answer-bearing
  output fields;
- bounds repeated collections and records the exact omitted count as `_omitted_items`;
- leaves the full execution trace used by logs, metrics and reports unchanged.

Neither repair selects an answer, recognizes a benchmark family, or changes an operator result.
Regression tests use synthetic output shapes and 45 fabricated places rather than benchmark
answers. Because `src/` changed after the run, every existing accuracy is again training-set
evidence for this revision; the next result must use a fresh untouched draw.

### Preserve a structural repair on the lenient attempt

The three-pass run at `b508631` verified both changes above: context overflows fell from six to
zero, mean prompt tokens fell 28.6%, and `poi_farthest_of_three` rose from 14/30 to 26/30 repeated
answers as the origin was included in the distance calculations. It also exposed an ordering bug
in the architecture-wide fallback. One graph used impossible named fields on a `batch_geocode`
list; the repair rewrote them as valid indexed references, then failed only this port's declared
type check. The old fallback discarded that structurally valid repair and applied lenient
validation to the original graph, reproducing the structural error it had just repaired.

The fallback order is now `repaired strict -> repaired lenient -> original lenient`. Trace entries
record `plan_source` for every accepted or rejected lenient attempt. Structural rules are never
relaxed, and the original graph remains available when the repair itself is malformed. This is one
policy for every intent, not the removed family-specific recovery path.

The follow-up was exercised before it was committed. Consequently the eight reports below carry
`code_revision=b5086310a651`, which identifies their checked-out commit but not the modified source
loaded by the process. The new `plan_source` trace field, which does not exist at `b508631`, pins the
executed behaviour to this follow-up. This provenance limitation is why a commit hash alone must
not be read as proof of a clean worktree.

- `test_20260825T135117Z.json` through `test_20260825T141310Z.json` are three paired passes over
  the spent 100-row v7 set at concurrency 32, temperature 0 and 15 steps. ReAct scored
  50/45/52 (147/300, 49.0%) and Spatial-Agent 83/80/85 (248/300, 82.7%), a 33.7-point aggregate
  gap. The preceding `b508631` Spatial-Agent passes scored 77/86/82 (245/300, 81.7%); the
  one-point change is inside the endpoint's measured spread and is not attributed to this fix.
  Spatial-Agent retained zero context overflows. Its six reasoning failures became two reasoning
  failures plus one unrelated 61,315-completion-token truncation, while total tokens were
  effectively unchanged (5,236,731 versus 5,237,697 over the three passes).
- `test_20260825T142821Z.json` and `test_20260825T143739Z.json` are one paired pass over the spent
  283-row v7 set under the same conditions. ReAct scored 141/283 (49.8%) and Spatial-Agent
  228/283 (80.6%). One pass is not a result, and this draw's known `nearby_kth_nearest` audit
  failure also rules out quoting that family.

The trace is the direct regression check. Repaired-lenient plans were accepted and executed 10
times in the 100-row passes (7 correct) and 9 times in the 283-row pass (5 correct); none became an
`agent_reasoning_failure`. The remaining five reasoning failures across these Spatial-Agent runs
were all nearby plans where both the repair and the original projected a named field such as
`place_ids` or `candidates` directly from a `batch_geocode` list. Validation correctly rejected
those structurally impossible references. They are unresolved planner/repair misses, not evidence
that the repaired-first fallback was skipped. These are training-set diagnostics only: all named
datasets were already spent, so the revision still needs a fresh untouched draw for a holdout
accuracy.

### Validate complete argument contracts and report partial execution failures

The post-`b508631` diagnostics above also exposed failures that final accuracy and
`failure_type` did not show. Across the three 100-row Spatial-Agent passes and the one 283-row
pass, 137 of 583 question attempts contained at least one failed execution step. Some were
expected missing-evidence probes on `unanswerable_*`, and some later steps merely cascaded from an
earlier failure, but 119 of 541 answerable attempts still contained one. A question could then be
answered from another branch, so these errors stayed only in per-query logs even when the report
row said the answer was correct.

Three architecture-wide changes close that observability and validation gap:

1. `OperatorContract` now declares every canonical runtime argument, not only the required ones.
   Graph validation rejects unsupported names before execution, so values such as
   `steps_analysis(step_index=...)` and `extract_distance(route_index=...)` enter the ordinary
   repair round instead of raising `TypeError` inside an otherwise "valid" graph. A regression
   test compares the contracts with every tool schema and deterministic operator signature so a
   newly added implementation argument cannot silently make the validator stricter than the
   executor. Exact aliases already accepted by the implementation remain accepted by validation.
2. A bare string becomes an internal reference only when it is in a data-bearing argument,
   exactly names a real node, and exactly names one of that step's declared dependencies. Thus
   `origins: "locations"`, `nodes: "locations"` and `distance_matrix: "legs"` become
   `$locations`/`$legs`, while literals such as `metric: "distance"`, route labels and question
   option text cannot be rewritten. `tsp_tw` also canonicalizes exact measure/unit synonyms:
   `time`, `travel_time`, `duration_s` and `seconds` mean `duration`; `distance_m`, `metres` and
   `meters` mean `distance`. These mappings do not inspect a question, option or gold answer.
3. Both agents now return a compact `execution_errors` list even when they recover an answer.
   Every report row carries that list and `execution_error_count`; run statistics carry
   `execution_error_question_count`, `execution_error_question_ids`,
   `execution_error_step_count` and counts by operator. The terminal summary and per-question log
   print the same totals. `failure_type` remains reserved for the question-level outcome, so an
   intermediate error and a recovered correct answer are both represented rather than one
   overwriting the other.

No live benchmark was run for these repairs. They change validation and execution observability,
so all existing datasets are spent again; their logs diagnose the defects but cannot measure this
revision as a holdout.

#### Post-change benchmark diagnostics

After the implementation above, but before it was committed, four single-pass reports were run
against the already-spent v7 datasets. All used concurrency 32, temperature 0, 15 reasoning steps
and the reference ReAct surface; none recorded a truncation, context overflow or iteration limit.
Their `code_revision=6b2deaf72574` identifies the checked-out parent rather than the modified
source loaded by the process. The presence of the new `execution_error_*` report fields and the
contract-validation traces pins the executed behaviour to this change, but the dirty-worktree
provenance means these are diagnostics, not reproducible accuracy results.

| report | agent | correct | question failures | questions / steps with execution errors |
|---|---:|---:|---:|---:|
| `test_20260825T153356Z.json` (v7 100) | ReAct | 49/100 | 0 | 27 / 31 |
| `test_20260825T153718Z.json` (v7 100) | Spatial-Agent | 84/100 | 0 | 14 / 29 |
| `test_20260825T154450Z.json` (v7 283) | ReAct | 134/283 | 0 | 87 / 102 |
| `test_20260825T155421Z.json` (v7 283) | Spatial-Agent | 238/283 | 7 | 44 / 77 |

The 100-row scores sit inside the immediately preceding three-pass ranges (ReAct 45--52,
Spatial-Agent 80--85). The 283-row Spatial-Agent score is 84.1% versus 80.6% in the preceding
single pass, but one pass on a set already used to change the code cannot attribute that movement.
The 283-row draw also retains its known `nearby_kth_nearest` audit failure. None of these numbers
is a new holdout result.

The traces directly exercise all three repairs:

- Ten plans supplied unsupported argument names: two in the 100-row run and eight in the 283-row
  run. Every one was rejected before execution. Six repairs then produced executable graphs (four
  answered correctly); four repair responses remained structurally or contractually invalid and
  ended as `agent_reasoning_failure`. No execution trace contains the old `unexpected keyword
  argument` failure.
- Twelve plans passed a bare `locations` or `all_locations` dependency into both sides of
  `distance_matrix` (three in the 100-row run and nine in the 283-row run). All twelve matrices
  received resolved place lists and completed. Eight `tsp_tw` plans used `metric="time"` (three
  and five respectively); all eight completed with the canonical `duration_s` result. The former
  bare-reference validation errors and `tsp_tw metric must be one of ...` errors are absent.
- Counting failed steps in the raw execution traces gives exactly 14 questions / 29 steps for the
  100-row Spatial-Agent run and 44 / 77 for the 283-row run, matching both report summaries.
  Twelve of those 14 and 38 of those 44 questions still answered correctly, confirming that the
  new fields preserve partial failures that the question-level `failure_type` intentionally does
  not classify as terminal failures.

The remaining Spatial-Agent execution errors are value-shape or plan-semantics problems: empty or
out-of-range selections, list/scalar confusion in arithmetic, malformed `batch_place_details`
inputs, and two 16-place matrices exceeding the declared limit of 15 (one in each Spatial-Agent
run). The seven terminal failures in the 283-row run split into three impossible named-field
projections from a `batch_geocode` list and four unsupported-argument plans whose one repair
response was still invalid. These are visible planner/repair misses rather than validation
bypasses. ReAct's partial errors are separately explained by the provider boundary: all 31 in the
100-row run, and 100 of 102 in the 283-row run, are explicit `UnsupportedTravelModeError` results
because Kakao routing supports driving only; the other two are explicit `RouteNotFoundError`
results for endpoints within five metres.

#### Follow-up: value contracts without broader evidence

A second review separated repeatable contract defects in those partial errors from planner misses
that have no safe deterministic interpretation. Three general repairs follow in that order:

1. A whole `batch_geocode` output is already a collection. Repeating the same whole-node reference
   inside one argument list is now a structural data-availability error, so four copies of a
   four-place node cannot flatten into sixteen endpoints. A single whole-list wrapper, distinct
   node references and indexed record references remain valid. The validator rejects the
   ambiguous cardinality rather than guessing whether the planner wanted one collection or four
   copies of it.
2. `extract_distance` and `extract_duration` now return one amount for a one-route wrapper and a
   collection only for several routes. `sum_amounts` flattens list nesting while preserving each
   measurement record, and `difference` totals either operand when it is a measurement
   collection. Distance and duration collections remain distinct; mixing the two fails instead of
   acquiring whichever unit happened to be inspected first.
3. Validation now checks values that are statically knowable before execution: literal collection
   limits, pair/route object shapes, ranking keys as literal field names, `tsp_tw` node/clock list
   lengths and indexes, and mutually incompatible tour arguments. Exact metric synonyms live in
   one contract module shared by validation and execution. Grounding binds any stated clock
   constraint to `metric="duration"`, removes a fixed end from a closed tour, and continues to
   remove clock arguments from a distance objective. `batch_place_details` may read the existing
   `place_id` from a normalized Place or geocode row, but it never searches a name or invents an
   id. A mixed list containing an unresolved row fails rather than silently fetching only its
   resolved subset. Values behind unresolved graph references are deliberately not pre-judged.

The cross-argument review also covered all `return_to_start` execution branches. A stated-order
closed tour now includes the closing leg without counting the start as another visit, and both
stated-order and greedy partial tours reserve enough budget to return. Open stated-order tours are
unchanged. `fixed_order` plus `end_index` is rejected rather than accepting an argument the
operator would ignore, while question grounding removes that redundant endpoint before
validation.

The regression suite pins the negative space of each change: repeated whole-list references fail
while a single wrapper and indexed references pass; nested distance amounts compose while mixed
distance/duration amounts fail; invalid literal `tsp_tw` combinations fail before execution while
an exact `time` alias canonicalizes to `duration`; and missing place evidence remains an error.
No live benchmark had been run for this follow-up at this point. The reports above diagnose it but
remain scores for the pre-follow-up dirty worktree, and every existing dataset remains spent.

#### Dirty-worktree verification of the value contracts

Four more single passes were run with the follow-up above loaded from the dirty worktree. As with
the preceding diagnostics, `code_revision=a18719eb2413` records only the checked-out parent and
does not identify the uncommitted source that the processes imported. All four used concurrency
32, temperature 0 and 15 reasoning steps; ReAct used the reference surface. No run recorded an
LLM truncation or context overflow. The 100-row ReAct run reached the iteration limit twice; the
other three did not.

| report | agent | correct | question failures | questions / steps with execution errors |
|---|---:|---:|---:|---:|
| `test_20260825T224021Z.json` (v7 100) | ReAct | 50/100 | 2 | 30 / 35 |
| `test_20260825T224415Z.json` (v7 100) | Spatial-Agent | 83/100 | 1 | 16 / 36 |
| `test_20260825T215842Z.json` (v7 283) | ReAct | 140/283 | 0 | 88 / 95 |
| `test_20260825T220751Z.json` (v7 283) | Spatial-Agent | 227/283 | 6 | 31 / 62 |

The scores do not identify a treatment effect. Against the immediately preceding dirty-worktree
single passes, ReAct moved 49% to 50% on 100 rows and 47.3% to 49.5% on 283; Spatial-Agent moved
84% to 83% and 84.1% to 80.2%. Those directions disagree, both sets were already spent, and one
pass cannot separate a code effect from the endpoint's measured run-to-run variation. The useful
evidence is narrower and lives in the execution traces:

- The preceding two Spatial-Agent reports contained ten `float(list)` arithmetic failures, two
  16-endpoint validation failures produced from four-place nodes, four attempts to combine a
  distance objective with clock arguments, and six `batch_place_details` list/row validation
  failures. None of those exact failure classes occurs in either new Spatial-Agent report.
- The old 16-endpoint cases now pass one four-place collection to each side of the matrix. Both
  produce a 4 by 4 matrix and answer correctly. The four old distance-plus-clock TSP cases also
  execute without an error and answer correctly. These are contract-level observations, not a
  claim that every affected answer must improve on another stochastic pass.
- Spatial-Agent execution-error totals fell from 44 questions / 77 steps to 31 / 62 on the
  283-row pass, but rose from 14 / 29 to 16 / 36 on the 100-row pass. ReAct's totals also moved in
  both directions. The totals therefore remain diagnostics of the sampled plans rather than a
  score for the repair.

The traces exposed one final instance of the same generic amount-shape contract. Five plans used
`sum_amounts([$leg_1, $leg_2])`, where each resolved item was a `distance_matrix` result wrapping
one route. Top-level wrappers and nested lists were normalized, but wrappers *inside* a list were
not, so those five plans failed while the generation call could still read the routes and answer
all five correctly. Amount collection normalization now recursively opens collection wrappers at
every list level while preserving route measurement records and their units. A regression test
covers both `sum_amounts` and `difference` over that exact structural shape. This last correction
postdates the four reports and has unit-test, not live-benchmark, evidence.

The seven new terminal Spatial-Agent failures are not silent contract bypasses: six are rejected
projections of named fields such as `place_ids`, `places` or `candidates` directly from a
`batch_geocode` list, and one retains an unsupported `select_by_index(sort_by=...)` argument. In
every case the single repair response remained invalid. Other partial errors are explicit missing
evidence, out-of-range selections, unsupported subjective fields, malformed planner expressions
or downstream dependency failures. No question-specific fallback was added. These runs spend the
datasets again and do not create a new holdout result.

#### Three passes at `98fb7d0`, and what a single pass could not have told us

The value contracts above were committed as `98fb7d0` and then measured properly: three passes a
side on `dataset/seoul_kmapeval_v7_mcq_300.jsonl` (283 rows), all at concurrency 32, temperature
0, `MAX_REASONING_STEPS=15`, ReAct on the reference surface with neither parallel tool calls nor a
forced final answer. No truncation and no context overflow in any pass.

| agent | pass 1 | pass 2 | pass 3 | mean | spread |
|---|---:|---:|---:|---:|---:|
| ReAct | 48.4 | 45.9 | 46.6 | 47.0 | 2.5 |
| Spatial-Agent | 82.3 | 81.3 | 84.8 | 82.8 | 3.5 |

Gap 35.8 against a floor of 28.8. A fourth pass a side at the same revision, run before these
three, agrees: ReAct 47.0 and Spatial-Agent 83.8.

**These numbers are a level, not a lift.** Every revision before `98fb7d0` in this stack has one
pass, and those single passes read 80.6 (`b508631`), 84.1 (`6b2deaf`) and 80.2 (`a18719e` plus a
dirty worktree) for Spatial-Agent. The pre-fix `6b2deaf` pass sits *above* the post-fix three-pass
mean. Three passes pin where `98fb7d0` sits; they cannot attribute the distance from any of those
points to the code, and the earlier sections' single-pass comparisons should be read the same way.
Spatial-Agent's own three-pass spread here is 3.5 points, nearly double the 1.8 the same dataset
showed at `6bae55c`, so even a three-pass mean carries about ±2.

What the three passes *can* settle is whether the committed repairs held, because a failure class
either appears in the traces or it does not. Counting execution errors across the three
Spatial-Agent passes (849 questions, 176 error steps over 98 questions):

| class the commit repaired | `6b2deaf` (1 pass) | `98fb7d0` (3 passes) |
|---|---:|---:|
| `float(list)` in amount arithmetic | 7 | **0** |
| `distance_matrix` over the 15-endpoint limit | 1 | **0** |
| distance objective combined with clock arguments | 4 | **0** |
| `batch_place_details` list/row shape | 4 | **0** |
| distance and duration amounts mixed | 0 | **0** |

Two `float()` TypeErrors do remain, in `timezone` and `calculate_proportion` — the same
unguarded-input shape in two operators the commit did not touch, not a return of the amount
arithmetic it did. Three `batch_place_details` errors remain and all three are the contract
working: an empty id list, a failed dependency, and a geocode row whose `place` is `{}`, which is
refused rather than silently fetched as a shorter list. The new static checks fired on 4 of 383
questions in the single-pass runs and never became a terminal failure there.

##### Two defects the three passes exposed

*The value checks were ending questions on the lenient pass.* Four of the eighteen terminal
Spatial-Agent failures across the three passes were `98fb7d0`'s own new checks: `kmapeval_181`
twice for a list where `steps_analysis` reads one route, `kmapeval_189` for the same, and
`kmapeval_255` for five service times against six nodes. Each check is correct about the step —
`steps_analysis` calls `route.get`, and a list raises `AttributeError` — but it was enforced on
the lenient pass too, which is the last thing tried before a question is given up on. The executor
already records a step that raises and carries on: an `open_at_time` `AttributeError` in the same
set left its question answered correctly. So a one-argument miss was trading a partial answer for
none. These checks are this port's own, exactly like declared output-type compatibility and
functional-role ordering, and they now step aside on `strict_types=False` alongside them. The
formal constraints do not: data availability still refuses leniently, and its message is pinned
both ways.

*A trip stop the planner cut short.* `kmapeval_211` failed in every run of five revisions with
`tsp_tw distance_matrix must be square and match nodes`, which named neither the place nor the
problem. The question writes `백련산꿈마을숲정이를`; the planner segmented the particle wrongly and
geocoded `백련산꿈마을숲정`, Kakao found nothing, and the loss surfaced three nodes later as a
matrix that would not square. The short form is still a substring of the question, so the fallback
that repairs a mis-*typed* name left it alone — only a list of the names the question states can
separate a truncation from a legitimate span. A trip's stops are stated exactly as its anchor is,
and `_extract_trip_schedule` already reads each one to bind its stay, so its keys now feed the
same list the anchor does. The negative space is pinned too: a name resembling no stated stop is
left as the planner wrote it.

The refusal message was wrong independently of the grounding, and is fixed separately.
`build_duration_matrix` already reports `missing_legs`, and `tsp_tw` was discarding it; the
refusal now names the absent legs, or the endpoint count it was given against the node count. An
earlier reading of this row — that the repair round had produced an incomplete matrix — was
mistaken: the repair round is not involved, the plan validated on the first attempt, and the
matrix was incomplete because one endpoint never resolved.

Both changes postdate the three passes, so 47.0/82.8 is what `98fb7d0` scored and every dataset in
`dataset/` is spent again. Measured over three more passes with both of them in, Spatial-Agent
scored 80.9 / 84.1 / 85.2, mean **83.4** against `98fb7d0`'s 82.8 — inside a spread of 4.2, so the
level is unchanged and only the failure accounting moved: value-check terminal failures went 3 to
0, and terminal failures overall 18 to 14 across three passes.

#### The empty-evidence hole: a measured negative result

On 17 of 849 questions the generation stage picked a candidate while its own reason said the
evidence was empty — an `[]` retrieval answered as "한 곳" because zero was not among the
candidates, or the largest number picked after every distance step failed. Fifteen of the
seventeen were already wrong. The invariants forbid choosing the least-bad match, so the stage was
given somewhere to say so: a `predicted_answer: null` plus an `insufficient_evidence` flag, its own
failure type so the refusal would not read as an unparseable answer, and a prompt that spelled out
the difference from a question the *map* cannot answer.

Three passes killed it.

| | baseline | with the decline |
|---|---:|---:|
| passes | 80.9 / 84.1 / 85.2 | 79.9 / 74.9 / 78.1 |
| mean | **83.4** | **77.6** |

Two independent reasons to revert, and the second matters more than the first.

*It cost 5.8 points.* The flag fired 20–22 times a pass where about 6 deserved it, and 39 of the 63
declines were on questions the baseline answered correctly. The reasons say why: `kmapeval_223`
declined after computing "17.827 km" with all six steps green, `kmapeval_138` after computing the
detour difference, `kmapeval_194` after counting three left turns. What the model calls
insufficient evidence is *"my computed value is not among the candidates"* — the same reflex as
"zero is not an option", generalized. No wording fixes that; the judgement was put in the wrong
place.

*It was family-skewed, and skewed against exactly the families it was written to protect.* Per
family, over three passes a side:

| family | baseline | with the decline | delta |
|---|---:|---:|---:|
| `unanswerable_opening_hours` | 66.7 | 16.7 | **−50.0** |
| `unanswerable_review_count` | 66.7 | 16.7 | **−50.0** |
| `unanswerable_price_level` | 66.7 | 33.3 | **−33.3** |
| `nearby_within_radius_count` | 63.9 | 47.2 | −16.7 |
| `routing_detour_cost` | 91.7 | 76.4 | −15.3 |
| `trip_optimal_order` | 81.9 | 68.1 | −13.9 |

The three families whose gold answer *is* "주어진 지도 정보로는 알 수 없음" lost half their rows to a
mechanism whose prompt explicitly told the model those rows were not it.

The obvious repair — honour the decline only when the execution log corroborates it, say when every
`measure`-role step errored or came back empty — is disqualified on the same grounds. Those
families' measure step is a `select_max` on an absent `rating`, which *always* errors, so the
corroboration rule would license declines precisely where it must not. There is a third reason to
leave it alone: an abstain path exists in Spatial-Agent's generation stage and has no counterpart
in ReAct, so adding one changes the comparison rather than measuring either architecture.

Reverted in full. The finding is worth more than the change: the guessing is real, it is small
(about 2% of questions, three quarters of them already wrong), and it cannot be closed from inside
the generation stage without either paying six points or biasing five families.

#### Checking a change for family bias

Prompted by the same review, both surviving changes were audited for whether they favour a
question type rather than the architecture. The method is a replay: take the 846 planner graphs
the three `98fb7d0` passes actually produced, run the changed code and the unchanged code over
each, and count where the outputs differ, by family.

*The lenient-pass gating* keys on `strict_types`, a validation mode, and names no family. Its
measured footprint is the three value-check terminal failures it removes, which fell in
`routing_turn_count_via` and two `trip_*` families — mixed, and the rule cannot see a family.

*The stated-literal gathering* was caught by the audit. As first written it read the stays only
`if intent == "trip"`, which is a family condition in the code, so the gate was removed: the
grounding now gathers every literal the question states however the question is classified, the
way the anchor and the compared places already were, and the regex simply finds nothing in a
question that states no stay. Ungating is a byte-identical no-op on all 648 non-trip graphs in the
replay. The mechanism's whole measured footprint is 6 of 846 graphs — `백련산꿈마을숲정` →
`백련산꿈마을숲정이` and `서울아레나` → `서울아레나 (2027년 예정)`, two questions in 283 — and both
happen to be trip questions because that is where this draw's planner truncated a name, not
because the rule looks for one. Two rows over three passes bound what it can be worth:
`trip_feasible_count_five` and `trip_optimal_order` each have at most three flips available, so
this change can account for at most about half of the +7.9 and +5.6 those families moved, and the
rest is the per-pass noise a 21-row family carries.

One asymmetry the audit surfaced and did not fix: this pre-registry name repair belongs to GeoFlow
grounding and has no counterpart in ReAct, whose planner writes names straight into tool arguments.
The *shared* name resolution below the tools — query variants, the evidence floor, branch handling
in `src/tools/registry.py` — is equivalent, which is what the invariant requires. But a repair
layer that grows on one side only would widen the gap for a reason that is not the architecture, so
its footprint belongs in any report that quotes a trip number.

#### The `unanswerable_*` regression, half of it closed

Reading the whole v7-283 trajectory by family showed the four `unanswerable_*` families falling
while everything else rose: 82.5 as a group at `41513ae` down to 65.1, on the 21 rows those five
families hold. The cause was in this port, not in the model. `a18719e` added a reference-shape
check that refuses a named field projected off a `batch_geocode` list, and geocoding the options
and then asking for their details is the standard plan shape there — the refusal fired on 22 of
their 124 graphs against 42 of the other 1,651, and because it was classed structural it refused on
the lenient pass too and ended the question.

It should not have. `$geo.place_ids` names no field of that list, so the resolver degrades it to
the whole list, which is exactly what the legal `$geo` resolves to; and since `98fb7d0` taught
`batch_place_details` to read the id off a geocode row, the plan executes and answers. A registry
test drives that end to end. The three refusals sharing this validator were given separate
verdicts: an out-of-range record index and a repeated whole-list reference each substitute
*different* evidence with no legal spelling that produces what the planner meant, so both stay
structural; a named field is a spelling, so strict validation still refuses it for the repair round
and it steps aside leniently.

Three passes a side at concurrency 32, against the same three that preceded it:

| | before | after |
|---|---|---|
| passes | 80.9 / 84.1 / 85.2 | 82.0 / 84.5 / 85.5 |
| overall | 83.4 | 84.0 |
| terminal failures, 3 passes | 14 | **3** |
| `unanswerable_*`, 21 rows | 65.1 | 69.8 |

The overall move is inside the spread and is not the claim. The claim is the failure count: twelve
questions stopped being terminal failures, spread across eight families, and what was left is three.
`trip_optimal_order` reads −11.1 and `trip_feasible_count_five` −7.9 over the same passes, which is
pass-to-pass movement rather than this change — the per-pass numbers overlap (79/75/92 against
62/75/75), and the replay says two graphs per family were newly admitted, which cannot move eight
rows.

Half the regression is still there: 69.8 against `41513ae`'s 82.5, and it is a different defect that
this fix exposed rather than caused. `unanswerable_rating` fell to 11.1 at `98fb7d0` and has stayed
there. Its rows used to fail terminally and now run: `batch_place_details` returns records whose
`rating` is `None`, `select_max(key="rating")` refuses with "No item contains comparable key:
rating", and the generation stage — told by its own prompt that "an operator that reports an error
contributed no evidence; fall back to the surviving steps" — falls back to the retrieval and names
a café. The absence of the field across every record *is* the evidence those questions are about,
and the operator's message does not say so. That is a message-and-prompt change in the generation
stage, the same place a decline flag cost 5.8 points, so it wants its own footprint replay and its
own passes before anything is written.

## Removing the runtime intent router

Upstream Spatial-Agent's official implementation carries an intent classification stage and
per-intent prompts. The paper's chapter 3 has neither: the claim there is that decomposing a
question into scientific core concepts and functional roles *replaces* intent-style pattern
matching. So "does the port answer as well without an intent router" is a direct check on that
claim rather than a refactor, and it is worth its own entry.

This port had two intent surfaces. The concept and factorization core never had one —
`src/agent/geoflow.py` has no per-intent branch outside template retrieval, and all four prompts
(`ANALYSIS`, `GRAPH`, `REPAIR`, `EVALUATOR`) are intent-independent. The router lived in
`_ground_graph_literals`, which took `analysis["intent"]` and gated eleven branches on it.

### The label is a guess, and it misses

Every recorded run logs the Analysis stage's intent beside the question, so how often the router
routes wrong is measurable from the logs alone. Over 849 graphs from three v7-283 passes at
`643fe24`:

| family | rows | what the stage called them |
|---|---|---|
| `nearby_subtype_kth` | 90 | 69 `nearby`, **21 `poi`** |
| `nearby_kth_nearest` | 72 | 57 `nearby`, **14 `poi`**, 1 `distance` |
| `routing_detour_cost` | 72 | 10 `routing`, **53 `poi`**, 8 `distance`, 1 `trip` |
| `routing_turn_count_via` | 63 | 21 `routing`, **42 `trip`** |
| `unanswerable_*` (all five) | 63 | 41 `nearby`, 21 `poi`, 1 `type` |

A mislabel is not a failure. Grounding does not raise when it declines to bind something; it
simply hands the executor a graph with no `required_type` on its ranking, or no anchor on its
geocode, and every stage afterwards reports success.

### Replaying grounding instead of re-running the benchmark

Grounding is a pure function of the planner's graph, the Analysis output, the question and its
options, and a recorded run wrote all four to `logs/`. `data/replay_grounding.py` re-runs it
offline at any revision — no LLM calls, no Kakao quota, and the *same* graphs on both sides,
which three fresh passes cannot offer against a ±8-point endpoint. It counts changed graphs by
`template_id` and by `mapeval_class`, because "the accuracy did not move" and "no family moved"
are different claims and only the second is evidence.

The corpus below is 2,577 graphs: the latest Spatial-Agent report for each of the 16 benchmarks
in `dataset/`, plus all three v7-283 passes at `643fe24`.

### A-class: the conjuncts the operator already implied

`calculate_start_time`, `calculate_finish_time` and `tsp_tw` appear only in a trip plan and
`filter_by_direction` only in a direction plan, so `operator == X and intent == Y` never admitted
a node `operator == X` would not have. It could only *exclude* one — and given the table above,
it did: a `tsp_tw` node in a plan the stage called `routing` ran with the stays, the budget, the
stated order and the closure all withheld, which is a confident wrong answer rather than a
failure. The trip itinerary lookup lost its gate too; the structural test underneath it (a
`batch_geocode` node listing more than two places) is what identifies an itinerary.

**Zero of 2,577 graphs changed**, on every one of 28 templates and all five task categories.
Byte-identical grounded graphs mean identical execution downstream, so this step needs no
benchmark run to be established as equivalent — a stronger statement than three passes could
make.

### B/C-class: presence as the gate

`GroundingFacts` is one reading of the question — anchor, kind of place, radius, sector, compared
pair, route preference, closing leg, stated order — taken once by `extract_facts`, and each
branch now asks whether the fact it needs is present. Which source leads is decided per fact:

- A radius, a sector, a pair of names, a route preference, a closure and a fixed order are
  written in the sentence verbatim, so the scan over the question goes first and the concept
  graph is consulted only to recover what the scan missed. An LLM re-transcription of a literal
  can only add error, which is already why option texts and place names are bound from the
  question rather than accepted from the planner.
- `target_type` is the other kind. "우산을 사야 합니다" names no category, and inferring 편의점 is
  the Analysis stage's job, not a regex's — so the question's words win when it names one and
  `analysis["target_type"]` fills the rest.

Three intent-keyed tables became single ordered lists (target-type leads; anchor relations, with
the sector pattern ahead of the nearest-of one because a sector question also says "가장 가까운";
anchor phrase splits). The option splice, which asked `intent in {"nearby", "direction",
"routing"}`, asks the graph instead: do this batch's places flow into an option ranking rather
than into a tour? That was what the intent set meant — the excluded label that mattered was
`trip`, because overwriting an itinerary's stops with the option texts answers a different trip.

Footprint over the same 2,577 graphs: **821 changed**, none in any `trip_*` family and none in
`routing_nth_turn`.

| argument slot | nodes changed | what changed |
|---|---|---|
| `batch_geocode.anchor` | 965 | 908 `None` → a name; 57 a `$reference` → a name; none the other way |
| `nearest.required_type` | 140 | the kind the question names, reaching the ranking |
| `nearby_places.*` | 142 | retrieval specs built from that kind |
| `batch_geocode.place_names` | 16 | verbatim-name repairs the anchor path now reaches |
| `batch_geocode.strict_names` | 8 | |

The option splice fires on the **same 61 nodes** before and after, neither more nor fewer.

### Two regressions the replay caught before they shipped

Both come from a scan that now runs on every question instead of on a labelled subset, which is
the predictable cost of ungating and the reason to measure rather than reason about it.

"헤이갤러리 근처에서 분위기가 가장 좋은 카페는?" bound `헤이갤러리 근처` as the anchor — a place
that does not exist — and wrote it over an option the plan had already geocoded. A vicinity word
is stripped from the anchor's tail now.

"A에서 B까지의 직선거리와 A에서 C까지의 직선거리는 얼마나 차이가 나나요?" read `(A, B)` as its
compared pair and bound `place_names` to it, deleting C — so `poi_distance_difference`, the
largest `poi` family, would have measured one of the two distances it was asked to compare. That
was *already* happening to the 15 of 99 rows the stage happened to label `distance`; a pair is
refused outright now when the question states more than one separation.

### What it scored, and the check the zero-footprint families provide

Three passes of Spatial-Agent over v7-283 at concurrency 32, against the three that preceded them
on the same rows at `643fe24`. ReAct is not run: nothing here touches it.

| | A0 `643fe24` | A2 `c07b998` |
|---|---|---|
| passes | 82.0 / 84.5 / 85.5 | 82.3 / 84.5 / 86.6 |
| overall | **84.0** | **84.5** |
| terminal failures, 3 passes | 3 | 3 |
| LLM calls | 2,658 | 2,669 |
| total tokens | 15.29 M | 15.75 M |
| intermediate execution errors | 240 | 228 |

+0.5 overall, inside the per-pass spread. On its own that number says nothing, and the
interesting part is what happens when the family deltas are put beside the *footprint on those
same rows* — 293 of the 849 graphs changed, and which ones is known exactly:

| family | graphs changed | accuracy Δ |
|---|---|---|
| `trip_optimal_order` | **0 / 72** | **+9.7** |
| `unanswerable_rating` | 8 / 9 | +22.2 |
| `unanswerable_price_level` | 3 / 6 | −33.3 |
| `nearby_kth_nearest` | 71 / 72 | +4.2 |
| `routing_detour_cost` | 57 / 72 | −4.2 |
| `nearby_within_radius_count` | **0 / 36** | −2.8 |
| `routing_nth_turn` | **0 / 63** | +1.6 |
| `trip_total_distance` | **0 / 63** | +1.6 |
| `trip_feasible_count_five` | **0 / 63** | 0.0 |
| `poi_distance_difference` | 56 / 99 | −1.0 |
| `routing_turn_count_via` | 42 / 63 | −1.6 |

The whole `trip` class is 0 of 198 graphs changed and moved **+4.0 points**, and
`trip_optimal_order` inside it moved +9.7 on graphs that are byte-identical between the two
revisions. That is not an effect of this change; it cannot be. It is this endpoint's pass-to-pass
noise, measured on the same rows in the same run condition — and it sets the scale against which
every other row of that table has to be read. No family whose graphs *did* change moved further
than one whose graphs did not.

So the result is a negative one, and it is the one worth having: **grounding stopped consulting
the intent label on a third of the questions and the score did not move.** The concept-and-role
decomposition carries what the router was carrying. That is the paper's chapter-3 claim, checked
rather than assumed — and it is a stronger check than "accuracy held", because the footprint says
exactly which 293 questions were being asked to hold it.

Two caveats belong with it. The `unanswerable_*` swings are 6- and 9-row families where one
question is 11 to 17 points, so neither the +22.2 nor the −33.3 is worth a sentence. And these
three passes carry the `ANALYSIS_PROMPT` change as well as the grounding change; they are
separate commits and could be separated by another three passes, but nothing in this result
depends on which of the two moved the 0.5.

### What is measured and what is not

Two things remain unmeasured and are deliberately separate commits. The `ANALYSIS_PROMPT` now
names `radius_m` and `direction` as typed attributes, because the recovery path had nothing to
read — 7 of 979 recorded analyses carried a radius attribute at all, under a free-form key,
against 36 questions that state one. And `retrieve_templates` still scores templates with a
weight-4 term for a matching intent against weight-1 keyword hits. Removing that term is *not*
the same kind of change as the ones above: it alters what the planner is shown, so no replay can
predict the graph it then writes, and dropping it outright leaves keyword matching only. The
principled replacement scores a template against the concept graph rather than against a label,
which means re-authoring every template's affinity. That is the remaining work, and it should be
done with a benchmark run beside it rather than on the argument alone.

### A3: remove the remaining planner and evaluator label paths

The remaining path is now removed in a separate change. `retrieve_templates` takes the normalized
concept analysis rather than an intent string. Each template declares semantic affinity terms and,
where needed, a graph/question shape such as a multi-place itinerary or road-network wording. The
retriever projects only the requested measure, target type, CONDITION/SUBCOND/MEASURE text and
concept attributes; it deliberately never traverses the top-level `intent`. Exact question
keywords remain the fallback when Analysis omitted the relevant concept hint.

The final evaluator no longer receives `Intent: ...` or an intent-specific suffix. The eight
selection rules are one shape-conditional prompt: a rule applies because the evidence contains a
route, bearing-filtered candidates, a set, pairwise metrics or itinerary totals. Analysis still
predicts the label, but `SpatialAgent.answer` pops it before template retrieval and passes the
intent-free dictionary to composition, factorization, repair and evaluation. `normalize_analysis`
also stopped using a missing intent as a missing measure; it now uses the MEASURE concept text or
the neutral `answer choice` fallback.

This stage cannot be evaluated by grounding replay: it changes the templates shown to the planner
and therefore the graph the LLM writes. A read-only replay of the three A2 passes does establish
the intended footprint before a live run: the first retrieved template changes on 229 of 849
recorded analyses. The largest changes are the mislabeled shapes this stage is meant to expose:
`routing_detour_cost` (61), `routing_turn_count_via` (40), and fixed-order
`trip_total_distance` (26). In those rows the label-selected example and the question structure
disagree; the live A3 result must decide whether the new planner context helps. Unit coverage is
546 passing tests, and both `spatial.py` and `geoflow.py` contain zero `intent ==`/`intent in`
branches. The live A3 run below measures the resulting planner change.

Three live A3 passes at `af51e93`, on the same v7-283 rows and run conditions as A0/A2, read
81.6/81.3/83.4 (pooled **82.1%**). A2 read 82.3/84.5/86.6 (pooled **84.5%**), so the observed
difference is -2.4 points. A question-cluster paired interval is approximately [-5.8, +1.1]
points: the runs do not distinguish an overall change from endpoint variation, but they also do
not prove equivalence under a narrow predeclared margin.

| `mapeval_class` | A2 | A3 | delta |
|---|---:|---:|---:|
| nearby | 80.2 | 78.6 | -1.6 |
| poi | 89.1 | 94.2 | +5.1 |
| routing | 89.4 | 88.9 | -0.5 |
| trip | 85.9 | 74.2 | -11.6 |
| unanswerable (port-added) | 71.4 | 73.0 | +1.6 |

The family split is the useful warning. `trip_total_distance`, the fixed-order shape, held at
95.2%; `trip_optimal_order` moved 80.6 -> 56.9 and `trip_feasible_count_five` 85.7 -> 73.0.
Conversely `nearby_cuisine_subtype` moved 55.6 -> 83.3. Those rows say template affinity is now a
material planner input, not that any one direction is architectural signal: the endpoint's known
cross-pass family variance is large, and these datasets are spent. Investigate the split on a
fresh draw rather than tuning the affinity table against it.

Repair occurrence rose from 122/849 (14.4%) at A2 to 141/849 (16.6%) at A3; terminal
`agent_reasoning_failure` rows rose from 3 to 8. The requested robustness runs on the three already
spent holdout files read 83.7% (v7h), 82.0% (v7h2) and 84.0% (v7h3), each pooled over three passes.
Their pass spreads were 7, 11 and 3 points respectively, and there are no same-revision A0/A2
runs on those files, so they are A3 levels rather than controlled deltas. The complete A0-A3
table, all source report names, three label axes, failure distribution and repair rates are in
`reports/intent_removal_a0_a3.md`.

## Two spatial relations the semantic vocabulary could not say

The Concept Transformation stage (see `reports/semantic_transformation_ablation.md`) recovered to
60.4% on v7-283 and stalled there, with two families carrying most of the remaining gap for the
same kind of reason: the planner could describe the question correctly and the vocabulary had no
symbol for what it described.

`routing_turn_count_via` asks about the drive from A through B to C. `ROUTE_MEASURE` had an
origin and a destination and nothing between them, so a resolved [A, B, C] was measured A to B --
a real route, fully executed, and the wrong one -- and the turns were counted on it. That is a
confident wrong answer, not a failure, which is why it cost 69.8 points on the family without
showing up in any error count.

`trip_total_distance` asks for the whole drive of a stated itinerary. `ROUTE_MATRIX` over n stops
routes n^2 pairs and the trip drives n-1 of them, so a total over the matrix answered a question
about every pair. Two of that family's 21 rows reported an errored step; the rest ran cleanly and
returned a number several times too large.

Both are added as *relations*, not as operators -- `directions` has taken waypoints since the
port began and every leg was already in the matrix:

- **`via`** on a route node: a list of ids in the order the route reaches them, wired to
  `directions` waypoints in that order. The ends of the route are the inputs and positions `via`
  did not claim. Nothing infers a waypoint from position, deliberately: "A와 B 중 C에 더 가까운
  곳" has a middle too, and a rule that made middles into waypoints would route it through the
  answer.
- **`SELECT_LEGS`** and the `select_legs` operator: the consecutive legs of a square matrix,
  optionally under a stated order. A graph that totals a bare square matrix gets the selection
  composed in as its own node, so the grouping stays visible in the graph rather than implied by
  a factor.

Three defects surfaced alongside them, none about waypoints or legs. All three were `inputs[0]`
standing in for every input: a route reader handed geocoded endpoints wired
`extract_distance(route=$A)`; `distance_matrix` over four resolved stops built a 1x1 grid; and
`tsp_tw` took its node list from whichever input was object-typed, giving a five-stop trip one
place beside a six-place matrix. They were invisible because the families they broke were already
broken for the stated reason.

Offline replay over the 283 recorded first-compose graphs: 52 changed -- `trip_total_distance`
13/21, `routing_detour_cost` 11/24, `trip_feasible_count_five` 10/21, `trip_optimal_order` 10/24,
`routing_turn_count_via` 8/21, and **0 of the 151** `nearby_*`/`poi_*`/`unanswerable_*` rows.
(First reported as 42, each family 1-4 lower: the "before" side imported the main checkout's
`src` instead of the worktree's. Re-measured under the provenance banner the replay now prints.
The 0 of 151 -- the part the reading rests on -- is unchanged.)
The replay caught one regression of this work before a benchmark did: the first version of the
route-composition rule composed a drive from `compare_routes` to itself. Note the limit of that
measurement -- `via` and `SELECT_LEGS` are planner vocabulary, so graphs recorded before they
existed never use them. The replay measures the wiring fixes; only a run measures the vocabulary.

Three Spatial-Agent passes at `53d28d1` on v7-283, concurrency 32, read 68.6/70.3/67.8 (mean
**68.9%**) against the single pass at `689734d` reading 60.4%. Validation 97.5/98.9/97.9,
semantic-vocabulary ratio 100%, concrete operator leakage 0 of 4,937 nodes. The two target
families moved +41.3 each (`routing_turn_count_via` 14.3 -> 55.6, `trip_total_distance`
52.4 -> 93.7), and the wiring fixes moved two families that were not targets
(`routing_detour_cost` +23.6, `trip_optimal_order` +6.9). By class, `routing` +22.2 and `trip`
+15.7 against `nearby` -1.2 and `poi` -2.2 -- and the offline footprint says nothing in this
change reaches a `nearby` or `poi` graph, so read those two as the draw, on this repository's own
rule that a family number belongs to the draw it was measured on.

Against `af51e93`'s 82.1% the remaining gap is about 13 points, from 22. It sits where the replay
says this change never reached: `nearby_within_radius_count` 22.2, `nearby_cuisine_subtype` 38.9,
`poi_distance_difference` 62.6 -- retrieve-then-narrow and pairwise-measure shapes, not routes.

## Concept-GeoFlow regression, and what closing it changed

`876c772` rewrote the planner's wire format to the Concept/Edge IR and left behind the
conventions that made the previous one work. It scored 14/100 on
`dataset/seoul_kmapeval_v7a_mcq_100.jsonl`; `58c0aad` before it read 76.95 on
`seoul_kmapeval_v7a_mcq_300`. `reports/final_validation_876c772.md` records the state it was
found in.

The repairs divide into three kinds, and the division is the useful part.

**Completion the step format had and the IR did not.** `plan_to_geoflow` completed the
step-shaped reply — Analysis concepts in scope, factors derived from their attributes, a
dangling reference dropped, a missing output typed by its transformation — and the IR path did
none of it. `src/agent/canonicalization.py` is that completion for the IR: every repair is
additive and last-resort, so a draft that already passes canonicalizes to itself, and a produced
concept's type comes from the vocabulary rather than from the planner. It also resolves the
places a measure reads when the graph never said to, hands a radius narrowing what it narrows,
and reads a detour's second route as the one with a stop — that last from the subset relation
between two routes' inputs, never from where an input sat.

**Conventions the prompt used to state.** `via`, "copy a place name verbatim", "the first input
of a search or a route is the place it is measured from", and "an itinerary is one ROUTE_MATRIX"
were all in the old `GRAPH_PROMPT` and none survived the rewrite. Restoring them helped;
`AGENTS.md`'s warning against tuning prompts still holds, and the one addition that was a
taxonomy rather than a convention — the semantic factor vocabulary — cost 9 points and was
reverted.

**Stages that could not see their own input.** `retrieve_macro_templates` ranked against factors
derived from concept attributes alone, so `ROUTE-OPTIMIZE` was never offered for the itinerary
questions that state a time budget and stays. Grounding read `len(names) == len(options) + 1` as
proof that a batch is [anchor, *options] long after MCQ matching left the core and `options`
became always empty. `tsp_tw` computed an order and reported only indices, and a total only as
`total_cost`.

`MCQAdapter` reconciles by rounding rather than nearness, matches an ordering by the order the
answer names it in, and reads a decline against `알 수 없음` only when nothing else matched, the
core produced no value at all, and no step raised. That last guard is what keeps it from
becoming the escape hatch `AGENTS.md` records: measured, it fires 13 times and is right 10.

Held out at `ca41713`, six passes at concurrency 32 and budget 15:
`seoul_kmapeval_v7h3_holdout_100` reads 69.2, against 72.0 for that set at `8797217` and 84.0 at
`af51e93`. The step budget is not what separates them —
`reports/step_budget_ablation_354e3dd.md` shows budget 30 leaving the overall flat while lifting
one family 38 points.


## The five Appendix E macro families the typed catalogue was missing

`src/agent/templates.py` carried five of Appendix E's ten macro templates —
`FILTER-AGGREGATE-MEASURE`, `OBJECT-FIELD-MEASURE`, `ROUTE-OPTIMIZE`, `MULTI-ROUTE-COMPARE`,
`MULTI-SEGMENT-AGGREGATE`. The other five existed only in `TEMPLATES` in `src/agent/geoflow.py`,
which that file's own comment marks as offline replay/regression compatibility: the Spatial-Agent
runtime never retrieves it, so no planner prompt has ever carried a bearing, an attribute lookup,
a turn-count or an arrival-time shape. A question of one of those shapes was composed from
whichever of the five ranked highest.

The missing five are now typed fragments beside the others, written in the semantic vocabulary of
`src/agent/semantics.py` and carrying no operator, no argument and no provider category:

| Template | Transformation edges | Factor affinity |
|---|---|---|
| `GEOCODE-BATCH-COMPARE` | `RESOLVE_PLACES -> DISTANCE_MEASURE -> EXTREME_SELECT` | `measure`, `extreme` |
| `LOCATION-BEARING-CLASSIFY` | `RESOLVE_PLACES -> FILTER (a stated sector) -> MEASURE` | `direction` |
| `ROUTE-STEP-EXTRACT` | `ROUTE_MEASURE -> ROUTE_STEPS -> MEASURE` | `metric` |
| `PLACE-ATTRIBUTE-QUERY` | `PLACE_SEARCH -> PLACE_DETAILS -> MEASURE` | none |
| `TIME-WINDOW-REVERSE` | `ROUTE_MEASURE -> SCHEDULE -> MEASURE` | `stays`, `stay_duration_s` |

`TIME-WINDOW-REVERSE` is the one that needed new ports: its contextual input is an `event` typed
`temporal_extent` and its measure is an `event`, and neither fits `_SPATIAL_INPUT` or
`_MEASURE_OUTPUT`. `PLACE-ATTRIBUTE-QUERY` declares an empty `factor_affinity` deliberately —
no radius, ordinal, direction, budget or route objective characterizes "what kind of place is
this", and inventing one for it would be a family patch wearing a template's clothes.

Each of the ten composes through `compose_templates` into a graph that passes G1–G5 strictly,
pinned per template in `tests/test_plan_conformance.py`, and each has a worked demonstration in
`default_example_store()` with one factor bound.

**What this changes and what it does not.** Retrieval is unchanged: the same typed scorer over the
same concepts and factors, now ranking ten candidates instead of five for the same two slots. That
is not a neutral addition — on a question whose concepts tie, `GEOCODE-BATCH-COMPARE` now sorts
ahead of `MULTI-ROUTE-COMPARE` and `OBJECT-FIELD-MEASURE` on the alphabetical tie-break, so the
prompt a tied question receives is different. Nothing about grounding, factorization or the
operator set moved, so `data/replay_grounding.py` cannot measure this: templates change the
planner's *vocabulary of shapes*, and graphs recorded before a shape was retrievable never use it.
Reading it costs a run, which is what the next section is.

### Measured: ten templates against five, same revision

`seoul_kmapeval_v7h3_holdout_100`, Spatial-Agent only, three passes a side at concurrency 32 and
budget 15, the two arms differing only in `src/agent/templates.py` and the example-store rows in
`src/agent/retrieval.py`.

| arm | passes | mean | terminal failures over 300 rows |
| --- | --- | --- | --- |
| ten templates | 66 / 67 / 70 | **67.7** | 44 parse, 33 graph-validation, 10 provider |
| five templates | 70 / 72 / 62 | **68.0** | 45 parse, 33 graph-validation, 11 provider |

The overall difference is -0.3 points against a five-pass spread that puts this set's own three
passes 10 points apart in the baseline arm alone, so it is not a difference. The failure
counts are the readable half: the five added priors changed neither the number of graphs the
validator refused nor the number of answers that failed to parse, which is what "the planner can
use these fragments" looks like. Every family number moved — `trip_feasible_count_five` +23.8,
`unanswerable_price_level` -66.7 on three rows — and none of them is quotable: `AGENTS.md`
requires six passes before a family is quoted on this set, and the `unanswerable_*` families are
three rows each, where one question is 33 points.

The prompt really did change. Over the 290 questions of the ten-template arm that reached
retrieval, `GEOCODE-BATCH-COMPARE` was offered 95 times and `TIME-WINDOW-REVERSE` 79 — so more
than half the set was planned from a prior that did not exist before, and the score did not move.
That is the same shape as the intent-removal result above: a large change in what the planner is
shown, no change in what it scores.

**Three of the five were never retrieved once.** `LOCATION-BEARING-CLASSIFY` and
`PLACE-ATTRIBUTE-QUERY` have no family in this set, which is expected. `ROUTE-STEP-EXTRACT` does
— `routing_nth_turn` and `routing_turn_count_via` are 28 of the 100 rows and its edges are
exactly their shape — and it is *structurally unreachable* for them: on a turn-count concept set
it ties `MULTI-ROUTE-COMPARE` and `ROUTE-OPTIMIZE` at the same score, both sort ahead of it
alphabetically, and the two slots are gone. Its `metric` affinity cannot break the tie because
`MULTI-ROUTE-COMPARE` claims `metric` too. The template exists and cannot be offered for the
questions it describes. That is a retrieval-scorer defect, not a template defect, and fixing it
by hand-weighting a family would be the family patch `AGENTS.md` forbids; it is recorded here
rather than tuned around.


### The retrieval scorer ranked on declared ports, and two defects fell out of it

Chasing the flat result above found the mechanism, and it is not in the templates.
`retrieve_macro_templates` scored fit as `2 * |question cores ∩ union(input port core_concepts)|`.
A port says what a fragment may be *handed*; taking the union made **declaring a port a ranking
advantage**.

*D1 — port inflation.* `TIME-WINDOW-REVERSE` needs a second, temporal input port because its clock
input is an `event`. That alone scored it 3 where every rival scored 2 on any question carrying an
`amount` or an `event`, which is most of them. Replayed over the 332 recorded analyses it was
offered 79 times — 22 of the 30 `poi_farthest_of_three` rows, a straight-line distance family, and
all three `unanswerable_opening_hours` rows. Its overlap with what it actually computes over is 1
on those questions, the lowest of the ten.

*D2 — an unreachable fragment.* `ROUTE-STEP-EXTRACT` was offered zero times for the 28
`routing_nth_turn` + `routing_turn_count_via` rows whose shape it is: it tied `MULTI-ROUTE-COMPARE`
and `ROUTE-OPTIMIZE` and lost both slots to the alphabetical tie-break on the dict key. This
predates the five additions — under the old scorer `MULTI-SEGMENT-AGGREGATE` and
`OBJECT-FIELD-MEASURE` could not retrieve themselves from their own concept sets either.

**The fix is one term.** Rank on the core concepts of the fragment's own `concept_nodes` — what it
computes over — and leave the ports as the I/O declaration composition needs them to be. Both
defects close. Three regression tests in `tests/test_plan_conformance.py` pin it, and all three
fail against the old term.

**Footprint**, replayed offline over the recorded analyses at no LLM or Kakao cost: 256 of 332 rows
are offered a different pair. `TIME-WINDOW-REVERSE` 90 → 11, `ROUTE-STEP-EXTRACT` 0 → 33,
`PLACE-ATTRIBUTE-QUERY` 0 → 121, `GEOCODE-BATCH-COMPARE` 114 → 58.

### Measured: six passes a side

`seoul_kmapeval_v7h3_holdout_100`, Spatial-Agent only, concurrency 32, budget 15.

| arm | passes | mean | graph-validation failures / 600 |
| --- | --- | --- | --- |
| five templates, old scorer | 70 / 72 / 62 / 77 / 68 / 67 | 69.3 | 61 |
| ten templates, fixed scorer | 69 / 71 / 74 / 64 / 73 / 73 | **70.7** | 51 |

**+1.4 is not a result.** The baseline's own six passes span 62 to 77, so the interval swallows the
difference several times over. Neither the ten templates nor the scorer fix moves this set's
overall accuracy, and the fix is not justified by it — it is justified by D1 and D2, which are
provable from the scorer alone and are pinned by tests that need no benchmark.

What the family split does show is that the fix moved what its footprint said it would. The two
families `ROUTE-STEP-EXTRACT` newly reaches are the two that gain most among the routing families:
`routing_nth_turn` 61.9 → 73.8 and `routing_turn_count_via` 42.9 → 54.8, both +11.9. `trip_total_distance`
+19.0 and `trip_feasible_count_five` +14.3 come with `TIME-WINDOW-REVERSE` no longer taking the
second slot on itinerary questions. Against them, `poi_distance_difference` -13.6 and
`nearby_subtype_kth` -10.0 — and those are the residual defect below, not noise about which nothing
can be said. Six passes is what `AGENTS.md` requires before quoting a family on this set; it is
still one draw.

**The residual defect, recorded and not acted on.** Type overlap barely separates shapes. A
`nearby_kth_nearest` question and `PLACE-ATTRIBUTE-QUERY` have the same core-concept signature,
`{location, object}`, so the generic fragment wins the generic questions — the fix changes which
template free-rides (`GEOCODE-BATCH-COMPARE` 114 → `PLACE-ATTRIBUTE-QUERY` 121), not that one does,
and `poi_distance_difference` is where that cost shows. The signal that would separate them is in
the typed facts retrieval never sees: `GroundingFacts` carries `turn_field`, `listed_places`,
`compared_pair`, `target_subtype` and `via_place`, and `grounding_factor_nodes` exports seven fact
types that do not include any of them. That is the same shape as the recorded `ROUTE-OPTIMIZE`
finding — a stage that cannot see its own input — and the same repair would apply. It is left
undone deliberately: handing a fact to the template that would win a family with it is a family
patch by effect, and the export also feeds `attach_grounding_factors`, so widening it changes
grounding and not only ranking.

## `--count` was a request, not a count, and it moved the class mix

`--count 300` produced 281, 282, 282, 283 and 283 rows on the five draws in `dataset/`. The
apportionment was never the problem — `apportion` splits the quotas by largest remainder and sums
to exactly what was asked for — and neither was any single unlucky draw: every one of the five is
short in the same family.

| family | apportioned | drawn (v7, v7a, v7b, v7c, v7d) |
| --- | --- | --- |
| `poi_farthest_of_three` | 30 | 13 / 12 / 13 / 12 / 13 |
| every other family | as apportioned | as apportioned, but for `nearby_subtype_kth` 28 once |

Thirteen is not a coincidence. The family scanned `itertools.combinations(landmarks[:55], 4)` and
retires all four of a row's places into a `used` set, so 55 landmarks is 13 disjoint quadruples and
the fourteenth row could not exist at any `--count`. The same constant-slice scan sits in four
other families (`poi_distance_difference` at 150, `routing_detour_cost` at 80, `routing_nth_turn`
at 140, `routing_turn_count_via` at 110); none of them binds at 300 rows, and all of them bind
somewhere above it. It is the defect `AGENTS.md` already records for `nearby_kth_nearest` — a scan
bound that does not grow with `count` — in a second place.

**What it cost is not the seventeen rows.** The quotas are MapEval-API's own class mix, counted off
`mapeval-api/dataset.json`: nearby 83, poi 64, routing 66, trip 67, unanswerable 20 over 300
questions. `poi` is the only class with one wide family and one narrow one, so seventeen rows off
`poi_farthest_of_three` came off `poi` entire: those files carry 45 or 46 `poi` rows against the 63
the quotas encode, 16% of the set instead of 21%. Upstream's 71.07% is a mean over its own mix, and
a mean over a different one is not comparable to it.

Two changes, in `data/benchmark_core.py` and `data/builder_cli.py`:

- **`candidate_groups(items, size, rng, count)`** replaces the five constant-slice scans. It offers
  disjoint tuples drawn from a reshuffled pool, budgeted at `max(24, count * 3) * 8` offers, so it
  grows with the build the way `_scan_limit` already did and spends the whole pool rather than its
  first *N* entries. Raising the constant instead is not available: 360 landmarks taken four at a
  time is 1.7 billion tuples and `used` skips nearly all of them.
- **Top-up rounds.** A family that still comes up short — because the city genuinely ran out of
  anchors it can use, which no scan bound fixes — is frozen at what it drew and its remaining rows
  are handed to families that filled their share, *of the same MapEval-API class first*. So the
  file holds the count the caller asked for and keeps the mix. A redraw runs under its own stream
  (`f"{seed}:{name}:{attempt}"`) and rows are deduplicated on question text. If after three rounds
  the pool still cannot supply the count, the build exits non-zero naming the short families rather
  than writing a file that is quietly smaller than the number on its command line.

`data/audit_dataset.py` gained one more check while this was being fixed: a generated row must not
carry a `context` field. No builder here has written one since v1 — MapEval-API is MapEval-Textual
with exactly that field removed, and every run here answers from live Kakao — so the check pins
what is already true rather than changing it. The v1 legacy file keeps its own for provenance and
is exempt, which its missing `mapeval_class` is what identifies.

**What this does to the earlier draws.** Nothing: v7 through v7d are the files they were, and their
recorded accuracies are what the code scored on them. But their `poi` class was measured on 45 rows
where the quotas asked for 63, so a `poi` number from one of those files and a `poi` number from a
set built after this fix are not the same measurement, and the overall accuracies sit on slightly
different mixes. Read them as this document already requires family numbers to be read — as
belonging to the draw they were measured on.

## v8: the first draw that is the size it was asked for, and the first `poi` at upstream's share

Built by `data/build_kmapeval_dataset.py --count 300` with the two fixes above, run at `81efc7b`
with nothing under `src/` touched, so it is held out. Three passes a side, concurrency 32,
`--react-tools reference`, temperature 0, `MAX_REASONING_STEPS` 15.

| | passes | mean | spread |
| --- | --- | --- | --- |
| no-tool floor | 80/300 | **26.7** | — |
| floor excluding `unanswerable_*` | 62/279 | **22.2** | — |
| ReAct | 46.3 / 48.0 / 44.3 | **46.2** | 3.7 |
| Spatial-Agent | 73.3 / 71.7 / 71.7 | **72.2** | 1.7 |

Gap 26.0, outside either agent's own spread. That sits at the low end of the 27-33 the five
281-283-row draws reported, and the reason is visible in the class table rather than in the
overall: the class whose row count changed is the class that moved.

| class | rows | rows on v7-v7d | ReAct | Spatial-Agent | gap |
| --- | --- | --- | --- | --- | --- |
| nearby | 84 | 82-84 | 55.2 | 77.8 | 22.6 |
| poi | **63** | **45-46** | 27.0 | 82.5 | **55.6** |
| routing | 66 | 66 | 36.4 | 69.2 | 32.8 |
| trip | 66 | 66 | 49.5 | 63.1 | 13.6 |
| unanswerable | 21 | 21 | 88.9 | 57.1 | -31.7 |

`poi` is the first measurement of that class at the proportion the quotas encode — 21% of the set
rather than 16% — and it is the widest class gap here, ReAct 27.0 against Spatial-Agent 82.5. Both
its families agree and neither is close: `poi_distance_difference` 26.3 / 81.8 and
`poi_farthest_of_three` 27.8 / 83.3, +55.6 apiece. Against a floor of 26.7 the baseline is not
measuring on `poi` at all — 27.0 is the floor — which is the same reading v7d recorded for
`distance` and `routing`, now on the class upstream weights second-heaviest.

**Three families invert, and all three are cheap to guess.** `nearby_within_radius_count` (ReAct
77.8, Spatial-Agent 41.7), `trip_feasible_count_five` (69.8 / 46.0) and the whole `unanswerable`
class (88.9 / 57.1). All three are ladders or refusals whose floor is already high — 5/12, 10/21
and 18/21 respectively on the no-tool run — so what they measure is closer to a prior over four
fixed strings than to a map. `trip` as a class is dragged by exactly this: it is 63.1 for
Spatial-Agent against 70.8 and 71.4 on its two non-ladder families.

**One draw, one caveat.** `data/audit_dataset.py` exits non-zero on `nearby_kth_nearest`, 17 of 24
rows at k=2. That is the sixth consecutive draw this family has skewed and the second-best of the
six; the cause is recorded in `AGENTS.md` — it anchors on Seoul's four densest chain categories,
where half of consecutive rank gaps fall under `ORDINAL_MARGIN_M` — and the lever is the pool or
the margin, not the scan this change fixed. That family's 44.4 / 77.8 is not quotable; nothing
else on the set is affected.

## The ordinal family could not ask its own question, and five draws said so

`nearby_kth_nearest` audited as concentrated on k=2 in six draws for six — 19 of 24 on v7, 20 of 24
on v7c, 17 of 24 on v8 — while `nearby_subtype_kth`, which draws its ordinal by the same rule in
the same file, balanced every time. Two repairs had already been made and neither moved it: the
ordinal stopped being keyed on the anchor loop index, and `_scan_limit` started growing with
`count`. This file recorded the remaining suspicion as "the lever is the pool or the margin, not
the scan". Measured, it is the pool, and not marginally.

Over 108 anchors at `ORDINAL_MARGIN_M = 90`, by the category the question asks for:

| code | kind | usable anchors | separable past k=2 | can supply k=4 |
| --- | --- | --- | --- | --- |
| `CE7` | 카페 | 0 | 0 | 0 |
| `FD6` | 음식점 | 0 | 0 | 0 |
| `CS2` | 편의점 | 2 | **0** | 0 |
| `BK9` | 은행 | 2 | **0** | 0 |
| `PM9` | 약국 | 2 | **0** | 0 |
| `MT1` | 대형마트 | 43 | 23 | 11 |
| `PO3` | 공공기관 | 27 | 7 | 3 |
| `SC4` | 학교 | 23 | 10 | 4 |
| `CT1` | 문화시설 | 21 | 9 | 5 |
| `SW8` | 지하철역 | 37 | 17 | 8 |

The family asked by `CE7`, `BK9`, `PM9` and `CS2`. Those four supplied six usable neighbourhoods
between them and *not one* separable past k=2, so k was never being drawn — it was being dictated
by the category, and no balancing code could have fixed it. Seoul puts four cafes inside 90 m of
each other; the ordinal question needs a kind of place the city spaces out. The family now asks by
`MT1`, `SC4`, `PO3` and `CT1`, and draws 8/8/8 at count 24 on four different seeds, at 601 Kakao
calls against the 5,799 the dense pool spent failing.

**A second defect underneath it, worth the general note.** The gap tests are nested — k=4 requires
every gap k=3 requires and one more — so `feasible` is always a prefix of `(2, 3, 4)`: every usable
anchor supplies k=2 and only a few supply k=4. `min(feasible, key=lambda k: (produced[k], k))`
therefore broke every early tie toward k=2 and spent the scarce anchors on the value that needed no
scarcity. `_scarcest_ordinal` takes the value furthest below its own target and breaks ties toward
the larger k. Replaying v8's exact draw off the cache — free and deterministic, because
`gold_evidence.ranked_m` records what each accepted anchor could have offered — separates the two
repairs:

| | spread | most-common share |
| --- | --- | --- |
| as v8 shipped | 17 / 4 / 3 | 0.71 (audit fails) |
| tie-break fixed, dense pool | 16 / 4 / 4 | 0.67 |
| tie-break fixed, sparse pool | **8 / 8 / 8** | 0.33 |

**What the family was hiding.** Split by k on v9, three passes a side:

| k | rows | ReAct | Spatial-Agent |
| --- | --- | --- | --- |
| 2 | 24 | 62.5 | 91.7 |
| 3 | 24 | 12.5 | 75.0 |
| 4 | 24 | 16.7 | 66.7 |

Ranking four options against each other — what an agent does when it does not retrieve — answers a
k-th question whenever the k-1 nearer places are all among the decoys: 60% of the time at k=2, 10%
at k=4. ReAct scores the rung that is answerable that way and collapses on the two that are not.
So a draw that was 17 to 19 of 24 rows at k=2 was not merely unbalanced; it was quoting the
baseline's single best rung as the family's number.

## v9: the first 300-row set that passes its own audit

Built at `d95a9bc` plus the ordinal fix, run at `d95a9bc` with `src/` untouched, so it is held out.
Three passes a side, concurrency 32, `--react-tools reference`, temperature 0, budget 15.

| | passes | mean | spread |
| --- | --- | --- | --- |
| no-tool floor | 79/300 | **26.3** | — |
| floor excluding `unanswerable_*` | 61/279 | **21.9** | — |
| ReAct | 44.7 / 47.7 / 44.0 | **45.4** | 3.7 |
| Spatial-Agent | 69.0 / 72.3 / 70.0 | **70.4** | 3.3 |

Gap 25.0, outside either agent's spread. `poi` is again the widest class gap — ReAct 21.7 against
Spatial-Agent 76.7, the baseline below its own floor on both of that class's families — and the
`unanswerable` class and `trip_feasible_count_five` again invert toward ReAct, which is what a
high-floor ladder does. Read v8 and v9 as two draws, not as a before and after: the only thing that
changed between them under `src/` is nothing, and the family numbers move by the usual per-draw
range.

## Three defects behind Spatial-Agent's failure buckets, and what fixing two of them did not do

A v9 pass reported `answer_parse_failure` 43, `graph_validation_failure` 19, `provider_failure` 10
— 72 of 300 rows. Read one at a time they are three unrelated things, and only two are defects.

**`answer_parse_failure` is a misnomer on this path.** Spatial-Agent never writes a `^^N^^` for
`parse_answer` to read: `response_text` is the MCQ adapter's selection or it is empty. Reaching
that branch means only that the adapter matched the grounded answer to none of the options. Read
charitably — the bare value as written, ×1000 and ÷1000 — **none of the 43 lands on any option**,
so the refusal is the anti-least-bad-match rule working and the question is simply answered
wrongly. It is now `grounded_answer_unmatched`. ReAct's `answer_parse_failure` keeps its name; there
a model really did write something the parser could not read.

**Inside it, one real defect with a clean fingerprint.** All seven `trip_total_distance` rows in
that bucket overshot the gold by *exactly their own first leg*:

| id | agent | gold | excess | legs |
| --- | --- | --- | --- | --- |
| v9_239 | 39913 | 35187 | 4726 | **4726**, 11299, 19162 |
| v9_246 | 30497 | 24629 | 5868 | **5868**, 11299, 7462 |
| v9_248 | 29561 | 20965 | 8596 | **8596**, 11327, 1042 |
| v9_249 | 45741 | 36371 | 9370 | **9370**, 22433, 4568 |

The plan says why: `sum_amounts(amounts=["$t4", "$t7", "$t10", "$t3"])` where `$t4` is
`extract_distance(route="$t3")`. A route and the distance taken off it are one measurement written
twice, so a three-leg drive totalled from four addends. `sum_amounts` cannot see it — by the time
it runs both are dicts carrying `distance_m` — but the executor still holds the unresolved
references and every step's arguments, so `_drop_subsumed_addends` drops an addend another addend
was measured from. Correct rows of the family carry exactly three addends; the seven carry four.

**`provider_failure` is a composite arriving where a name belongs.** Five of ten: three option
lists (`'CU 마곡아르디에점, CU 강서발산역점, GS25 강서등촌점, GS25 마곡루체점'`) and two itineraries
(`'쥬인스테이 → 삼모아트센터'`). `place_names` is the plural argument, so a `field_validator` splits
an element on an itinerary arrow or a comma *followed by a space* — the space matters, because
`종로5,6가동 주민센터` and `7,900파스타 용두역점` are single Kakao names and all seven comma-bearing
names in the 12,138-place pool write it without one. A single-name argument still fails: there the
composite is genuinely ambiguous. The three `'Located Hotel Isabel'` rows are left alone — repairing
a translated name is an ability added to one architecture, not a wiring fix.

**`graph_validation_failure` is mostly the step budget.** Seven of nineteen are
`exceeds MAX_REASONING_STEPS=15` on graphs of 17 to 23 transformations, and eleven of nineteen are
`trip_feasible_count_five`. That is Spatial-Agent's version of the `iteration_limit` v6's four-stop
families spent on ReAct: a family that cannot be said within the budget measures the budget. The
repair is to shrink the family, as v7 did, not to raise a budget that is one number for both
architectures. Nothing was changed for it.

**What the two fixes did to the score: nothing measurable.** Spatial-Agent on v9, concurrency 32:

| | passes | mean |
| --- | --- | --- |
| before (4 passes) | 69.0 / 72.3 / 70.0 / 71.7 | **71.2** |
| after (3 passes) | 71.0 / 73.0 / 69.0 | **71.0** |

−0.2, and the after-passes alone span 4.0. `trip_total_distance` went 78.6 → 84.1 and
`routing_detour_cost` 64.6 → 58.3 — the two families the rule fires on, moving opposite ways — while
`unanswerable_review_count` moved +20.8 on two rows and `unanswerable_subjective` −11.8, on
mechanisms neither fix can reach. So the benchmark says what it said before.

The justification is the replay, which needs no endpoint and no quota. Over every v9 spatial log
carrying a `sum_amounts` the rule fires on, recomputing the total without the subsumed addend
**corrects 20 and breaks none**: 20 fixed, 40 wrong either way, 0 correct totals lost. That is the
same shape as the intent-removal entry above — the claim is what the mechanism does, demonstrated
where it can be demonstrated exactly, and explicitly *not* a point on the accuracy.

**v9 is spent.** `src/` changed in response to what it showed, so 45.4 / 70.4 belongs to `d95a9bc`
and the numbers here are a level for the code after it. Build a fresh set before quoting a held-out
number again.
