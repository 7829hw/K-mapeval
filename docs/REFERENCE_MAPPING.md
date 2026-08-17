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
| Appendix C operators | `ToolRegistry`, `SpatialOperatorRegistry` | Adds reverse/batch details, waypoint directions, instruction route filtering, extractors, pairwise extremes, place filtering, travel-time nearest, temporal operators, and step analysis with paper-level semantics. |
| TSP-TW | `SpatialOperatorRegistry.tsp_tw` | Exhaustive optimization up to nine nodes with service times, windows, and budget; infeasible instances return the paper's nearest-unvisited partial feasible fallback. OR-Tools is not bundled. |
| Temporal operators | `timezone`, `open_at_time`, `calculate_finish_time`, `calculate_start_time` | Handles cross-midnight/24-hour periods. Multi-stop finish time queries cached/live route durations and adds stays. Latest-departure calculation is an explicit template helper. |
| Spatial-Agent evaluator/generator | Spatial-Agent `evaluate` and `generate` | The LLM conditions on question, final state, and full trace. Deterministic match evidence no longer overwrites the generated selection. |
| Google Maps client | `KakaoMapProvider` | Kakao Local handles search/geocode/reverse-geocode/nearby; Kakao Mobility handles driving and verified multi-waypoint routes with optional guides. |
| Spatial-Agent context cache | `SQLiteMapCache` | Both agents share one normalized Kakao cache. The upstream MapEval-Textual/Google context snapshot is not bundled. |
| SFT and DPO | Not implemented | This repository evaluates the off-the-shelf prompting path and does not claim fine-tuned Qwen results. |

## Remaining non-equivalences

- Template retrieval is keyword based, not embedding based.
- Factorization and concept binding are deterministic, not SFT/DPO learned.
- Kakao live POIs replace the upstream Google/MapEval-Textual evidence snapshot.
- Kakao Mobility support is driving-only.
- The benchmark router uses the LLM analysis intent; Korean heuristics are only a fallback when the
  returned intent is missing or unsupported.
- The bundled MCQ trip format primarily evaluates option-order comparison. `tsp_tw` is available
  for true free-order plans but is not silently substituted for option semantics.
- Kakao Local exposes phone and Kakao URL but not rating, price level, or structured opening hours.
  These optional `Place` fields can only be used when an authoritative enriched cache supplies them;
  missing values are never invented.
- Offline coordinate-to-timezone resolution deliberately covers the Korean benchmark extent only.

These differences mean the project implements the paper's explicit graph formalism and execution
constraints, but it does not claim numerical reproduction of the paper's trained model or original
map-data environment.

Reports from this repository must therefore be labeled **prompting-only**. They are not directly
comparable to the paper's SFT+DPO headline results, because the learned policy, original weights,
embedding example store, and original map evidence snapshot are not included.

## Kakao-specific constraints

- Raw Kakao JSON never reaches an agent.
- Kakao Local has no standalone place-details REST method; `place_details` and
  `batch_place_details` read normalized cached records from earlier retrieval/enrichment.
- `reverse_geocode` uses `/v2/local/geo/coord2address.json`; waypoint routes use the official
  `POST /v1/waypoints/directions` and verify returned waypoint names and coordinates.
- Unsupported travel modes fail explicitly.
- SQLite stores normalized `Place` and `Route` payloads, not raw Kakao responses or API keys.
