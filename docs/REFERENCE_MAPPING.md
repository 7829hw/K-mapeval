# Reference implementation mapping

| Upstream concept | K-MapEval implementation | Deliberate deviation |
|---|---|---|
| MapEval `Evaluator2.py` structured ReAct loop | `src/agent/react.py` | Removes the localhost backend, sleeps, and remote writes. Uses user-requested 0-based `^^N^^` answers. |
| MapEval tools/backend | `src/tools/registry.py`, `src/tools/map.py`, `src/tools/kakao.py` | Provider injection replaces the separate HTTP backend; tools expose normalized JSON. |
| Spatial information theory analysis | `normalize_analysis`, `ConceptGraph` | Preserves all seven core-concept labels and six roles. Missing EXTENT/MEASURE concepts are made explicit and marked synthetic. Dataset classification metadata is not passed in. |
| Concept transformation drafting | `retrieve_templates`, Spatial-Agent `compose` | Implements the ten Appendix E macro names. Retrieval remains deterministic intent/keyword scoring rather than embedding cosine similarity. |
| Concept graph `G` | `ConceptGraph`, `ConceptNode` | Stores concept IDs, `lambda` types, `rho` roles, attributes, and dependency edges from Analysis. |
| Operator-concept hypergraph `G'` | `factorize_geoflow`, `OperatorHyperedge` | Factorization is deterministic rather than learned. Each hyperedge records input concepts and one or more output-path bindings. Derived intermediate concepts are explicitly marked. |
| Five GeoFlow constraints | `normalize_and_validate_graph` | Enforces G1 acyclicity, G2 role ordering, G3 typed references, G4 executable contracts/data arguments, and both halves of G5: `EXTENT/TEXTENT → v → MEASURE` for every operator node. It also rejects unbound concepts. |
| Core-concept execution types | `OPERATOR_CONTRACTS`, `SpatialOperatorRegistry` | LOCATION, OBJECT, FIELD, EVENT, NETWORK, AMOUNT, and PROPORTION all have executable producers. The current benchmark directly exercises only a subset. |
| Contextual/functional roles | `factorize_geoflow`, `ROLE_PRIORITY` | Root evidence is EXTENT/TEXTENT; restriction, filtering, structure, and answer stages use SUBCOND, COND, SUPPORT, and MEASURE. |
| Topological executor | Spatial-Agent `execute` stage | Records operator state and separately materializes concept state through output bindings, including multiple bindings from one operator. |
| TSP-TW | `SpatialOperatorRegistry.tsp_tw` | Deterministic exhaustive optimization with time-window feasibility, limited to nine nodes; it does not depend on OR-Tools. MCQ trip questions may still compare option-provided visit orders. |
| Temporal operators | `open_at_time`, `timezone_convert`, `calculate_finish_time` | Implemented and typed as EVENT outputs; the bundled Korean 100-question set contains no temporal questions. |
| Spatial-Agent evaluator/generator | Spatial-Agent `evaluate` and `generate` | Evaluation chooses a 0-based option; generation emits `^^N^^`. |
| Google Maps client | `KakaoMapProvider` | Kakao Local handles search/geocode/nearby and Kakao Mobility handles driving routes. |
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

These differences mean the project implements the paper's explicit graph formalism and execution
constraints, but it does not claim numerical reproduction of the paper's trained model or original
map-data environment.

## Kakao-specific constraints

- Raw Kakao JSON never reaches an agent.
- Kakao Local has no standalone place-details REST method; `place_details` reads normalized cached
  places from an earlier retrieval.
- Unsupported travel modes fail explicitly.
- SQLite stores normalized `Place` and `Route` payloads, not raw Kakao responses or API keys.
