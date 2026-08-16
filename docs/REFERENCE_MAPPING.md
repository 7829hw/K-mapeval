# Reference implementation mapping

| Upstream concept | K-MapEval implementation | Deliberate deviation |
|---|---|---|
| MapEval `Evaluator2.py` structured ReAct loop | `src/agent/react.py` | Removes the localhost backend, 30/60-second sleeps, and remote result writes. Uses user-requested 0-based multiple choice with `^^N^^`. |
| MapEval `FormattedTools.py` | `src/tools/registry.py` | Thin tools return canonical JSON rather than provider-specific prose. No embedded bearer token. |
| MapEval `Tools.py` / MapQaTor backend | `src/tools/map.py`, `src/tools/kakao.py` | Provider dependency injection replaces the separate HTTP backend. |
| Spatial information theory analysis | Spatial-Agent `analyze` stage | Extracts all seven core concepts and six functional roles together with intent; classification metadata is not passed in. |
| Concept transformation drafting | `src/agent/geoflow.py` template retrieval | Retrieves pre-validated macros for place attributes, bearing, geocode comparison, radius filtering, route comparison, and multi-segment aggregation. Uses deterministic intent/keyword similarity instead of an embedding service. |
| GeoFlow construction and factorization | Spatial-Agent `compose` stage | LLM composes a typed operator-concept DAG from the analysis, candidate options, and retrieved templates. |
| Five GeoFlow constraints | Spatial-Agent `validate` stage | Enforces acyclicity, role ordering, operator input/output type compatibility, executable operator contracts, and connectivity to a Measure node. Invalid graphs get one repair pass and are never silently truncated. |
| Topological executor | Spatial-Agent `execute` stage | Executes role-prioritized topological order and records every intermediate result plus the final concept state. |
| Spatial-Agent operators | `src/tools/registry.py`, `src/tools/spatial.py` | Includes batch geocoding, route distance matrix, distance/bearing, nearest/radius, route comparison, and multi-segment aggregation over Kakao-normalized fields. |
| Spatial-Agent evaluator/generator | Spatial-Agent `evaluate` and `generate` stages | Evaluation chooses a 0-based option; generation deterministically emits `^^N^^`. |
| Google Maps client | `KakaoMapProvider` | Kakao Local handles search/geocode/nearby and Kakao Mobility handles driving routes. |
| Spatial-Agent local context cache | `SQLiteMapCache` | Both agent architectures share the same normalized cache rather than giving Spatial-Agent a private cache. |

The paper's SFT and DPO stages are optional. This repository implements the paper's off-the-shelf
LLM prompting path and does not claim the separately fine-tuned Qwen SFT+DPO result.

## Kakao-specific constraints

- Raw Kakao JSON never reaches an agent.
- Kakao Local does not expose a standalone place-details REST method. `place_details` reads places already normalized and cached during the current run.
- The MVP supports driving routes. Unsupported modes fail explicitly instead of silently substituting a different metric.
- The user-requested persistent SQLite cache stores only normalized `Place` and `Route` payloads, not raw Kakao JSON.
