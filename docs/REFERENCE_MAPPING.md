# Reference implementation mapping

| Upstream concept | K-MapEval implementation | Deliberate deviation |
|---|---|---|
| MapEval `Evaluator2.py` structured ReAct loop | `src/agent/react.py` | Removes the localhost backend, 30/60-second sleeps, and remote result writes. Keeps 1-based multiple choice and `^^N^^`. |
| MapEval `FormattedTools.py` | `src/tools/registry.py` | Thin tools return canonical JSON rather than provider-specific prose. No embedded bearer token. |
| MapEval `Tools.py` / MapQaTor backend | `src/tools/map.py`, `src/tools/kakao.py` | Provider dependency injection replaces the separate HTTP backend. |
| Spatial-Agent intent routing | Spatial-Agent `route` stage | Preserved as a separate LLM stage. Classification metadata is not passed in. |
| Spatial-Agent planner and transformation executor | Spatial-Agent `plan` and `execute` stages | Uses a compact JSON plan DSL and common tool registry. |
| Spatial-Agent operators | `src/tools/spatial.py` | Reimplements semantically stable distance, bearing, sort, min/max and route aggregation without Google-specific fields. |
| Spatial-Agent evaluator/generator | Spatial-Agent `evaluate` and `generate` stages | Evaluation chooses a 1-based option; generation deterministically emits MapEval format. |
| Google Maps client | `KakaoMapProvider` | Kakao Local handles search/geocode/nearby and Kakao Mobility handles driving routes. |
| Spatial-Agent local context cache | `SQLiteMapCache` | Both agent architectures share the same normalized cache rather than giving Spatial-Agent a private cache. |

## Kakao-specific constraints

- Raw Kakao JSON never reaches an agent.
- Kakao Local does not expose a standalone place-details REST method. `place_details` reads places already normalized and cached during the current run.
- The MVP supports driving routes. Unsupported modes fail explicitly instead of silently substituting a different metric.
- The user-requested persistent SQLite cache stores only normalized `Place` and `Route` payloads, not raw Kakao JSON.
