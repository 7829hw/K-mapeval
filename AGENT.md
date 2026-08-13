# AGENT.md

## Project

Build a Korean MapEval-style evaluation environment using Kakao map APIs.
Compare a MapEval-style ReAct agent with Spatial-Agent under the same map tools.
Research whether Spatial-Agent's gains reproduce in Korean geography and POI settings.
This phase implements infrastructure, not the full benchmark dataset.
Create only a small sample dataset for end-to-end validation.

## References

Use https://github.com/MapEval/MapEval-API as the main ReAct reference.
Review `Evaluator2.py` for agent execution and multiple-choice evaluation.
Review `FormattedTools.py` and `Tools.py` for tool interfaces and formatting.
Use https://github.com/MapQaTor/mapqator-backend as an architectural reference only.
Do not reproduce the full MapQaTor backend for the MVP.
Use https://github.com/ecerybao/Spatial-Agent as the main Spatial-Agent reference.
Review `src/agent/spatial_agent.py`, `src/agent/operators.py`, `src/tools/google_maps.py`, and `test_agent.py`.

## Architecture

Both agents must use the same normalized map-tool layer.
Canonical flow: `Agent -> Common Tools -> KakaoMapProvider -> Kakao APIs`.
Do not give ReAct and Spatial-Agent separate Kakao implementations.
Do not implement a separate HTTP backend server in the MVP.

## Map Provider

Define a provider-neutral `MapProvider` interface.
Implement `KakaoMapProvider` as the Kakao-backed implementation.
All Kakao API calls must go through `KakaoMapProvider`.
Do not call Kakao APIs directly from agents, evaluators, or datasets.
Use Kakao Local REST API for place, category, address, and coordinate queries.
Use Kakao Mobility APIs for driving directions when needed.
Keep API keys in environment variables and provide `.env.example`.

## Normalized Schemas

Do not expose raw Kakao JSON directly to agents.
Normalize places to `place_id`, `name`, `latitude`, `longitude`, `address`, and `category`.
Normalize routes to `origin`, `destination`, `distance_m`, `duration_s`, and `steps`.
Both agents must consume the same normalized schemas.

## ReAct Baseline

Implement a MapEval-style ReAct baseline using the reference code.
Provide tools comparable to `PlaceSearch`, `PlaceDetails`, `NearbyPlaces`, `Directions`, and `TravelTime`.
Tool wrappers must be thin and delegate to `KakaoMapProvider`.
Preserve MapEval multiple-choice evaluation behavior where practical.
Never expose the gold answer to the ReAct agent.

## Spatial-Agent Port

Reuse the existing Spatial-Agent architecture as much as possible.
Preserve `Route -> Plan -> Execute -> Evaluate -> Generate`.
Replace Google Maps-specific dependencies with `MapProvider`.
Prefer dependency injection over hardcoded provider clients.
Reuse existing spatial operators when semantically valid.
Only modify planner, state, or operators when Kakao compatibility requires it.
Never expose the gold answer to Spatial-Agent.

## Spatial Operations

Separate external retrieval from deterministic calculations.
Use the provider for place lookup, nearby search, geocoding, directions, and travel time.
Perform Haversine distance, sorting, filtering, min/max, and route comparison locally.
Do not spend external API calls on deterministic calculations.

## Sample Dataset

Do not build the full benchmark dataset in this phase.
Create only approximately 8-12 sample questions.
Cover `nearby`, `poi`, `routing`, and `trip`.
Use about 2-3 samples per classification.
Use fields `id`, `question`, `options`, `answer`, and `classification`.
Prefer 1-based answer indices for MapEval compatibility.
Treat samples as development fixtures, not research benchmark data.
Manually verify sample answers before pipeline testing.

## Evaluation

Run both agents on exactly the same sample questions.
Use the same LLM, model version, temperature, provider, and normalized outputs.
The intended independent variable is agent architecture.
Report multiple-choice accuracy as the primary metric.
Also record classification accuracy, tool calls, API calls, latency, and failures.
Keep per-question logs for debugging and later error analysis.

## Logging and Errors

Log `question_id`, `classification`, `agent_type`, `predicted_answer`, and correctness.
Log tool names, normalized arguments, execution status, and API-call counts.
Never log API keys or secrets.
Distinguish provider failures from agent reasoning failures.
Handle place-not-found, timeout, auth failure, rate limit, and route-not-found.

## Testing

Test Kakao response normalization and canonical schemas.
Test deterministic spatial operators, answer parsing, and tool wrappers.
Mock Kakao APIs for normal unit tests.
Keep live API tests separate and optional.

## Repository Guidance

Prefer folders for `providers`, `schemas`, `tools`, `agents`, `evaluation`, `dataset`, `scripts`, and `tests`.
Keep Kakao-specific logic inside the provider layer.
Keep agent-facing interfaces provider-neutral.
Avoid unnecessary databases, web UIs, distributed systems, or production infrastructure.
Do not create persistent Kakao response dumps unless usage rights are explicitly confirmed.

## Definition of Done

`KakaoMapProvider` is implemented and tested.
MapEval-style ReAct runs against Kakao-backed tools.
Spatial-Agent runs after replacing Google Maps dependencies.
Both agents evaluate the same 8-12 sample questions.
Accuracy, tool calls, API calls, latency, and failures are reported.
Core tests pass without requiring live Kakao API access.
The full Korean benchmark dataset remains a separate future task.

## Research Integrity

Do not hardcode answers or special-case individual questions.
Do not reveal evaluation-only metadata unless the protocol requires it.
Do not give one agent richer map evidence than the other.
Keep tool schemas and normalization identical across agents.
Document any deviation from the reference implementations.
Optimize for a fair `ReAct vs Spatial-Agent` comparison.
