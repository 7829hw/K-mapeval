# K-MapEval PRD

## 1. Overview

**K-MapEval** is a research project that builds a MapEval-style spatial-reasoning evaluation
environment for Korean geography, POIs, and domestic map APIs. It tests whether the performance
improvements reported for Spatial-Agent can be reproduced in a Korean map environment.

The project has three goals:

1. Build a **Korean MapEval-API-style execution environment** using Korean geographic/POI data and
   the Kakao Map API.
2. Compare a **MapEval-style ReAct Agent and Spatial-Agent** under the same Korean map conditions.
3. Conduct a preliminary test of whether Spatial-Agent's reported gains generalize to Korean maps
   and domestic POI data.

> The primary goal of this phase is **not to complete the benchmark dataset itself, but to
> implement a Korean evaluation system informed by MapEval and Spatial-Agent**.
> The complete benchmark dataset will be created separately in a later phase.

---

## 2. Reference Implementations

The structure and code of the existing projects should be consulted actively during
implementation.

### MapEval-API

Repository:

`https://github.com/MapEval/MapEval-API`

Primary references:

- `Evaluator2.py`
  - ReAct-based benchmark-agent construction
  - Question/options input and answer-evaluation flow
- `FormattedTools.py`
  - Exposing map APIs as LLM tools
  - Formatting API responses
- `Tools.py`
  - Map-tool interfaces such as Place Search, Nearby, and Directions
- `dataset.json`
  - MapEval benchmark item structure and classification vocabulary

K-MapEval's **ReAct baseline should preserve the MapEval-API evaluation structure as closely as
possible**, replacing its Google/MapQaTor dependencies with a Kakao-based tool layer.

### MapQaTor Backend

Repository:

`https://github.com/MapQaTor/mapqator-backend`

Primary references:

- How MapEval-API abstracts map functionality behind a separate backend
- API structures for Place Search, Nearby, and Directions
- Separating the benchmark from the map provider

The K-MapEval MVP does not implement a separate backend server.

The Python `KakaoMapProvider` assumes the role that MapQaTor's backend would have provided.

### Spatial-Agent

Repository:

`https://github.com/ecerybao/Spatial-Agent`

Primary references:

- `src/agent/spatial_agent.py`
  - `Route -> Plan -> Execute -> Evaluate -> Generate` workflow
- `src/agent/operators.py`
  - Spatial-operator implementation and execution structure
- `src/tools/google_maps.py`
  - Google Maps client abstraction
- `src/tools/local_context_db.py`
  - Local spatial-data interface
- `test_agent.py`
  - MapEval-API dataset evaluation and accuracy calculation

The K-MapEval Spatial-Agent implementation should **preserve the original Spatial-Agent code as
far as possible**, replacing only the map-provider layer with Kakao-based components.

---

## 3. Core Research Question

> When both agents receive the same Korean spatial questions and the same Kakao map information,
> does Spatial-Agent achieve higher spatial-reasoning performance than MapEval's ReAct Agent?

Primary comparison:

```text
ReAct Agent
vs
Spatial-Agent
```

The following conditions should be kept identical across both systems wherever possible:

- LLM
- Benchmark questions
- Kakao Map API
- Tool-output schema
- Temperature
- Maximum number of reasoning/tool-call steps
- Evaluation metric

The goal is therefore for **agent architecture** to be the primary independent variable.

---

## 4. System Structure

```text
             K-MapEval Sample Questions
            question / options / answer
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
      MapEval-style         Spatial-Agent
       ReAct Agent
             │                   │
             └─────────┬─────────┘
                       ▼
              Common Tool Layer
                       │
                KakaoMapProvider
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
     Kakao Local API       Kakao Mobility API
       POI / Geocode          Directions
```

The MVP does not build a separate MapQaTor-style backend server.

Both agents directly call the same Python `KakaoMapProvider` implementation.

---

## 5. Kakao Map Provider

Define a common interface to separate the map provider from the agents.

```python
class MapProvider:
    def search_place(self, query): ...
    def geocode(self, address): ...
    def nearby_search(self, ...): ...
    def place_details(self, place_id): ...
    def directions(self, origin, destination, mode): ...
```

Kakao implementation:

```python
class KakaoMapProvider(MapProvider):
    ...
```

Primary data sources:

- Kakao Local REST API
  - Keyword place search
  - Category search
  - Address search
  - Coordinate-related functions
- Kakao Mobility API
  - Automobile route planning

`KakaoMapProvider` replaces the role previously played by Spatial-Agent's `GoogleMapsClient`.

---

## 6. Canonical Tool Interface

Use a common schema so ReAct and Spatial-Agent do not receive Kakao responses in different forms.

### Place

```python
{
    "place_id": "...",
    "name": "경복궁",
    "address": "서울특별시 종로구 ...",
    "latitude": 37.5796,
    "longitude": 126.9770,
    "category": "관광명소"
}
```

### Route

```python
{
    "origin": "...",
    "destination": "...",
    "distance_m": 8500,
    "duration_s": 1320,
    "steps": [...]
}
```

Kakao raw JSON must be normalized by `KakaoMapProvider` before it reaches an agent.

---

## 7. MapEval-style ReAct Baseline

Implement the ReAct baseline with `Evaluator2.py`, `FormattedTools.py`, and `Tools.py` from
MapEval-API as references.

Primary tools:

```text
PlaceSearch
PlaceDetails
NearbyPlaces
Directions
TravelTime / Distance
```

Execution flow:

```text
Question
   ↓
Thought
   ↓
Tool Call
   ↓
Observation
   ↓
Thought
   ↓
Tool Call
   ↓
Observation
   ↓
Final Answer
```

Each tool uses the shared `KakaoMapProvider` rather than calling the Kakao API directly.

```python
@tool
def place_search(query: str):
    return provider.search_place(query)
```

This ensures that ReAct and Spatial-Agent use the same map-data layer.

---

## 8. Spatial-Agent Porting

Preserve the existing Spatial-Agent architecture as far as possible.

```text
Question
   ↓
Route
   ↓
Plan
   ↓
Execute
   ↓
Evaluate
   ↓
Generate
```

The primary change is the **map-provider dependency**.

Before:

```text
GoogleMapsClient
      ↓
Google Maps API
```

After:

```text
KakaoMapProvider
      ↓
Kakao APIs
```

The planner, workflow, state management, and spatial-operator structure should remain as close as
possible to the original code.

This makes any performance difference interpretable as a **generalization result caused by the
map environment**, rather than as a consequence of unrelated changes to Spatial-Agent itself.

---

## 9. Spatial Operators

Separate API retrieval from deterministic spatial computation.

### Retrieval Tools

```text
Place Search
Nearby Search
Geocoding
Directions
```

### Deterministic Operators

```text
Haversine Distance
Minimum / Maximum
Sorting
Candidate Filtering
Route Comparison
Travel-Time Comparison
```

Example:

```text
Kakao Place Search
       ↓
Candidate Coordinates
       ↓
Haversine Operator
       ↓
Distance Comparison
       ↓
Answer
```

Reuse Spatial-Agent's existing operators where possible, modifying only the parts incompatible
with Kakao's data format.

---

## 10. Benchmark Dataset

### MVP Scope

**The complete benchmark dataset is not created during this implementation phase.**

Dataset construction is a separate follow-up research phase.

At this stage, create only a **small number of sample questions** to verify that the system works.

Recommended coverage:

```text
Support nearby / poi / routing / trip, together with
type / direction / distance / radius question types.

Approximately 8–12 questions in total.
```

The sample dataset is intended to verify:

- Kakao API integration
- ReAct tool calls
- Spatial-Agent workflow
- Answer-evaluation pipeline
- Whether both agents can process the same questions

### Sample Schema

Follow the MapEval-API format.

`answer` is a 0-based index into `options`; agent outputs, logs, and reports use the same
convention.

```json
{
  "id": "nearby_001",
  "question": "경복궁에서 가장 가까운 지하철역은 어디인가?",
  "options": [
    "광화문역",
    "경복궁역",
    "안국역",
    "종각역"
  ],
  "answer": 1,
  "classification": "nearby"
}
```

Development metadata may be added when needed.

```json
{
  "region": "Seoul",
  "difficulty": "easy",
  "verified_at": "YYYY-MM-DD"
}
```

This metadata must not be included in the agent input.

---

## 11. Future Benchmark Dataset

Build the complete Korean benchmark dataset in a separate phase after the MVP is complete.

Future considerations:

- Geographic coverage across the country
- Avoiding excessive concentration in Seoul
- POI-category diversity
- Balance across `nearby / poi / routing / trip / type / direction / distance / radius`
- Easy / medium / hard difficulty
- Single-hop / multi-hop reasoning
- Answer verification
- Benchmark consistency under API changes
- Data storage and Kakao API usage-policy review

The final dataset size will be decided during a separate dataset-design phase.

---

## 12. Evaluation Features

The MVP evaluation records the following.

### Primary Metric

- Multiple-choice accuracy

### Additional Metrics

- Accuracy by classification
- Number of tool calls
- Number of Kakao API calls
- Number of reasoning steps
- Execution failures
- Latency

Example:

```text
Question ID: routing_001

ReAct
Answer: 3
Correct: True
Tool Calls: 5
API Calls: 4

Spatial-Agent
Answer: 3
Correct: True
Tool Calls: 3
API Calls: 3
```

---

## 13. Repository Structure

```text
k-mapeval/
├── main.py
│
├── src/
│   ├── agent/
│   │   ├── base.py
│   │   ├── react.py
│   │   └── spatial.py
│   │
│   ├── tools/
│   │   ├── map.py
│   │   ├── kakao.py
│   │   ├── cache.py
│   │   ├── registry.py
│   │   └── spatial.py
│   │
│   ├── config.py
│   ├── dataset.py
│   ├── evaluator.py
│   ├── metrics.py
│   ├── models.py
│   └── parsing.py
│
├── dataset/
│   └── sample.jsonl
│
├── tests/
└── README.md
```

---

## 14. Implementation Phases

### Phase 1 — Reference Code Analysis

Analyze the following repositories and document their correspondence:

```text
MapEval/MapEval-API
MapQaTor/mapqator-backend
ecerybao/Spatial-Agent
```

In particular, identify these relationships:

```text
MapEval Tool
        ↕
Kakao Tool

GoogleMapsClient
        ↕
KakaoMapProvider

MapEval Evaluator
        ↕
K-MapEval Evaluator
```

### Phase 2 — KakaoMapProvider

- Implement the `MapProvider` interface
- Connect to the Kakao Local API
- Connect to the Kakao Mobility API
- Normalize API responses
- Implement API error handling

### Phase 3 — Sample Dataset

- Follow the MapEval-API schema
- Write 2–3 samples for each of four classifications
- Build only approximately 8–12 questions
- Verify answers manually

**Do not build the complete benchmark dataset in this phase.**

### Phase 4 — ReAct Baseline

- Use the MapEval-API ReAct implementation as a reference
- Reproduce the same tool structure
- Connect `KakaoMapProvider`
- Implement final-answer parsing

### Phase 5 — Spatial-Agent Porting

- Fork or reuse the Spatial-Agent repository
- Remove the `GoogleMapsClient` dependency
- Connect `KakaoMapProvider`
- Make the existing spatial operators compatible
- Preserve the existing workflow as far as possible

### Phase 6 — MVP Evaluation

Run both agents on the same sample questions:

```text
ReAct + Kakao
vs
Spatial-Agent + Kakao
```

This phase is intended to verify **the feasibility of the complete evaluation pipeline**, not to
produce statistically significant performance results.

### Phase 7 — Full Benchmark

After the MVP works, build a separate dataset and conduct the main experiment.

---

## 15. MVP Completion Criteria

The implementation MVP is complete when all of the following are satisfied:

- MapEval-related code structure has been analyzed
- Spatial-Agent code structure has been analyzed
- `KakaoMapProvider` is implemented
- Kakao Local API calls work
- Kakao Mobility API calls work
- A canonical response schema is implemented
- The MapEval-style ReAct Agent runs
- The Kakao-based Spatial-Agent runs
- 8–12 sample questions are prepared
- Both agents can be evaluated on the same samples
- Accuracy, tool calls, and API calls are recorded

**Building a benchmark dataset with hundreds of questions is not part of the MVP completion
criteria.**

---

## 16. Final Deliverables

The MVP deliverables are:

```text
1. KakaoMapProvider
   A Kakao-based common spatial-data interface

2. MapEval-style ReAct Baseline
   A Korean baseline informed by the existing MapEval code

3. Kakao Spatial-Agent
   A port of Spatial-Agent to the Kakao environment

4. Sample Evaluation Dataset
   8–12 questions for system validation

5. Common Evaluator
   An execution environment for comparing ReAct and Spatial-Agent
```

The Korean MapEval benchmark will be expanded and evaluated in earnest through a separate
benchmark-dataset construction phase.
