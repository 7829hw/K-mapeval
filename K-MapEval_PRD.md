# K-MapEval PRD

## 1. 개요

**K-MapEval**은 한국 지리·POI 환경과 국내 지도 API를 기반으로 MapEval 계열의 공간 추론 평가 환경을 구축하고, 기존 Spatial-Agent의 성능 향상이 한국 지도 환경에서도 재현되는지 검증하기 위한 연구용 프로젝트다.

본 프로젝트의 목적은 다음과 같다.

1. 한국 지리·POI 환경과 Kakao Map API를 사용하는 **한국형 MapEval-API 실행 환경**을 구현한다.
2. 동일한 한국 지도 환경에서 **MapEval 방식의 ReAct Agent와 Spatial-Agent를 비교**한다.
3. 기존 Spatial-Agent의 성능 향상이 한국 지도 및 국내 POI 환경에서도 재현되는지 예비 검증한다.

> 본 단계의 주목적은 **benchmark dataset 자체를 완성하는 것이 아니라, MapEval과 Spatial-Agent를 참고하여 한국형 평가 시스템을 구현하는 것​**이다.  
> 전체 Benchmark Dataset은 후속 단계에서 별도로 제작한다.

---

## 2. Reference Implementations

구현 시 기존 프로젝트의 구조와 코드를 적극적으로 참고한다.

### MapEval-API

Repository:

`https://github.com/MapEval/MapEval-API`

주요 참고 대상:

- `Evaluator2.py`
  - ReAct 기반 benchmark agent 구성
  - question/options 입력 및 answer evaluation 방식
- `FormattedTools.py`
  - 지도 API를 LLM tool 형태로 제공하는 방식
  - API response formatting 방식
- `Tools.py`
  - Place Search, Nearby, Directions 등 지도 tool interface
- `dataset.json`
  - MapEval benchmark 문항 구조 및 classification 참고

K-MapEval의 **ReAct baseline은 가능한 한 MapEval-API의 평가 구조를 유지**하면서 Google/MapQaTor 의존 부분을 Kakao 기반 tool layer로 교체한다.

### MapQaTor Backend

Repository:

`https://github.com/MapQaTor/mapqator-backend`

주요 참고 대상:

- MapEval-API가 지도 기능을 별도 backend로 추상화한 방식
- Place Search / Nearby / Directions 등의 API 구조
- benchmark와 지도 provider를 분리하는 설계

단, K-MapEval MVP에서는 별도의 backend server를 구현하지 않는다.

MapQaTor의 역할은 Python 내부의 `KakaoMapProvider`가 담당한다.

### Spatial-Agent

Repository:

`https://github.com/ecerybao/Spatial-Agent`

주요 참고 대상:

- `src/agent/spatial_agent.py`
  - `Route → Plan → Execute → Evaluate → Generate` workflow
- `src/agent/operators.py`
  - spatial operator 구현 및 실행 구조
- `src/tools/google_maps.py`
  - Google Maps client abstraction
- `src/tools/local_context_db.py`
  - local spatial data interface
- `test_agent.py`
  - MapEval-API dataset 평가 및 accuracy 계산 방식

K-MapEval의 Spatial-Agent 구현은 **원본 Spatial-Agent 코드를 최대한 유지**하면서 지도 provider 부분만 Kakao 기반으로 교체하는 것을 원칙으로 한다.

---

## 3. 핵심 연구 질문

> 동일한 한국형 공간 질의와 동일한 Kakao 지도 정보를 사용할 때, Spatial-Agent가 MapEval의 ReAct Agent보다 높은 공간 추론 성능을 보이는가?

주요 비교 조건:

```text
ReAct Agent
vs
Spatial-Agent
```

두 시스템에서 다음 조건은 가능한 한 동일하게 유지한다.

- LLM
- benchmark questions
- Kakao Map API
- tool output schema
- temperature
- 최대 reasoning/tool-call 횟수
- evaluation metric

즉 핵심 독립 변수는 **Agent Architecture**가 되도록 한다.

---

## 4. 시스템 구조

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

MVP에서는 별도의 MapQaTor 형태 backend server를 구축하지 않는다.

두 Agent가 동일한 `KakaoMapProvider` Python implementation을 직접 호출하도록 한다.

---

## 5. Kakao Map Provider

지도 provider와 Agent를 분리하기 위해 공통 interface를 정의한다.

```python
class MapProvider:
    def search_place(self, query): ...
    def geocode(self, address): ...
    def nearby_search(self, ...): ...
    def place_details(self, place_id): ...
    def directions(self, origin, destination, mode): ...
```

Kakao 구현:

```python
class KakaoMapProvider(MapProvider):
    ...
```

주요 데이터 소스:

- Kakao Local REST API
  - 키워드 장소 검색
  - 카테고리 검색
  - 주소 검색
  - 좌표 관련 기능
- Kakao Mobility API
  - 자동차 경로 탐색

Spatial-Agent 코드의 `GoogleMapsClient`가 담당하던 역할을 `KakaoMapProvider`가 대체한다.

---

## 6. Canonical Tool Interface

ReAct와 Spatial-Agent가 서로 다른 형태의 Kakao API 응답을 받는 것을 방지하기 위해 공통 schema를 사용한다.

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

Kakao raw JSON은 Agent에게 직접 제공하지 않고 `KakaoMapProvider`에서 정규화한다.

---

## 7. MapEval-style ReAct Baseline

MapEval-API의 `Evaluator2.py`, `FormattedTools.py`, `Tools.py`를 참고하여 ReAct baseline을 구현한다.

주요 tool:

```text
PlaceSearch
PlaceDetails
NearbyPlaces
Directions
TravelTime / Distance
```

동작 구조:

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

각 tool은 직접 Kakao API를 호출하지 않고 공통 `KakaoMapProvider`를 사용한다.

```python
@tool
def place_search(query: str):
    return provider.search_place(query)
```

이를 통해 ReAct와 Spatial-Agent가 동일한 지도 데이터 layer를 사용하도록 한다.

---

## 8. Spatial-Agent Porting

Spatial-Agent repository의 기존 architecture를 최대한 유지한다.

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

주요 수정 대상은 **지도 provider dependency**다.

기존:

```text
GoogleMapsClient
      ↓
Google Maps API
```

변경:

```text
KakaoMapProvider
      ↓
Kakao APIs
```

Planner, workflow, state management, spatial operator 구조는 가능한 한 원본 코드와 동일하게 유지한다.

이를 통해 성능 차이가 Spatial-Agent 자체 변경이 아니라 **지도 환경 변화에 따른 generalization 결과**로 해석될 수 있도록 한다.

---

## 9. Spatial Operators

API retrieval과 deterministic spatial calculation을 분리한다.

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

예:

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

Spatial-Agent의 기존 operator 구현을 우선 재사용하고, Kakao 데이터 형식과 호환되지 않는 부분만 수정한다.

---

## 10. Benchmark Dataset

### MVP 범위

**본 구현 단계에서는 전체 Benchmark Dataset을 제작하지 않는다.**

Dataset 구축은 별도의 후속 연구 단계로 분리한다.

현재 단계에서는 시스템이 정상 작동하는지를 확인하기 위한 **소수의 sample questions만 작성한다.**

권장:

```text
nearby / poi / routing / trip과 함께
type / direction / distance / radius 유형을 지원한다.

총 약 8~12문항
```

이 sample dataset의 목적은:

- Kakao API integration 확인
- ReAct tool-call 확인
- Spatial-Agent workflow 확인
- answer evaluation pipeline 확인
- 두 Agent가 동일 문제를 처리할 수 있는지 확인

이다.

### Sample Schema

MapEval-API 형식을 참고한다.

`answer`는 `options`의 0-based index이며, 에이전트 출력과 로그·보고서도 같은 기준을 사용한다.

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

필요한 경우 개발용 metadata를 추가할 수 있다.

```json
{
  "region": "서울",
  "difficulty": "easy",
  "verified_at": "YYYY-MM-DD"
}
```

단, 이러한 metadata는 Agent input에 포함하지 않는다.

---

## 11. 향후 Benchmark Dataset 구축

전체 한국형 benchmark dataset 제작은 MVP 완료 후 별도 단계에서 수행한다.

향후 고려 대상:

- 전국 지역 분포
- 서울 편중 방지
- POI category diversity
- `nearby / poi / routing / trip / type / direction / distance / radius` 균형
- easy / medium / hard 난이도
- single-hop / multi-hop reasoning
- 정답 검증
- API 변화에 대한 benchmark consistency
- 데이터 저장 및 Kakao API 이용정책 검토

최종 목표 규모는 별도 dataset design 단계에서 결정한다.

---

## 12. 평가 기능

MVP evaluation에서는 다음을 기록한다.

### Primary Metric

- Multiple-choice Accuracy

### Additional Metrics

- classification별 Accuracy
- tool-call 횟수
- Kakao API 호출 횟수
- reasoning step 수
- execution failure
- latency

예:

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

## 13. Repository 구조

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

## 14. 구현 단계

### Phase 1 — Reference Code Analysis

다음 repository의 코드를 분석하고 대응 관계를 정리한다.

```text
MapEval/MapEval-API
MapQaTor/mapqator-backend
ecerybao/Spatial-Agent
```

특히 다음 관계를 파악한다.

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

- `MapProvider` interface 구현
- Kakao Local API 연결
- Kakao Mobility API 연결
- response normalization 구현
- API error handling 구현

### Phase 3 — Sample Dataset

- MapEval-API schema 참고
- 4개 classification별 2~3개 sample 작성
- 약 8~12문항만 구성
- 사람이 직접 정답 검증

**전체 benchmark dataset은 이 단계에서 제작하지 않는다.**

### Phase 4 — ReAct Baseline

- MapEval-API ReAct implementation 참고
- 동일한 tool 구조 재현
- KakaoMapProvider 연결
- final answer parsing 구현

### Phase 5 — Spatial-Agent Porting

- Spatial-Agent repository fork 또는 code reuse
- `GoogleMapsClient` dependency 제거
- `KakaoMapProvider` 연결
- 기존 spatial operators 호환
- 기존 workflow 최대한 유지

### Phase 6 — MVP Evaluation

동일 sample question에 대해:

```text
ReAct + Kakao
vs
Spatial-Agent + Kakao
```

를 실행한다.

이 단계의 목적은 통계적으로 유의한 성능 검증이 아니라 **전체 evaluation pipeline의 feasibility 확인**이다.

### Phase 7 — Full Benchmark

MVP가 정상 동작한 이후 별도 dataset 구축 과정을 거쳐 본 실험을 수행한다.

---

## 15. MVP 완료 조건

다음을 만족하면 구현 MVP가 완료된 것으로 본다.

- MapEval 관련 코드 구조 분석 완료
- Spatial-Agent 코드 구조 분석 완료
- `KakaoMapProvider` 구현
- Kakao Local API 호출 가능
- Kakao Mobility API 호출 가능
- canonical response schema 구현
- MapEval-style ReAct Agent 실행 가능
- Kakao 기반 Spatial-Agent 실행 가능
- 8~12개의 sample questions 준비
- 동일 sample에 대해 두 Agent 평가 가능
- Accuracy / tool calls / API calls 기록 가능

**수백 문항 규모의 Benchmark Dataset 구축은 MVP 완료 조건에 포함하지 않는다.**

---

## 16. 최종 산출물

MVP 단계의 최종 산출물은 다음과 같다.

```text
1. KakaoMapProvider
   Kakao 기반 공통 spatial data interface

2. MapEval-style ReAct Baseline
   기존 MapEval 코드를 참고한 한국형 baseline

3. Kakao Spatial-Agent
   기존 Spatial-Agent를 Kakao 환경으로 porting

4. Sample Evaluation Dataset
   시스템 검증용 8~12개 문항

5. Common Evaluator
   ReAct와 Spatial-Agent 비교 실행 환경
```

이후 별도의 Benchmark Dataset 구축 단계를 통해 한국형 MapEval benchmark를 확장하고 본격적인 성능 비교를 수행한다.
