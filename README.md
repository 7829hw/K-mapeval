# K-MapEval

Kakao 지도 정보 위에서 MapEval 방식 ReAct와 Spatial-Agent의 공간 추론 성능을 같은 조건으로 비교하는 연구용 MVP입니다. 전체 한국형 벤치마크가 아니라, 공통 provider·도구·평가 파이프라인과 8개 개발용 샘플을 제공합니다.

## 구조

```text
K-MapEval/
├── main.py              # 단일 실행 진입점
├── src/
│   ├── agent/           # ReAct / Spatial-Agent
│   ├── tools/           # Kakao provider, SQLite cache, 공간 연산
│   ├── config.py        # .env 설정
│   ├── dataset.py       # JSONL 로더
│   ├── evaluator.py     # 평가 및 결과 기록
│   └── models.py        # Place / Route 스키마
├── dataset/             # 평가 데이터셋
├── tests/               # 단위 테스트
└── results/             # 실행 결과(자동 생성)
```

두 에이전트 모두 `Place`와 `Route` 정규화 스키마만 봅니다. 정답, classification, region, difficulty, verified_at은 에이전트 입력에 포함되지 않습니다.

## SQLite 캐시

두 에이전트는 동일한 `KakaoMapProvider`와 SQLite 캐시를 사용합니다.

```text
ReAct ───────┐
             ├─→ KakaoMapProvider → SQLite cache
Spatial ─────┘                         │
                                   cache miss
                                       ▼
                                   Kakao API
```

`search_place`, `geocode`, `nearby_search`, `place_details`, `directions` 및 이를 사용하는 `travel_time`은 항상 캐시를 먼저 조회합니다. 기본 DB는 `data/kakao_cache.db`, 기본 TTL은 24시간입니다. `.env`에서 다음 값을 바꿀 수 있습니다.

```ini
KAKAO_CACHE_DB_PATH=data/kakao_cache.db
KAKAO_CACHE_TTL_SECONDS=86400
```

TTL을 `0`으로 설정하면 만료하지 않습니다. DB에는 정규화된 `Place`/`Route`와 캐시 키 생성용 요청 인자만 저장하며 Kakao API 키, 원본 응답, LLM 프롬프트는 저장하지 않습니다. 평가 결과에는 `cache_hits`, `cache_misses`, 실제 `api_calls`가 별도로 기록됩니다.

## 설치

Python 3.11 이상을 권장합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp example.env .env
```

`.env`에 LLM과 `KAKAO_REST_API_KEY`를 입력합니다. 하나의 Kakao REST API 키를 Local API와 Mobility API에 함께 사용합니다. `LLM_BASE_URL`을 비워두면 OpenAI 기본 endpoint를 사용하고, 입력하면 OpenAI 호환 Chat Completions endpoint를 사용합니다. Temperature, 출력 토큰 수, 요청 timeout은 별도로 전달하지 않고 연결된 LLM/API의 기본값을 사용합니다. `MAX_REASONING_STEPS`로 문항당 reasoning/tool-call 단계 상한을 설정할 수 있습니다.

## 실행

```bash
python main.py --agent react
python main.py --agent spatial
python main.py --agent both
```

일부 문항만 실행할 수 있습니다.

```bash
python main.py --agent both --dataset dataset/test.jsonl --ids nearby_001 poi_001
```

결과는 기본적으로 `results/` 아래에 에이전트별 summary와 문항별 trace로 저장됩니다. 로그에는 API 키가 들어가지 않습니다.
실행 중에는 각 QA의 시작·완료, 전체 진행률, 정답 여부와 소요시간이 터미널에 표시됩니다.
`results/<agent>_report.json`은 QA 하나가 끝날 때마다 누적 갱신되므로 중간에 실행이
중단되어도 완료된 문항까지의 summary와 trace를 확인할 수 있습니다.

## 평가 항목

- 전체 및 classification별 multiple-choice accuracy
- tool-call, Kakao API-call, reasoning-step 수
- SQLite cache hit/miss 수
- latency와 failure 수
- 문항별 normalized arguments, 실행 상태, 예측값

`dataset/sample.jsonl`은 통합 동작 확인용 fixture이며 `verification_status=needs_live_kakao_check`로 표시되어 있습니다. 실험용 benchmark로 사용하기 전에 본인의 Kakao 키로 최신 POI·경로 결과를 확인하고 `verified_at`을 기록해야 합니다.

## 테스트

일반 테스트는 Kakao API를 mock하므로 키나 네트워크가 필요 없습니다.

```bash
pytest
ruff check .
```

원본 코드와의 대응 및 의도적인 차이는 `docs/REFERENCE_MAPPING.md`에 정리되어 있습니다.
