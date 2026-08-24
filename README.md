# K-MapEval

MapEval 방식 ReAct와 Spatial-Agent의 공간 추론 성능을 같은 조건으로 비교하는 연구용 MVP입니다. 공통 provider·도구·평가 파이프라인을 제공하며, 독립변수는 에이전트 구조 하나뿐입니다.

두 에이전트의 근거(evidence)는 항상 `KakaoMapProvider`에서 옵니다. Kakao Local / Kakao
Mobility 실호출과 두 에이전트가 공유하는 SQLite 캐시만 사용합니다.

지원 classification은 `nearby`, `poi`, `routing`, `trip`, `type`, `direction`,
`distance`, `radius`입니다. `answer`와 에이전트의 선택지 번호는 모두 `options`의
0-based index입니다.

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
├── logs/                # 문항별 실행 로그(자동 생성)
└── reports/             # 배치 평가 보고서(자동 생성)
```

두 에이전트 모두 `Place`와 `Route` 정규화 스키마만 봅니다. 정답, classification, region,
difficulty, verified_at 및 과거 데이터셋의 **context**는 에이전트 입력이나 provider 설정에
포함되지 않습니다. `BenchmarkItem.agent_input()`은 `(question, options)`만 반환합니다.

Spatial-Agent는 `공간 개념/기능 역할 분석 → 매크로 검색 → ConceptGraph 구성 →
operator-concept hypergraph factorization → 5개 제약 검증 → 위상순 실행 → 근거 기반 선택`
순서로 동작합니다. 실행 결과 개념은 operator output에, 반경·방향·카테고리 같은 보조
상수 개념은 논문의 factor node에 해당하는 hyperedge parameter/input에 바인딩됩니다.
G5는 모든 노드에 대해 `EXTENT/TEXTENT → node → MEASURE` 양쪽 도달성을 검사합니다.
LLM이 부여한 절차 역할은 실제 데이터 의존 순서를 위반하지 않도록 정규화합니다.
그래프가 잘못되면 한 번 수리한 뒤 다시 factorization·검증합니다. Routing/Trip은
`distance_matrix`, `aggregate_route_groups`를 사용해
선택지별 경로와 다중 구간 합계를 보존하므로 단계 상한 때문에 계획 뒷부분이 잘리지
않습니다. EVENT/NETWORK/PROPORTION 및 시간/TSP-TW 연산도 실행 계층에 포함되지만 현재
100문항 MCQ가 모두 이를 직접 평가하지는 않습니다. 논문의 SFT+DPO와 임베딩 검색은
구현하지 않았으며, Kakao 데이터로 평가하므로 논문 수치 재현을 주장하지 않습니다.

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

`search_place`, `geocode`, `reverse_geocode`, `nearby_search`, `place_details`, `directions` 및 이를 사용하는 `travel_time`은 항상 캐시를 먼저 조회합니다. 경유지 경로와 turn-by-turn guide도 같은 SQLite 캐시에 저장됩니다. 기본 DB는 `data/kakao_cache.db`, 기본 TTL은 24시간입니다. `.env`에서 다음 값을 바꿀 수 있습니다.

장소/반경 검색은 Kakao Maps Local의 공식 `keyword.json`·`category.json`을 사용합니다.
자동차 경로는 Kakao Maps의 도보·대중교통 API가 아니라 Kakao Mobility의 공식
`/v1/directions`를 사용하며, 벤치마크에 필요한 거리·시간 요약만 요청합니다.

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

`.env`에 LLM과 `KAKAO_REST_API_KEY`를 입력합니다. 하나의 Kakao REST API 키를 Local API와 Mobility API에 함께 사용합니다. `LLM_BASE_URL`을 비워두면 OpenAI 기본 endpoint를 사용하고, 입력하면 OpenAI 호환 Chat Completions endpoint를 사용합니다. Temperature는 두 upstream과 동일하게 `LLM_TEMPERATURE=0`으로 전달합니다(전달하지 않으면 endpoint 기본값이 적용되어 동일 벤치마크의 두 실행이 11점 차이가 났습니다). 출력 토큰 상한은 전달하지 않고 vLLM 서버 설정을 따릅니다(두 upstream도 동일합니다). 서버 상한에 걸려 잘린 문항은 `llm_output_truncated`로 기록됩니다. 요청 timeout은 `LLM_TIMEOUT_SECONDS`입니다. `MAX_REASONING_STEPS`(기본 15, langchain 기본값)는 문항당 추론 단계 상한이며 ReAct의 loop 반복 수와 Spatial-Agent 그래프의 노드 수에 함께 적용됩니다. `BENCHMARK_CONCURRENCY`의 기본값은 `4`이며, 각 worker가 독립 LLM 클라이언트·에이전트·Kakao provider를 사용해 네 문항을 동시에 처리합니다.

## 실행

```bash
python main.py --agent react
python main.py --agent spatial
python main.py --agent both
python main.py --agent spatial --concurrency 4
```

일부 문항만 실행할 수 있습니다.

```bash
python main.py --agent both --ids seoul_mapqa_v0_000907 seoul_mapqa_v0_000009
```

모든 실행에 `KAKAO_REST_API_KEY`가 필요합니다. provider 선택 옵션은 없으며
`KakaoMapProvider`가 항상 사용됩니다.

로그 생성 방식은 Spatial-Agent 원본과 같습니다. 각 문항의 실행 trace는
`logs/<UTC>_id<문항ID>_<질문-slug>.log`에 기록되고, 배치가 끝나면
`reports/test_<UTC>.json` 보고서 하나가 생성됩니다. 보고서는 `metadata`, `statistics`,
`results`로 구성됩니다. 입력 JSONL, 에이전트 출력, 문항 로그와 보고서의
`correct_answer`/`predicted_option`은 모두 0-based입니다. 로그에는 API 키가 들어가지
않습니다.

## 평가 항목

- 전체 및 classification별 multiple-choice accuracy
- tool-call, Kakao API-call, reasoning-step 수
- SQLite cache hit/miss 수
- latency와 failure 수
- 문항별 normalized arguments, 실행 상태, 예측값

`dataset/seoul_mapeval_v1_mcq_100.jsonl`이 평가 데이터셋입니다. OSM 기반 서울 pool인
`dataset/seoul_mapeval_v1.json`에서 seed `20260818`으로 template별 quota를 두고 100문항을 뽑았고,
행마다 anchor가 서로 다르며 선택지는 문항 id로 시드된 순서로 섞여 있습니다(원본 생성기가 거리
선택지를 오름차순 고정으로 만들어 `distance_between` 전체 정답이 index 2였습니다).
이 파일의 `context` 필드는 초기 context 실험의 출처 보존용 레거시 메타데이터이며 현재 실행에서는
사용하지 않습니다. 보고서 `metadata.provider`에는 항상 `kakao`가 기록됩니다.

## 테스트

일반 테스트는 Kakao API를 mock하므로 키나 네트워크가 필요 없습니다.

```bash
pytest
ruff check .
```

원본 코드와의 대응 및 의도적인 차이는 `docs/REFERENCE_MAPPING.md`에 정리되어 있습니다.
