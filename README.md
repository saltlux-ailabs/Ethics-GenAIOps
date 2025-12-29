# GOAT Attack Evaluation API Server

AI 모델의 윤리성을 평가하는 FastAPI 기반 백엔드 시스템입니다.
GOAT(Generative Offensive Agent Tester) 방식을 활용하여 LLM의 안전성과 유해성을 종합적으로 평가합니다.

## 주요 기능

- **Benchmark 평가**: Toxicity/Safety 벤치마크를 통한 정량적 평가
- **GOAT 공격 시뮬레이션**: 다양한 공격 기법을 사용한 레드팀 테스트
- **턴별 평가 및 보고서 생성**: LLM 기반 상세 평가 리포트
- **Full Evaluation**: Benchmark + Report 통합 평가 파이프라인
- **세션 관리**: 평가 세션 생성, 상태 추적, 결과 저장
- **Leaderboard 연동**: 평가 결과 저장 및 비교

## 기술 스택

- **Framework**: FastAPI
- **Database**: MongoDB
- **LLM**: OpenAI API / Custom OpenAI-compatible API
- **Language**: Python 3.10+

## 프로젝트 구조

```
api-server-v2/
├── main_api.py              # FastAPI 앱 진입점
├── core.py                  # LLM 통신 및 핵심 로직
├── requirements.txt         # Python 의존성
├── docker-compose.yaml      # Docker 설정
│
├── api/
│   ├── models/
│   │   └── schemas.py       # Pydantic 스키마 정의
│   ├── routes/
│   │   ├── sessions.py      # 세션 관리 API
│   │   ├── config.py        # 설정 관리 API
│   │   ├── benchmark.py     # 벤치마크 API
│   │   └── full_evaluation.py  # Full Evaluation API
│   └── services/
│       ├── session_manager.py      # 세션 상태 관리
│       ├── benchmark_runner.py     # 벤치마크 실행
│       ├── evaluation_runner.py    # GOAT 평가 실행
│       ├── full_evaluation_runner.py  # 통합 평가 실행
│       ├── report_generator.py     # 보고서 생성
│       ├── leaderboard_client.py   # Leaderboard API 클라이언트
│       └── mongodb_service.py      # MongoDB 연동
│
├── config/
│   ├── goat_config.yaml     # GOAT 카테고리/질문 설정
│   └── prompts/             # 시스템 프롬프트 템플릿
│
├── resource/
│   ├── benchmark/           # 벤치마크 데이터셋
│   │   ├── toxicity_fmt/    # Toxicity 벤치마크 (easy/medium/hard)
│   │   └── safety_fmt/      # Safety 벤치마크 (easy/medium/hard)
│   └── sessions/            # 세션별 결과 저장
│
└── docs/
    ├── API_SPECIFICATION.md    # API 명세서 (Markdown)
    └── API_SPECIFICATION.docx  # API 명세서 (Word)
```

## 설치 및 실행

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd api-server-v2

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
# .env 파일 생성
OPENAI_API_KEY=your-openai-api-key
MONGODB_URI=mongodb://localhost:27017/
LEADERBOARD_API_URL=http://localhost:8080
LEADERBOARD_PASSWORD=your-password
```

### 3. 서버 실행

```bash
# 개발 모드 (자동 리로드)
uvicorn main_api:app --host 0.0.0.0 --port 8000 --reload

# 프로덕션 모드
uvicorn main_api:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4. Docker 실행 (선택)

```bash
docker-compose up -d
```

## API 엔드포인트

### 기본

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/` | API 정보 |
| GET | `/health` | 헬스 체크 |
| GET | `/docs` | Swagger UI |

### 세션 관리 (`/sessions`)

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/sessions/` | 세션 생성 |
| POST | `/sessions/{id}/run` | 평가 실행 |
| POST | `/sessions/{id}/report` | 보고서 생성 |
| GET | `/sessions/{id}/status` | 상태 조회 |
| GET | `/sessions/{id}/results` | 결과 조회 |

### 벤치마크 (`/benchmark`)

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/benchmark/{id}/run` | 벤치마크 실행 |
| GET | `/benchmark/{id}/status` | 상태 조회 |
| GET | `/benchmark/{id}/results` | 결과 조회 |

### Full Evaluation (`/full-evaluation`)

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/full-evaluation/{id}/run` | 통합 평가 실행 |
| GET | `/full-evaluation/{id}/status` | 상태 조회 |

### 설정 (`/config`)

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET/PUT | `/config/prompts/{name}` | 프롬프트 관리 |
| GET/PUT | `/config/goat` | GOAT 설정 관리 |
| GET/PUT | `/config/mongodb` | MongoDB 설정 |

## 사용 예시

### 세션 생성 및 평가 실행

```bash
# 1. 세션 생성
curl -X POST http://localhost:8000/sessions/ \
  -H "Content-Type: application/json" \
  -d '{"session_id": "my-test-001"}'

# 2. 평가 실행 (OpenAI 모드)
curl -X POST http://localhost:8000/sessions/my-test-001/run \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "openai",
    "provider": {"model": "gpt-4o-mini"},
    "max_workers": 4
  }'

# 3. 상태 확인
curl http://localhost:8000/sessions/my-test-001/status

# 4. 결과 조회
curl http://localhost:8000/sessions/my-test-001/results
```

### Full Evaluation (Benchmark + Report)

```bash
# 통합 평가 실행
curl -X POST http://localhost:8000/full-evaluation/eval-001/run \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "custom",
    "provider": {
      "model": "luxia3-deep-32b",
      "url": "http://api.example.com/v1/chat/completions"
    },
    "max_workers": 8,
    "metadata": {
      "experiment": "v1.0",
      "notes": "Initial evaluation"
    }
  }'

# 상태 확인
curl http://localhost:8000/full-evaluation/eval-001/status
```

## 벤치마크 메트릭

### Toxicity Benchmark
- **평가 대상**: User 메시지의 유해성
- **Label**: 0-5 (유해성 점수), HD (판정 불가)
- **메트릭**: Accuracy, Precision, Recall, F1 Score

### Safety Benchmark
- **평가 대상**: Assistant 응답의 안전성
- **Label**: SAFE, UNSAFE
- **메트릭**: Accuracy, Precision, Recall, F1 Score

## 상태 흐름

### 세션 상태
```
init → create_conversation → evaluation → finished
                                       ↘ error
```

### Full Evaluation 상태
```
Phase 1: Benchmark
  init → running → finished (또는 error)
                      ↓
Phase 2: Report (Benchmark 성공 시)
  create_conversation → evaluation → turn_evaluation → report_generation → finished
```

## 문서

- [API 명세서 (Markdown)](docs/API_SPECIFICATION.md)
- [API 명세서 (Word)](docs/API_SPECIFICATION.docx)
- [Swagger UI](http://localhost:8000/docs) (서버 실행 후)
- [ReDoc](http://localhost:8000/redoc) (서버 실행 후)

## 라이선스

Copyright (c) 2025 Saltlux. All rights reserved.
