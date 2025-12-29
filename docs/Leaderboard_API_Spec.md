# Leaderboard API 명세서

## 기본 정보

| 항목 | 내용 |
|------|------|
| Base URL | `{{base_url}}` |
| 인증 방식 | JWT Bearer Token (Authorization 헤더) |
| Content-Type | `application/json` |

---

## 1. 리더보드 API

### 1.1 리더보드 생성

| 항목 | 내용 |
|------|------|
| **Endpoint** | `POST /leaderboards` |
| **설명** | 새로운 리더보드 생성 |

#### Request Headers

| Header | Type | Required | Description |
|--------|------|----------|-------------|
| Content-Type | string | Y | `application/json` |
| Authorization | string | Y | JWT 토큰 |

#### Request Body

```json
{
  "email": "string (필수)",
  "status": "string (필수) - init | running | completed 등",
  "results": [],
  "evaluation_results": {
    "user_toxicity_distribution_score": 0.85,
    "assistant_safety_distribution_score": 0.92
  },
  "evaluated_dialogues": [],
  "dialogue_reports": [],
  "metadata": {
    "model": "string - 모델명",
    "model_url": "string - 모델 URL",
    "model_author": "string - 모델 제작자",
    "description": "string - 설명"
  },
  "benchmark_results": {
    "toxicity": {
      "overall_metrics": {
        "accuracy": 0.95,
        "precision": 0.93,
        "recall": 0.91,
        "f1_score": 0.92
      }
    },
    "safety": {
      "overall_metrics": {
        "accuracy": 0.88,
        "precision": 0.87,
        "recall": 0.89,
        "f1_score": 0.88
      }
    }
  },
  "benchmark_status": "string",
  "benchmark_progress": {
    "current": 100,
    "total": 100
  },
  "benchmark_elapsed_time_seconds": 120,
  "global_report": "string",
  "report_timestamp": "string (ISO 8601)"
}
```

#### Response (201 Created)

```json
{
  "_id": "string - 생성된 리더보드 ID"
}
```

---

### 1.2 리더보드 최소 파라미터 생성

| 항목 | 내용 |
|------|------|
| **Endpoint** | `POST /leaderboards` |
| **설명** | 최소 파라미터로 리더보드 생성 |

#### Request Body (최소)

```json
{
  "email": "string (필수)",
  "status": "string (필수)"
}
```

---

### 1.3 리더보드 전체 변경 (PUT)

| 항목 | 내용 |
|------|------|
| **Endpoint** | `PUT /leaderboards/{id}` |
| **설명** | 리더보드 전체 데이터 교체 |

#### Path Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| id | string | Y | 리더보드 ID (예: `slxailabs20251211_69393c78c9a559d16c5095c6`) |

#### Request Body

1.1과 동일한 구조

#### Response (200 OK)

```json
{
  "_id": "string"
}
```

---

### 1.4 전체 리더보드 조회

| 항목 | 내용 |
|------|------|
| **Endpoint** | `GET /leaderboards` |
| **설명** | 모든 리더보드 목록 조회 |

#### Request Headers

| Header | Type | Required | Description |
|--------|------|----------|-------------|
| Authorization | string | Y | JWT 토큰 |

#### Response (200 OK)

```json
[
  {
    "_id": "string",
    "email": "string",
    "status": "string",
    "metadata": {...},
    "benchmark_results": {...},
    ...
  }
]
```

---

### 1.5 완료된 리더보드 조회 (페이지네이션)

| 항목 | 내용 |
|------|------|
| **Endpoint** | `GET /leaderboards/finished` |
| **설명** | 완료 상태의 리더보드 목록 조회 |

#### Query Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| page | number | N | 0 | 페이지 번호 (0부터 시작) |
| limit | number | N | - | 페이지당 항목 수 |

#### Example

```
GET /leaderboards/finished?page=0&limit=5
```

---

### 1.6 상태별 리더보드 조회

| 항목 | 내용 |
|------|------|
| **Endpoint** | `GET /leaderboards/status/{status}` |
| **설명** | 특정 상태의 리더보드 목록 조회 |

#### Path Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| status | string | Y | 상태값 (예: `init`, `running`, `completed`) |

#### Example

```
GET /leaderboards/status/init
```

---

### 1.7 ID로 리더보드 조회

| 항목 | 내용 |
|------|------|
| **Endpoint** | `GET /leaderboards/{id}` |
| **설명** | 특정 ID의 리더보드 조회 |

#### Path Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| id | string | Y | 리더보드 ID |

#### Example

```
GET /leaderboards/69396b61431b440027108b82
```

---

### 1.8 이메일로 리더보드 조회

| 항목 | 내용 |
|------|------|
| **Endpoint** | `GET /leaderboards/email` |
| **설명** | 특정 이메일의 리더보드 목록 조회 |

#### Query Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| addr | string | Y | 이메일 주소 |

#### Example

```
GET /leaderboards/email?addr=jjlee@saltlux.com
```

---

### 1.9 리더보드 삭제

| 항목 | 내용 |
|------|------|
| **Endpoint** | `DELETE /leaderboards/{id}` |
| **설명** | 특정 리더보드 삭제 |

#### Path Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| id | string | Y | 리더보드 ID |

#### Example

```
DELETE /leaderboards/slxailabs20251211_69393c78c9a559d16c5095c6
```

---

## 2. 세션 리더 API

### 2.1 세션 생성

| 항목 | 내용 |
|------|------|
| **Endpoint** | `POST /sessionleaders` |
| **설명** | 새 세션 리더 생성 |

#### Request Headers

| Header | Type | Required | Description |
|--------|------|----------|-------------|
| Authorization | string | Y | JWT 토큰 |

---

### 2.2 세션 실행

| 항목 | 내용 |
|------|------|
| **Endpoint** | `POST /sessionleaders/run/{id}` |
| **설명** | 특정 세션의 벤치마크 실행 |

#### Path Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| id | string | Y | 세션 ID |

#### Request Body

```json
{
  "metadata": {
    "model": "string - 모델명",
    "model_url": "string - 모델 엔드포인트 URL",
    "model_author": "string - 모델 제작자",
    "email": "string - 이메일",
    "description": "string - 설명"
  },
  "mode": "string - 실행 모드 (openai | custom)",
  "provider": {
    "model": "string - 모델명",
    "url": "string - 모델 URL"
  },
  "max_workers": 4
}
```

#### Example

```
POST /sessionleaders/run/aa
```

---

## API 요약표

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/leaderboards` | 리더보드 생성 |
| PUT | `/leaderboards/{id}` | 리더보드 전체 수정 |
| GET | `/leaderboards` | 전체 리더보드 조회 |
| GET | `/leaderboards/finished` | 완료된 리더보드 조회 |
| GET | `/leaderboards/status/{status}` | 상태별 리더보드 조회 |
| GET | `/leaderboards/{id}` | ID로 리더보드 조회 |
| GET | `/leaderboards/email?addr={email}` | 이메일로 리더보드 조회 |
| DELETE | `/leaderboards/{id}` | 리더보드 삭제 |
| POST | `/sessionleaders` | 세션 생성 |
| POST | `/sessionleaders/run/{id}` | 세션 실행 |

---

## 상태값 (Status)

| 상태 | 설명 |
|------|------|
| `init` | 초기 상태 |
| `running` | 실행 중 |
| `completed` | 완료 |
| `finished` | 완료 (동의어) |
| `error` | 오류 발생 |

---

## 에러 응답

| HTTP Status | 설명 |
|-------------|------|
| 400 | Bad Request - 잘못된 요청 |
| 401 | Unauthorized - 인증 실패 |
| 404 | Not Found - 리소스 없음 |
| 409 | Conflict - 중복 또는 충돌 |
| 500 | Internal Server Error - 서버 오류 |

---

*문서 생성일: 2025-12-12*
*원본: 251211_Leaderboard API.postman_collection 1.json*
