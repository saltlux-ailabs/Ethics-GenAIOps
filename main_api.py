# -*- coding: utf-8 -*-
"""GOAT Attack Evaluation API Server

FastAPI 기반 AI 모델 윤리성 평가 시스템 백엔드

Usage:
    uvicorn main_api:app --host 0.0.0.0 --port 8000 --reload
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.routes.sessions import router as sessions_router
from api.routes.config import router as config_router
from api.routes.benchmark import router as benchmark_router
from api.routes.full_evaluation import router as full_evaluation_router
from api.models.schemas import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 라이프사이클 관리

    시작 시 초기화, 종료 시 정리 작업을 수행합니다.
    """
    # 시작 시
    print("=" * 60)
    print("GOAT Attack Evaluation API Server Starting...")
    print("=" * 60)

    # resource 디렉토리 확인/생성
    from pathlib import Path
    resource_path = Path("./resource/sessions")
    resource_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Sessions directory ready: {resource_path.absolute()}")

    print("✓ API Server ready")
    print("=" * 60)

    yield  # 애플리케이션 실행

    # 종료 시
    print("\n" + "=" * 60)
    print("GOAT Attack Evaluation API Server Shutting down...")
    print("=" * 60)


# FastAPI 앱 초기화
app = FastAPI(
    title="GOAT Attack Evaluation API",
    description="""
## AI 모델 윤리성 평가 시스템

GOAT (Generative Offensive Agent Tester) 알고리즘을 활용한 AI 모델 윤리성 평가 API입니다.

### 주요 기능

- **세션 관리**: 평가 세션 생성 및 관리
- **평가 실행**: 백그라운드에서 GOAT 공격 시뮬레이션 실행
- **상태 조회**: 실시간 평가 진행 상황 확인
- **결과 조회**: 완료된 평가 결과 조회

### 평가 모드

- **OpenAI 모드**: OpenAI API를 사용하여 평가 (gpt-4o-mini 등)
- **Custom 모드**: 외부 OpenAI 호환 API 엔드포인트 사용

### 평가 지표

- **Toxicity Score**: User 메시지의 유해성 점수 (0-5, HD)
- **Safety Score**: Assistant 응답의 안전성 (SAFE/UNSAFE)
    """,
    version="1.0.0",
    contact={
        "name": "Saltlux AI Team",
        "url": "https://saltlux.com",
    },
    license_info={
        "name": "Proprietary",
    },
    lifespan=lifespan
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(sessions_router)
app.include_router(config_router)
app.include_router(benchmark_router)
app.include_router(full_evaluation_router)


@app.get(
    "/",
    summary="API 정보",
    description="API 서버 기본 정보를 반환합니다."
)
async def root():
    """API 루트 엔드포인트"""
    return {
        "name": "GOAT Attack Evaluation API",
        "version": "1.0.0",
        "description": "AI 모델 윤리성 평가 시스템",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="헬스 체크",
    description="서버 상태를 확인합니다."
)
async def health_check() -> HealthResponse:
    """헬스 체크 엔드포인트"""
    return HealthResponse(status="healthy", version="1.0.0")


@app.get(
    "/api-config",
    summary="현재 API 설정 조회",
    description="현재 API 모드 설정 상태를 조회합니다. (Target/Attacker 모델 설정)"
)
async def get_api_config():
    """현재 API 설정 조회 엔드포인트"""
    from core import get_current_config
    return get_current_config()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
