# -*- coding: utf-8 -*-
"""벤치마크 API 라우터

자체 벤치마크 평가 엔드포인트
- Toxicity 벤치마크: user 메시지 유해성 평가 (0-5, HD)
- Safety 벤치마크: assistant 응답 안전성 평가 (SAFE/UNSAFE)
"""
from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from api.models.schemas import (
    BenchmarkStatus,
    RunBenchmarkRequest,
    BenchmarkStatusResponse,
    BenchmarkResultsResponse,
    ProgressInfo
)
from api.services.session_manager import SessionManager
from api.services.benchmark_runner import BenchmarkRunner
from api.services.mongodb_service import mongodb_service

# 라우터 초기화
router = APIRouter(prefix="/benchmark", tags=["Benchmark"])

# 서비스 인스턴스
session_manager = SessionManager()
benchmark_runner = BenchmarkRunner(session_manager)


@router.post(
    "/{session_id}/run",
    response_model=BenchmarkStatusResponse,
    summary="벤치마크 실행",
    description="""
지정된 세션에서 자체 벤치마크 평가를 백그라운드로 실행합니다.

## 벤치마크 종류
- **Toxicity**: user 메시지의 유해성 평가 (0-5, HD)
- **Safety**: assistant 응답의 안전성 평가 (SAFE/UNSAFE)

## 난이도
- easy, medium, hard 전체 실행

## 메트릭
- Accuracy, Precision, Recall, F1 Score (난이도별 및 전체)
"""
)
async def run_benchmark(
    session_id: str,
    request: RunBenchmarkRequest,
    background_tasks: BackgroundTasks
) -> BenchmarkStatusResponse:
    """벤치마크 실행 엔드포인트

    Args:
        session_id: 세션 ID
        request: 벤치마크 실행 요청 (mode, provider, max_workers)
        background_tasks: FastAPI 백그라운드 태스크

    Returns:
        BenchmarkStatusResponse: 실행 시작 응답

    Raises:
        HTTPException: 이미 실행 중 (409)
    """
    # 현재 상태 확인
    current_data = mongodb_service.get_benchmark_status(session_id)
    if current_data:
        current_status = current_data.get("benchmark_status")
        if current_status == "running":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Benchmark already running for session: {session_id}"
            )

    # 초기 상태 설정
    mongodb_service.update_benchmark_status(
        session_id,
        status="init"
    )

    # provider 설정 구성
    provider_config = {
        "model": request.provider.model
    }
    if request.provider.url:
        provider_config["url"] = request.provider.url

    # 백그라운드 태스크 등록
    background_tasks.add_task(
        benchmark_runner.run_benchmark_sync,
        session_id,
        request.mode.value,
        provider_config,
        request.max_workers,
        request.metadata,
        request.email  # email 전달
    )

    return BenchmarkStatusResponse(
        session_id=session_id,
        benchmark_status=BenchmarkStatus.RUNNING,
        message="벤치마크가 백그라운드에서 시작되었습니다."
    )


@router.get(
    "/{session_id}/status",
    response_model=BenchmarkStatusResponse,
    summary="벤치마크 상태 조회",
    description="벤치마크 실행 상태를 조회합니다."
)
async def get_benchmark_status(session_id: str) -> BenchmarkStatusResponse:
    """벤치마크 상태 조회 엔드포인트

    Args:
        session_id: 세션 ID

    Returns:
        BenchmarkStatusResponse: 현재 상태

    Raises:
        HTTPException: 세션 없음 (404)
    """
    data = mongodb_service.get_benchmark_status(session_id)

    if not data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark session not found: {session_id}"
        )

    # 상태 변환
    status_str = data.get("benchmark_status", "init")
    try:
        benchmark_status = BenchmarkStatus(status_str)
    except ValueError:
        benchmark_status = BenchmarkStatus.INIT

    # 진행 정보
    progress = None
    progress_data = data.get("benchmark_progress")
    if progress_data:
        progress = ProgressInfo(
            total=progress_data.get("total", 0),
            completed=progress_data.get("completed", 0),
            failed=progress_data.get("failed", 0)
        )

    return BenchmarkStatusResponse(
        session_id=session_id,
        benchmark_status=benchmark_status,
        progress=progress,
        error_message=data.get("benchmark_error")
    )


@router.get(
    "/{session_id}/results",
    response_model=BenchmarkResultsResponse,
    summary="벤치마크 결과 조회",
    description="""
벤치마크 결과를 조회합니다.

## 결과 구조
```json
{
  "toxicity": {
    "easy": { "items": [...], "metrics": {...} },
    "medium": { "items": [...], "metrics": {...} },
    "hard": { "items": [...], "metrics": {...} },
    "overall_metrics": {...}
  },
  "safety": {
    "easy": { "items": [...], "metrics": {...} },
    "medium": { "items": [...], "metrics": {...} },
    "hard": { "items": [...], "metrics": {...} },
    "overall_metrics": {...}
  }
}
```

## 각 item 구조
```json
{
  "index": 0,
  "messages": [...],
  "label": "SAFE",
  "prediction": "SAFE",
  "correct": true
}
```
"""
)
async def get_benchmark_results(session_id: str) -> BenchmarkResultsResponse:
    """벤치마크 결과 조회 엔드포인트

    Args:
        session_id: 세션 ID

    Returns:
        BenchmarkResultsResponse: 벤치마크 결과

    Raises:
        HTTPException: 세션 없음 (404)
    """
    data = mongodb_service.get_benchmark_results(session_id)

    if not data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark session not found: {session_id}"
        )

    # 상태 변환
    status_str = data.get("benchmark_status", "init")
    try:
        benchmark_status = BenchmarkStatus(status_str)
    except ValueError:
        benchmark_status = BenchmarkStatus.INIT

    # 상태에 따른 응답
    if benchmark_status == BenchmarkStatus.FINISHED:
        return BenchmarkResultsResponse(
            session_id=session_id,
            benchmark_status=benchmark_status,
            benchmark_results=data.get("benchmark_results"),
            elapsed_time_seconds=data.get("benchmark_elapsed_time_seconds"),
            message="벤치마크가 완료되었습니다."
        )
    elif benchmark_status == BenchmarkStatus.ERROR:
        return BenchmarkResultsResponse(
            session_id=session_id,
            benchmark_status=benchmark_status,
            error=data.get("benchmark_error"),
            message="벤치마크 실행 중 오류가 발생했습니다."
        )
    elif benchmark_status == BenchmarkStatus.RUNNING:
        return BenchmarkResultsResponse(
            session_id=session_id,
            benchmark_status=benchmark_status,
            message="벤치마크가 실행 중입니다. 완료 후 다시 조회해주세요."
        )
    else:
        return BenchmarkResultsResponse(
            session_id=session_id,
            benchmark_status=benchmark_status,
            message="벤치마크가 아직 시작되지 않았습니다."
        )

