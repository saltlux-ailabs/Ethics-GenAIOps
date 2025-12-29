# -*- coding: utf-8 -*-
"""Full Evaluation API 라우터

Benchmark와 Report를 순차적으로 실행하는 End-to-End 평가 엔드포인트
결과는 Leaderboard API를 통해 저장됩니다.
"""
import asyncio
from typing import Optional, List, Any, Dict

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, model_validator

from api.models.schemas import (
    ModeType,
    ProviderConfig,
    ProgressInfo
)
from api.services.session_manager import SessionManager
from api.services.full_evaluation_runner import FullEvaluationRunner
from api.services.leaderboard_client import leaderboard_client


# ============================================================================
# Full Evaluation 전용 스키마
# ============================================================================

class RunFullEvaluationRequest(BaseModel):
    """Full Evaluation 실행 요청

    Benchmark와 Report 모두에 동일한 설정이 적용됩니다.

    Attributes:
        mode: API 모드 (openai 또는 custom)
        provider: API 제공자 설정
        max_workers: 병렬 처리 worker 수 (기본값: 4)
        metadata: 사용자 정의 메타데이터 (자유 형식)
        email: 사용자 이메일 (metadata에서 자동 추출)
    """
    mode: ModeType = Field(..., description="API 모드 (openai/custom)")
    provider: ProviderConfig = Field(..., description="API 제공자 설정")
    max_workers: int = Field(4, ge=1, le=16, description="병렬 처리 worker 수")
    metadata: Optional[Dict[str, Any]] = Field(None, description="사용자 정의 메타데이터 (자유 형식)")
    email: Optional[str] = Field(None, description="사용자 이메일")

    @model_validator(mode='before')
    @classmethod
    def extract_email_from_metadata(cls, data: Any) -> Any:
        """metadata에서 email을 추출하여 최상단 필드로 이동"""
        if isinstance(data, dict):
            metadata = data.get('metadata')
            if metadata and isinstance(metadata, dict) and 'email' in metadata:
                if not data.get('email'):
                    data['email'] = metadata.pop('email')
                else:
                    metadata.pop('email', None)
        return data


class FullEvaluationStartResponse(BaseModel):
    """Full Evaluation 실행 시작 응답

    Attributes:
        session_id: 세션 ID
        message: 시작 메시지
        phases: 실행 단계 목록
    """
    session_id: str
    message: str
    phases: List[str] = ["benchmark", "report"]


class FullEvaluationStatusResponse(BaseModel):
    """Full Evaluation 상태 조회 응답

    Attributes:
        session_id: 세션 ID
        current_phase: 현재 실행 단계 (benchmark/report/completed/error)
        benchmark_status: Benchmark 상태
        report_status: Report 상태
        benchmark_progress: Benchmark 진행 상황
        report_progress: Report 진행 상황
        error_message: 에러 메시지 (에러 시)
    """
    session_id: str
    current_phase: str
    benchmark_status: Optional[str] = None
    report_status: Optional[str] = None
    benchmark_progress: Optional[ProgressInfo] = None
    report_progress: Optional[ProgressInfo] = None
    error_message: Optional[str] = None


# ============================================================================
# 라우터 초기화
# ============================================================================

router = APIRouter(prefix="/full-evaluation", tags=["Full Evaluation"])

# 서비스 인스턴스
session_manager = SessionManager()
full_evaluation_runner = FullEvaluationRunner(session_manager)


# ============================================================================
# 엔드포인트
# ============================================================================

@router.post(
    "/{session_id}/run",
    response_model=FullEvaluationStartResponse,
    summary="Full Evaluation 실행",
    description="""
Benchmark와 Report를 순차적으로 백그라운드에서 실행합니다.
결과는 Leaderboard API를 통해 저장됩니다.

## 실행 순서
1. **Leaderboard 생성**: POST /leaderboards 호출하여 초기 상태 저장
2. **Benchmark 실행**: Toxicity/Safety 벤치마크 평가
3. **Report 실행**: GOAT 공격 시뮬레이션 + 턴별 평가 + 보고서 생성
4. **결과 저장**: PUT /leaderboards/{id} 호출하여 최종 결과 저장

## 실패 처리
- Benchmark 실패 시: Report 건너뛰고 종료, Leaderboard 에러 상태로 업데이트

## 동일한 설정 적용
- mode, provider, max_workers는 Benchmark와 Report 모두에 동일하게 적용됩니다.

## metadata에서 email 자동 추출
- metadata 내에 email이 포함된 경우 자동으로 최상단 필드로 추출됩니다.
"""
)
async def run_full_evaluation(
    session_id: str,
    request: RunFullEvaluationRequest
) -> FullEvaluationStartResponse:
    """Full Evaluation 실행 엔드포인트

    Args:
        session_id: 세션 ID
        request: Full Evaluation 실행 요청

    Returns:
        FullEvaluationStartResponse: 실행 시작 응답

    Raises:
        HTTPException: 이미 실행 중 (409)
    """
    # 현재 상태 확인 - Leaderboard API를 통해 확인
    try:
        existing_data = await leaderboard_client.get_leaderboard(session_id)
        if existing_data:
            current_status = existing_data.get("status")
            if current_status in ["init", "running"]:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Evaluation already running for session: {session_id}"
                )
    except HTTPException:
        raise
    except Exception:
        # Leaderboard API 연결 실패 등은 무시하고 진행
        pass

    # provider 설정 구성
    provider_config = {
        "model": request.provider.model
    }
    if request.provider.url:
        provider_config["url"] = request.provider.url

    # 비동기 백그라운드 태스크 실행
    asyncio.create_task(
        full_evaluation_runner.run_full_evaluation_async(
            session_id,
            request.mode.value,
            provider_config,
            request.max_workers,
            request.metadata,
            request.email  # email 전달
        )
    )

    return FullEvaluationStartResponse(
        session_id=session_id,
        message="Full Evaluation이 백그라운드에서 시작되었습니다. (Benchmark → Report) 결과는 Leaderboard API를 통해 저장됩니다.",
        phases=["benchmark", "report"]
    )


@router.get(
    "/{session_id}/status",
    response_model=FullEvaluationStatusResponse,
    summary="Full Evaluation 상태 조회",
    description="""
Full Evaluation의 현재 상태를 Leaderboard API를 통해 조회합니다.

## 상태 흐름
1. `init` - 초기화
2. `running` - 실행 중 (Benchmark 또는 Report)
3. `finished` - 전체 완료
4. `error` - 오류 발생

## 응답 필드
- `current_phase`: 현재 실행 단계
- `benchmark_status`: Benchmark 상태 (init/running/finished/error)
- `status`: 전체 상태 (init/running/finished/error)
"""
)
async def get_full_evaluation_status(session_id: str) -> FullEvaluationStatusResponse:
    """Full Evaluation 상태 조회 엔드포인트

    Args:
        session_id: 세션 ID

    Returns:
        FullEvaluationStatusResponse: 현재 상태

    Raises:
        HTTPException: 세션 없음 (404)
    """
    # Leaderboard API에서 상태 조회
    try:
        data = await leaderboard_client.get_leaderboard(session_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to connect to Leaderboard API: {str(e)}"
        )

    if data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Full Evaluation session not found: {session_id}"
        )

    # 상태 추출
    overall_status = data.get("status")
    benchmark_status = data.get("benchmark_status")
    error_message = data.get("error_message")

    # 현재 단계 결정
    current_phase = _determine_current_phase(overall_status, benchmark_status)

    # 진행 상황 파싱
    benchmark_progress = None
    benchmark_progress_data = data.get("benchmark_progress")
    if benchmark_progress_data:
        benchmark_progress = ProgressInfo(
            total=benchmark_progress_data.get("total", 0),
            completed=benchmark_progress_data.get("current", 0),
            failed=0
        )

    return FullEvaluationStatusResponse(
        session_id=session_id,
        current_phase=current_phase,
        benchmark_status=benchmark_status,
        report_status=overall_status,
        benchmark_progress=benchmark_progress,
        error_message=error_message
    )


def _determine_current_phase(overall_status: Optional[str], benchmark_status: Optional[str]) -> str:
    """현재 실행 단계 결정

    Args:
        overall_status: 전체 상태
        benchmark_status: Benchmark 상태

    Returns:
        str: 현재 단계 (init/benchmark/report/completed/error)
    """
    # 에러 상태 확인
    if overall_status == "error":
        return "error"

    # 초기 상태
    if overall_status == "init":
        return "init"

    # Benchmark 실행 중
    if benchmark_status in ["init", "running"]:
        return "benchmark"

    # Benchmark 완료 & 전체 실행 중 = Report 진행 중
    if benchmark_status == "finished" and overall_status == "running":
        return "report"

    # 전체 완료
    if overall_status == "finished":
        return "completed"

    # 기타
    return "unknown"
