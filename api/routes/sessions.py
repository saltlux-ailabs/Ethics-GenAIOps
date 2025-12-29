# -*- coding: utf-8 -*-
"""세션 관리 API 라우터

세션 생성, 평가 실행, 상태 조회, 결과 조회 엔드포인트
"""
from datetime import datetime
from typing import List

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from api.models.schemas import (
    SessionStatus,
    ReportStatus,
    CreateSessionRequest,
    CreateSessionResponse,
    RunEvaluationRequest,
    StatusResponse,
    ResultsResponse,
    ReportStatusResponse,
    ProgressInfo
)
from api.services.session_manager import SessionManager
from api.services.evaluation_runner import EvaluationRunner
from api.services.report_generator import ReportGenerator
from api.services.mongodb_service import mongodb_service

# 라우터 초기화
router = APIRouter(prefix="/sessions", tags=["Sessions"])

# 서비스 인스턴스
session_manager = SessionManager()
evaluation_runner = EvaluationRunner(session_manager)
report_generator = ReportGenerator(session_manager)


@router.post(
    "/",
    response_model=CreateSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="세션 생성",
    description="새로운 평가 세션을 생성합니다. session_id를 지정하지 않으면 자동 생성됩니다."
)
async def create_session(request: CreateSessionRequest) -> CreateSessionResponse:
    """세션 생성 엔드포인트

    Args:
        request: 세션 생성 요청 (session_id 선택사항)

    Returns:
        CreateSessionResponse: 생성된 세션 정보

    Raises:
        HTTPException: 세션 ID 중복 시 409 Conflict
    """
    try:
        session_id = session_manager.create_session(request.session_id)
        return CreateSessionResponse(
            session_id=session_id,
            status=SessionStatus.INIT,
            created_at=datetime.now()
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )


@router.post(
    "/{session_id}/report",
    response_model=StatusResponse,
    summary="평가 실행",
    description="지정된 세션에서 GOAT 공격 평가를 백그라운드로 실행합니다."
)
async def generate_report(
    session_id: str,
    request: RunEvaluationRequest,
    background_tasks: BackgroundTasks
) -> StatusResponse:
    """평가 실행 엔드포인트

    Args:
        session_id: 세션 ID
        request: 평가 실행 요청 (mode, provider, max_workers)
        background_tasks: FastAPI 백그라운드 태스크

    Returns:
        StatusResponse: 실행 시작 응답

    Raises:
        HTTPException: 세션 없음 (404), 이미 실행 중 (409)
    """
    # 세션 존재 확인
    if not session_manager.session_exists(session_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}"
        )

    # 현재 상태 확인
    try:
        status_data = session_manager.get_status(session_id)
        current_status = status_data.get("status")

        # 이미 실행 중이거나 완료된 경우
        if current_status in [
            SessionStatus.CREATE_CONVERSATION.value,
            SessionStatus.EVALUATION.value
        ]:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Evaluation already running for session: {session_id}"
            )

    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session status not found: {session_id}"
        )

    # provider 설정 구성
    provider_config = {
        "model": request.provider.model
    }
    if request.provider.url:
        provider_config["url"] = request.provider.url

    # 백그라운드 태스크 등록
    background_tasks.add_task(
        evaluation_runner.run_evaluation_sync,
        session_id,
        request.mode.value,
        provider_config,
        request.max_workers,
        request.metadata,
        request.email  # email 전달
    )

    return StatusResponse(
        session_id=session_id,
        status=SessionStatus.CREATE_CONVERSATION,
        message="평가가 백그라운드에서 시작되었습니다."
    )


@router.get(
    "/{session_id}/report/status",
    response_model=ReportStatusResponse,
    summary="보고서 생성 상태 조회",
    description="보고서 생성의 현재 상태와 진행 상황을 조회합니다."
)
async def get_report_status(session_id: str) -> ReportStatusResponse:
    """보고서 생성 상태 조회 엔드포인트

    Args:
        session_id: 세션 ID

    Returns:
        ReportStatusResponse: 보고서 생성 상태 정보

    Raises:
        HTTPException: 세션 없음 (404)
    """
    # MongoDB에서 세션 데이터 확인
    session_data = mongodb_service.get_results(session_id)

    if session_data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found in MongoDB: {session_id}"
        )

    # 보고서 상태 확인
    report_status = session_data.get("report_status", "init")
    report_progress = session_data.get("report_progress")
    report_error = session_data.get("report_error")

    # 진행 상황 파싱
    progress = None
    if report_progress:
        progress = ProgressInfo(**report_progress)

    return ReportStatusResponse(
        session_id=session_id,
        report_status=ReportStatus(report_status),
        report_progress=progress,
        report_error=report_error
    )


@router.get(
    "/{session_id}/status",
    response_model=StatusResponse,
    summary="상태 조회",
    description="세션의 현재 상태와 진행 상황을 조회합니다."
)
async def get_session_status(session_id: str) -> StatusResponse:
    """상태 조회 엔드포인트

    Args:
        session_id: 세션 ID

    Returns:
        StatusResponse: 현재 상태 정보

    Raises:
        HTTPException: 세션 없음 (404)
    """
    if not session_manager.session_exists(session_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}"
        )

    try:
        status_data = session_manager.get_status(session_id)

        # 진행 상황 파싱
        progress = None
        if "progress" in status_data:
            progress = ProgressInfo(**status_data["progress"])

        return StatusResponse(
            session_id=session_id,
            status=SessionStatus(status_data["status"]),
            progress=progress,
            error_message=status_data.get("error_message")
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session status not found: {session_id}"
        )


@router.get(
    "/{session_id}/results",
    response_model=ResultsResponse,
    summary="결과 조회",
    description="평가가 완료된 세션의 결과를 조회합니다. 상태가 'finished'일 때만 결과를 반환합니다."
)
async def get_session_results(session_id: str) -> ResultsResponse:
    """결과 조회 엔드포인트

    Args:
        session_id: 세션 ID

    Returns:
        ResultsResponse: 평가 결과 (finished일 때만 results/summary 포함)

    Raises:
        HTTPException: 세션 없음 (404)
    """
    if not session_manager.session_exists(session_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}"
        )

    try:
        status_data = session_manager.get_status(session_id)
        current_status = SessionStatus(status_data["status"])

        # finished 상태가 아닌 경우
        if current_status != SessionStatus.FINISHED:
            status_messages = {
                SessionStatus.INIT: "평가가 아직 시작되지 않았습니다.",
                SessionStatus.CREATE_CONVERSATION: "대화 생성 중입니다.",
                SessionStatus.EVALUATION: "평가 진행 중입니다.",
                SessionStatus.ERROR: f"평가 중 오류가 발생했습니다: {status_data.get('error_message', 'Unknown error')}"
            }

            return ResultsResponse(
                session_id=session_id,
                status=current_status,
                error="Results not available",
                message=status_messages.get(current_status, "평가가 진행 중입니다.")
            )

        # 결과 로드
        results = session_manager.get_results(session_id)

        if results is None:
            return ResultsResponse(
                session_id=session_id,
                status=current_status,
                error="Results file not found",
                message="결과 파일을 찾을 수 없습니다."
            )

        return ResultsResponse(
            session_id=session_id,
            status=current_status,
            results=results.get("results"),
            evaluation_results=results.get("evaluation_results")
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}"
        )


@router.get(
    "/",
    response_model=List[dict],
    summary="세션 목록 조회",
    description="모든 세션의 목록을 조회합니다."
)
async def list_sessions() -> List[dict]:
    """세션 목록 조회 엔드포인트

    Returns:
        List[dict]: 세션 목록 (session_id, status, created_at, updated_at)
    """
    return session_manager.list_sessions()
