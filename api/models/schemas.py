# -*- coding: utf-8 -*-
"""Pydantic 스키마 정의

세션 관리 및 평가 API를 위한 요청/응답 모델 정의
"""
from pydantic import BaseModel, Field, model_validator
from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime


class SessionStatus(str, Enum):
    """세션 상태 열거형"""
    INIT = "init"
    CREATE_CONVERSATION = "create_conversation"
    EVALUATION = "evaluation"
    ERROR = "error"
    FINISHED = "finished"


class ReportStatus(str, Enum):
    """보고서 생성 상태 열거형"""
    INIT = "init"
    GENERATING = "generating"
    FINISHED = "finished"
    FINISHED_WITH_ERRORS = "finished_with_errors"
    ERROR = "error"


class ModeType(str, Enum):
    """API 모드 타입"""
    OPENAI = "openai"
    CUSTOM = "custom"


class ProviderConfig(BaseModel):
    """API 제공자 설정

    Attributes:
        model: 사용할 모델명
        url: custom 모드일 때 API URL (선택사항)
    """
    model: str = Field(..., description="사용할 모델명")
    url: Optional[str] = Field(None, description="custom 모드일 때 API URL")


class CreateSessionRequest(BaseModel):
    """세션 생성 요청

    Attributes:
        session_id: 사용자 지정 세션 ID (선택사항, 없으면 자동 생성)
    """
    session_id: Optional[str] = Field(None, description="사용자 지정 세션 ID")


class CreateSessionResponse(BaseModel):
    """세션 생성 응답

    Attributes:
        session_id: 생성된 세션 ID
        status: 현재 상태 (init)
        created_at: 생성 시각
    """
    session_id: str
    status: SessionStatus
    created_at: datetime


class RunEvaluationRequest(BaseModel):
    """평가 실행 요청

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


class ProgressInfo(BaseModel):
    """진행 상황 정보

    Attributes:
        total: 전체 작업 수
        completed: 완료된 작업 수
        failed: 실패한 작업 수
    """
    total: int
    completed: int
    failed: int


class StatusResponse(BaseModel):
    """상태 조회 응답

    Attributes:
        session_id: 세션 ID
        status: 현재 상태
        progress: 진행 상황 (선택사항)
        error_message: 에러 메시지 (error 상태일 때)
        message: 추가 메시지 (선택사항)
    """
    session_id: str
    status: SessionStatus
    progress: Optional[ProgressInfo] = None
    error_message: Optional[str] = None
    message: Optional[str] = None


class ResultsResponse(BaseModel):
    """결과 조회 응답

    Attributes:
        session_id: 세션 ID
        status: 현재 상태
        results: 평가 결과 목록 (finished일 때만)
        evaluation_results: 요약 정보 (finished일 때만)
        error: 에러 메시지 (에러 시)
        message: 추가 메시지
    """
    session_id: str
    status: SessionStatus
    results: Optional[List[Dict[str, Any]]] = None
    evaluation_results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    message: Optional[str] = None


class ReportStatusResponse(BaseModel):
    """보고서 생성 상태 응답

    Attributes:
        session_id: 세션 ID
        report_status: 보고서 생성 상태
        report_progress: 진행 상황 (선택사항)
        report_error: 에러 메시지 (error 상태일 때)
        message: 추가 메시지 (선택사항)
    """
    session_id: str
    report_status: ReportStatus
    report_progress: Optional[ProgressInfo] = None
    report_error: Optional[str] = None
    message: Optional[str] = None


class HealthResponse(BaseModel):
    """헬스 체크 응답"""
    status: str = "healthy"
    version: str = "1.0.0"


# ============================================================================
# Config 관리 스키마
# ============================================================================

class PromptInfo(BaseModel):
    """프롬프트 파일 정보

    Attributes:
        name: 파일명
        size: 파일 크기 (bytes)
        modified_at: 수정 시각
    """
    name: str
    size: int
    modified_at: str


class PromptContent(BaseModel):
    """프롬프트 내용

    Attributes:
        name: 파일명
        content: 프롬프트 내용
        size: 파일 크기 (bytes)
        modified_at: 수정 시각
    """
    name: str
    content: str
    size: int
    modified_at: str


class UpdatePromptRequest(BaseModel):
    """프롬프트 수정 요청

    Attributes:
        content: 새 프롬프트 내용
    """
    content: str = Field(..., description="새 프롬프트 내용")


class UpdatePromptResponse(BaseModel):
    """프롬프트 수정 응답

    Attributes:
        name: 파일명
        size: 파일 크기 (bytes)
        modified_at: 수정 시각
        message: 결과 메시지
    """
    name: str
    size: int
    modified_at: str
    message: str


class CategoryConfig(BaseModel):
    """카테고리 설정

    Attributes:
        name: 카테고리명
        goal: 공격 목표
        questions: 질문 목록
        attack_strategy: 공격 전략
    """
    name: str = Field(..., description="카테고리명")
    goal: str = Field(..., description="공격 목표")
    questions: List[str] = Field(..., description="질문 목록")
    attack_strategy: str = Field(..., description="공격 전략")


class GoatConfig(BaseModel):
    """GOAT 전체 설정

    Attributes:
        categories: 카테고리 설정 딕셔너리
        default_goal: 기본 목표
    """
    categories: Dict[str, CategoryConfig]
    default_goal: str


class UpdateCategoryResponse(BaseModel):
    """카테고리 수정 응답

    Attributes:
        category_id: 카테고리 ID
        message: 결과 메시지
        modified_at: 수정 시각
    """
    category_id: str
    message: str
    modified_at: str


class UpdateGoatConfigResponse(BaseModel):
    """GOAT 설정 수정 응답

    Attributes:
        message: 결과 메시지
        modified_at: 수정 시각
    """
    message: str
    modified_at: str


# ============================================================================
# Benchmark 스키마
# ============================================================================

class BenchmarkStatus(str, Enum):
    """벤치마크 실행 상태 열거형"""
    INIT = "init"
    RUNNING = "running"
    FINISHED = "finished"
    ERROR = "error"


class RunBenchmarkRequest(BaseModel):
    """벤치마크 실행 요청

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


class BenchmarkMetrics(BaseModel):
    """벤치마크 메트릭

    Attributes:
        accuracy: 정확도
        precision: 정밀도
        recall: 재현율
        f1_score: F1 점수
        total_samples: 전체 샘플 수
        hd_excluded: HD로 제외된 샘플 수 (toxicity 전용)
    """
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    total_samples: int
    hd_excluded: int = 0


class BenchmarkStatusResponse(BaseModel):
    """벤치마크 상태 조회 응답

    Attributes:
        session_id: 세션 ID
        benchmark_status: 벤치마크 실행 상태
        progress: 진행 상황 (선택사항)
        error_message: 에러 메시지 (error 상태일 때)
        message: 추가 메시지 (선택사항)
    """
    session_id: str
    benchmark_status: BenchmarkStatus
    progress: Optional[ProgressInfo] = None
    error_message: Optional[str] = None
    message: Optional[str] = None


class BenchmarkResultsResponse(BaseModel):
    """벤치마크 결과 조회 응답

    Attributes:
        session_id: 세션 ID
        benchmark_status: 벤치마크 실행 상태
        benchmark_results: 벤치마크 결과 (finished일 때만)
        elapsed_time_seconds: 소요 시간
        error: 에러 메시지 (에러 시)
        message: 추가 메시지
    """
    session_id: str
    benchmark_status: BenchmarkStatus
    benchmark_results: Optional[Dict[str, Any]] = None
    elapsed_time_seconds: Optional[float] = None
    error: Optional[str] = None
    message: Optional[str] = None


# ============================================================================
# MongoDB 설정 스키마
# ============================================================================

class MongoDBConfigRequest(BaseModel):
    """MongoDB 설정 변경 요청

    Attributes:
        mongo_uri: MongoDB 연결 URI
        db_name: 데이터베이스 이름
        collection_name: 컬렉션 이름
    """
    mongo_uri: Optional[str] = Field(None, description="MongoDB 연결 URI")
    db_name: Optional[str] = Field(None, description="데이터베이스 이름")
    collection_name: Optional[str] = Field(None, description="컬렉션 이름")


class MongoDBConnectionTest(BaseModel):
    """MongoDB 연결 테스트 결과

    Attributes:
        success: 연결 성공 여부
        message: 결과 메시지
        collections: 컬렉션 목록 (성공 시)
    """
    success: bool
    message: str
    collections: Optional[List[str]] = None


class MongoDBConfigResponse(BaseModel):
    """MongoDB 설정 조회 응답

    Attributes:
        mongo_uri: MongoDB 연결 URI (마스킹됨)
        db_name: 데이터베이스 이름
        collection_name: 컬렉션 이름
        connected: 현재 연결 여부
    """
    mongo_uri: str
    db_name: str
    collection_name: str
    connected: bool


class MongoDBConfigUpdateResponse(BaseModel):
    """MongoDB 설정 변경 응답

    Attributes:
        mongo_uri: MongoDB 연결 URI (마스킹됨)
        db_name: 데이터베이스 이름
        collection_name: 컬렉션 이름
        connection_test: 연결 테스트 결과
        message: 결과 메시지
    """
    mongo_uri: str
    db_name: str
    collection_name: str
    connection_test: MongoDBConnectionTest
    message: str
