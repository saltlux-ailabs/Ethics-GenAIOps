# -*- coding: utf-8 -*-
"""설정 관리 라우터

프롬프트(txt)와 GOAT 설정(yaml) 파일 관리 API 엔드포인트
"""
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, status, UploadFile, File

from api.models.schemas import (
    PromptInfo,
    PromptContent,
    UpdatePromptRequest,
    UpdatePromptResponse,
    CategoryConfig,
    GoatConfig,
    UpdateCategoryResponse,
    UpdateGoatConfigResponse,
    MongoDBConfigRequest,
    MongoDBConfigResponse,
    MongoDBConfigUpdateResponse,
    MongoDBConnectionTest
)
from api.services.config_service import config_service
from api.services.mongodb_service import mongodb_service


router = APIRouter(prefix="/config", tags=["Config"])


# =============================================================================
# 프롬프트 관리 API
# =============================================================================

@router.get("/prompts", response_model=List[PromptInfo])
async def list_prompts() -> List[PromptInfo]:
    """프롬프트 파일 목록 조회

    Returns:
        List[PromptInfo]: 프롬프트 파일 정보 목록
    """
    prompts = config_service.list_prompts()
    return [PromptInfo(**p) for p in prompts]


@router.get("/prompts/{name}", response_model=PromptContent)
async def get_prompt(name: str) -> PromptContent:
    """특정 프롬프트 내용 조회

    Args:
        name: 프롬프트 파일명

    Returns:
        PromptContent: 프롬프트 내용

    Raises:
        400: 유효하지 않은 파일명
        404: 파일이 없는 경우
    """
    try:
        result = config_service.get_prompt(name)
        return PromptContent(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@router.put("/prompts/{name}", response_model=UpdatePromptResponse)
async def update_prompt(name: str, request: UpdatePromptRequest) -> UpdatePromptResponse:
    """프롬프트 수정

    Args:
        name: 프롬프트 파일명
        request: 수정 요청 (content)

    Returns:
        UpdatePromptResponse: 수정 결과

    Raises:
        400: 유효하지 않은 파일명 또는 크기 초과
        404: 파일이 없는 경우
    """
    try:
        result = config_service.update_prompt(name, request.content)
        return UpdatePromptResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@router.post("/prompts/upload", response_model=UpdatePromptResponse)
async def upload_prompt(file: UploadFile = File(...)) -> UpdatePromptResponse:
    """프롬프트 파일 업로드

    파일명이 기존 프롬프트 파일명과 일치해야만 업데이트됩니다.

    Constraints:
        - .txt 파일만 허용
        - 파일명이 기존 프롬프트 목록에 있어야 함
        - 최대 100KB
        - UTF-8 인코딩

    Args:
        file: 업로드할 프롬프트 파일

    Returns:
        UpdatePromptResponse: 업로드 결과

    Raises:
        400: 파일명 불일치, 확장자 오류, 크기 초과, 인코딩 오류
        404: 기존 파일이 없는 경우
    """
    try:
        content = await file.read()
        result = config_service.upload_prompt(file.filename, content)
        return UpdatePromptResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


# =============================================================================
# GOAT 설정 관리 API
# =============================================================================

@router.get("/goat", response_model=Dict[str, Any])
async def get_goat_config() -> Dict[str, Any]:
    """GOAT 설정 전체 조회

    Returns:
        Dict: GOAT 설정 데이터

    Raises:
        404: 설정 파일이 없는 경우
    """
    try:
        return config_service.get_goat_config()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@router.put("/goat", response_model=UpdateGoatConfigResponse)
async def update_goat_config(config: Dict[str, Any]) -> UpdateGoatConfigResponse:
    """GOAT 설정 전체 수정

    Args:
        config: 새 설정 데이터

    Returns:
        UpdateGoatConfigResponse: 수정 결과

    Raises:
        404: 설정 파일이 없는 경우
    """
    try:
        result = config_service.update_goat_config(config)
        return UpdateGoatConfigResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@router.get("/goat/categories", response_model=List[str])
async def list_categories() -> List[str]:
    """카테고리 목록 조회

    Returns:
        List[str]: 카테고리 ID 목록
    """
    return config_service.list_categories()


@router.get("/goat/categories/{category_id}", response_model=CategoryConfig)
async def get_category(category_id: str) -> CategoryConfig:
    """특정 카테고리 조회

    Args:
        category_id: 카테고리 ID

    Returns:
        CategoryConfig: 카테고리 설정

    Raises:
        404: 카테고리가 없는 경우
    """
    try:
        result = config_service.get_category(category_id)
        return CategoryConfig(**result)
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@router.put("/goat/categories/{category_id}", response_model=UpdateCategoryResponse)
async def update_category(category_id: str, data: CategoryConfig) -> UpdateCategoryResponse:
    """특정 카테고리 수정

    Args:
        category_id: 카테고리 ID
        data: 새 카테고리 데이터

    Returns:
        UpdateCategoryResponse: 수정 결과

    Raises:
        404: 카테고리가 없는 경우
    """
    try:
        result = config_service.update_category(category_id, data.model_dump())
        return UpdateCategoryResponse(**result)
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


# =============================================================================
# MongoDB 설정 관리 API
# =============================================================================

@router.get("/mongodb", response_model=MongoDBConfigResponse)
async def get_mongodb_config() -> MongoDBConfigResponse:
    """MongoDB 현재 설정 조회

    Returns:
        MongoDBConfigResponse: 현재 MongoDB 설정 (URI는 마스킹 처리)
    """
    config = mongodb_service.get_config()
    return MongoDBConfigResponse(**config)


@router.put("/mongodb", response_model=MongoDBConfigUpdateResponse)
async def update_mongodb_config(request: MongoDBConfigRequest) -> MongoDBConfigUpdateResponse:
    """MongoDB 설정 변경 (런타임)

    API 재시작 없이 MongoDB 연결 설정을 변경합니다.
    설정 변경 시 기존 연결을 종료하고 새 설정으로 재연결합니다.

    Args:
        request: 변경할 설정 (미입력 항목은 기존 값 유지)

    Returns:
        MongoDBConfigUpdateResponse: 변경된 설정 및 연결 테스트 결과
    """
    result = mongodb_service.configure(
        mongo_uri=request.mongo_uri,
        db_name=request.db_name,
        collection_name=request.collection_name
    )

    return MongoDBConfigUpdateResponse(
        mongo_uri=result["mongo_uri"],
        db_name=result["db_name"],
        collection_name=result["collection_name"],
        connection_test=MongoDBConnectionTest(**result["connection_test"]),
        message=result["message"]
    )


@router.post("/mongodb/test", response_model=MongoDBConnectionTest)
async def test_mongodb_connection() -> MongoDBConnectionTest:
    """MongoDB 연결 테스트

    현재 설정으로 MongoDB 연결을 테스트합니다.

    Returns:
        MongoDBConnectionTest: 연결 테스트 결과
    """
    result = mongodb_service._test_connection()
    return MongoDBConnectionTest(**result)
