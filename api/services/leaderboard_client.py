# -*- coding: utf-8 -*-
"""Leaderboard API 클라이언트

별도 Leaderboard API 서버와 통신하여 리더보드 데이터를 관리합니다.
"""
import os
import httpx
from typing import Dict, Any, Optional


class LeaderboardClient:
    """Leaderboard API 클라이언트

    환경변수:
        LEADERBOARD_API_URL: Leaderboard API 서버 URL
        LEADERBOARD_JWT_TOKEN: JWT 인증 토큰
    """

    def __init__(self):
        """LeaderboardClient 초기화"""
        self._base_url = os.getenv("LEADERBOARD_API_URL", "http://localhost:8080")
        self.password = os.getenv("LEADERBOARD_PASSWORD", "")
        self._timeout = 30.0  # 30초 타임아웃

    def _get_headers(self) -> Dict[str, str]:
        """요청 헤더 생성

        Returns:
            Dict[str, str]: Authorization 헤더 포함
        """
        return {
            "Content-Type": "application/json"
        }

    async def create_leaderboard(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """리더보드 생성

        POST /leaderboards

        Args:
            data: 리더보드 데이터
                - email: 사용자 이메일 (필수)
                - status: 상태 (필수)
                - metadata: 메타데이터
                - 기타 평가 결과 필드

        Returns:
            Dict[str, Any]: 생성된 리더보드 정보 (_id 포함)

        Raises:
            httpx.HTTPStatusError: API 호출 실패 시
        """
        url = f"{self._base_url}/leaderboards/engine"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                url,
                json=data,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()

    async def update_leaderboard(
        self,
        leaderboard_id: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """리더보드 업데이트 (전체 변경)

        PUT /leaderboards/{id}

        Args:
            leaderboard_id: 리더보드 ID
            data: 업데이트할 데이터

        Returns:
            Dict[str, Any]: 업데이트된 리더보드 정보

        Raises:
            httpx.HTTPStatusError: API 호출 실패 시
        """
        url = f"{self._base_url}/leaderboards/{self.password}_{leaderboard_id}"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.put(
                url,
                json=data,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()

    async def get_leaderboard(self, leaderboard_id: str) -> Optional[Dict[str, Any]]:
        """리더보드 조회

        GET /leaderboards/{id}

        Args:
            leaderboard_id: 리더보드 ID

        Returns:
            Dict[str, Any]: 리더보드 데이터, 없으면 None

        Raises:
            httpx.HTTPStatusError: API 호출 실패 시 (404 제외)
        """
        url = f"{self._base_url}/leaderboards/{self.password}_{leaderboard_id}"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(
                url,
                headers=self._get_headers()
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()

    async def delete_leaderboard(self, leaderboard_id: str) -> bool:
        """리더보드 삭제

        DELETE /leaderboards/{id}

        Args:
            leaderboard_id: 리더보드 ID

        Returns:
            bool: 삭제 성공 여부

        Raises:
            httpx.HTTPStatusError: API 호출 실패 시
        """
        url = f"{self._base_url}/leaderboards/{leaderboard_id}"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.delete(
                url,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return True

    def create_leaderboard_sync(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """리더보드 생성 (동기 버전)

        POST /leaderboards/engine

        Args:
            data: 리더보드 데이터

        Returns:
            Dict[str, Any]: 생성된 리더보드 정보 (_id 포함)
        """
        url = f"{self._base_url}/leaderboards/engine"

        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                url,
                json=data,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()

    def update_leaderboard_sync(
        self,
        leaderboard_id: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """리더보드 업데이트 (동기 버전)

        PUT /leaderboards/{password}_{id}

        Args:
            leaderboard_id: 리더보드 ID
            data: 업데이트할 데이터

        Returns:
            Dict[str, Any]: 업데이트된 리더보드 정보
        """
        url = f"{self._base_url}/leaderboards/{self.password}_{leaderboard_id}"

        with httpx.Client(timeout=self._timeout) as client:
            response = client.put(
                url,
                json=data,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()

    def get_leaderboard_sync(self, leaderboard_id: str) -> Optional[Dict[str, Any]]:
        """리더보드 조회 (동기 버전)

        GET /leaderboards/{password}_{id}

        Args:
            leaderboard_id: 리더보드 ID

        Returns:
            Dict[str, Any]: 리더보드 데이터, 없으면 None
        """
        url = f"{self._base_url}/leaderboards/{self.password}_{leaderboard_id}"

        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(
                url,
                headers=self._get_headers()
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()


# 싱글톤 인스턴스
leaderboard_client = LeaderboardClient()
