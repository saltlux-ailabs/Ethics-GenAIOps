# -*- coding: utf-8 -*-
"""세션 관리 서비스

세션 폴더 생성, 상태 관리, 결과 저장 등을 담당
"""
import json
import uuid
import re
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from api.models.schemas import SessionStatus, ProgressInfo


class SessionManager:
    """세션 관리 클래스

    세션 폴더 생성, status.json 관리, results.json 저장/조회
    스레드 안전성을 위해 파일 접근 시 잠금 사용
    """

    BASE_PATH = Path("./resource/sessions")

    def __init__(self):
        """SessionManager 초기화"""
        # 기본 디렉토리 생성
        self.BASE_PATH.mkdir(parents=True, exist_ok=True)
        # 세션별 파일 잠금 관리 (session_id -> Lock)
        self._locks: Dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()  # _locks 딕셔너리 접근용 잠금

    def _get_session_lock(self, session_id: str) -> threading.Lock:
        """세션별 잠금 객체 반환 (없으면 생성)

        Args:
            session_id: 세션 ID

        Returns:
            threading.Lock: 해당 세션의 잠금 객체
        """
        with self._locks_lock:
            if session_id not in self._locks:
                self._locks[session_id] = threading.Lock()
            return self._locks[session_id]

    def _sanitize_filename(self, filename: str) -> str:
        """파일명에서 사용할 수 없는 특수 문자를 안전한 문자로 치환

        Args:
            filename: 원본 파일명

        Returns:
            str: 안전한 파일명
        """
        # Windows/Unix에서 사용할 수 없는 문자들 치환
        sanitized = re.sub(r'[<>:"|?*\\/]', '_', filename)
        sanitized = re.sub(r'_+', '_', sanitized)
        sanitized = sanitized.strip(' _')

        if not sanitized:
            sanitized = 'default_session'

        return sanitized

    def _generate_session_id(self) -> str:
        """고유 세션 ID 생성

        Returns:
            str: 타임스탬프 기반 세션 ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        return f"{timestamp}_{unique_id}"

    def _get_session_path(self, session_id: str) -> Path:
        """세션 폴더 경로 반환

        Args:
            session_id: 세션 ID

        Returns:
            Path: 세션 폴더 경로
        """
        sanitized_id = self._sanitize_filename(session_id)
        return self.BASE_PATH / sanitized_id

    def _get_status_file(self, session_id: str) -> Path:
        """status.json 파일 경로 반환"""
        return self._get_session_path(session_id) / "status.json"

    def _get_results_file(self, session_id: str) -> Path:
        """results.json 파일 경로 반환"""
        return self._get_session_path(session_id) / "results.json"

    def session_exists(self, session_id: str) -> bool:
        """세션 존재 여부 확인

        Args:
            session_id: 세션 ID

        Returns:
            bool: 세션 존재 여부
        """
        return self._get_session_path(session_id).exists()

    def create_session(self, session_id: Optional[str] = None) -> str:
        """세션 폴더 생성 및 초기화

        Args:
            session_id: 사용자 지정 세션 ID (None 또는 빈 문자열이면 UUID 자동 생성)

        Returns:
            str: 생성된 세션 ID

        Raises:
            ValueError: 세션이 이미 존재하는 경우
        """
        # 세션 ID 생성 또는 검증 (None 또는 빈 문자열이면 자동 생성)
        if not session_id:
            session_id = self._generate_session_id()
        else:
            session_id = self._sanitize_filename(session_id)

        # 중복 체크
        if self.session_exists(session_id):
            raise ValueError(f"Session already exists: {session_id}")

        # 세션 폴더 생성
        session_path = self._get_session_path(session_id)
        session_path.mkdir(parents=True, exist_ok=True)

        # 초기 상태 파일 생성
        initial_status = {
            "status": SessionStatus.INIT.value,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        self._save_status_file(session_id, initial_status)

        return session_id

    def _save_status_file(self, session_id: str, data: Dict[str, Any]) -> None:
        """status.json 파일 저장

        Args:
            session_id: 세션 ID
            data: 저장할 데이터
        """
        status_file = self._get_status_file(session_id)
        with open(status_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_status_file(self, session_id: str) -> Dict[str, Any]:
        """status.json 파일 로드

        Args:
            session_id: 세션 ID

        Returns:
            Dict: 상태 데이터

        Raises:
            FileNotFoundError: 파일이 없는 경우
        """
        status_file = self._get_status_file(session_id)
        if not status_file.exists():
            raise FileNotFoundError(f"Status file not found for session: {session_id}")

        try:
            with open(status_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            # 파일이 손상된 경우 기본 상태로 복구
            print(f"[WARNING] Corrupted status file for session {session_id}: {e}")
            print(f"[WARNING] Resetting to default status...")
            default_status = {
                "status": SessionStatus.ERROR.value,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "error_message": f"Status file was corrupted: {type(e).__name__}"
            }
            self._save_status_file(session_id, default_status)
            return default_status

    def get_status(self, session_id: str) -> Dict[str, Any]:
        """세션 상태 조회

        Args:
            session_id: 세션 ID

        Returns:
            Dict: 상태 정보 (status, progress, error_message 등)

        Raises:
            FileNotFoundError: 세션이 없는 경우
        """
        if not self.session_exists(session_id):
            raise FileNotFoundError(f"Session not found: {session_id}")

        return self._load_status_file(session_id)

    def update_status(
        self,
        session_id: str,
        status: SessionStatus,
        progress: Optional[Dict[str, int]] = None,
        error_msg: Optional[str] = None,
        elapsed_time_seconds: Optional[float] = None
    ) -> None:
        """세션 상태 업데이트 (스레드 안전)

        Args:
            session_id: 세션 ID
            status: 새 상태
            progress: 진행 상황 (total, completed, failed)
            error_msg: 에러 메시지 (error 상태일 때)
            elapsed_time_seconds: 전체 실행 시간 (초 단위, finished 상태일 때)
        """
        lock = self._get_session_lock(session_id)
        with lock:
            try:
                current_data = self._load_status_file(session_id)
            except FileNotFoundError:
                current_data = {}

            current_data["status"] = status.value
            current_data["updated_at"] = datetime.now().isoformat()

            if progress is not None:
                current_data["progress"] = progress

            if error_msg is not None:
                current_data["error_message"] = error_msg

            if elapsed_time_seconds is not None:
                current_data["elapsed_time_seconds"] = elapsed_time_seconds

            self._save_status_file(session_id, current_data)

    def update_progress(
        self,
        session_id: str,
        progress: Dict[str, int]
    ) -> None:
        """진행률만 업데이트 (상태는 유지, 스레드 안전)

        Args:
            session_id: 세션 ID
            progress: 진행 상황 (total, completed, failed)
        """
        lock = self._get_session_lock(session_id)
        with lock:
            try:
                current_data = self._load_status_file(session_id)
            except FileNotFoundError:
                return  # 상태 파일이 없으면 무시

            if progress is not None:
                current_data["progress"] = progress
                current_data["updated_at"] = datetime.now().isoformat()
                self._save_status_file(session_id, current_data)

    def save_results(self, session_id: str, results: Dict[str, Any]) -> None:
        """평가 결과 저장

        Args:
            session_id: 세션 ID
            results: main() 함수 반환값 (results, evaluation_results)
        """
        results_file = self._get_results_file(session_id)

        # 저장 시간 추가
        results["saved_at"] = datetime.now().isoformat()

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def get_results(self, session_id: str) -> Optional[Dict[str, Any]]:
        """평가 결과 조회

        Args:
            session_id: 세션 ID

        Returns:
            Dict: 결과 데이터 (없으면 None)
        """
        results_file = self._get_results_file(session_id)

        if not results_file.exists():
            return None

        with open(results_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_session_log_path(self, session_id: str) -> Path:
        """세션 로그 파일 경로 반환

        Args:
            session_id: 세션 ID

        Returns:
            Path: 로그 파일 경로
        """
        sanitized_id = self._sanitize_filename(session_id)
        return self._get_session_path(session_id) / f"{sanitized_id}.log"

    def list_sessions(self) -> list:
        """모든 세션 목록 조회

        Returns:
            list: 세션 정보 목록
        """
        sessions = []
        if not self.BASE_PATH.exists():
            return sessions

        for session_dir in self.BASE_PATH.iterdir():
            if session_dir.is_dir():
                try:
                    status_data = self._load_status_file(session_dir.name)
                    sessions.append({
                        "session_id": session_dir.name,
                        "status": status_data.get("status"),
                        "created_at": status_data.get("created_at"),
                        "updated_at": status_data.get("updated_at")
                    })
                except FileNotFoundError:
                    continue

        # 생성 시간순 정렬
        sessions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return sessions


# 싱글톤 인스턴스
session_manager = SessionManager()
