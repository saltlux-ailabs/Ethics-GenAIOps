# -*- coding: utf-8 -*-
"""MongoDB 결과 저장 서비스

평가 결과를 MongoDB에 저장하는 서비스
런타임에 설정 변경 가능 (API 재시작 불필요)
"""
import threading
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from typing import Dict, Any, Optional


class MongoDBService:
    """MongoDB 결과 저장 서비스

    런타임에 설정 변경 가능:
    - configure() 메서드로 URI, DB, Collection 변경
    - 설정 변경 시 기존 연결 자동 종료 후 재연결
    """

    # 기본값 (초기 설정)
    DEFAULT_MONGO_URI = "mongodb://saltlux:AI%40dlsrhdwlsmd!24@211.109.9.82:27017/"
    DEFAULT_DB_NAME = "leaderboard"
    DEFAULT_COLLECTION_NAME = "leaderboard"

    def __init__(self):
        """MongoDBService 초기화"""
        self._client: Optional[MongoClient] = None
        self._lock = threading.Lock()  # 스레드 안전성

        # 인스턴스 설정 (동적 변경 가능)
        self._mongo_uri = self.DEFAULT_MONGO_URI
        self._db_name = self.DEFAULT_DB_NAME
        self._collection_name = self.DEFAULT_COLLECTION_NAME

    def configure(
        self,
        mongo_uri: Optional[str] = None,
        db_name: Optional[str] = None,
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """MongoDB 설정 변경 (런타임)

        설정 변경 시 기존 연결을 종료하고 새 설정으로 재연결합니다.
        API 재시작 없이 설정 변경 가능.

        Args:
            mongo_uri: MongoDB 연결 URI
            db_name: 데이터베이스 이름
            collection_name: 컬렉션 이름

        Returns:
            Dict: 변경된 설정 및 연결 테스트 결과
        """
        with self._lock:
            # 기존 연결 종료
            if self._client:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None

            # 새 설정 적용
            if mongo_uri is not None:
                self._mongo_uri = mongo_uri
            if db_name is not None:
                self._db_name = db_name
            if collection_name is not None:
                self._collection_name = collection_name

            # 연결 테스트
            test_result = self._test_connection()

            return {
                "mongo_uri": self._mask_uri(self._mongo_uri),
                "db_name": self._db_name,
                "collection_name": self._collection_name,
                "connection_test": test_result,
                "message": "MongoDB configuration updated successfully"
            }

    def get_config(self) -> Dict[str, Any]:
        """현재 MongoDB 설정 조회

        Returns:
            Dict: 현재 설정 (URI는 마스킹 처리)
        """
        return {
            "mongo_uri": self._mask_uri(self._mongo_uri),
            "db_name": self._db_name,
            "collection_name": self._collection_name,
            "connected": self._client is not None
        }

    def _mask_uri(self, uri: str) -> str:
        """URI에서 비밀번호 마스킹

        Args:
            uri: MongoDB URI

        Returns:
            str: 마스킹된 URI
        """
        try:
            # mongodb://user:password@host:port/ 형식에서 password 마스킹
            if "@" in uri and "://" in uri:
                prefix = uri.split("://")[0] + "://"
                rest = uri.split("://")[1]
                if "@" in rest:
                    auth_part = rest.split("@")[0]
                    host_part = rest.split("@")[1]
                    if ":" in auth_part:
                        user = auth_part.split(":")[0]
                        return f"{prefix}{user}:****@{host_part}"
            return uri
        except Exception:
            return "****"

    def _test_connection(self) -> Dict[str, Any]:
        """연결 테스트

        Returns:
            Dict: 테스트 결과
        """
        try:
            client = MongoClient(self._mongo_uri, serverSelectionTimeoutMS=5000)
            # ping 테스트
            client.admin.command('ping')
            # 컬렉션 접근 테스트
            db = client[self._db_name]
            collections = db.list_collection_names()
            client.close()

            return {
                "success": True,
                "message": "Connection successful",
                "collections": collections
            }
        except ServerSelectionTimeoutError:
            return {
                "success": False,
                "message": "Connection timeout - server not reachable"
            }
        except ConnectionFailure as e:
            return {
                "success": False,
                "message": f"Connection failed: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error: {str(e)}"
            }

    def _get_client(self) -> MongoClient:
        """MongoDB 클라이언트 반환 (lazy initialization, 스레드 안전)

        Returns:
            MongoClient: MongoDB 클라이언트
        """
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = MongoClient(self._mongo_uri)
        return self._client

    def _get_collection(self):
        """컬렉션 반환

        Returns:
            Collection: MongoDB 컬렉션
        """
        return self._get_client()[self._db_name][self._collection_name]

    def save_results(self, session_id: str, results: Dict[str, Any]) -> bool:
        """결과 저장 (upsert)

        Args:
            session_id: 세션 ID (_id로 사용)
            results: 저장할 결과 데이터

        Returns:
            bool: 성공 여부
        """
        try:
            collection = self._get_collection()
            document = {
                "_id": session_id,
                **results
            }
            collection.replace_one(
                {"_id": session_id},
                document,
                upsert=True
            )
            return True
        except Exception as e:
            print(f"[MongoDB ERROR] Failed to save results: {e}")
            return False

    def update_status(
        self,
        session_id: str,
        status: Optional[str] = None,
        progress: Optional[Dict[str, int]] = None,
        error_msg: Optional[str] = None,
        elapsed_time_seconds: Optional[float] = None
    ) -> bool:
        """상태 실시간 업데이트 ($set으로 부분 업데이트, upsert)

        프론트엔드에서 session_id로 조회하여 상태에 따른 UI 이벤트 처리 가능

        Args:
            session_id: 세션 ID (_id로 사용)
            status: 상태값 (init, create_conversation, evaluation, finished, error)
            progress: 진행 정보 {"total": int, "completed": int, "failed": int}
            error_msg: 에러 메시지
            elapsed_time_seconds: 소요 시간

        Returns:
            bool: 성공 여부
        """
        try:
            collection = self._get_collection()

            update_data = {
                "updated_at": datetime.now().isoformat()
            }

            if status is not None:
                update_data["status"] = status
            if progress is not None:
                update_data["progress"] = progress
            if error_msg is not None:
                update_data["error_message"] = error_msg
            if elapsed_time_seconds is not None:
                update_data["elapsed_time_seconds"] = elapsed_time_seconds

            collection.update_one(
                {"_id": session_id},
                {"$set": update_data},
                upsert=True
            )
            return True
        except Exception as e:
            print(f"[MongoDB ERROR] Failed to update status: {e}")
            return False

    def get_results(self, session_id: str) -> Optional[Dict[str, Any]]:
        """session_id로 results 조회

        Args:
            session_id: 세션 ID

        Returns:
            Dict: 세션 데이터 (results 포함), 없으면 None
        """
        try:
            collection = self._get_collection()
            document = collection.find_one({"_id": session_id})
            return document
        except Exception as e:
            print(f"[MongoDB ERROR] Failed to get results: {e}")
            return None

    def update_result_report(
        self,
        session_id: str,
        result_index: int,
        report: str
    ) -> bool:
        """특정 result에 report 필드 추가

        Args:
            session_id: 세션 ID
            result_index: results 배열 내 인덱스
            report: LLM이 생성한 보고서 내용

        Returns:
            bool: 성공 여부
        """
        try:
            collection = self._get_collection()
            # results 배열의 특정 인덱스에 report 필드 추가
            collection.update_one(
                {"_id": session_id},
                {
                    "$set": {
                        f"results.{result_index}.report": report,
                        "updated_at": datetime.now().isoformat()
                    }
                }
            )
            return True
        except Exception as e:
            print(f"[MongoDB ERROR] Failed to update result report: {e}")
            return False

    def update_report_status(
        self,
        session_id: str,
        report_status: str,
        report_progress: Optional[Dict[str, int]] = None,
        report_error: Optional[str] = None
    ) -> bool:
        """보고서 생성 상태 업데이트

        Args:
            session_id: 세션 ID
            report_status: 보고서 상태 (generating, finished, error)
            report_progress: 진행 정보 {"total": int, "completed": int, "failed": int}
            report_error: 에러 메시지

        Returns:
            bool: 성공 여부
        """
        try:
            collection = self._get_collection()
            update_data = {
                "report_status": report_status,
                "updated_at": datetime.now().isoformat()
            }
            if report_progress is not None:
                update_data["report_progress"] = report_progress
            if report_error is not None:
                update_data["report_error"] = report_error

            collection.update_one(
                {"_id": session_id},
                {"$set": update_data},
                upsert=True
            )
            return True
        except Exception as e:
            print(f"[MongoDB ERROR] Failed to update report status: {e}")
            return False

    def close(self):
        """연결 종료"""
        with self._lock:
            if self._client:
                self._client.close()
                self._client = None

    # =========================================================================
    # Benchmark 관련 메서드
    # =========================================================================

    def save_benchmark_results(
        self,
        session_id: str,
        benchmark_results: Dict[str, Any],
        elapsed_time_seconds: float
    ) -> bool:
        """벤치마크 결과 저장

        Args:
            session_id: 세션 ID
            benchmark_results: 벤치마크 결과 데이터
            elapsed_time_seconds: 소요 시간

        Returns:
            bool: 성공 여부
        """
        try:
            collection = self._get_collection()
            collection.update_one(
                {"_id": session_id},
                {
                    "$set": {
                        "benchmark_results": benchmark_results,
                        "benchmark_elapsed_time_seconds": elapsed_time_seconds,
                        "updated_at": datetime.now().isoformat()
                    }
                },
                upsert=True
            )
            return True
        except Exception as e:
            print(f"[MongoDB ERROR] Failed to save benchmark results: {e}")
            return False

    def update_benchmark_status(
        self,
        session_id: str,
        status: Optional[str] = None,
        progress: Optional[Dict[str, int]] = None,
        error_msg: Optional[str] = None,
        elapsed_time_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        email: Optional[str] = None
    ) -> bool:
        """벤치마크 상태 업데이트

        Args:
            session_id: 세션 ID
            status: 벤치마크 상태 (init, running, finished, error)
            progress: 진행 정보 {"total": int, "completed": int, "failed": int}
            error_msg: 에러 메시지
            elapsed_time_seconds: 소요 시간
            metadata: 사용자 정의 메타데이터
            email: 사용자 이메일 (최상단 필드로 저장)

        Returns:
            bool: 성공 여부
        """
        try:
            collection = self._get_collection()

            update_data = {
                "updated_at": datetime.now().isoformat()
            }

            if status is not None:
                update_data["benchmark_status"] = status
            if progress is not None:
                update_data["benchmark_progress"] = progress
            if error_msg is not None:
                update_data["benchmark_error"] = error_msg
            if elapsed_time_seconds is not None:
                update_data["benchmark_elapsed_time_seconds"] = elapsed_time_seconds
            if metadata is not None:
                update_data["metadata"] = metadata
            if email is not None:
                update_data["email"] = email

            collection.update_one(
                {"_id": session_id},
                {"$set": update_data},
                upsert=True
            )
            return True
        except Exception as e:
            print(f"[MongoDB ERROR] Failed to update benchmark status: {e}")
            return False

    def get_benchmark_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """벤치마크 상태 조회

        Args:
            session_id: 세션 ID

        Returns:
            Dict: 벤치마크 상태 정보, 없으면 None
        """
        try:
            collection = self._get_collection()
            document = collection.find_one(
                {"_id": session_id},
                {
                    "benchmark_status": 1,
                    "benchmark_progress": 1,
                    "benchmark_error": 1,
                    "benchmark_elapsed_time_seconds": 1
                }
            )
            return document
        except Exception as e:
            print(f"[MongoDB ERROR] Failed to get benchmark status: {e}")
            return None

    def get_benchmark_results(self, session_id: str) -> Optional[Dict[str, Any]]:
        """벤치마크 결과 조회

        Args:
            session_id: 세션 ID

        Returns:
            Dict: 벤치마크 결과 데이터, 없으면 None
        """
        try:
            collection = self._get_collection()
            document = collection.find_one(
                {"_id": session_id},
                {
                    "benchmark_status": 1,
                    "benchmark_results": 1,
                    "benchmark_elapsed_time_seconds": 1,
                    "benchmark_error": 1
                }
            )
            return document
        except Exception as e:
            print(f"[MongoDB ERROR] Failed to get benchmark results: {e}")
            return None


# 싱글톤 인스턴스
mongodb_service = MongoDBService()
