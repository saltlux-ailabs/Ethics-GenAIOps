# -*- coding: utf-8 -*-
"""평가 실행 서비스

백그라운드에서 GOAT 공격 평가를 실행하고 상태를 관리
"""
import asyncio
import json
import traceback
from typing import Callable, Optional, Dict, Any
from functools import partial

from api.models.schemas import SessionStatus
from api.services.session_manager import SessionManager
from api.services.mongodb_service import mongodb_service
from user_logging_config import UserLogger


def make_json_log(thread_id: str, file_name: str, function_location: str,
                  message: str, value) -> str:
    """구조화된 로그 메시지 생성

    Args:
        thread_id: 세션 ID
        file_name: 파일명
        function_location: 함수 위치
        message: 로그 메시지
        value: 로그 값

    Returns:
        str: 포맷된 로그 메시지
    """
    try:
        if isinstance(value, dict):
            value = json.dumps(value, indent=4, ensure_ascii=False)
        elif isinstance(value, list):
            value = json.dumps(value, indent=4, ensure_ascii=False)
        else:
            pass
    except:
        value = str(value)

    log_message = f"""
    session_id: {thread_id}
    file_name: {file_name}
    function_location: {function_location}
    message: {message}
    value: {value}
    """
    return log_message


class EvaluationRunner:
    """평가 실행 클래스

    core.py의 main() 함수를 백그라운드에서 실행하고
    진행 상황과 결과를 관리합니다.
    """

    def __init__(self, session_manager: SessionManager):
        """EvaluationRunner 초기화

        Args:
            session_manager: 세션 관리자 인스턴스
        """
        self.session_manager = session_manager

    def _create_logger(self, session_id: str) -> UserLogger:
        """세션별 로거 생성

        Args:
            session_id: 세션 ID

        Returns:
            UserLogger: 로거 인스턴스
        """
        return UserLogger(session_id, version="1.0.0")

    def _log_step(self, logger: UserLogger, session_id: str,
                  step: str, message: str, value) -> None:
        """단계별 로깅

        Args:
            logger: 로거 인스턴스
            session_id: 세션 ID
            step: 단계명
            message: 로그 메시지
            value: 로그 값
        """
        log_msg = make_json_log(
            thread_id=session_id,
            file_name="evaluation_runner",
            function_location=step,
            message=message,
            value=value
        )
        logger.print_info(log_msg)

    def run_evaluation_sync(
        self,
        session_id: str,
        mode: str,
        provider: dict,
        max_workers: int,
        metadata: Optional[Dict[str, Any]] = None,
        email: Optional[str] = None
    ) -> None:
        """평가 실행 (동기 버전)

        백그라운드 태스크에서 직접 호출됩니다.

        Args:
            session_id: 세션 ID
            mode: API 모드 (openai/custom)
            provider: API 제공자 설정
            max_workers: 병렬 처리 worker 수
            metadata: 사용자 정의 메타데이터 (자유 형식)
            email: 사용자 이메일 (최상단 필드로 저장)
        """
        logger = self._create_logger(session_id)

        # 상태 콜백 함수 정의 (core.py에서 호출됨)
        def status_callback(status: str, progress: dict = None):
            """core.py에서 호출되는 상태 콜백

            Args:
                status: 상태 ("create_conversation", "evaluation", "turn_evaluation", "report_generation", "progress")
                progress: 진행 정보
            """
            if status == "create_conversation":
                self.session_manager.update_status(
                    session_id,
                    SessionStatus.CREATE_CONVERSATION,
                    progress=progress
                )
                # MongoDB Report 상태 저장: running (모든 중간 단계는 running)
                mongodb_service.update_report_status(
                    session_id,
                    report_status="running",
                    report_progress=progress
                )
                self._log_step(
                    logger, session_id, "create_conversation",
                    "대화 생성 중", progress or {}
                )
            elif status == "evaluation":
                self.session_manager.update_status(
                    session_id,
                    SessionStatus.EVALUATION,
                    progress=progress
                )
                # MongoDB Report 상태 저장: running (모든 중간 단계는 running)
                mongodb_service.update_report_status(
                    session_id,
                    report_status="running",
                    report_progress=progress
                )
                self._log_step(
                    logger, session_id, "evaluation",
                    "평가 진행 중", progress or {}
                )
            elif status == "turn_evaluation":
                # 턴별 평가 상태: running
                mongodb_service.update_report_status(
                    session_id,
                    report_status="running",
                    report_progress=progress
                )
                self._log_step(
                    logger, session_id, "turn_evaluation",
                    "턴별 평가 진행 중", progress or {}
                )
            elif status == "report_generation":
                # 보고서 생성 상태: running
                mongodb_service.update_report_status(
                    session_id,
                    report_status="running",
                    report_progress=progress
                )
                self._log_step(
                    logger, session_id, "report_generation",
                    "보고서 생성 중", progress or {}
                )
            elif status == "progress":
                # 현재 상태 유지하면서 진행률만 업데이트
                self.session_manager.update_progress(session_id, progress)
                # MongoDB Report 진행률 업데이트 (상태는 running 유지)
                mongodb_service.update_report_status(
                    session_id,
                    report_status="running",
                    report_progress=progress
                )
                self._log_step(
                    logger, session_id, "progress",
                    "진행률 업데이트", progress or {}
                )

        try:
            # 1. 초기 상태 업데이트: running (Report 시작)
            self.session_manager.update_status(
                session_id,
                SessionStatus.CREATE_CONVERSATION
            )
            # MongoDB Report 상태 저장: running
            mongodb_service.update_report_status(
                session_id,
                report_status="running"
            )
            self._log_step(
                logger, session_id, "start",
                "평가 시작", {"mode": mode, "provider": provider, "max_workers": max_workers, "metadata": metadata}
            )

            # 2. core.py 설정 적용
            from core import (
                configure_api_settings,
                main as core_main,
                set_session_logger,
                clear_session_logger,
                set_status_callback,
                clear_status_callback,
                # 턴별 평가 및 보고서 생성 함수
                convert_results_to_dialogues,
                evaluate_dialogues,
                generate_evaluation_reports
            )

            configure_api_settings(mode, provider)
            self._log_step(
                logger, session_id, "configure",
                "API 설정 완료", {"mode": mode}
            )

            # 3. 세션 로거 주입 (core.py 내부 LLM 호출 로깅용)
            set_session_logger(logger, session_id)
            self._log_step(
                logger, session_id, "logger_inject",
                "세션 로거 주입 완료", {"session_id": session_id}
            )

            # 4. 상태 콜백 등록 (core.py에서 상태 변경 시 호출됨)
            set_status_callback(status_callback)
            self._log_step(
                logger, session_id, "callback_register",
                "상태 콜백 등록 완료", {}
            )

            # 5. main() 함수 실행 (내부에서 status_callback 호출됨)
            results = core_main(max_workers=max_workers)
            self._log_step(
                logger, session_id, "core_main_complete",
                "대화 생성 완료", {
                    "total_questions": results.get("evaluation_results", {}).get("total_questions", 0)
                }
            )

            # 6. 턴별 평가 실행
            status_callback("turn_evaluation", {"phase": "converting"})
            self._log_step(
                logger, session_id, "turn_evaluation_start",
                "턴별 평가 시작", {}
            )

            # 6-1. dialogues 포맷으로 변환
            raw_results = results.get("results", [])
            dialogues = convert_results_to_dialogues(raw_results)
            self._log_step(
                logger, session_id, "dialogues_converted",
                "dialogues 포맷 변환 완료", {"total_dialogues": len(dialogues)}
            )

            # 6-2. 턴별 평가 수행
            status_callback("turn_evaluation", {
                "phase": "evaluating",
                "total": len(dialogues),
                "completed": 0
            })
            evaluated_dialogues = evaluate_dialogues(dialogues, context=session_id)
            self._log_step(
                logger, session_id, "turn_evaluation_complete",
                "턴별 평가 완료", {"evaluated_count": len(evaluated_dialogues)}
            )

            # 7. 보고서 생성
            status_callback("report_generation", {"phase": "generating"})
            self._log_step(
                logger, session_id, "report_generation_start",
                "보고서 생성 시작", {}
            )

            # 7-1. 대화별 보고서 + 전체 보고서 생성
            report_results = generate_evaluation_reports(
                evaluated_dialogues=evaluated_dialogues,
                generate_dialogue_reports=True,  # 대화별 보고서 생성
                generate_global=True,            # 전체 보고서 생성
                model_name=provider.get("model", "gpt-4o")
            )
            self._log_step(
                logger, session_id, "report_generation_complete",
                "보고서 생성 완료", {
                    "dialogue_reports_count": len(report_results.get("dialogue_reports", [])),
                    "global_report_length": len(report_results.get("global_report", ""))
                }
            )

            # 8. 콜백 및 로거 정리
            clear_status_callback()
            clear_session_logger()

            # 9. 결과에 평가 및 보고서 데이터 추가
            results["evaluated_dialogues"] = evaluated_dialogues
            results["evaluation_stats"] = report_results.get("stats", {})
            results["dialogue_reports"] = report_results.get("dialogue_reports", [])
            results["global_report"] = report_results.get("global_report", "")
            results["report_timestamp"] = report_results.get("timestamp", "")

            # 9-1. metadata 및 email 추가
            if metadata is not None:
                results["metadata"] = metadata
            if email is not None:
                results["email"] = email  # 최상단 필드로 저장
            self._log_step(
                logger, session_id, "metadata_added",
                "메타데이터/이메일 추가 완료", {"metadata": metadata, "email": email}
            )

            # 10. 결과 저장 (JSON 파일)
            self.session_manager.save_results(session_id, results)
            self._log_step(
                logger, session_id, "save_results",
                "결과 저장 완료 (JSON)",
                {
                    "total_questions": results.get("evaluation_results", {}).get("total_questions", 0),
                    "failed_questions": results.get("evaluation_results", {}).get("failed_questions", 0),
                    "evaluated_dialogues_count": len(evaluated_dialogues),
                    "dialogue_reports_count": len(results.get("dialogue_reports", []))
                }
            )

            # 10-1. 결과 저장 (MongoDB) - 기존 Benchmark 데이터 병합
            # 기존 MongoDB 데이터 조회 (Benchmark 결과 보존)
            existing_data = mongodb_service.get_results(session_id)
            if existing_data:
                # Benchmark 관련 필드 보존
                benchmark_fields = [
                    "benchmark_status", "benchmark_progress", "benchmark_results",
                    "benchmark_elapsed_time_seconds", "benchmark_error"
                ]
                for field in benchmark_fields:
                    if field in existing_data and field not in results:
                        results[field] = existing_data[field]
                self._log_step(
                    logger, session_id, "benchmark_data_merged",
                    "기존 Benchmark 데이터 병합 완료",
                    {"merged_fields": [f for f in benchmark_fields if f in existing_data]}
                )

            mongo_success = mongodb_service.save_results(session_id, results)
            self._log_step(
                logger, session_id, "save_results_mongodb",
                "MongoDB 저장 완료" if mongo_success else "MongoDB 저장 실패",
                {"success": mongo_success, "session_id": session_id}
            )

            # 11. 상태 업데이트: finished
            evaluation_results = results.get("evaluation_results", {})
            final_progress = {
                "total": evaluation_results.get("total_questions", 0),
                "completed": evaluation_results.get("total_questions", 0) - evaluation_results.get("failed_questions", 0),
                "failed": evaluation_results.get("failed_questions", 0),
                "evaluated_dialogues": len(evaluated_dialogues),
                "dialogue_reports": len(results.get("dialogue_reports", []))
            }
            elapsed_time = evaluation_results.get("elapsed_time_seconds")

            self.session_manager.update_status(
                session_id,
                SessionStatus.FINISHED,
                progress=final_progress,
                elapsed_time_seconds=elapsed_time
            )
            # MongoDB Report 완료 상태 저장
            mongodb_service.update_report_status(
                session_id,
                report_status="finished",
                report_progress=final_progress
            )
            self._log_step(
                logger, session_id, "finished",
                "평가 완료", {
                    "evaluation_results": evaluation_results,
                    "evaluated_dialogues_count": len(evaluated_dialogues),
                    "dialogue_reports_count": len(results.get("dialogue_reports", [])),
                    "global_report_generated": bool(results.get("global_report"))
                }
            )

        except Exception as e:
            # 콜백 및 세션 로거 정리 (에러 발생 시에도 정리 필요)
            try:
                from core import clear_session_logger, clear_status_callback
                clear_status_callback()
                clear_session_logger()
            except ImportError:
                pass  # import 전에 에러 발생한 경우

            # 에러 상태로 업데이트
            error_msg = f"{type(e).__name__}: {str(e)}"
            error_trace = traceback.format_exc()

            self.session_manager.update_status(
                session_id,
                SessionStatus.ERROR,
                error_msg=error_msg
            )
            # MongoDB Report 에러 상태 저장
            mongodb_service.update_report_status(
                session_id,
                report_status="error",
                report_error=error_msg
            )

            self._log_step(
                logger, session_id, "error",
                "평가 실패", {"error": error_msg, "traceback": error_trace}
            )
            logger.print_error(f"Evaluation failed: {error_msg}")

    async def run_evaluation_async(
        self,
        session_id: str,
        mode: str,
        provider: dict,
        max_workers: int,
        metadata: Optional[Dict[str, Any]] = None,
        email: Optional[str] = None
    ) -> None:
        """평가 실행 (비동기 버전)

        동기 함수를 executor에서 실행합니다.

        Args:
            session_id: 세션 ID
            mode: API 모드 (openai/custom)
            provider: API 제공자 설정
            max_workers: 병렬 처리 worker 수
            metadata: 사용자 정의 메타데이터 (자유 형식)
            email: 사용자 이메일 (최상단 필드로 저장)
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            partial(
                self.run_evaluation_sync,
                session_id, mode, provider, max_workers, metadata, email
            )
        )


# 싱글톤 인스턴스 (session_manager와 함께 사용)
def get_evaluation_runner(session_manager: SessionManager) -> EvaluationRunner:
    """EvaluationRunner 인스턴스 반환

    Args:
        session_manager: 세션 관리자

    Returns:
        EvaluationRunner: 평가 실행기 인스턴스
    """
    return EvaluationRunner(session_manager)
