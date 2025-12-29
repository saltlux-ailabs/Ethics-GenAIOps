# -*- coding: utf-8 -*-
"""평가 보고서 생성 서비스

history_evaluation_prompt.txt를 사용하여 LLM이 평가 보고서를 생성하고
MongoDB와 로컬 파일에 저장하는 서비스
"""
import json
import time
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional

from api.services.session_manager import SessionManager
from api.services.mongodb_service import mongodb_service
from user_logging_config import UserLogger

# 카테고리 코드 -> 한글 이름 매핑
CATEGORY_NAMES = {
    "A": "법적 위험성",
    "B": "혐오 및 차별표현",
    "C": "무책임한 조언",
    "D": "사회문화적 규범위반"
}


def make_json_log(
    session_id: str,
    file_name: str,
    function_location: str,
    message: str,
    value: Any = None
) -> str:
    """구조화된 로그 메시지 생성

    Args:
        session_id: 세션 ID
        file_name: 파일명
        function_location: 함수 위치
        message: 로그 메시지
        value: 로그 값

    Returns:
        str: 포맷된 로그 메시지
    """
    try:
        if isinstance(value, dict):
            value_str = json.dumps(value, indent=4, ensure_ascii=False)
        elif isinstance(value, list):
            value_str = json.dumps(value, indent=4, ensure_ascii=False)
        elif value is None:
            value_str = "None"
        else:
            value_str = str(value)
    except Exception:
        value_str = str(value)

    log_message = f"""
    timestamp: {datetime.now().isoformat()}
    session_id: {session_id}
    file_name: {file_name}
    function_location: {function_location}
    message: {message}
    value: {value_str}
    """
    return log_message


class ReportGenerator:
    """평가 보고서 생성 클래스

    MongoDB의 results를 읽어 history_evaluation_prompt.txt를 사용하여
    LLM이 평가 보고서를 생성하고 저장합니다.
    """

    def __init__(self, session_manager: SessionManager):
        """ReportGenerator 초기화

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

    def _format_conversation_for_evaluation(
        self,
        messages: List[Dict],
        category: str,
        dialogue_num: int = 1
    ) -> str:
        """대화 데이터를 평가용 포맷으로 변환

        Args:
            messages: 대화 메시지 리스트 [{"role": "system/user/assistant", "content": "..."}]
            category: 카테고리 코드 (A, B, C, D)
            dialogue_num: 대화 번호 (기본값 1)

        Returns:
            str: 포맷팅된 대화 텍스트
        """
        category_name = CATEGORY_NAMES.get(category, category)
        formatted = f"## 평가 대상 대화\n\n**범주**: {category} ({category_name})\n\n"

        turn_num = 0
        pending_user = None

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                # system 메시지는 건너뜀 (별도로 처리)
                continue
            elif role == "user":
                # user 메시지는 Interlocutor로 표시
                pending_user = content
            elif role == "assistant":
                # assistant와 함께 한 턴으로 출력
                if pending_user is not None:
                    turn_num += 1
                    formatted += f"**턴 [{dialogue_num}-{turn_num}]**\n"
                    formatted += f"- Interlocutor: \"{pending_user}\"\n"
                    formatted += f"- Target: \"{content}\"\n\n"
                    pending_user = None

        # 마지막 user 메시지가 남아있으면 (assistant 응답 없이)
        if pending_user is not None:
            turn_num += 1
            formatted += f"**턴 [{dialogue_num}-{turn_num}]**\n"
            formatted += f"- Interlocutor: \"{pending_user}\"\n"
            formatted += f"- Target: (응답 없음)\n\n"

        return formatted

    def generate_report_for_result(
        self,
        result: Dict[str, Any],
        result_index: int,
        dialogue_num: int = 1,
        context: str = ""
    ) -> str:
        """단일 result에 대한 보고서 생성

        Args:
            result: 단일 대화 결과 {"category": str, "message": List, ...}
            result_index: results 배열 내 인덱스
            dialogue_num: 대화 번호
            context: 로깅용 컨텍스트

        Returns:
            str: LLM이 생성한 평가 보고서
        """
        from core import load_prompt_template, query_llm

        # history_evaluation_prompt.txt 로드
        system_prompt = load_prompt_template("history_evaluation_prompt.txt")

        # 대화 데이터 포맷팅
        messages = result.get("message", [])
        category = result.get("category", "A")
        formatted_conversation = self._format_conversation_for_evaluation(
            messages, category, dialogue_num
        )

        # LLM 메시지 구성
        eval_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_conversation}
        ]

        # LLM 호출
        report = query_llm(eval_messages, model_name="target", context=f"report_gen:{context}")

        return report

    def _log(
        self,
        logger: UserLogger,
        session_id: str,
        function_location: str,
        message: str,
        value: Any = None,
        level: str = "info"
    ) -> None:
        """구조화된 로그 기록

        Args:
            logger: 로거 인스턴스
            session_id: 세션 ID
            function_location: 함수 위치
            message: 로그 메시지
            value: 로그 값
            level: 로그 레벨 (info, warning, error)
        """
        log_msg = make_json_log(
            session_id=session_id,
            file_name="report_generator",
            function_location=function_location,
            message=message,
            value=value
        )

        if level == "info":
            logger.print_info(log_msg)
        elif level == "warning":
            logger.print_warning(log_msg)
        elif level == "error":
            logger.print_error(log_msg)

    def _process_single_result(
        self,
        session_id: str,
        result: Dict[str, Any],
        result_index: int,
        logger: UserLogger
    ) -> Dict[str, Any]:
        """단일 result 처리 (병렬 처리용)

        Args:
            session_id: 세션 ID
            result: 단일 대화 결과
            result_index: results 배열 내 인덱스
            logger: 로거 인스턴스

        Returns:
            Dict: 처리 결과 {"index": int, "success": bool, "error": str|None, "report": str|None}
        """
        start_time = time.time()
        category = result.get("category", "Unknown")
        initial_question = result.get("initial_question", "")[:50]

        try:
            context = f"session_{session_id}_result_{result_index}"

            # 1. 보고서 생성 시작 로깅
            self._log(
                logger, session_id, "_process_single_result",
                f"보고서 생성 시작 - Result {result_index}",
                {
                    "result_index": result_index,
                    "category": category,
                    "initial_question": initial_question,
                    "message_count": len(result.get("message", []))
                }
            )

            # 2. 대화 포맷팅
            messages = result.get("message", [])
            formatted_conversation = self._format_conversation_for_evaluation(
                messages, category, result_index + 1
            )

            self._log(
                logger, session_id, "_process_single_result",
                f"대화 포맷팅 완료 - Result {result_index}",
                {
                    "result_index": result_index,
                    "formatted_length": len(formatted_conversation),
                    "formatted_preview": formatted_conversation[:500] + "..." if len(formatted_conversation) > 500 else formatted_conversation
                }
            )

            # 3. LLM 호출하여 보고서 생성
            from core import load_prompt_template, query_llm

            system_prompt = load_prompt_template("history_evaluation_prompt.txt")

            eval_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted_conversation}
            ]

            self._log(
                logger, session_id, "_process_single_result",
                f"LLM 호출 시작 - Result {result_index}",
                {
                    "result_index": result_index,
                    "system_prompt_length": len(system_prompt),
                    "user_message_length": len(formatted_conversation)
                }
            )

            report = query_llm(eval_messages, model_name="target", context=f"report_gen:{context}")

            llm_elapsed = time.time() - start_time
            self._log(
                logger, session_id, "_process_single_result",
                f"LLM 보고서 생성 완료 - Result {result_index}",
                {
                    "result_index": result_index,
                    "report_length": len(report) if report else 0,
                    "llm_elapsed_seconds": round(llm_elapsed, 2),
                    "report_preview": report[:500] + "..." if report and len(report) > 500 else report
                }
            )

            # 4. MongoDB에 report 필드 업데이트
            success = mongodb_service.update_result_report(
                session_id=session_id,
                result_index=result_index,
                report=report
            )

            total_elapsed = time.time() - start_time

            if success:
                self._log(
                    logger, session_id, "_process_single_result",
                    f"MongoDB 저장 성공 - Result {result_index}",
                    {
                        "result_index": result_index,
                        "category": category,
                        "report_length": len(report) if report else 0,
                        "total_elapsed_seconds": round(total_elapsed, 2)
                    }
                )
            else:
                self._log(
                    logger, session_id, "_process_single_result",
                    f"MongoDB 저장 실패 - Result {result_index}",
                    {
                        "result_index": result_index,
                        "category": category
                    },
                    level="warning"
                )

            return {
                "index": result_index,
                "success": success,
                "error": None,
                "report": report
            }

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            error_trace = traceback.format_exc()
            elapsed = time.time() - start_time

            self._log(
                logger, session_id, "_process_single_result",
                f"보고서 생성 실패 - Result {result_index}",
                {
                    "result_index": result_index,
                    "category": category,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "error_trace": error_trace,
                    "elapsed_seconds": round(elapsed, 2)
                },
                level="error"
            )

            return {
                "index": result_index,
                "success": False,
                "error": error_msg,
                "report": None
            }

    def run_report_generation_sync(
        self,
        session_id: str,
        mode: str,
        provider: dict,
        max_workers: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """보고서 생성 실행 (동기 버전, 병렬 처리)

        백그라운드 태스크에서 직접 호출됩니다.
        MongoDB와 로컬 파일(results.json) 모두에 저장합니다.

        Args:
            session_id: 세션 ID
            mode: API 모드 (openai/custom)
            provider: API 제공자 설정
            max_workers: 병렬 처리 worker 수
            metadata: 사용자 정의 메타데이터 (자유 형식, 선택)
        """
        logger = self._create_logger(session_id)
        start_time = time.time()

        try:
            # ========================================
            # 1. 초기화 및 설정
            # ========================================
            self._log(
                logger, session_id, "run_report_generation_sync",
                "보고서 생성 시작",
                {
                    "mode": mode,
                    "provider": provider,
                    "max_workers": max_workers,
                    "metadata": metadata
                }
            )

            # API 설정 적용
            from core import configure_api_settings

            configure_api_settings(mode, provider)

            self._log(
                logger, session_id, "run_report_generation_sync",
                "API 설정 완료",
                {
                    "mode": mode,
                    "model": provider.get("model"),
                    "url": provider.get("url", "N/A (OpenAI)")
                }
            )

            # ========================================
            # 2. MongoDB에서 results 조회
            # ========================================
            self._log(
                logger, session_id, "run_report_generation_sync",
                "MongoDB 데이터 조회 시작",
                {"session_id": session_id}
            )

            session_data = mongodb_service.get_results(session_id)

            if session_data is None:
                raise ValueError(f"Session not found in MongoDB: {session_id}")

            results = session_data.get("results", [])
            if not results:
                raise ValueError(f"No results found for session: {session_id}")

            total_results = len(results)

            # 카테고리별 통계
            category_counts = {}
            for r in results:
                cat = r.get("category", "Unknown")
                category_counts[cat] = category_counts.get(cat, 0) + 1

            self._log(
                logger, session_id, "run_report_generation_sync",
                "MongoDB 데이터 조회 완료",
                {
                    "total_results": total_results,
                    "category_counts": category_counts,
                    "session_status": session_data.get("status"),
                    "has_evaluation_results": "evaluation_results" in session_data
                }
            )

            # ========================================
            # 3. 초기 상태 업데이트
            # ========================================
            mongodb_service.update_report_status(
                session_id=session_id,
                report_status="generating",
                report_progress={"total": total_results, "completed": 0, "failed": 0}
            )

            self._log(
                logger, session_id, "run_report_generation_sync",
                "보고서 생성 상태 초기화",
                {
                    "report_status": "generating",
                    "total": total_results
                }
            )

            # ========================================
            # 4. 병렬 처리로 보고서 생성
            # ========================================
            completed = 0
            failed = 0
            failed_indices = []
            generated_reports = {}  # index -> report 저장용

            self._log(
                logger, session_id, "run_report_generation_sync",
                "병렬 처리 시작",
                {
                    "max_workers": max_workers,
                    "total_tasks": total_results
                }
            )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 작업 제출
                future_to_index = {
                    executor.submit(
                        self._process_single_result,
                        session_id,
                        result,
                        idx,
                        logger
                    ): idx
                    for idx, result in enumerate(results)
                }

                self._log(
                    logger, session_id, "run_report_generation_sync",
                    "모든 작업 제출 완료",
                    {"submitted_tasks": len(future_to_index)}
                )

                # 결과 수집
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        result = future.result()
                        if result["success"]:
                            completed += 1
                            generated_reports[idx] = result.get("report")
                        else:
                            failed += 1
                            failed_indices.append(idx)

                        # 진행률 업데이트
                        mongodb_service.update_report_status(
                            session_id=session_id,
                            report_status="generating",
                            report_progress={
                                "total": total_results,
                                "completed": completed,
                                "failed": failed
                            }
                        )

                        # 10% 단위로 진행률 로깅
                        progress_pct = (completed + failed) / total_results * 100
                        if (completed + failed) == 1 or progress_pct % 10 < (1 / total_results * 100):
                            self._log(
                                logger, session_id, "run_report_generation_sync",
                                f"진행률 업데이트",
                                {
                                    "progress": f"{completed + failed}/{total_results}",
                                    "progress_percent": round(progress_pct, 1),
                                    "completed": completed,
                                    "failed": failed
                                }
                            )

                    except Exception as e:
                        failed += 1
                        failed_indices.append(idx)
                        self._log(
                            logger, session_id, "run_report_generation_sync",
                            f"작업 수집 중 예외 발생 - Result {idx}",
                            {
                                "result_index": idx,
                                "error_type": type(e).__name__,
                                "error_message": str(e)
                            },
                            level="error"
                        )

            # ========================================
            # 5. 완료 상태 업데이트
            # ========================================
            elapsed_time = time.time() - start_time
            final_status = "finished" if failed == 0 else "finished_with_errors"

            mongodb_service.update_report_status(
                session_id=session_id,
                report_status=final_status,
                report_progress={
                    "total": total_results,
                    "completed": completed,
                    "failed": failed
                }
            )

            self._log(
                logger, session_id, "run_report_generation_sync",
                "MongoDB 상태 업데이트 완료",
                {
                    "final_status": final_status,
                    "total": total_results,
                    "completed": completed,
                    "failed": failed,
                    "failed_indices": failed_indices if failed_indices else "None",
                    "elapsed_seconds": round(elapsed_time, 2)
                }
            )

            # ========================================
            # 6. 로컬 파일(results.json)에도 저장
            # ========================================
            self._log(
                logger, session_id, "run_report_generation_sync",
                "로컬 파일 저장 시작",
                {"session_id": session_id}
            )

            try:
                # MongoDB에서 최신 데이터 다시 조회 (보고서 포함)
                updated_session_data = mongodb_service.get_results(session_id)

                if updated_session_data:
                    # _id 필드 제거 (JSON 직렬화 문제 방지)
                    if "_id" in updated_session_data:
                        del updated_session_data["_id"]

                    # 로컬 파일에 저장
                    self.session_manager.save_results(session_id, updated_session_data)

                    self._log(
                        logger, session_id, "run_report_generation_sync",
                        "로컬 파일 저장 완료",
                        {
                            "session_id": session_id,
                            "results_count": len(updated_session_data.get("results", [])),
                            "reports_included": sum(
                                1 for r in updated_session_data.get("results", [])
                                if r.get("report")
                            )
                        }
                    )
                else:
                    self._log(
                        logger, session_id, "run_report_generation_sync",
                        "로컬 파일 저장 실패 - MongoDB 데이터 조회 실패",
                        {"session_id": session_id},
                        level="warning"
                    )

            except Exception as e:
                self._log(
                    logger, session_id, "run_report_generation_sync",
                    "로컬 파일 저장 중 예외 발생",
                    {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "error_trace": traceback.format_exc()
                    },
                    level="error"
                )

            # ========================================
            # 7. 최종 완료 로깅
            # ========================================
            total_elapsed = time.time() - start_time

            self._log(
                logger, session_id, "run_report_generation_sync",
                "보고서 생성 완료",
                {
                    "final_status": final_status,
                    "total_results": total_results,
                    "completed": completed,
                    "failed": failed,
                    "failed_indices": failed_indices if failed_indices else "None",
                    "total_elapsed_seconds": round(total_elapsed, 2),
                    "avg_time_per_report": round(total_elapsed / total_results, 2) if total_results > 0 else 0,
                    "mode": mode,
                    "model": provider.get("model"),
                    "max_workers": max_workers
                }
            )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            error_trace = traceback.format_exc()
            elapsed = time.time() - start_time

            mongodb_service.update_report_status(
                session_id=session_id,
                report_status="error",
                report_error=error_msg
            )

            self._log(
                logger, session_id, "run_report_generation_sync",
                "보고서 생성 치명적 오류",
                {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "error_trace": error_trace,
                    "elapsed_seconds": round(elapsed, 2)
                },
                level="error"
            )


# 팩토리 함수
def get_report_generator(session_manager: SessionManager) -> ReportGenerator:
    """ReportGenerator 인스턴스 반환

    Args:
        session_manager: 세션 관리자

    Returns:
        ReportGenerator: 보고서 생성기 인스턴스
    """
    return ReportGenerator(session_manager)

