# -*- coding: utf-8 -*-
"""Full Evaluation 실행 서비스

Benchmark와 Report를 순차적으로 실행하는 End-to-End 평가 서비스
- Benchmark 먼저 실행
- Benchmark 완료 후 자동으로 Report 실행
- Benchmark 실패 시 Report 건너뛰고 종료
- 결과는 Leaderboard API를 통해 저장
"""
import asyncio
import json
import os
import time
import traceback
from datetime import datetime
from typing import Optional, Dict, Any

from api.services.session_manager import SessionManager
from api.services.benchmark_runner import BenchmarkRunner
from api.services.evaluation_runner import EvaluationRunner
from api.services.leaderboard_client import leaderboard_client
from user_logging_config import UserLogger


def make_json_log(thread_id: str, file_name: str, function_location: str,
                  message: str, value) -> str:
    """구조화된 로그 메시지 생성"""
    try:
        if isinstance(value, dict):
            value = json.dumps(value, indent=4, ensure_ascii=False)
        elif isinstance(value, list):
            value = json.dumps(value, indent=4, ensure_ascii=False)
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


class FullEvaluationRunner:
    """Full Evaluation 실행 클래스

    Benchmark와 Report를 순차적으로 백그라운드에서 실행합니다.
    결과는 Leaderboard API를 통해 저장됩니다.
    """

    def __init__(self, session_manager: SessionManager):
        """FullEvaluationRunner 초기화

        Args:
            session_manager: 세션 관리자 인스턴스
        """
        self.session_manager = session_manager
        self.benchmark_runner = BenchmarkRunner(session_manager)
        self.evaluation_runner = EvaluationRunner(session_manager)

    def _create_logger(self, session_id: str) -> UserLogger:
        """세션별 로거 생성

        Args:
            session_id: 세션 ID

        Returns:
            UserLogger: 로거 인스턴스
        """
        return UserLogger(session_id, version="1.0.0")

    def _log(self, logger: UserLogger, session_id: str,
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
            file_name="full_evaluation_runner",
            function_location=step,
            message=message,
            value=value
        )
        logger.print_info(log_msg)

    async def run_full_evaluation_async(
        self,
        session_id: str,
        mode: str,
        provider: dict,
        max_workers: int,
        metadata: Optional[Dict[str, Any]] = None,
        email: Optional[str] = None
    ) -> None:
        """Full Evaluation 실행 (비동기 버전)

        Benchmark 완료 후 자동으로 Report 실행.
        Benchmark 실패 시 Report 건너뛰고 종료.
        결과는 Leaderboard API를 통해 저장.

        Args:
            session_id: 세션 ID
            mode: API 모드 (openai/custom)
            provider: API 제공자 설정
            max_workers: 병렬 처리 worker 수
            metadata: 사용자 정의 메타데이터
            email: 사용자 이메일
        """
        logger = self._create_logger(session_id)
        leaderboard_id = None
        start_time = time.time()

        try:
            # ================================================================
            # Phase 0: Leaderboard 생성 (init 상태)
            # ================================================================
            self._log(
                logger, session_id, "full_evaluation_start",
                "Full Evaluation 시작 - Leaderboard 생성",
                {"mode": mode, "provider": provider, "max_workers": max_workers}
            )

            # Leaderboard API에 초기 상태 생성
            init_data = {
                "email": email or "",
                "pwd": os.getenv("LEADERBOARD_PASSWORD", ""),
                "status": "init",
                "metadata": metadata,
                "benchmark_status": "init",
                "benchmark_results": {},
                "benchmark_progress": {"current": 0, "total": 0},
                "results": [],
                "evaluation_results": {},
                "evaluated_dialogues": [],
                "dialogue_reports": [],
                "global_report": "",
                "report_timestamp": datetime.now().isoformat()
            }

            create_response = await leaderboard_client.create_leaderboard(init_data)
            leaderboard_id = create_response.get("_id")

            self._log(
                logger, session_id, "leaderboard_created",
                "Leaderboard 생성 완료",
                {"leaderboard_id": leaderboard_id}
            )

            # ================================================================
            # Phase 1: Benchmark 실행
            # ================================================================
            self._log(
                logger, session_id, "benchmark_start",
                "Phase 1: Benchmark 실행 시작",
                {}
            )

            # Benchmark 상태 업데이트 (running)
            await leaderboard_client.update_leaderboard(
                leaderboard_id,
                {
                    "email": email or "",
                    "pwd": os.getenv("LEADERBOARD_PASSWORD", ""),
                    "status": "running",
                    "metadata": metadata,
                    "benchmark_status": "running"
                }
            )

            # Benchmark 실행 (CPU-bound 작업이므로 스레드풀에서 실행)
            benchmark_results = await asyncio.to_thread(
                self._run_benchmark_internal,
                session_id=session_id,
                mode=mode,
                provider=provider,
                max_workers=max_workers,
                metadata=metadata,
                logger=logger
            )

            # Benchmark 결과 확인
            if benchmark_results is None or benchmark_results.get("status") == "error":
                # Benchmark 실패 - Leaderboard 에러 상태로 업데이트
                error_msg = benchmark_results.get("error", "Unknown benchmark error") if benchmark_results else "Benchmark failed"
                await leaderboard_client.update_leaderboard(
                    leaderboard_id,
                    {
                        "email": email or "",
                        "pwd": os.getenv("LEADERBOARD_PASSWORD", ""),
                        "status": "error",
                        "metadata": metadata,
                        "benchmark_status": "error",
                        "error_message": error_msg
                    }
                )
                self._log(
                    logger, session_id, "benchmark_failed",
                    "Benchmark 실패 - Report 건너뛰고 종료",
                    {"error": error_msg}
                )
                return

            # Benchmark 성공 - 결과 업데이트
            elapsed_benchmark = time.time() - start_time
            await leaderboard_client.update_leaderboard(
                leaderboard_id,
                {
                    "email": email or "",
                    "pwd": os.getenv("LEADERBOARD_PASSWORD", ""),
                    "status": "running",
                    "metadata": metadata,
                    "benchmark_status": "finished",
                    "benchmark_results": benchmark_results.get("results", {}),
                    "benchmark_elapsed_time_seconds": elapsed_benchmark
                }
            )

            self._log(
                logger, session_id, "benchmark_complete",
                "Benchmark 완료",
                {"elapsed_seconds": elapsed_benchmark}
            )

            # ================================================================
            # Phase 2: Report 실행 (Benchmark 성공 시에만)
            # ================================================================
            self._log(
                logger, session_id, "report_start",
                "Phase 2: Report 실행 시작",
                {}
            )

            # Report 실행 (CPU-bound 작업이므로 스레드풀에서 실행)
            report_results = await asyncio.to_thread(
                self._run_report_internal,
                session_id=session_id,
                mode=mode,
                provider=provider,
                max_workers=max_workers,
                metadata=metadata,
                logger=logger
            )

            # Report 결과 확인
            if report_results is None or report_results.get("status") == "error":
                # Report 실패
                error_msg = report_results.get("error", "Unknown report error") if report_results else "Report failed"
                await leaderboard_client.update_leaderboard(
                    leaderboard_id,
                    {
                        "email": email or "",
                        "pwd": os.getenv("LEADERBOARD_PASSWORD", ""),
                        "status": "error",
                        "metadata": metadata,
                        "benchmark_status": "finished",
                        "benchmark_results": benchmark_results.get("results", {}),
                        "error_message": error_msg
                    }
                )
                self._log(
                    logger, session_id, "report_failed",
                    "Report 실패",
                    {"error": error_msg}
                )
                return

            # ================================================================
            # Phase 3: 전체 완료 - 최종 결과 저장
            # ================================================================
            elapsed_total = time.time() - start_time

            final_data = {
                "email": email or "",
                "pwd": os.getenv("LEADERBOARD_PASSWORD", ""),
                "status": "finished",
                "metadata": metadata,
                # Benchmark 결과
                "benchmark_status": "finished",
                "benchmark_results": benchmark_results.get("results", {}),
                "benchmark_elapsed_time_seconds": elapsed_benchmark,
                "benchmark_progress": {"current": 100, "total": 100},
                # Report 결과
                "results": report_results.get("results", []),
                "evaluation_results": report_results.get("evaluation_results", {}),
                "evaluated_dialogues": report_results.get("evaluated_dialogues", []),
                "dialogue_reports": report_results.get("dialogue_reports", []),
                "global_report": report_results.get("global_report", ""),
                "report_timestamp": datetime.now().isoformat(),
                # 총 소요 시간
                "total_elapsed_time_seconds": elapsed_total
            }

            await leaderboard_client.update_leaderboard(leaderboard_id, final_data)

            self._log(
                logger, session_id, "full_evaluation_complete",
                "Full Evaluation 완료",
                {
                    "leaderboard_id": leaderboard_id,
                    "total_elapsed_seconds": elapsed_total
                }
            )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            error_trace = traceback.format_exc()

            # Leaderboard 에러 상태 업데이트 (ID가 있는 경우에만)
            if leaderboard_id:
                try:
                    await leaderboard_client.update_leaderboard(
                        leaderboard_id,
                        {
                            "email": email or "",
                            "pwd": os.getenv("LEADERBOARD_PASSWORD", ""),
                            "status": "error",
                            "metadata": metadata,
                            "error_message": error_msg
                        }
                    )
                except Exception as update_error:
                    self._log(
                        logger, session_id, "leaderboard_update_error",
                        "Leaderboard 에러 상태 업데이트 실패",
                        {"error": str(update_error)}
                    )

            self._log(
                logger, session_id, "full_evaluation_error",
                "Full Evaluation 실패",
                {"error": error_msg, "traceback": error_trace}
            )
            logger.print_error(f"Full Evaluation failed: {error_msg}")

    def _run_benchmark_internal(
        self,
        session_id: str,
        mode: str,
        provider: dict,
        max_workers: int,
        metadata: Optional[Dict[str, Any]],
        logger: UserLogger
    ) -> Optional[Dict[str, Any]]:
        """Benchmark 내부 실행 (Leaderboard API 호출 없이)

        Args:
            session_id: 세션 ID
            mode: API 모드
            provider: 제공자 설정
            max_workers: 워커 수
            metadata: 메타데이터
            logger: 로거

        Returns:
            Dict: Benchmark 결과 또는 None (실패 시)
        """
        try:
            from core import (
                configure_api_settings,
                load_prompt_template,
                query_llm,
                set_session_logger,
                clear_session_logger
            )

            configure_api_settings(mode, provider)
            set_session_logger(logger, session_id)

            # 프롬프트 로드
            toxicity_prompt = load_prompt_template("user_toxicity_analysis_system.txt")
            safety_prompt = load_prompt_template("assistant_toxicity_analysis_system.txt")

            # BenchmarkRunner의 내부 로직 실행
            from api.services.benchmark_runner import BenchmarkRunner, DIFFICULTY_LEVELS
            benchmark_runner = BenchmarkRunner(self.session_manager)

            benchmark_results = {
                "toxicity": {},
                "safety": {}
            }

            # Toxicity 벤치마크
            for difficulty in DIFFICULTY_LEVELS:
                data = benchmark_runner._load_benchmark_data("toxicity", difficulty)
                results = []
                from concurrent.futures import ThreadPoolExecutor, as_completed

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            benchmark_runner._evaluate_single_toxicity_item,
                            item, idx, query_llm, toxicity_prompt
                        ): idx for idx, item in enumerate(data)
                    }
                    for future in as_completed(futures):
                        results.append(future.result())

                results.sort(key=lambda x: x["index"])
                metrics = benchmark_runner._calculate_metrics(results, "toxicity")
                benchmark_results["toxicity"][difficulty] = {
                    "items": results,
                    "metrics": metrics
                }

            benchmark_results["toxicity"]["overall_metrics"] = benchmark_runner._calculate_overall_metrics(
                benchmark_results["toxicity"], "toxicity"
            )

            # Safety 벤치마크
            for difficulty in DIFFICULTY_LEVELS:
                data = benchmark_runner._load_benchmark_data("safety", difficulty)
                results = []

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            benchmark_runner._evaluate_single_safety_item,
                            item, idx, query_llm, safety_prompt
                        ): idx for idx, item in enumerate(data)
                    }
                    for future in as_completed(futures):
                        results.append(future.result())

                results.sort(key=lambda x: x["index"])
                metrics = benchmark_runner._calculate_metrics(results, "safety")
                benchmark_results["safety"][difficulty] = {
                    "items": results,
                    "metrics": metrics
                }

            benchmark_results["safety"]["overall_metrics"] = benchmark_runner._calculate_overall_metrics(
                benchmark_results["safety"], "safety"
            )

            clear_session_logger()

            return {
                "status": "finished",
                "results": benchmark_results
            }

        except Exception as e:
            try:
                from core import clear_session_logger
                clear_session_logger()
            except:
                pass

            return {
                "status": "error",
                "error": str(e)
            }

    def _run_report_internal(
        self,
        session_id: str,
        mode: str,
        provider: dict,
        max_workers: int,
        metadata: Optional[Dict[str, Any]],
        logger: UserLogger
    ) -> Optional[Dict[str, Any]]:
        """Report (GOAT 평가) 내부 실행 (Leaderboard API 호출 없이)

        Args:
            session_id: 세션 ID
            mode: API 모드
            provider: 제공자 설정
            max_workers: 워커 수
            metadata: 메타데이터
            logger: 로거

        Returns:
            Dict: Report 결과 또는 None (실패 시)
        """
        try:
            from core import (
                configure_api_settings,
                main as core_main,
                set_session_logger,
                clear_session_logger,
                set_status_callback,
                clear_status_callback,
                convert_results_to_dialogues,
                evaluate_dialogues,
                generate_evaluation_reports
            )

            configure_api_settings(mode, provider)
            set_session_logger(logger, session_id)

            # 빈 콜백 (상태 업데이트는 Leaderboard API로만)
            def empty_callback(status: str, progress: dict = None):
                pass

            set_status_callback(empty_callback)

            # main() 실행 (대화 생성)
            results = core_main(max_workers=max_workers)

            # 턴별 평가
            raw_results = results.get("results", [])
            dialogues = convert_results_to_dialogues(raw_results)
            evaluated_dialogues = evaluate_dialogues(dialogues, context=session_id)

            # 보고서 생성
            report_results = generate_evaluation_reports(
                evaluated_dialogues=evaluated_dialogues,
                generate_dialogue_reports=True,
                generate_global=True,
                model_name=provider.get("model", "gpt-4o")
            )

            clear_status_callback()
            clear_session_logger()

            return {
                "status": "finished",
                "results": results.get("results", []),
                "evaluation_results": results.get("evaluation_results", {}),
                "evaluated_dialogues": evaluated_dialogues,
                "dialogue_reports": report_results.get("dialogue_reports", []),
                "global_report": report_results.get("global_report", "")
            }

        except Exception as e:
            try:
                from core import clear_session_logger, clear_status_callback
                clear_status_callback()
                clear_session_logger()
            except:
                pass

            return {
                "status": "error",
                "error": str(e)
            }


def get_full_evaluation_runner(session_manager: SessionManager) -> FullEvaluationRunner:
    """FullEvaluationRunner 인스턴스 반환

    Args:
        session_manager: 세션 관리자

    Returns:
        FullEvaluationRunner: Full Evaluation 실행기 인스턴스
    """
    return FullEvaluationRunner(session_manager)
