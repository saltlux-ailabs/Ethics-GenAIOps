# -*- coding: utf-8 -*-
"""벤치마크 실행 서비스

자체 벤치마크 데이터를 사용하여 LLM의 판단 정확도를 측정합니다.
- Toxicity Benchmark: user 메시지 유해성 평가 (0-5, HD)
- Safety Benchmark: assistant 응답 안전성 평가 (SAFE/UNSAFE)
"""
import json
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

from api.models.schemas import BenchmarkStatus
from api.services.session_manager import SessionManager
from api.services.mongodb_service import mongodb_service
from user_logging_config import UserLogger


# 벤치마크 리소스 경로
BENCHMARK_DIR = Path(__file__).parent.parent.parent / "resource" / "benchmark"
TOXICITY_DIR = BENCHMARK_DIR / "toxicity_fmt"
SAFETY_DIR = BENCHMARK_DIR / "safety_fmt"

# 난이도 레벨
DIFFICULTY_LEVELS = ["easy", "medium", "hard"]


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


class BenchmarkRunner:
    """벤치마크 실행 클래스

    toxicity_fmt와 safety_fmt 벤치마크를 실행하고
    정확도, Precision, Recall, F1 Score를 계산합니다.
    """

    def __init__(self, session_manager: SessionManager):
        """BenchmarkRunner 초기화

        Args:
            session_manager: 세션 관리자 인스턴스
        """
        self.session_manager = session_manager

    def _create_logger(self, session_id: str) -> UserLogger:
        """세션별 로거 생성"""
        return UserLogger(session_id, version="1.0.0")

    def _log(self, logger: UserLogger, session_id: str,
             step: str, message: str, value) -> None:
        """단계별 로깅"""
        log_msg = make_json_log(
            thread_id=session_id,
            file_name="benchmark_runner",
            function_location=step,
            message=message,
            value=value
        )
        logger.print_info(log_msg)

    def _load_benchmark_data(self, benchmark_type: str, difficulty: str) -> List[Dict]:
        """벤치마크 데이터 로드

        Args:
            benchmark_type: "toxicity" 또는 "safety"
            difficulty: "easy", "medium", "hard"

        Returns:
            List[Dict]: 벤치마크 아이템 리스트
        """
        if benchmark_type == "toxicity":
            file_path = TOXICITY_DIR / f"{difficulty}.json"
        else:
            file_path = SAFETY_DIR / f"{difficulty}.json"

        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _evaluate_single_toxicity_item(
        self,
        item: Dict,
        index: int,
        query_llm_func,
        system_prompt: str
    ) -> Dict[str, Any]:
        """단일 toxicity 아이템 평가

        Args:
            item: 벤치마크 아이템 {"messages": [...], "label": "0-5 or HD"}
            index: 아이템 인덱스
            query_llm_func: LLM 쿼리 함수
            system_prompt: 시스템 프롬프트

        Returns:
            Dict: 평가 결과
        """
        try:
            messages = item["messages"]
            label = item["label"]

            # 평가용 메시지 구성
            eval_messages = [{"role": "system", "content": system_prompt}]
            for msg in messages:
                if msg["role"] in ["user", "assistant"]:
                    eval_messages.append(msg)

            # LLM 호출
            prediction = query_llm_func(
                eval_messages,
                model_name="target",
                context=f"benchmark_toxicity_{index}"
            )
            prediction = prediction.strip()

            # 결과 정규화 (공백, 개행 제거)
            prediction = prediction.replace(" ", "").replace("\n", "")

            return {
                "index": index,
                "messages": messages,
                "label": label,
                "prediction": prediction,
                "correct": label == prediction,
                "error": None
            }
        except Exception as e:
            return {
                "index": index,
                "messages": item.get("messages", []),
                "label": item.get("label", ""),
                "prediction": "N/A",
                "correct": False,
                "error": str(e)
            }

    def _evaluate_single_safety_item(
        self,
        item: Dict,
        index: int,
        query_llm_func,
        system_prompt: str
    ) -> Dict[str, Any]:
        """단일 safety 아이템 평가

        Args:
            item: 벤치마크 아이템 {"messages": [...], "label": "SAFE or UNSAFE"}
            index: 아이템 인덱스
            query_llm_func: LLM 쿼리 함수
            system_prompt: 시스템 프롬프트

        Returns:
            Dict: 평가 결과
        """
        try:
            messages = item["messages"]
            label = item["label"]

            # 평가용 메시지 구성
            eval_messages = [{"role": "system", "content": system_prompt}]
            for msg in messages:
                if msg["role"] in ["user", "assistant"]:
                    eval_messages.append(msg)

            # LLM 호출
            prediction = query_llm_func(
                eval_messages,
                model_name="target",
                context=f"benchmark_safety_{index}"
            )
            prediction = prediction.strip().upper()

            # 결과 정규화
            if "UNSAFE" in prediction:
                prediction = "UNSAFE"
            elif "SAFE" in prediction:
                prediction = "SAFE"

            return {
                "index": index,
                "messages": messages,
                "label": label,
                "prediction": prediction,
                "correct": label == prediction,
                "error": None
            }
        except Exception as e:
            return {
                "index": index,
                "messages": item.get("messages", []),
                "label": item.get("label", ""),
                "prediction": "N/A",
                "correct": False,
                "error": str(e)
            }

    def _calculate_metrics(
        self,
        results: List[Dict],
        benchmark_type: str
    ) -> Dict[str, Any]:
        """메트릭 계산

        Args:
            results: 평가 결과 리스트
            benchmark_type: "toxicity" 또는 "safety"

        Returns:
            Dict: accuracy, precision, recall, f1_score, total_samples, hd_excluded
        """
        if benchmark_type == "toxicity":
            # HD 제외 (Option B)
            filtered_results = [
                r for r in results
                if r["label"] != "HD" and r["prediction"] != "HD"
            ]
            hd_excluded = len(results) - len(filtered_results)

            if not filtered_results:
                return {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                    "total_samples": len(results),
                    "hd_excluded": hd_excluded
                }

            # Multi-class metrics (macro average)
            labels = set(r["label"] for r in filtered_results)
            correct = sum(1 for r in filtered_results if r["correct"])
            accuracy = correct / len(filtered_results)

            # Per-class precision, recall
            precisions = []
            recalls = []
            for label in labels:
                tp = sum(1 for r in filtered_results if r["label"] == label and r["prediction"] == label)
                fp = sum(1 for r in filtered_results if r["label"] != label and r["prediction"] == label)
                fn = sum(1 for r in filtered_results if r["label"] == label and r["prediction"] != label)

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

                precisions.append(precision)
                recalls.append(recall)

            avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
            avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
            f1_score = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0

            return {
                "accuracy": round(accuracy, 4),
                "precision": round(avg_precision, 4),
                "recall": round(avg_recall, 4),
                "f1_score": round(f1_score, 4),
                "total_samples": len(filtered_results),
                "hd_excluded": hd_excluded
            }

        else:  # safety (binary)
            if not results:
                return {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                    "total_samples": 0,
                    "hd_excluded": 0
                }

            correct = sum(1 for r in results if r["correct"])
            accuracy = correct / len(results)

            # Binary: UNSAFE를 positive로 간주
            tp = sum(1 for r in results if r["label"] == "UNSAFE" and r["prediction"] == "UNSAFE")
            fp = sum(1 for r in results if r["label"] == "SAFE" and r["prediction"] == "UNSAFE")
            fn = sum(1 for r in results if r["label"] == "UNSAFE" and r["prediction"] == "SAFE")

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            return {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1_score, 4),
                "total_samples": len(results),
                "hd_excluded": 0
            }

    def _calculate_overall_metrics(
        self,
        difficulty_results: Dict[str, Dict],
        benchmark_type: str
    ) -> Dict[str, Any]:
        """전체 메트릭 계산 (모든 난이도 합산)

        Args:
            difficulty_results: 난이도별 결과 {"easy": {...}, "medium": {...}, "hard": {...}}
            benchmark_type: "toxicity" 또는 "safety"

        Returns:
            Dict: 전체 메트릭
        """
        all_results = []
        # 난이도별 결과만 처리 (overall_metrics 제외)
        for difficulty in DIFFICULTY_LEVELS:
            if difficulty in difficulty_results:
                data = difficulty_results[difficulty]
                all_results.extend(data.get("items", []))

        return self._calculate_metrics(all_results, benchmark_type)

    def run_benchmark_sync(
        self,
        session_id: str,
        mode: str,
        provider: dict,
        max_workers: int,
        metadata: Optional[Dict[str, Any]] = None,
        email: Optional[str] = None
    ) -> None:
        """벤치마크 실행 (동기 버전)

        Args:
            session_id: 세션 ID
            mode: API 모드 (openai/custom)
            provider: API 제공자 설정
            max_workers: 병렬 처리 worker 수
            metadata: 사용자 정의 메타데이터
            email: 사용자 이메일 (최상단 필드로 저장)
        """
        logger = self._create_logger(session_id)
        start_time = time.time()

        try:
            # 1. 초기 상태 업데이트 (email은 최상단 필드로 저장)
            mongodb_service.update_benchmark_status(
                session_id,
                status="running",
                metadata=metadata,
                email=email
            )
            self._log(
                logger, session_id, "start",
                "벤치마크 시작",
                {"mode": mode, "provider": provider, "max_workers": max_workers}
            )

            # 2. core.py 설정 적용
            from core import (
                configure_api_settings,
                load_prompt_template,
                query_llm,
                set_session_logger,
                clear_session_logger
            )

            configure_api_settings(mode, provider)
            set_session_logger(logger, session_id)

            self._log(
                logger, session_id, "configure",
                "API 설정 완료",
                {"mode": mode}
            )

            # 3. 프롬프트 로드
            toxicity_prompt = load_prompt_template("user_toxicity_analysis_system.txt")
            safety_prompt = load_prompt_template("assistant_toxicity_analysis_system.txt")

            # 4. 벤치마크 결과 저장 구조
            benchmark_results = {
                "toxicity": {},
                "safety": {}
            }

            # 전체 아이템 수 계산 (진행률 표시용)
            total_items = 0
            for difficulty in DIFFICULTY_LEVELS:
                toxicity_data = self._load_benchmark_data("toxicity", difficulty)
                safety_data = self._load_benchmark_data("safety", difficulty)
                total_items += len(toxicity_data) + len(safety_data)

            completed_items = 0
            failed_items = 0

            # 5. Toxicity 벤치마크 실행
            self._log(logger, session_id, "toxicity_start", "Toxicity 벤치마크 시작", {})

            for difficulty in DIFFICULTY_LEVELS:
                data = self._load_benchmark_data("toxicity", difficulty)
                self._log(
                    logger, session_id, f"toxicity_{difficulty}",
                    f"Toxicity {difficulty} 데이터 로드",
                    {"count": len(data)}
                )

                results = []
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            self._evaluate_single_toxicity_item,
                            item, idx, query_llm, toxicity_prompt
                        ): idx for idx, item in enumerate(data)
                    }

                    for future in as_completed(futures):
                        result = future.result()
                        results.append(result)
                        completed_items += 1

                        if result.get("error"):
                            failed_items += 1

                        # 진행률 업데이트 (10% 단위)
                        if completed_items % max(1, total_items // 10) == 0:
                            mongodb_service.update_benchmark_status(
                                session_id,
                                progress={
                                    "total": total_items,
                                    "completed": completed_items,
                                    "failed": failed_items
                                }
                            )

                # 인덱스 순으로 정렬
                results.sort(key=lambda x: x["index"])

                # 메트릭 계산
                metrics = self._calculate_metrics(results, "toxicity")

                benchmark_results["toxicity"][difficulty] = {
                    "items": results,
                    "metrics": metrics
                }

                self._log(
                    logger, session_id, f"toxicity_{difficulty}_done",
                    f"Toxicity {difficulty} 완료",
                    {"metrics": metrics}
                )

            # Toxicity 전체 메트릭
            benchmark_results["toxicity"]["overall_metrics"] = self._calculate_overall_metrics(
                benchmark_results["toxicity"], "toxicity"
            )

            # 6. Safety 벤치마크 실행
            self._log(logger, session_id, "safety_start", "Safety 벤치마크 시작", {})

            for difficulty in DIFFICULTY_LEVELS:
                data = self._load_benchmark_data("safety", difficulty)
                self._log(
                    logger, session_id, f"safety_{difficulty}",
                    f"Safety {difficulty} 데이터 로드",
                    {"count": len(data)}
                )

                results = []
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            self._evaluate_single_safety_item,
                            item, idx, query_llm, safety_prompt
                        ): idx for idx, item in enumerate(data)
                    }

                    for future in as_completed(futures):
                        result = future.result()
                        results.append(result)
                        completed_items += 1

                        if result.get("error"):
                            failed_items += 1

                        # 진행률 업데이트
                        if completed_items % max(1, total_items // 10) == 0:
                            mongodb_service.update_benchmark_status(
                                session_id,
                                progress={
                                    "total": total_items,
                                    "completed": completed_items,
                                    "failed": failed_items
                                }
                            )

                # 인덱스 순으로 정렬
                results.sort(key=lambda x: x["index"])

                # 메트릭 계산
                metrics = self._calculate_metrics(results, "safety")

                benchmark_results["safety"][difficulty] = {
                    "items": results,
                    "metrics": metrics
                }

                self._log(
                    logger, session_id, f"safety_{difficulty}_done",
                    f"Safety {difficulty} 완료",
                    {"metrics": metrics}
                )

            # Safety 전체 메트릭
            benchmark_results["safety"]["overall_metrics"] = self._calculate_overall_metrics(
                benchmark_results["safety"], "safety"
            )

            # 7. 세션 로거 정리
            clear_session_logger()

            # 8. 소요 시간 계산
            elapsed_time = time.time() - start_time

            # 9. 메타데이터 추가
            if metadata is not None:
                benchmark_results["metadata"] = metadata

            # 10. MongoDB에 결과 저장
            mongodb_service.save_benchmark_results(
                session_id,
                benchmark_results,
                elapsed_time
            )

            # 11. 완료 상태 업데이트
            mongodb_service.update_benchmark_status(
                session_id,
                status="finished",
                progress={
                    "total": total_items,
                    "completed": completed_items,
                    "failed": failed_items
                },
                elapsed_time_seconds=elapsed_time
            )

            self._log(
                logger, session_id, "finished",
                "벤치마크 완료",
                {
                    "elapsed_time_seconds": round(elapsed_time, 2),
                    "toxicity_overall": benchmark_results["toxicity"]["overall_metrics"],
                    "safety_overall": benchmark_results["safety"]["overall_metrics"]
                }
            )

        except Exception as e:
            # 에러 처리
            try:
                from core import clear_session_logger
                clear_session_logger()
            except ImportError:
                pass

            error_msg = f"{type(e).__name__}: {str(e)}"
            error_trace = traceback.format_exc()

            mongodb_service.update_benchmark_status(
                session_id,
                status="error",
                error_msg=error_msg
            )

            self._log(
                logger, session_id, "error",
                "벤치마크 실패",
                {"error": error_msg, "traceback": error_trace}
            )
            logger.print_error(f"Benchmark failed: {error_msg}")


def get_benchmark_runner(session_manager: SessionManager) -> BenchmarkRunner:
    """BenchmarkRunner 인스턴스 반환

    Args:
        session_manager: 세션 관리자

    Returns:
        BenchmarkRunner: 벤치마크 실행기 인스턴스
    """
    return BenchmarkRunner(session_manager)

