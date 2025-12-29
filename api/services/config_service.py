# -*- coding: utf-8 -*-
"""설정 파일 관리 서비스

프롬프트(txt)와 GOAT 설정(yaml) 파일을 관리하는 서비스
"""
import os
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional


class ConfigService:
    """설정 파일 관리 서비스

    프롬프트 템플릿과 GOAT 설정 파일의 조회/수정을 담당합니다.
    """

    CONFIG_DIR = Path("./config")
    PROMPTS_DIR = CONFIG_DIR / "prompts"
    GOAT_CONFIG_FILE = CONFIG_DIR / "goat_config.yaml"

    # 허용된 프롬프트 파일 목록 (보안: 화이트리스트)
    ALLOWED_PROMPTS = [
        "attacker_system.txt",
        "techniques_guide.txt",
        "decision_framework.txt",
        "output_format.txt",
        "user_toxicity_analysis_system.txt",
        "assistant_toxicity_analysis_system.txt"
    ]

    # 최대 파일 크기 (100KB)
    MAX_FILE_SIZE = 100 * 1024

    def __init__(self):
        """ConfigService 초기화"""
        # 디렉토리 존재 확인
        if not self.CONFIG_DIR.exists():
            raise FileNotFoundError(f"Config directory not found: {self.CONFIG_DIR}")
        if not self.PROMPTS_DIR.exists():
            raise FileNotFoundError(f"Prompts directory not found: {self.PROMPTS_DIR}")

    def _validate_prompt_name(self, name: str) -> bool:
        """프롬프트 파일명 검증

        Args:
            name: 파일명

        Returns:
            bool: 유효 여부
        """
        # Path Traversal 방지
        if ".." in name or "/" in name or "\\" in name:
            return False
        # 화이트리스트 검증
        return name in self.ALLOWED_PROMPTS

    def _get_prompt_path(self, name: str) -> Path:
        """프롬프트 파일 경로 반환

        Args:
            name: 파일명

        Returns:
            Path: 파일 경로

        Raises:
            ValueError: 유효하지 않은 파일명
        """
        if not self._validate_prompt_name(name):
            raise ValueError(f"Invalid prompt name: {name}")
        return self.PROMPTS_DIR / name

    # =========================================================================
    # 프롬프트 관리
    # =========================================================================

    def list_prompts(self) -> List[Dict[str, Any]]:
        """프롬프트 파일 목록 조회

        Returns:
            List[Dict]: 프롬프트 정보 목록
        """
        prompts = []
        for name in self.ALLOWED_PROMPTS:
            path = self.PROMPTS_DIR / name
            if path.exists():
                stat = path.stat()
                prompts.append({
                    "name": name,
                    "size": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        return prompts

    def get_prompt(self, name: str) -> Dict[str, Any]:
        """프롬프트 내용 조회

        Args:
            name: 파일명

        Returns:
            Dict: 프롬프트 정보 및 내용

        Raises:
            ValueError: 유효하지 않은 파일명
            FileNotFoundError: 파일이 없는 경우
        """
        path = self._get_prompt_path(name)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {name}")

        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        stat = path.stat()
        return {
            "name": name,
            "content": content,
            "size": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
        }

    def update_prompt(self, name: str, content: str) -> Dict[str, Any]:
        """프롬프트 수정

        Args:
            name: 파일명
            content: 새 내용

        Returns:
            Dict: 수정된 프롬프트 정보

        Raises:
            ValueError: 유효하지 않은 파일명 또는 크기 초과
            FileNotFoundError: 파일이 없는 경우
        """
        path = self._get_prompt_path(name)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {name}")

        # 크기 검증
        content_size = len(content.encode('utf-8'))
        if content_size > self.MAX_FILE_SIZE:
            raise ValueError(f"Content too large: {content_size} bytes (max: {self.MAX_FILE_SIZE})")

        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

        stat = path.stat()
        return {
            "name": name,
            "size": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "message": "Prompt updated successfully"
        }

    def upload_prompt(self, filename: str, content: bytes) -> Dict[str, Any]:
        """파일 업로드로 프롬프트 수정

        업로드된 파일명이 기존 프롬프트 파일명과 동일해야만 업데이트 허용

        Args:
            filename: 업로드된 파일명 (기존 파일과 일치해야 함)
            content: 파일 내용 (bytes)

        Returns:
            Dict: 수정 결과

        Raises:
            ValueError: 파일명 불일치, 확장자 오류, 크기 초과, 인코딩 오류
            FileNotFoundError: 기존 파일이 없는 경우
        """
        # 1. 파일 확장자 검증
        if not filename.endswith('.txt'):
            raise ValueError(f"Only .txt files are allowed: {filename}")

        # 2. 파일명 검증 (화이트리스트)
        if not self._validate_prompt_name(filename):
            raise ValueError(f"Invalid prompt filename: {filename}. Must be one of: {', '.join(self.ALLOWED_PROMPTS)}")

        # 3. 기존 파일 존재 확인
        path = self.PROMPTS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {filename}")

        # 4. 파일 크기 검증
        if len(content) > self.MAX_FILE_SIZE:
            raise ValueError(f"File too large: {len(content)} bytes (max: {self.MAX_FILE_SIZE})")

        # 5. UTF-8 인코딩 검증
        try:
            text_content = content.decode('utf-8')
        except UnicodeDecodeError:
            raise ValueError("File must be UTF-8 encoded")

        # 6. 파일 저장
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text_content)

        stat = path.stat()
        return {
            "name": filename,
            "size": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "message": "Prompt uploaded successfully"
        }

    # =========================================================================
    # GOAT 설정 관리
    # =========================================================================

    def get_goat_config(self) -> Dict[str, Any]:
        """GOAT 설정 전체 조회

        Returns:
            Dict: GOAT 설정 데이터

        Raises:
            FileNotFoundError: 설정 파일이 없는 경우
        """
        if not self.GOAT_CONFIG_FILE.exists():
            raise FileNotFoundError(f"GOAT config file not found: {self.GOAT_CONFIG_FILE}")

        with open(self.GOAT_CONFIG_FILE, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def update_goat_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """GOAT 설정 전체 수정

        Args:
            config: 새 설정 데이터

        Returns:
            Dict: 수정 결과

        Raises:
            FileNotFoundError: 설정 파일이 없는 경우
        """
        if not self.GOAT_CONFIG_FILE.exists():
            raise FileNotFoundError(f"GOAT config file not found: {self.GOAT_CONFIG_FILE}")

        with open(self.GOAT_CONFIG_FILE, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        return {
            "message": "GOAT config updated successfully",
            "modified_at": datetime.now().isoformat()
        }

    def list_categories(self) -> List[str]:
        """카테고리 목록 조회

        Returns:
            List[str]: 카테고리 ID 목록
        """
        config = self.get_goat_config()
        return list(config.get("categories", {}).keys())

    def get_category(self, category_id: str) -> Dict[str, Any]:
        """특정 카테고리 조회

        Args:
            category_id: 카테고리 ID

        Returns:
            Dict: 카테고리 설정

        Raises:
            KeyError: 카테고리가 없는 경우
        """
        config = self.get_goat_config()
        categories = config.get("categories", {})

        if category_id not in categories:
            raise KeyError(f"Category not found: {category_id}")

        return categories[category_id]

    def update_category(self, category_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """특정 카테고리 수정

        Args:
            category_id: 카테고리 ID
            data: 새 카테고리 데이터

        Returns:
            Dict: 수정 결과

        Raises:
            KeyError: 카테고리가 없는 경우
        """
        config = self.get_goat_config()
        categories = config.get("categories", {})

        if category_id not in categories:
            raise KeyError(f"Category not found: {category_id}")

        # 카테고리 업데이트
        categories[category_id] = data
        config["categories"] = categories

        # 저장
        with open(self.GOAT_CONFIG_FILE, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        return {
            "category_id": category_id,
            "message": "Category updated successfully",
            "modified_at": datetime.now().isoformat()
        }


# 싱글톤 인스턴스
config_service = ConfigService()
