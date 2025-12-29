import logging
from logging.handlers import TimedRotatingFileHandler
import logging.handlers as lh
import os
import re
from typing import NoReturn

class UserLogger:
    def __init__(self, session_id, version, level="debug"):
        self.session_id = session_id
        self.version = version
        # Sanitize thread name for use in file path
        self.sanitized_session_id = self._sanitize_filename(session_id)
        
        logger = logging.getLogger(f'[TEST-API-{session_id}]')
        if(level == "debug"):
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.DEBUG)

        # Check if handlers already exist to prevent duplication
        # If the logger already has handlers, it means this session logger was already initialized
        if not logger.handlers:
            formatter = logging.Formatter('[TEST-API] %(levelname)s:\t%(asctime)s - %(filename)s:%(lineno)d - %(message)s', '%Y-%m-%d %H:%M:%S')


            LOG_MAX_SIZE = 1024*1024*10  # 1GB
            LOG_FILE_CNT = 10
            LOG_DIR = f'./resource/sessions/{session_id}'

            # Create log directory if it doesn't exist
            if not os.path.exists(LOG_DIR):
                os.makedirs(LOG_DIR)
                print("logs 폴더 생성")

            # Regular log file handler - use os.path.join with sanitized filename
            log_file_path = os.path.join(LOG_DIR, f"{self.sanitized_session_id}.log")
            regular_handler = lh.RotatingFileHandler(log_file_path, maxBytes=LOG_MAX_SIZE, backupCount=LOG_FILE_CNT, encoding='utf-8')
            regular_handler.setFormatter(formatter)
            # regular_handler.setLevel(logging.DEBUG)
            logger.addHandler(regular_handler)


        self.logger = logger
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        파일명에서 사용할 수 없는 특수 문자를 안전한 문자로 치환합니다.
        Windows와 Unix 시스템 모두에서 사용할 수 없는 문자들을 처리합니다.
        """
        # Windows에서 사용할 수 없는 문자들: < > : " | ? * \ /
        # Unix에서 사용할 수 없는 문자들: / (null character도 있지만 일반적으로 문제되지 않음)
        
        # 특수 문자를 언더스코어로 치환
        sanitized = re.sub(r'[<>:"|?*\\/]', '_', filename)
        
        # 연속된 언더스코어를 하나로 줄임
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # 앞뒤 공백 및 언더스코어 제거
        sanitized = sanitized.strip(' _')
        
        # 빈 문자열이 되는 경우 기본값 설정
        if not sanitized:
            sanitized = 'default_log'
            
        return sanitized
        
    def print_info(self, message: str) -> NoReturn:
        self.logger.setLevel(logging.INFO)
        self.logger.info(message)
        
    def print_warning(self, message: str) -> NoReturn:
        self.logger.setLevel(logging.INFO)
        self.logger.warning(message)        
        
    def print_error(self, message: str, exc_info=True) -> NoReturn:
        """
        에러 로그를 출력합니다. 기본적으로 스택 트레이스 정보와 실제 에러 발생 위치를 포함합니다.
        
        Args:
            message: 에러 메시지
            exc_info: True이면 스택 트레이스 정보도 함께 출력 (기본값: True)
        """
        import sys
        
        self.logger.setLevel(logging.ERROR)
        
        # 현재 예외 정보가 있는지 확인하고 실제 에러 발생 위치 추출
        enhanced_message = message
        exc_type, exc_value, exc_traceback = sys.exc_info()
        
        if exc_traceback is not None:
            # 실제 예외가 발생한 위치 찾기
            tb = exc_traceback
            while tb.tb_next:
                tb = tb.tb_next
            
            filename = tb.tb_frame.f_code.co_filename
            line_number = tb.tb_lineno
            function_name = tb.tb_frame.f_code.co_name
            
            # 파일명에서 경로 제거하고 파일명만 추출
            import os
            filename = os.path.basename(filename)
            
            enhanced_message = f"{message} [에러 발생 위치: {filename}:{line_number} in {function_name}()]"
        
        if exc_info:
            self.logger.error(enhanced_message, exc_info=True)
        else:
            self.logger.error(enhanced_message)
    
    def print_exception(self, message: str, exc_info=None) -> NoReturn:
        """
        예외 정보와 함께 상세한 에러 로그를 출력합니다.
        
        Args:
            message: 에러 메시지
            exc_info: 예외 정보 (None이면 현재 예외 정보를 자동으로 수집)
        """
        self.logger.setLevel(logging.ERROR)
        if exc_info is None:
            self.logger.exception(message)
        else:
            self.logger.error(f"{message} - Exception: {exc_info}", exc_info=exc_info)

    def print_debug(self, message: str) -> NoReturn:
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug(message)
    

