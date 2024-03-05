import os
import logging
from datetime import datetime

def setup_logger():
    log_directory = "logs"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # 오늘 날짜를 포함한 로그 파일명 생성
    today = datetime.now().strftime("%Y-%m-%d")
    log_filename = os.path.join(log_directory, f"log_{today}.log")

    # 로거 생성
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 로그 포맷 설정
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)

    # 파일 핸들러 생성
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")  # 여기서 인코딩 설정
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # 로거에 파일 핸들러 추가
    logger.addHandler(file_handler)
    return logger
