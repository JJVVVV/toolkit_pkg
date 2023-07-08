import logging
import sys
from pathlib import Path

import colorlog


def _getLogger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.DEBUG)

    # 定义日志输出格式
    log_format = "%(asctime)s - %(log_color)s %(levelname)s > %(name)s: %(message)s"

    # 定义日志输出颜色
    log_colors = {"DEBUG": "cyan", "INFO": "green", "WARNING": "yellow", "ERROR": "red", "CRITICAL": "red,bg_white"}
    formatter = colorlog.ColoredFormatter(log_format, log_colors=log_colors)
    # formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def getLogger(name: str, file_path: Path | str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.DEBUG)

    # 定义日志输出格式
    log_format = "%(asctime)s - %(log_color)s %(levelname)s > %(name)s: %(message)s"

    # 定义日志输出颜色
    log_colors = {"DEBUG": "cyan", "INFO": "green", "WARNING": "yellow", "ERROR": "red", "CRITICAL": "red,bg_white"}
    formatter = colorlog.ColoredFormatter(log_format, log_colors=log_colors)
    # formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 添加handler
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
