import logging
import sys
from pathlib import Path

import colorlog

# class CustomLogRecord(logging.LogRecord):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.levelname = self.levelname.center(7)


# logging.setLogRecordFactory(CustomLogRecord)


def _getLogger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.DEBUG)

    # 定义日志输出格式
    log_format = "%(asctime)s%(log_color)s <%(levelname)s> %(name)s: %(message)s"

    # 定义日志输出颜色
    # "DEBUG": "white",
    log_colors = {"INFO": "green", "WARNING": "yellow", "ERROR": "red", "CRITICAL": "red,bg_white"}
    formatter = colorlog.ColoredFormatter(log_format, log_colors=log_colors)
    # formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # if logger.hasHandlers():
    #     return logger
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def getFileHandler(file_path: Path | str, level=logging.INFO):
    # 添加handler
    log_format = "%(asctime)s%(log_color)s <%(levelname)s> %(name)s: %(message)s"
    # "DEBUG": "white",
    log_colors = {"INFO": "green", "WARNING": "yellow", "ERROR": "red", "CRITICAL": "red,bg_white"}
    formatter = colorlog.ColoredFormatter(log_format, log_colors=log_colors)
    file_handler = logging.FileHandler(file_path, mode="a")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    return file_handler


def getLogger(name: str, file_path: Path | str | None = None) -> logging.Logger:
    logger: logging.Logger = _getLogger(name)
    if file_path is not None:
        file_handler = getFileHandler(file_path)
        logger.addHandler(file_handler)

    return logger
