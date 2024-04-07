import logging
import sys

import colorlog

from .config import ConfigBase
from .enums import ConversationStyle, Split
from .logger import _getLogger, getLogger

toolkit_logger = _getLogger("[toolkit]")
file_logger_path = None


def set_file_logger(file_path: str):
    global file_logger_path
    file_logger_path = file_path
    log_format = "%(asctime)s - %(log_color)s %(levelname)s > %(name)s: %(message)s"
    log_colors = {"DEBUG": "cyan", "INFO": "green", "WARNING": "yellow", "ERROR": "red", "CRITICAL": "red,bg_white"}
    formatter = colorlog.ColoredFormatter(log_format, log_colors=log_colors)

    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    toolkit_logger.addHandler(file_handler)
