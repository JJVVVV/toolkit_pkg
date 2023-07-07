import logging
import sys

import colorlog

from .config import ConfigBase
from .enums import ConversationStyle, Split
from .logger import getLogger

toolkit_logger = logging.getLogger(__name__)
toolkit_logger.setLevel(level=logging.DEBUG)
# 定义日志输出格式
log_format = "%(asctime)s - %(log_color)s %(levelname)s > %(name)s: %(message)s"
# 定义日志输出颜色
log_colors = {"DEBUG": "cyan", "INFO": "green", "WARNING": "yellow", "ERROR": "red", "CRITICAL": "red,bg_white"}
formatter = colorlog.ColoredFormatter(log_format, log_colors=log_colors)
# formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")


console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
toolkit_logger.addHandler(console_handler)


def set_file_logger(file_path: str):
    # 添加handler
    # global file_handler
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    toolkit_logger.addHandler(file_handler)
