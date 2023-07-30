import os
from typing import Any, Dict, List, Tuple


def type_to_str(data):
    return str(type(data))[8:-2]


def get_data_types(data: List | Tuple | Any) -> str | Dict[Any, str]:
    """
    获取一个嵌套结构中的所有数据类型，并保留嵌套结构
    """
    if isinstance(data, dict):
        return {k: get_data_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return f"List[{', '.join([get_data_types(item) for item in data])}]"
    elif isinstance(data, tuple):
        return f"Tuple[{', '.join([get_data_types(item) for item in data])}]"
    else:
        return type_to_str(data)


def search_file(directory, filename):
    """
    从目录中按名字递归查找
    """
    file_paths = []
    # warkos.walk遍历目录下的所有子目录和文件: root为某个目录, dirs为root下的目录, files为root下的文件
    for root, dirs, files in os.walk(directory):
        if filename in files:
            file_path = os.path.join(root, filename)
            file_paths.append(file_path)
    return file_paths


def find_file(directory: str, filename: str, depth: int = 0) -> str | None:
    """Deep first search, stop when the the first matched file is found.\n
    Only root dirctory will be scanned, when `depth=0`"""
    if depth < 0:  # 递归深度限制
        return None

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)  # 获取项目路径

        if os.path.isfile(item_path) and item == filename:  # 如果项目是文件并且文件名匹配
            return item_path

        elif os.path.isdir(item_path):  # 如果项目是目录，则递归搜索
            found_file = find_file(item_path, filename, depth - 1)
            if found_file is not None:  # 如果在子目录中找到了文件，返回其路径
                return found_file

    return None  # 如果在这个目录中没有找到文件，返回None


def find_files(dir: str, filename: str, depth: int = 0) -> List[str]:
    """Deep first search, find all matched files.\n
    Only root dirctory will be scanned, when `depth=0`"""
    found_files = []

    if depth < 0:  # 递归深度限制
        return found_files

    for item in os.listdir(dir):
        item_path = os.path.join(dir, item)  # 获取项目路径

        if os.path.isfile(item_path) and item == filename:  # 如果项目是文件并且文件名匹配
            found_files.append(item_path)

        elif os.path.isdir(item_path):  # 如果项目是目录，则递归搜索
            found_files.extend(find_files(item_path, filename, depth - 1))  # 将找到的文件添加到列表中

    return found_files  #
