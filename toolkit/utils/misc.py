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
