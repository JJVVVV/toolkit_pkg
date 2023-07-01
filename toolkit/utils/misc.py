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
