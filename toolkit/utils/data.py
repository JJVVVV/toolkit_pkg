import random
from typing import List, Tuple


def split_data(data: List, ratio: float = 0.1) -> Tuple[List, List]:
    "Before using this function, please set `seed` with `random.sedd()` or `toolkit.trainer.setup_seed()`"
    random.shuffle(data)
    num_val = round(len(data) * ratio)
    return data[num_val:], data[:num_val]
