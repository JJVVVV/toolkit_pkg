import json
from collections import UserDict
from functools import reduce
from heapq import nlargest
from pathlib import Path
from typing import List, Tuple

METRIC_DICT_DATA_NAME = "performance.json"

MetricDictGroup = List["MetricDictGroup"] | Tuple["MetricDictGroup"]


class MetricDict(UserDict):
    ("Support metrics: `accuracy`, `F1-score`, `loss`, `bleu(1~4)`, `rougeL`, `hit@1`, `MRR`, `rouge(1~3)`, `self-bleu(1~4)`")
    __metric_for_compare = None

    __metric_scale_map = {
        "accuracy": 1,
        "F1-score": 1,
        "loss": -1,
        "bleu1": 1,
        "bleu2": 1,
        "bleu3": 1,
        "bleu4": 1,
        "rougeL": 1,
        "hit@1": 1,
        "MRR": 1,
        "rouge1": 1,
        "rouge2": 1,
        "rouge3": 1,
        "self-bleu1": 1,
        "self-bleu2": 1,
        "self-bleu3": 1,
        "self-bleu4": 1,
    }

    # def __init__(self, *args):
    #     super().__init__(*args)
    #     self.metric_used_to_comp = MetricsDict.metric_used_to_comp
    #     self.scale = MetricsDict.scale

    @classmethod
    def support_metrics(cls):
        "Return a set which contains all supported metrics."
        return cls.__metric_scale_map.keys()

    def __setitem__(self, key: str, value: float | int):
        if key not in self.__metric_scale_map:
            raise KeyError(f"Key '{key}' is not allowed.")
        super().__setitem__(key, value)

    @classmethod
    def set_metric_for_compare(cls, value: str):
        if value not in cls.__metric_scale_map:
            raise ValueError(
                f"The value for attribute `metric_for_compare` was not understood: received `{value}` "
                f"but only {[key for key in cls.__metric_scale_map.keys()]} are valid."
            )
        cls.__metric_for_compare = value

    @classmethod
    def metric_for_compare(cls):
        return cls.__metric_for_compare

    @classmethod
    def _check(cls):
        if cls.__metric_for_compare is None:
            raise TypeError("The metric used to comparison `MetricDict.metric_used_to_comp` is undefined.")
        if cls.__metric_for_compare not in cls.__metric_scale_map:
            raise KeyError("The metric' scale is undefined in `MetricDict.metric_scale` dict.")

    def round(self, precision=4):
        "Round the values to a given precision in decimal digits."
        for key, value in self.items():
            self[key] = round(value, precision)

    def __eq__(self, other):
        self._check()
        if isinstance(other, MetricDict):
            return (
                self.get(MetricDict.__metric_for_compare) * MetricDict.__metric_scale_map[self.__metric_for_compare]
                == other.get(MetricDict.__metric_for_compare) * MetricDict.__metric_scale_map[self.__metric_for_compare]
            )
        raise NotImplementedError()

    def __ne__(self, other):
        try:
            result = self.__eq__(other)
        except NotImplementedError as e:
            raise e
        return not result

    def __lt__(self, other):
        self._check()
        if isinstance(other, MetricDict):
            return (
                self.get(MetricDict.__metric_for_compare) * MetricDict.__metric_scale_map[self.__metric_for_compare]
                < other.get(MetricDict.__metric_for_compare) * MetricDict.__metric_scale_map[self.__metric_for_compare]
            )
        raise NotImplementedError()

    def __le__(self, other):
        self._check()
        if isinstance(other, MetricDict):
            return (
                self.get(MetricDict.__metric_for_compare) * MetricDict.__metric_scale_map[self.__metric_for_compare]
                <= other.get(MetricDict.__metric_for_compare) * MetricDict.__metric_scale_map[self.__metric_for_compare]
            )
        raise NotImplementedError()

    def __gt__(self, other):
        self._check()
        if isinstance(other, MetricDict):
            return (
                self.get(MetricDict.__metric_for_compare) * MetricDict.__metric_scale_map[self.__metric_for_compare]
                > other.get(MetricDict.__metric_for_compare) * MetricDict.__metric_scale_map[self.__metric_for_compare]
            )
        raise NotImplementedError()

    def __ge__(self, other):
        self._check()
        if isinstance(other, MetricDict):
            return (
                self.get(MetricDict.__metric_for_compare) * MetricDict.__metric_scale_map[self.__metric_for_compare]
                >= other.get(MetricDict.__metric_for_compare) * MetricDict.__metric_scale_map[self.__metric_for_compare]
            )
        raise NotImplementedError()

    def __neg__(self):
        return MetricDict({key: -value for key, value in self.items()})

    def __add__(self, addend):
        if isinstance(addend, MetricDict):
            result = MetricDict()
            if not self.keys() == addend.keys():
                raise KeyError(f"The keys of the two MetricsDict {self.keys()}, {addend.keys()} are not exactly the same.")
            for key, value in self.items():
                result[key] = value + addend[key]
            return result
        elif isinstance(addend, (int, float)):
            result = MetricDict()
            for key, value in self.items():
                result[key] = value + addend
            return result
        raise NotImplementedError()

    def __radd__(self, augend):
        if isinstance(augend, MetricDict):
            result = MetricDict()
            if not self.keys() == augend.keys():
                raise KeyError(f"The keys of the two MetricsDict {self.keys()}, {augend.keys()} are not exactly the same.")
            for key, value in augend.items():
                result[key] = value + self[key]
            return result
        elif isinstance(augend, (int, float)):
            result = MetricDict()
            for key, value in self.items():
                result[key] = augend + value
            return result
        raise NotImplementedError()

    def __sub__(self, subtrahend):
        return self + (-subtrahend)

    def __rsub__(self, minuend):
        return minuend + (-self)

    def __truediv__(self, divisor):
        if isinstance(divisor, (int, float)):
            result = MetricDict()
            for key, value in self.items():
                result[key] = value / divisor
            return result
        raise NotImplementedError()

    def __rtruediv__(self, dividend):
        if isinstance(dividend, (int, float)):
            result = MetricDict()
            for key, value in self.items():
                result[key] = dividend / value
            return result
        raise NotImplementedError()

    def __mul__(self, multiplier):
        if isinstance(multiplier, (int, float)):
            result = MetricDict()
            for key, value in self.items():
                result[key] = value * multiplier
            return result
        raise NotImplementedError()

    def __rmul__(self, multiplicand):
        if isinstance(multiplicand, (int, float)):
            return self * multiplicand
        raise NotImplementedError()

    @staticmethod
    def mean_top_k(metric_dicts: MetricDictGroup, top_k: int | None = None) -> "MetricDict" | MetricDictGroup:
        if not metric_dicts:
            ret = None
        elif isinstance(metric_dicts[0], MetricDict):
            if top_k is None:
                ret = reduce(lambda x, y: x + y, metric_dicts) / len(metric_dicts)
            else:
                metric_dicts_topk = nlargest(top_k, metric_dicts)
                ret = reduce(lambda x, y: x + y, metric_dicts_topk) / len(metric_dicts_topk)
            for key, value in ret.items():
                ret[key] = round(value, 2)
            return ret
        elif isinstance(metric_dicts[0], List | Tuple):
            ret = [MetricDict.mean_top_k(md) for md in metric_dicts]
        return ret

    def save(self, save_dir: Path | str, file_name: str = METRIC_DICT_DATA_NAME):
        """
        Save data as a json file.
        """
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
        assert save_dir.exists(), f"The directory `{save_dir}` dose not exists."
        save_path = save_dir / file_name
        data = {"metric_for_compare": self.__metric_for_compare, "data": dict(self)}
        with save_path.open("w", encoding="utf8") as file:
            json.dump(data, file, indent=2)

    @classmethod
    def load(cls, load_dir_or_path: Path | str, file_name: str = METRIC_DICT_DATA_NAME) -> "MetricDict":
        """
        load data from a json file.
        """
        if isinstance(load_dir_or_path, str):
            load_dir_or_path = Path(load_dir_or_path)
        if load_dir_or_path.is_file():
            resolved_path = load_dir_or_path
        else:
            resolved_path = load_dir_or_path / file_name
        with resolved_path.open("r", encoding="utf8") as file:
            o = json.load(file)
        cls.__metric_for_compare = o["metric_for_compare"]
        return cls(o["data"])

    # def to_json(self) -> Dict:
    #     """
    #     Return a python dict that can be encoded by json.
    #     """
    #     ret = dict()
    #     for key, value in self.items():
    #         ret[key.name] = value
    #     return ret

    # @classmethod
    # def from_json(cls, data: Dict) -> "MetricDict":
    #     """
    #     Return a MetricDict with the data from a python dict
    #     """
    #     ret = cls()
    #     for key, value in data.items():
    #         ret[Metric[key]] = value
    #     return ret
