import json
from collections import UserDict
from copy import deepcopy
from functools import reduce
from heapq import nlargest
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

METRIC_DICT_DATA_NAME = "performance.json"

MetricDictGroup = Iterable["MetricDictGroup"] | Dict[int, "MetricDict"]


class MetricDict(UserDict):
    ("Support metrics: `accuracy`, `F1-score`, `loss`, `bleu(1~4)`, `rougeL`, `hit@1`, `MRR`, `rouge(1~3)`, `self-bleu(1~4)`")

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
        "rouge4": 1,
        "rouge5": 1,
        "rouge6": 1,
        "rouge7": 1,
        "rouge8": 1,
        "rouge9": 1,
        "self-bleu1": 1,
        "self-bleu2": 1,
        "self-bleu3": 1,
        "self-bleu4": 1,
    }
    custom_metric_scale_map = dict()

    def __init__(self, a_dict=None, /, metric_for_compare=None, **kwargs):
        super().__init__(a_dict, **kwargs)
        if isinstance(a_dict, MetricDict):
            self.__metric_for_compare = a_dict.__metric_for_compare
            if metric_for_compare is not None:
                self.__metric_for_compare = metric_for_compare
        else:
            self.__metric_for_compare = metric_for_compare

    @classmethod
    def support_metrics(cls):
        "Return a set which contains all supported metrics."
        return set(cls.__metric_scale_map.keys()) | set(cls.custom_metric_scale_map.keys())

    def __setitem__(self, key: str, value: float | int):
        if key not in self.__metric_scale_map and key not in self.custom_metric_scale_map:
            raise KeyError(f"Key '{key}' is not allowed. Supported keys: {self.support_metrics()}")
        super().__setitem__(key, value)

    @property
    def metric_for_compare(self):
        return self.__metric_for_compare

    @metric_for_compare.setter
    def metric_for_compare(self, value: str):
        if value not in self.__metric_scale_map and value not in self.custom_metric_scale_map:
            raise ValueError(
                f"The value for attribute `metric_for_compare` is invalid. "
                f"Only {self.support_metrics()} are valid, or maybe you can define the custom metric in `MetricDict.custom_metric_scale_map`"
            )
        self.__metric_for_compare = value

    def _check(self):
        if self.__metric_for_compare is None:
            raise TypeError("The `metric_for_compare` is undefined.")
        if self.__metric_for_compare in self.__metric_scale_map:
            return self.__metric_scale_map
        elif self.__metric_for_compare in self.custom_metric_scale_map:
            return self.custom_metric_scale_map
        else:
            raise ValueError(
                f"The value for attribute `metric_for_compare` is invalid. "
                f"Only {self.support_metrics()} are valid, or maybe you can define the custom metric in `MetricDict.custom_metric_scale_map`"
            )

    def round(self, precision=4):
        "Round the values to a given precision in decimal digits."
        for key, value in self.items():
            self[key] = round(value, precision)
        return self

    def __eq__(self, other):
        assert (
            self.__metric_for_compare == other.__metric_for_compare
        ), f"Only the MetricDict with same `metric_for_compare` can be compare, got `{self.__metric_for_compare }!={other.__metric_for_compare}`"
        metric_scale_map = self._check()
        if isinstance(other, MetricDict):
            return (
                self.get(self.__metric_for_compare) * metric_scale_map[self.__metric_for_compare]
                == other.get(other.__metric_for_compare) * metric_scale_map[other.__metric_for_compare]
            )
        raise NotImplementedError()

    def __ne__(self, other):
        try:
            result = self.__eq__(other)
        except NotImplementedError as e:
            raise e
        return not result

    def __lt__(self, other):
        assert (
            self.__metric_for_compare == other.__metric_for_compare
        ), f"Only the MetricDict with same `metric_for_compare` can be compare, got `{self.__metric_for_compare }!={other.__metric_for_compare}`"
        metric_scale_map = self._check()
        if isinstance(other, MetricDict):
            return (
                self.get(self.__metric_for_compare) * metric_scale_map[self.__metric_for_compare]
                < other.get(other.__metric_for_compare) * metric_scale_map[other.__metric_for_compare]
            )
        raise NotImplementedError()

    def __le__(self, other):
        assert (
            self.__metric_for_compare == other.__metric_for_compare
        ), f"Only the MetricDict with same `metric_for_compare` can be compare, got `{self.__metric_for_compare }!={other.__metric_for_compare}`"
        metric_scale_map = self._check()
        if isinstance(other, MetricDict):
            return (
                self.get(self.__metric_for_compare) * metric_scale_map[self.__metric_for_compare]
                <= other.get(other.__metric_for_compare) * metric_scale_map[other.__metric_for_compare]
            )
        raise NotImplementedError()

    def __gt__(self, other):
        assert (
            self.__metric_for_compare == other.__metric_for_compare
        ), f"Only the MetricDict with same `metric_for_compare` can be compare, got `{self.__metric_for_compare }!={other.__metric_for_compare}`"
        metric_scale_map = self._check()
        if isinstance(other, MetricDict):
            return (
                self.get(self.__metric_for_compare) * metric_scale_map[self.__metric_for_compare]
                > other.get(other.__metric_for_compare) * metric_scale_map[other.__metric_for_compare]
            )
        raise NotImplementedError()

    def __ge__(self, other):
        assert (
            self.__metric_for_compare == other.__metric_for_compare
        ), f"Only the MetricDict with same `metric_for_compare` can be compare, got `{self.__metric_for_compare }!={other.__metric_for_compare}`"
        metric_scale_map = self._check()
        if isinstance(other, MetricDict):
            return (
                self.get(self.__metric_for_compare) * metric_scale_map[self.__metric_for_compare]
                >= other.get(other.__metric_for_compare) * metric_scale_map[other.__metric_for_compare]
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

    # # ! deprecated
    # @staticmethod
    # def _mean_top_k(metric_dicts: Dict[int, "MetricDict"], top_k: int | None = None) -> Tuple[List | "MetricDict"] | None:
    #     """
    #     metric_dicts: `metric_dicts[seed] = MetricDict`
    #     """
    #     if not metric_dicts:
    #         return None
    #     elif isinstance(metric_dicts, Dict):
    #         if top_k is None:
    #             seeds = list(metric_dicts.keys())
    #             ret = reduce(lambda x, y: x + y, metric_dicts.values()) / len(metric_dicts)
    #         else:
    #             metric_dicts_topk = dict(nlargest(top_k, metric_dicts.items(), key=lambda item: item[1]))
    #             seeds = list(metric_dicts_topk.keys())
    #             ret = reduce(lambda x, y: x + y, metric_dicts_topk.values()) / len(metric_dicts_topk)
    #         for key, value in ret.items():
    #             ret[key] = round(value, 2)
    #         return ret, seeds

    # # ! deprecated
    # @staticmethod
    # def mean_topk(metric_dicts_group: Dict[str, Dict], top_k: int | None = None) -> Dict[str, Tuple[List | "MetricDict"] | None]:
    #     "Deprecated! ! !"
    #     "metric_dicts_group: `metric_dicts_group[split][seed] = MetricDict`"
    #     ret = dict()
    #     for key, value in metric_dicts_group.items():
    #         ret[key] = MetricDict._mean_top_k(value, top_k)
    #     return ret

    def _get_attridata(self) -> dict:
        "获得属性字典, 并加入类属性. 返回`字典`"
        data = deepcopy(self.__dict__)
        data["custom_metric_scale_map"] = self.custom_metric_scale_map
        return data

    def save(self, save_dir: Path | str, file_name: str = METRIC_DICT_DATA_NAME):
        """
        Save data as a json file.
        """
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
        assert save_dir.exists(), f"The directory `{save_dir}` dose not exist."
        save_path = save_dir / file_name
        data = self._get_attridata()
        with save_path.open("w", encoding="utf8") as file:
            json.dump(data, file, indent=2)

    @classmethod
    def _load_attridata(cls, data: dict) -> "MetricDict":
        "加载属性字典, 并加载类属性. 返回`MetricDict`"
        cls.custom_metric_scale_map = data.pop("custom_metric_scale_map")
        metricdict = cls()
        metricdict.__dict__ = data
        return metricdict

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
            data = json.load(file)
        return cls._load_attridata(data)

    @staticmethod
    def recur_set_metric_for_compare(data: list | dict, metric_for_compare: str):
        if isinstance(data, list):
            for item in data:
                MetricDict.recur_set_metric_for_compare(item, metric_for_compare)
        elif isinstance(data, dict):
            for value in data.values():
                MetricDict.recur_set_metric_for_compare(value, metric_for_compare)
        elif isinstance(data, MetricDict):
            data.metric_for_compare = metric_for_compare
        else:
            raise ValueError(f"不支持的类型: {type(data)}")

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


if __name__ == "__main__":
    d1 = MetricDict({"accuracy": 98, "loss": 1}, metric_for_compare="accuracy")
    d3 = MetricDict({"accuracy": 99, "loss": 2}, metric_for_compare="accuracy")
    attridata = d3._get_attridata()
    d2 = MetricDict._load_attridata(attridata)
    print(d2 == d3)
    print(d1 == d2)
    print(d1 < d2)
    print(d1 > d2)
    print(-d2)
    print(d2 + 1)
    print(d2 - 1)
    print(d2 * 2)
    print(d2 / 2)
    print(d1 + d2)
    print(d1 - d2)
    print(d2.__dict__)

    MetricDict.recur_set_metric_for_compare({"a": {1: [d1, d3]}}, "loss")
    print(d1.metric_for_compare, d3.metric_for_compare, d2.metric_for_compare)
    print(d1 > d3)

    # print(d1 * d2)
    # print(d1 / d2)
