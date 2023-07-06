from collections import UserDict
from functools import reduce
from heapq import nlargest
from typing import List


class MetricDict(UserDict):
    metric_for_compare = None
    # metric_scale = defaultdict(lambda: 1, {2:-1})
    metric_scale_map = {"Accuracy": 1, "F1-score": 1, "Loss": -1}

    # def __init__(self, *args):
    #     super().__init__(*args)
    #     self.metric_used_to_comp = MetricsDict.metric_used_to_comp
    #     self.scale = MetricsDict.scale

    def __setitem__(self, key, value):
        if key not in self.metric_scale_map:
            raise KeyError(f"Key '{key}' is not allowed.")
        super().__setitem__(key, value)

    @classmethod
    def set_metric_for_compare(cls, value):
        if value not in cls.metric_scale_map:
            raise ValueError(
                f"The value for attribute `metric_for_compare` was not understood: received `{value}` "
                f"but only {[key for key in cls.metric_scale_map.keys()]} are valid."
            )
        cls.metric_for_compare = value

    @classmethod
    def _check(cls):
        if cls.metric_for_compare is None:
            raise TypeError("The metric used to comparison `MetricDict.metric_used_to_comp` is undefined.")
        if cls.metric_for_compare not in cls.metric_scale_map:
            raise KeyError("The metric' scale is undefined in `MetricDict.metric_scale` dict.")

    def __eq__(self, other):
        self._check()
        if isinstance(other, MetricDict):
            return (
                self.get(MetricDict.metric_for_compare) * MetricDict.metric_scale_map[self.metric_for_compare]
                == other.get(MetricDict.metric_for_compare) * MetricDict.metric_scale_map[self.metric_for_compare]
            )
        raise NotImplementedError()
        # return NotImplemented

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
                self.get(MetricDict.metric_for_compare) * MetricDict.metric_scale_map[self.metric_for_compare]
                < other.get(MetricDict.metric_for_compare) * MetricDict.metric_scale_map[self.metric_for_compare]
            )
        raise NotImplementedError()
        # return NotImplemented

    def __le__(self, other):
        self._check()
        if isinstance(other, MetricDict):
            return (
                self.get(MetricDict.metric_for_compare) * MetricDict.metric_scale_map[self.metric_for_compare]
                <= other.get(MetricDict.metric_for_compare) * MetricDict.metric_scale_map[self.metric_for_compare]
            )
        raise NotImplementedError()
        # return NotImplemented

    def __gt__(self, other):
        self._check()
        if isinstance(other, MetricDict):
            return (
                self.get(MetricDict.metric_for_compare) * MetricDict.metric_scale_map[self.metric_for_compare]
                > other.get(MetricDict.metric_for_compare) * MetricDict.metric_scale_map[self.metric_for_compare]
            )
        raise NotImplementedError()
        # return NotImplemented

    def __ge__(self, other):
        self._check()
        if isinstance(other, MetricDict):
            return (
                self.get(MetricDict.metric_for_compare) * MetricDict.metric_scale_map[self.metric_for_compare]
                >= other.get(MetricDict.metric_for_compare) * MetricDict.metric_scale_map[self.metric_for_compare]
            )
        raise NotImplementedError()
        # return NotImplemented

    def __add__(self, other):
        if isinstance(other, MetricDict):
            result = MetricDict()
            if not self.keys() == other.keys():
                raise KeyError(f"The keys of the two MetricsDict {self.keys()}, {other.keys()} are not exactly the same.")
            for key, value in self.items():
                result[key] = value + other[key]
            return result
        raise NotImplementedError()
        # return NotImplemented

    def __truediv__(self, divisor):
        if isinstance(divisor, (int, float)):
            result = MetricDict()
            for key, value in self.items():
                result[key] = value / divisor
            return result
        raise NotImplementedError()
        # return NotImplemented

    def __mul__(self, multiplier):
        if isinstance(multiplier, (int, float)):
            result = MetricDict()
            for key, value in self.items():
                result[key] = value * multiplier
            return result
        raise NotImplementedError()
        # return NotImplemented

    @staticmethod
    def mean_top_k(metric_dicts: List["MetricDict"], top_k: int | None = None) -> "MetricDict":
        if not metric_dicts:
            print("No metric dict.")
        if top_k is None:
            return reduce(lambda x, y: x + y, metric_dicts) / len(metric_dicts)
        metric_dicts_topk = nlargest(top_k, metric_dicts)
        return reduce(lambda x, y: x + y, metric_dicts_topk) / len(metric_dicts_topk)

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
