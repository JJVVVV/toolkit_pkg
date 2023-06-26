from collections import UserDict


class MetricDict(UserDict):
    metric_used_to_comp = None
    scale = 1

    # def __init__(self, *args):
    #     super().__init__(*args)
    #     self.metric_used_to_comp = MetricsDict.metric_used_to_comp
    #     self.scale = MetricsDict.scale

    def __eq__(self, other):
        if isinstance(other, MetricDict):
            return self.get(MetricDict.metric_used_to_comp) * MetricDict.scale == other.get(MetricDict.metric_used_to_comp) * MetricDict.scale
        return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __lt__(self, other):
        if isinstance(other, MetricDict):
            return self.get(MetricDict.metric_used_to_comp) * MetricDict.scale < other.get(MetricDict.metric_used_to_comp) * MetricDict.scale
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, MetricDict):
            return self.get(MetricDict.metric_used_to_comp) * MetricDict.scale <= other.get(MetricDict.metric_used_to_comp) * MetricDict.scale
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, MetricDict):
            return self.get(MetricDict.metric_used_to_comp) * MetricDict.scale > other.get(MetricDict.metric_used_to_comp) * MetricDict.scale
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, MetricDict):
            return self.get(MetricDict.metric_used_to_comp) * MetricDict.scale >= other.get(MetricDict.metric_used_to_comp) * MetricDict.scale
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, MetricDict):
            result = MetricDict()
            if not self.keys() == other.keys():
                raise KeyError(f"The keys of the two MetricsDict {self.keys()}, {other.keys()} are not exactly the same.")
            for key, value in self.items():
                result[key] = value + other[key]
            return result
        return NotImplemented

    def __truediv__(self, divisor):
        if isinstance(divisor, (int, float)):
            result = MetricDict()
            for key, value in self.items():
                result[key] = value / divisor
            return result
        return NotImplemented

    def __mul__(self, multiplier):
        if isinstance(multiplier, (int, float)):
            result = MetricDict()
            for key, value in self.items():
                result[key] = value * multiplier
            return result
        return NotImplemented
