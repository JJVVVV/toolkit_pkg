import json
from typing import Any

from .enums import Metric
from .utils import MetricDict


class JsonEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, MetricDict):
            print("MetricDict")
            return o.to_json()
        if isinstance(o, Metric):
            print("Metric")
            return o.name

        return super().default(o)
