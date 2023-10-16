import json
from typing import Any

from .metric import MetricDict


class JsonEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, MetricDict):
            print("MetricDict")
            return o.to_json()

        return super().default(o)
