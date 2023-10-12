from torchmetrics.functional.text.rouge import rouge_score
from .metricdict import MetricDict


def calculate_rouge(pred: list[str], tgt: list[str], rouge_keys: str | tuple[str, ...] = "rougeL", language="zh"):
    """
    rouge_keys that are allowed are `rougeL`, `rougeLsum`, and `rouge1` through `rouge9`.
    """
    ret = MetricDict({key: 0 for key in rouge_keys})
    if language == "zh":
        for p, t in zip(pred, tgt):
            ret_dict = rouge_score(pred, tgt, rouge_keys=rouge_keys, tokenizer=list, normalizer=lambda x: x)
            ret += MetricDict({key: ret_dict[key + "_fmeasure"].item() for key in rouge_keys})
        return ret / len(pred)
