from torchmetrics.functional.text.rouge import rouge_score
from tqdm.auto import tqdm

from . import MetricDict


def rouge(pred: list[str], tgt: list[str], rouge_keys: str | tuple[str, ...] = "rougeL", language="zh") -> MetricDict:
    """
    rouge_keys that are allowed are `rougeL`, `rougeLsum`, and `rouge1` through `rouge9`.
    """
    ret = MetricDict({key: 0 for key in rouge_keys})
    if language == "zh":
        tokenizer = list
        normalizer = lambda x: x
    elif language == "en":
        tokenizer = lambda s: s.split()
        normalizer = lambda x: x
    else:
        raise NotImplementedError()

    for p, t in tqdm(zip(pred, tgt), total=len(pred), desc="Calculating rouge: "):
        ret_dict = rouge_score(p, t, rouge_keys=rouge_keys, tokenizer=tokenizer, normalizer=normalizer)
        ret += MetricDict({key: ret_dict[key + "_fmeasure"].item() for key in rouge_keys})
    return ret / len(pred) * 100
