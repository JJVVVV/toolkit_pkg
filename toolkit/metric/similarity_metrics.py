import re

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from torchmetrics.functional.text.rouge import rouge_score
from tqdm.auto import tqdm

from . import MetricDict


def rouge(preds: str | list[str], tgts: str | list[str] | list[list[str]], language: str, rouge_keys: str | tuple[str, ...] = "rougeL") -> MetricDict:
    """
    rouge_keys that are allowed are `rougeL`, `rougeLsum`, and `rouge1` through `rouge9`.
    """
    if isinstance(preds, str):
        preds = [preds]

    if isinstance(tgts, str):
        tgts = [tgts]
    if isinstance(tgts[0], str):
        tgts = [[tgt] for tgt in tgts]
    # now preds: list[str], tgts: list[list[str]]
    assert len(preds) == len(tgts), f"The number of `preds` and `tgts` are not equal: len(preds)={len(preds)}, len(tgts)={len(tgts)}."

    ret = 0
    if language == "zh":
        tokenizer = list
        normalizer = lambda x: x
    elif language == "en":
        tokenizer = lambda s: s.split()
        normalizer = lambda x: x
    else:
        raise NotImplementedError(f"Do NOT support language: `{language}`")

    for pred, tgt_list in tqdm(zip(preds, tgts), total=len(preds), desc="Calculating rouge: "):
        a_pair = 0
        for tgt in tgt_list:
            rouge_dict = rouge_score(pred, tgt, rouge_keys=rouge_keys, tokenizer=tokenizer, normalizer=normalizer)
            a_pair += MetricDict({key: rouge_dict[key + "_fmeasure"].item() for key in rouge_keys})
        ret += (a_pair/len(tgt_list))

    return ret / len(preds)


# bleu_keys2weights = dict(
#     bleu1=(1.0,), bleu2=(1.0 / 2.0, 1.0 / 2.0), bleu3=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), bleu4=(1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0)
# )


def bleu_key2weights(bleu_key: str) -> tuple[float]:
    # 从字符串中提取以非数字字符开头的最后一组连续数字
    m = re.search(r"bleu(\d+)", bleu_key)
    assert m is not None, f"`{bleu_key}` is not a valid bleu key"
    no = int(m.group(1))
    assert no > 0, f"`{bleu_key}` is not a valid bleu key"
    return tuple([1.0 / no] * no)


def bleu(
    preds: str | list[str],
    tgts: str | list[str] | list[list[str]],
    language: str,
    bleu_keys: str | tuple[str, ...] = "bleu4",
    weights: tuple[float] | list[tuple[float]] | None = None,
    smoothing_level: int = 0,
) -> MetricDict | list[float]:
    """
    if `weights` is set, the `bleu_keys` will be ignored.\n
    if `bleu_keys` is ignored, then can not return MetricDict, just return the list of bleu score\n
    Smoothing level 0: No smoothing.\n
    Smoothing level 1: Add epsilon counts to precision with 0 counts.\n
    More detail about smoothing: `https://www.nltk.org/api/nltk.translate.bleu_score.html`
    """
    return_metricdict = weights is None
    if isinstance(preds, str):
        preds = [preds]

    if isinstance(tgts, str):
        tgts = [tgts]
    if isinstance(tgts[0], str):
        tgts = [[tgt] for tgt in tgts]
    # now preds: list[str], tgts: list[list[str]]
    assert len(preds) == len(tgts), f"The number of `preds` and `tgts` are not equal: len(preds)={len(preds)}, len(tgts)={len(tgts)}."

    if isinstance(bleu_keys, str):
        bleu_keys = (bleu_keys,)
    if weights is None:
        weights = []
        for key in bleu_keys:
            weights.append(bleu_key2weights(key))

    chencherry = SmoothingFunction()
    smoothing_function = getattr(chencherry, f"method{smoothing_level}")

    if language == "zh":
        tokenizer = list
    elif language == "en":
        tokenizer = lambda x: x.split()

    preds = [tokenizer(pred) for pred in preds]
    tgts = [[tokenizer(one_of_tgt) for one_of_tgt in tgt] for tgt in tgts]

    scores = corpus_bleu(tgts, preds, weights, smoothing_function)
    if isinstance(scores, float):
        scores = [scores]

    if return_metricdict:
        return MetricDict({key: value for key, value in zip(bleu_keys, scores)})
    else:
        return scores
