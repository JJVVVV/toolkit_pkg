# import numpy as np
# from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
# from numpy.typing import NDArray
from tqdm.auto import tqdm

from . import MetricDict, bleu
from .utils.utils_distinct_n import ngrams


def distinct_n_sentence_level(sentence, n):
    """
    Compute distinct-N for a single sentence.
    :param sentence: a list of words.
    :param n: int, ngram.
    :return: float, the metric value.
    """
    if len(sentence) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(ngrams(sentence, n))
    return len(distinct_ngrams) / len(sentence)


def distinct_n_corpus_level(sentences, n):
    """
    Compute average distinct-N of a list of sentences (the corpus).
    :param sentences: a list of sentence.
    :param n: int, ngram.
    :return: float, the average value.
    """
    return sum(distinct_n_sentence_level(sentence, n) for sentence in sentences) / len(sentences)


bleu_keys2weights = dict(
    bleu1=(1.0,), bleu2=(1.0 / 2.0, 1.0 / 2.0), bleu3=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), bleu4=(1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0)
)


def self_bleu_one_set(
    s: list[str],
    language: str,
    bleu_keys: str | tuple[str] = "bleu4",
    weights: tuple[float] | list[tuple[float]] | None = None,
    smoothing_level: int = 0,
) -> MetricDict | list[float]:
    """
    if `weights` is set, the `bleu_keys` will be ignored.\n
    if `bleu_keys` is ignored, then can not return MetricDict, just return the list of self-bleu score\n
    Smoothing level 0: No smoothing.\n
    Smoothing level 1: Add epsilon counts to precision with 0 counts.\n
    More detail about smoothing: `https://www.nltk.org/api/nltk.translate.bleu_score.html`
    language: str, Optional["zh", "en"]
    """
    refs = []
    hyps = []
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            hyps.append(s[i])
            refs.append(s[j])
    metric = bleu(refs, hyps, language, bleu_keys, weights, smoothing_level)
    if isinstance(metric, MetricDict):
        # return 1 - MetricDict({f"self-{key}": value for key, value in metric.items()})
        return MetricDict({f"self-{key}": value for key, value in metric.items()})

    else:
        # return [1 - score for score in metric]
        return metric


def self_bleu(
    sets_list: list[list[str]],
    language: str,
    bleu_keys: str | tuple[str] = "bleu4",
    weights: tuple[float] | list[tuple[float]] | None = None,
    smoothing_level: int = 0,
) -> MetricDict | list[float]:
    """
    if `weights` is set, the `bleu_keys` will be ignored.\n
    if `bleu_keys` is ignored, then can not return MetricDict, just return the list of self-bleu score\n
    Smoothing level 0: No smoothing.\n
    Smoothing level 1: Add epsilon counts to precision with 0 counts.\n
    More detail about smoothing: `https://www.nltk.org/api/nltk.translate.bleu_score.html`
    language: str, Optional["zh", "en"]
    """
    # sets_list: 一个列表, 其中的每个元素是一个集合, 要计算每个集合中各个元素之间的多样性程度, 然后求平均
    if weights is None:
        score = 0
        for s in tqdm(sets_list, desc="Calculate self-bleu: "):
            score += self_bleu_one_set(s, language, bleu_keys, weights, smoothing_level)
        return score / len(sets_list)
    else:
        raise NotImplementedError()
