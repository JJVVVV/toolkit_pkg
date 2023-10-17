import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from numpy.typing import NDArray

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
    s: list[str], bleu_keys: str | tuple[str] = "bleu4", weights: tuple[int] | list[tuple[int]] = None, smoothing_level: int = 0, language: str = "zh"
) -> NDArray:
    """
    if `weights` is set, the `bleu_keys` will be ignored.
    Smoothing level 0: No smoothing.\n
    Smoothing level 1: Add epsilon counts to precision with 0 counts.\n
    Detail: `https://www.nltk.org/api/nltk.translate.bleu_score.html`
    """
    if isinstance(bleu_keys, str):
        bleu_keys = (bleu_keys,)
    if weights is None:
        weights = []
        for key in bleu_keys:
            weights.append(bleu_keys2weights[key])

    chencherry = SmoothingFunction()
    smoothing_function = getattr(chencherry, f"method{smoothing_level}")

    if language == "zh":
        tokenizer = list
    elif language == "en":
        tokenizer = lambda x: x.split()
    s = [tokenizer(x) for x in s]
    refs = []
    hyps = []
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            hyps.append(s[i])
            refs.append([s[j]])
    scores = corpus_bleu(refs, hyps, weights, smoothing_function)

    return 1 - np.array(scores)


def self_bleu(
    sets_list: list[list[str]],
    bleu_keys: str | tuple[str] = "bleu4",
    weights: tuple[int] | list[tuple[int]] = None,
    smoothing_level: int = 0,
    language: str = "zh",
) -> NDArray:
    # sets_list: 一个列表, 其中的每个元素是一个集合, 要计算每个集合中各个元素之间的多样性程度, 然后求平均
    score = 0
    for s in sets_list:
        score += self_bleu_one_set(s, bleu_keys, weights, smoothing_level, language)
    return score / len(sets_list)
