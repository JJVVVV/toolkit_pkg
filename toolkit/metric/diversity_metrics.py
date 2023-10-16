from ._metricdict import MetricDict
from .utils.utils_distinct_n import ngrams
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import numpy as np
from numpy.typing import NDArray


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


def self_bleu_one_set(s: list[str], weights=(0.25, 0.25, 0.25, 0.25), smoothing_level: int = 0, language: str = "zh") -> NDArray:
    """
    Smoothing level 0: No smoothing.\n
    Smoothing level 1: Add epsilon counts to precision with 0 counts.\n
    Detail: `https://www.nltk.org/api/nltk.translate.bleu_score.html`
    """
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
    # if isinstance(scores, list):
    #     scores = [1 - score for score in scores]
    # else:
    #     scores = 1 - scores
    return 1 - np.array(scores)


def self_bleu(sets_list: list[list[str]], weights=(0.25, 0.25, 0.25, 0.25), smoothing_level: int = 0, language: str = "zh") -> NDArray:
    # sets_list: 一个列表, 其中的每个元素是一个集合, 要计算每个集合中各个元素之间的多样性程度, 然后求平均
    score = 0
    for s in sets_list:
        score += self_bleu_one_set(s, weights, smoothing_level, language)
    return score / len(sets_list)
