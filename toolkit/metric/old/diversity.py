import numpy as np

from .. import metric_base, utils


class DistinctNgrams(metric_base.DiversityMetric):
    default_config = {"n": 3}

    def __init__(self, config):
        super().__init__(config)

        # validate config
        self.uint_assert("n")

    def normalized_unique_ngrams(self, ngram_lists):
        """
        Calc the portion of unique n-grams out of all n-grams.
        :param ngram_lists: list of lists of ngrams
        :return: value in (0,1]
        """
        ngrams = [item for sublist in ngram_lists for item in sublist]  # flatten
        return len(set(ngrams)) / len(ngrams) if len(ngrams) > 0 else 0.0

    def __call__(self, response_set):
        super().__call__(response_set)
        return self.normalized_unique_ngrams(utils.lines_to_ngrams(response_set, n=self.config["n"]))


class AveragedDistinctNgrams(metric_base.AveragedNgramDiversityMetric):
    use_me = True
    default_config = {"n_min": 1, "n_max": 5}

    def __init__(self, config):
        super().__init__(config, DistinctNgrams)
