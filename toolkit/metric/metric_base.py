from abc import ABC, abstractmethod
import numpy as np
import os


global_score_cache = {}
similarity2diversity_function = lambda sim_score_list: -np.mean(sim_score_list)


class Metric(ABC):
    use_me = False  # static var indicates to run files whether or not to use this metric
    default_config = {}  # static var, specifies the default config for run files

    def __init__(self, config):
        self.config = config

        # validate config
        assert type(self.config) == dict, "Metric config must be dict type."

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def uint_assert(self, field_name):
        err_msg = "Required: {}(int) > 0".format(field_name)
        assert type(self.config.get(field_name, None)) == int, err_msg
        assert self.config[field_name] > 0, err_msg

    def input_path_assert(self, field_name):
        err_msg = "[{}] not exists.".format(field_name)
        assert os.path.exists(self.config.get(field_name, None)), err_msg


class DiversityMetric(Metric):
    required_input = "response_set"  # in most cases, the diversity metric input is the response set S_c

    def __init__(self, config):
        super().__init__(config)

    @abstractmethod
    def __call__(self, response_set):
        # validate input
        assert type(response_set) == list
        assert all([type(e) == str for e in response_set])

        # place holder
        diversity_score = None
        return diversity_score


class SimilarityMetric(Metric):
    def __init__(self, config):
        super().__init__(config)

    @abstractmethod
    def __call__(self, resp_a, resp_b):
        # validate input
        assert type(resp_a) == type(resp_b) == str

        # place holder
        similarity_score = None
        return similarity_score


class Similarity2DiversityMetric(DiversityMetric):
    """
    Implements the diversity to similarity reduction specified on section 5 in the paper
    (https://arxiv.org/pdf/2004.02990.pdf)
    for any similarity metric.

    config:
        shared with the original similarity metric.

    usage:
        metric = Similarity2DiversityMetric(config, SimilarityMetricClassName)
        metric(response_set)

    inheritance guidelines:
        implement __init__ only

    inheritance example:
        see CosineSimilarity2Diversity
    """

    def __init__(self, config, similarity_metric_class):
        super().__init__(config)
        assert issubclass(similarity_metric_class, SimilarityMetric)
        self.similarity_metric = similarity_metric_class(config)

    def __call__(self, response_set):
        super().__call__(response_set)

        similarity_list = []
        for i in range(len(response_set)):
            for j in range(i):
                similarity_list.append(self.similarity_metric(response_set[i], response_set[j]))
        diversity_score = similarity2diversity_function(similarity_list)
        return diversity_score


class AveragedNgramDiversityMetric(DiversityMetric):
    """
    Calculates the mean values of an n-gram based diversity metric in range n \in [n_min, n_max].

    config:
        shared with the original n-gram metric.
        n_min(int) > 0 - Specify the lowest n-gram value to be averaged
        n_max(int) > 0 - Specify the highest n-gram value to be averaged

    usage:
        metric = AveragedNgramDiversityMetric(config, NgramMetricClassName)
        metric(response_set)

    inheritance guidelines:
        implement __init__ only

    inheritance example:
        see AveragedDistinctNgrams
    """

    def __init__(self, config, ngram_metric_class):
        super().__init__(config)

        # validate config
        self.uint_assert("n_min")
        self.uint_assert("n_max")
        err_msg = "AveragedNgramMetric config must include n_max > n_min > 0 (int) representing n-gram size."
        assert self.config["n_max"] > self.config["n_min"] > 0, err_msg

        # add n field
        self.config["n"] = self.config["n_min"]

        # instance ngram metric
        assert issubclass(ngram_metric_class, DiversityMetric)
        self.ngram_metric = ngram_metric_class(self.config)

    def __call__(self, response_set):
        super().__call__(response_set)

        ngrams_results = []
        for n in range(self.config["n_min"], self.config["n_max"] + 1):
            self.config["n"] = n
            result = self.ngram_metric(response_set)
            # print('{}, {}'.format(self.ngram_metric.config['n'], result))
            ngrams_results.append(result)
        return np.mean(ngrams_results)
