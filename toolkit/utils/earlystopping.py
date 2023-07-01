import copy
import glob
import json
import os
import shutil
from functools import reduce
from heapq import nlargest
from pathlib import Path
from typing import Any, Dict

from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from ..logger import _getLogger
from .metricdict import MetricDict
from .trainconfig import TrainConfig

logger = _getLogger(__name__)

EARLYSTOPPING_DATA_NAME = "earlystopping_data.json"


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=5, metric="acc"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            如果是True, 为每个验证集损失值改善打印一条信息
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            监测数量的最小变化，以符合改进的要求
                            Default: 0
        """
        self.patience = patience
        self.counter = 0

        self.need_to_stop = False
        self.best_checkpoint = None

        self.metric_used_to_comp = metric
        MetricDict.metric_used_to_comp = self.metric_used_to_comp  # ? loss, acc,  f1
        # * 保证 MetricDict 中的值越大越好
        match metric:
            case "loss":
                self.scale = -1
            case _:
                self.scale = 1

        MetricDict.scale = self.scale

        self.optimal_dev_metrics_dict = None
        self.optimal_test_metrics_dict = None
        self.cheat_test_metrics_dict = None

    def __call__(
        self,
        dev_metrics_dict: MetricDict,
        test_metrics_dict: MetricDict | None,
        curCheckpoint: int,
        curStep: int,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        configs: TrainConfig,
    ):
        if self.optimal_dev_metrics_dict is None:
            self.best_checkpoint = curCheckpoint
            self.optimal_dev_metrics_dict = MetricDict(dev_metrics_dict)
            if test_metrics_dict is not None:
                self.optimal_test_metrics_dict = MetricDict(test_metrics_dict)
            self.save_checkpoint(model, tokenizer, configs)
        elif dev_metrics_dict > self.optimal_dev_metrics_dict:
            self.best_checkpoint = curCheckpoint
            self.optimal_dev_metrics_dict.update(dev_metrics_dict)
            if test_metrics_dict is not None:
                self.optimal_test_metrics_dict.update(test_metrics_dict)
            self.save_checkpoint(model, tokenizer, configs)
            self.counter = 0
        else:
            self.counter += 1
            logger.debug(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.need_to_stop = True

        if test_metrics_dict is not None:
            if self.cheat_test_metrics_dict is None:
                self.cheat_test_metrics_dict = MetricDict(test_metrics_dict)
            elif test_metrics_dict > self.cheat_test_metrics_dict:
                self.cheat_test_metrics_dict.update(test_metrics_dict)

    def save_checkpoint(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, configs: TrainConfig):
        output_dir = configs.checkpoints_dir + "/best_checkpoint"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        logger.debug(f"Saving the optimal model and tokenizer to {output_dir}.")
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        configs.save(output_dir)
        logger.debug(f"Save successfully.")

    def save(self, save_directory: Path | str, **kwargs):
        if isinstance(save_directory, str):
            save_directory = Path(save_directory)
        if save_directory.is_file():
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")
        save_directory.mkdir(parents=True, exist_ok=True)
        data_file_name = kwargs.pop("config_file_name", EARLYSTOPPING_DATA_NAME)
        output_config_file_path = save_directory / data_file_name

        self.to_json_file(output_config_file_path)
        logger.info(f"EarlyStopping data saved in {output_config_file_path}")

    @classmethod
    def load(cls, pretrained_model_dir_or_path: Path | str, silence=False, **kwargs) -> "EarlyStopping":
        json_file_name = kwargs.pop("json_file_name", EARLYSTOPPING_DATA_NAME)
        if isinstance(pretrained_model_dir_or_path, str):
            pretrained_model_dir_or_path = Path(pretrained_model_dir_or_path)

        if pretrained_model_dir_or_path.is_file():
            resolved_config_file = pretrained_model_dir_or_path
        else:
            resolved_config_file = pretrained_model_dir_or_path / json_file_name
        # Load config dict
        try:
            attributes_dict = cls._dict_from_json_file(resolved_config_file)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise EnvironmentError(f"It looks like the config file at '{resolved_config_file}' is not a valid JSON file.")
        if not silence:
            logger.info(f"loading configuration file {resolved_config_file}")
        attributes_dict.update(kwargs)
        early_stopping = cls.from_dict(attributes_dict)
        MetricDict.scale = early_stopping.scale
        MetricDict.metric_used_to_comp = early_stopping.metric_used_to_comp
        return early_stopping

    @staticmethod
    def _dict_from_json_file(json_file: Path | str) -> Dict:
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        attributes_dict = json.loads(text)
        for key, value in attributes_dict.items():
            if isinstance(value, dict):
                attributes_dict[key] = MetricDict(value)
        return attributes_dict

    @classmethod
    def from_dict(cls, attributes_dict: Dict[str, Any]) -> "EarlyStopping":
        early_stopping = cls()
        early_stopping._update(attributes_dict)
        return early_stopping

    def _update(self, attributes_dict: Dict[str, Any]):
        for key, value in attributes_dict.items():
            setattr(self, key, value)

    def to_json_file(self, json_file_path: Path | str):
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def to_json_string(self) -> str:
        attributes_dict = self.to_dict()
        for key, value in attributes_dict.items():
            if isinstance(value, MetricDict):
                attributes_dict[key] = dict(value)
        return json.dumps(attributes_dict, indent=2, sort_keys=False) + "\n"

    def to_dict(self) -> Dict[str, Any]:
        output = copy.deepcopy(self.__dict__)
        return output


def search_file(directory, filename):
    file_paths = []
    # warkos.walk遍历目录下的所有子目录和文件: root为某个目录, dirs为root下的目录, files为root下的文件
    for root, dirs, files in os.walk(directory):
        if filename in files:
            file_path = os.path.join(root, filename)
            file_paths.append(file_path)
    return file_paths


def load_metric_dicts_from_earlystopping(seeds_dir, json_file_name=EARLYSTOPPING_DATA_NAME):
    seed_dirs = glob.glob(seeds_dir + "/*")
    success = 0
    dev_metrics_dicts = []
    test_metrics_dicts = []
    cheat_metrics_dicts = []
    for seed_dir in seed_dirs:
        earlyStopping_path = search_file(seed_dir, json_file_name)
        if earlyStopping_path:
            if "checkpoint-" in earlyStopping_path[0]:
                print(seed_dir)
                continue
            earlyStopping = EarlyStopping.load(earlyStopping_path[0], silence=True)
            dev_metrics_dicts.append(earlyStopping.optimal_dev_metrics_dict)
            test_metrics_dicts.append(earlyStopping.optimal_test_metrics_dict)
            cheat_metrics_dicts.append(earlyStopping.cheat_test_metrics_dict)
            success += 1
        else:
            print(seed_dir)
    print(f"success/total: {success}/{len(seed_dirs)}")
    return dev_metrics_dicts, test_metrics_dicts, cheat_metrics_dicts


def top_k_mean_in_metric_dicts(metric_dicts: list[MetricDict], top_k=None):
    if not metric_dicts:
        print("No metric dict.")
    if top_k is None:
        return reduce(lambda x, y: x + y, metric_dicts) / len(metric_dicts)
    metric_dicts_topk = nlargest(top_k, metric_dicts)
    return reduce(lambda x, y: x + y, metric_dicts_topk) / len(metric_dicts_topk)
