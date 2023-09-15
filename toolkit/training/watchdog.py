import copy
import glob
import json
import shutil
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Tuple

from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from .. import toolkit_logger
from ..config.trainconfig import TrainConfig
from ..enums import Split
from ..logger import _getLogger
from ..metric.metricdict import MetricDict
from ..utils.misc import find_file

logger = _getLogger("WatchDog")

WATCHDOG_DATA_NAME = "watchdog_data.json"
OPTIMAL_CHECKPOINT_NAME = "optimal_checkpoint"


class WatchDog:
    """Watch dog monitor the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience: int, metric: str):
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

        self.metric_for_compare = metric
        if metric is not None:
            MetricDict.set_metric_for_compare(metric)

        self.optimal_val_metricdict = None
        self.optimal_test_metricdict = None
        self.cheat_test_metricdict = None
        self.finished = False

    def finish(self):
        "Indicates that the training is over."
        self.finished = True

    def is_finish(self) -> bool:
        return self.finished

    def __call__(
        self,
        val_metricdict: MetricDict,
        test_metricdict: MetricDict | None,
        epoch: int,
        step_global: int,
        model: PreTrainedModel,
        configs: TrainConfig,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        file_logger: Logger | None = None,
        silence: bool = False,
    ):
        # log some information
        if file_logger is not None:
            logger = file_logger
        else:
            logger = toolkit_logger
        logger.info(f"epoch={epoch:03d} step={step_global:06d}")
        self.report(val_metricdict, Split.VALIDATION, file_logger=logger)
        if test_metricdict is not None:
            self.report(test_metricdict, Split.TEST, file_logger=logger)
        logger.info("")

        if self.optimal_val_metricdict is None:
            self.best_checkpoint = (epoch, step_global)
            self.optimal_val_metricdict = MetricDict(val_metricdict)
            if test_metricdict is not None:
                self.optimal_test_metricdict = MetricDict(test_metricdict)
            self.save_optimal_checkpoint(model, configs, tokenizer, silence)
        elif val_metricdict > self.optimal_val_metricdict:
            self.best_checkpoint = (epoch, step_global)
            self.optimal_val_metricdict.update(val_metricdict)
            if test_metricdict is not None:
                self.optimal_test_metricdict.update(test_metricdict)
            self.counter = 0
            self.save_optimal_checkpoint(model, configs, tokenizer, silence)
        else:
            self.counter += 1
            logger.debug(f"WatchDog patience: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.need_to_stop = True

        if test_metricdict is not None:
            if self.cheat_test_metricdict is None:
                self.cheat_test_metricdict = MetricDict(test_metricdict)
            elif test_metricdict > self.cheat_test_metricdict:
                self.cheat_test_metricdict.update(test_metricdict)
        logger.debug(f"WatchDog: {self.optimal_val_metricdict[self.metric_for_compare]} {self.counter}/{self.patience}")

    @staticmethod
    def report(metricdict: MetricDict, split: Split, file_logger: Logger | None = None):
        "Report a metric dictionary."
        if file_logger is not None:
            logger = file_logger
        else:
            logger = toolkit_logger
        info = f"<{split.name:^14}>  {metricdict}"
        logger.info(info)

    def final_report(self, configs: TrainConfig, file_logger: Logger | None = None):
        "Report the final information after training finished."
        if file_logger is not None:
            logger = file_logger
        else:
            logger = toolkit_logger
        logger.info("------------------------------------Report------------------------------------")
        # * if early stop is triggered
        if configs.early_stop and self.need_to_stop:
            logger.info(f"trainning is stopped by WatchDog")
        # * If early stop is set but is not triggered, the epoch may have been set too small
        elif configs.early_stop:
            logger.info(f"All epochs are finished and the early stop is not triggered. The model may need further training!")
        logger.info(f"Cheat performance: {str(self.cheat_test_metricdict)}")
        logger.info(f"The best model at (epoch={self.best_checkpoint[0]}, step_global={self.best_checkpoint[1]})")
        logger.info(f"Dev performance: {str(self.optimal_val_metricdict)}")
        if self.optimal_test_metricdict is not None:
            logger.info(f"Test performance: {str(self.optimal_test_metricdict)}")

    def optimal_performance(self) -> Dict[str, float]:
        """
        Return a python dict which contain performances of all metrics on test, validation and cheat.
        """
        ret = dict()
        for metricdict, prefix in zip(
            (self.optimal_test_metricdict, self.optimal_val_metricdict, self.cheat_test_metricdict), ("Test_", "Val_", "Cheat_")
        ):
            if metricdict is not None:
                for key, value in metricdict.items():
                    ret[prefix + key] = value
        return ret

    # TODO 当前只支持 Transformers 中的 model 和 tokenizer
    def save_optimal_checkpoint(
        self, model: PreTrainedModel, configs: TrainConfig, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None, silence=True
    ):
        output_dir = Path(configs.save_dir, OPTIMAL_CHECKPOINT_NAME)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir()
        if not silence:
            logger.debug("🚩 Saving optimal checkpoint ...")
            logger.debug(f"❔ The optimal checkpoint will be saved in {output_dir}.")
            # logger.debug(f"💾 Saving the optimal model and tokenizer to {output_dir} ...")

        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
        configs.save(output_dir)
        with open(output_dir / "performance.json", "w", encoding="utf-8") as writer:
            writer.write(
                json.dumps(
                    {
                        "Validiton": dict(self.optimal_val_metricdict),
                        "Test": dict(self.optimal_test_metricdict) if self.optimal_test_metricdict is not None else None,
                    },
                    indent=2,
                    sort_keys=False,
                )
                + "\n"
            )

        if not silence:
            logger.debug(f"✅ Save optimal checkpoint successfully.")

        self.save(output_dir)

    def save(self, save_dir: Path | str, json_file_name: str = WATCHDOG_DATA_NAME, silence=True, **kwargs):
        if not silence:
            logger.debug(f"💾 Saving Watch Dog data ...")
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
        if save_dir.is_file():
            raise AssertionError(f"Provided path ({save_dir}) should be a directory, not a file")
        save_dir.mkdir(parents=True, exist_ok=True)
        json_file_path = save_dir / json_file_name

        self.to_json_file(json_file_path)
        if not silence:
            # logger.debug(f"Save WatchDog data in {json_file_path} successfully.")
            logger.debug(f"✔️  Save successfully.")

    @classmethod
    def load(cls, json_file_dir_or_path: Path | str, json_file_name: str = WATCHDOG_DATA_NAME, silence=True, **kwargs) -> "WatchDog":
        if not silence:
            logger.debug(f"💾 Loading Watch Dog data ...")

        if isinstance(json_file_dir_or_path, str):
            json_file_dir_or_path = Path(json_file_dir_or_path)

        if json_file_dir_or_path.is_file():
            json_file_path = json_file_dir_or_path
        else:
            json_file_path = json_file_dir_or_path / json_file_name
        # Load config dict
        try:
            attributes_dict = cls._dict_from_json_file(json_file_path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise EnvironmentError(f"It looks like the config file at '{json_file_path}' is not a valid JSON file.")
        attributes_dict.update(kwargs)
        watch_dog = cls.from_dict(attributes_dict)
        MetricDict.set_metric_for_compare(watch_dog.metric_for_compare)

        if not silence:
            logger.debug(f"✔️  Load successfully.")
        return watch_dog

    @staticmethod
    def _dict_from_json_file(json_file: Path | str) -> Dict:
        "Load a attributes dictionary from a json file."
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        attributes_dict = json.loads(text)
        for key in ("optimal_val_metricdict", "optimal_test_metricdict", "cheat_test_metricdict"):
            if attributes_dict[key] is not None:
                attributes_dict[key] = MetricDict(attributes_dict[key])
        return attributes_dict

    @classmethod
    def from_dict(cls, attributes_dict: Dict[str, Any]) -> "WatchDog":
        "Load a WatchDog from a attributes dictionary."
        watch_dog = cls(None, None)
        watch_dog._update(attributes_dict)
        return watch_dog

    def _update(self, attributes_dict: Dict[str, Any]):
        "Set attributes from a attribute dictionary."
        for key, value in attributes_dict.items():
            setattr(self, key, value)

    def to_json_file(self, json_file_path: Path | str):
        "Write json string to the given file path"
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def to_json_string(self) -> str:
        "Dump the attribute dictionary to json string."
        attributes_dict = self.to_dict()
        for key in ("optimal_val_metricdict", "optimal_test_metricdict", "cheat_test_metricdict"):
            if attributes_dict[key] is not None:
                attributes_dict[key] = dict(attributes_dict[key])
        return json.dumps(attributes_dict, indent=2, sort_keys=False) + "\n"

    def to_dict(self) -> Dict[str, Any]:
        "Return the atrribute dictionary"
        output = copy.deepcopy(self.__dict__)
        return output

    @classmethod
    def metric_dicts_from_diff_seeds(
        cls, seeds_dir: Path | str, json_file_name: str = WATCHDOG_DATA_NAME, silence=False
    ) -> Tuple[List[MetricDict], List[MetricDict], List[MetricDict]]:
        """
        Get a list of validation metricdicts, test metricdicts and cheat test metricdicts from different seed.
        """
        seed_dirs = glob.glob(seeds_dir + "/*")
        success = 0
        dev_metrics_dicts = []
        test_metrics_dicts = []
        cheat_metrics_dicts = []
        for seed_dir in seed_dirs:
            watchdog_data_path = find_file(seed_dir, json_file_name)
            if watchdog_data_path and (watch_dog := cls.load(watchdog_data_path, silence=True)).is_finish():
                if watch_dog.optimal_val_metricdict is not None:
                    dev_metrics_dicts.append(watch_dog.optimal_val_metricdict)
                if watch_dog.optimal_test_metricdict is not None:
                    test_metrics_dicts.append(watch_dog.optimal_test_metricdict)
                if watch_dog.cheat_test_metricdict is not None:
                    cheat_metrics_dicts.append(watch_dog.cheat_test_metricdict)
                success += 1
            else:
                logger.debug(f"❌ Failed: {seed_dir}")
        # print("xxxxxxx")
        if not silence:
            logger.info(f"success/total: {success}/{len(seed_dirs)}")
        return dev_metrics_dicts, test_metrics_dicts, cheat_metrics_dicts
