import copy
import json
import shutil
from collections import defaultdict
from functools import reduce
from heapq import nlargest
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import torch.distributed as dist
from transformers import PreTrainedTokenizer

from .. import toolkit_logger
from ..config.trainconfig import TrainConfig
from ..enums import Split
from ..logger import _getLogger
from ..metric import MetricDict
from ..utils.misc import find_file

logger = _getLogger("WatchDog")

WATCHDOG_DATA_NAME = "watchdog_data.json"
OPTIMAL_CHECKPOINT_NAME = "optimal_checkpoint"


class WatchDog:
    """Watch dog monitor the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience: int, metric: str, record_cheat: bool = True):
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
        self.local_rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.record_cheat = record_cheat
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
        model,
        configs: TrainConfig,
        tokenizer: PreTrainedTokenizer | None = None,
        file_logger: Logger | None = None,
        silence: bool = False,
    ):
        if epoch == configs.epochs - 1:
            self.finish()
        # log some information
        if file_logger is not None:
            pass
        else:
            file_logger = toolkit_logger
        if self.local_rank == 0:
            file_logger.info(f"epoch={epoch:03d} step={step_global:06d}")
            self.report(val_metricdict, Split.VALIDATION, file_logger=file_logger)
            if test_metricdict is not None:
                self.report(test_metricdict, Split.TEST, file_logger=file_logger)
            file_logger.info("")

        if self.record_cheat and test_metricdict is not None:
            if self.cheat_test_metricdict is None:
                self.cheat_test_metricdict = MetricDict(test_metricdict)
            elif test_metricdict > self.cheat_test_metricdict:
                self.cheat_test_metricdict.update(test_metricdict)

        if self.optimal_val_metricdict is None:
            self.best_checkpoint = (epoch, step_global)
            self.optimal_val_metricdict = MetricDict(val_metricdict)
            if test_metricdict is not None:
                self.optimal_test_metricdict = MetricDict(test_metricdict)
            self.save_optimal_model(model, configs, tokenizer, silence)
        elif val_metricdict > self.optimal_val_metricdict:
            self.best_checkpoint = (epoch, step_global)
            self.optimal_val_metricdict.update(val_metricdict)
            if test_metricdict is not None:
                self.optimal_test_metricdict.update(test_metricdict)
            self.counter = 0
            self.save_optimal_model(model, configs, tokenizer, silence)
        else:
            self.counter += 1
            if self.local_rank == 0:
                logger.debug(f"WatchDog patience: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.need_to_stop = True

        if self.local_rank == 0:
            file_logger.debug(f"WatchDog: {self.optimal_val_metricdict[self.metric_for_compare]} {self.counter}/{self.patience}")

    @staticmethod
    def report(metricdict: MetricDict, split: Split | Literal["TRAINING", "VALIDATION", "TEST", "ANY"], file_logger: Logger | None = None):
        "Report a metric dictionary."
        if not isinstance(split, Split):
            split = Split[split]
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
        if self.cheat_test_metricdict is not None:
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

    def save_hf_model(self, configs, output_dir, model, tokenizer):
        """
        Save the model and tokenizer.\n
        This function should be called in all subprocess.
        """
        # save model
        # 如果使用 deepspeed 的 ZeRO3 模式， 此时模型的参数在被分到了不同的卡上，需要save前先gather到同一卡上
        if configs.parallel_mode == "deepspeed" and model.zero_optimization_partition_weights():
            if model.zero_gather_16bit_weights_on_model_save():
                # consolidation is expensive in time and memory and therefore isn't a default
                state_dict = model._zero3_consolidated_16bit_state_dict()
            else:
                # the model will be bogus if not consolidated so don't confuse the user by saving it
                logger.error(f"Did not save the model {output_dir} because `stage3_gather_16bit_weights_on_model_save` is False")
                exit(1)
            if self.local_rank == 0:
                model.module.save_pretrained(output_dir, is_main_process=(self.local_rank == 0), state_dict=state_dict, max_shard_size="10GB")
                model.module.config.save_pretrained(output_dir, is_main_process=(self.local_rank == 0))
                # 此行代码解决一个奇怪的bug，该bug导致会多存一个没有用的 "pytorch_model.bin"
                if (output_dir / "pytorch_model.bin.index.json").exists() and (dummy_fie := (output_dir / "pytorch_model.bin")).exists():
                    dummy_fie.unlink()
        else:
            # 已知bug: 使用deepspeed 0-2时,无法shard模型,
            # 原因是transformers中关于shard model的实现中, 会将有相同 id_storage 的 state_dict 存到一起,
            # 而当使用deepspeed 0-2时, 不知为何所有的 state_dict 都有相同的 i_storage
            if self.local_rank == 0:
                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save_pretrained(output_dir, is_main_process=(self.local_rank == 0), max_shard_size="10GB")

        # save tokenizer
        if tokenizer is not None and self.local_rank == 0:
            tokenizer.save_pretrained(output_dir, is_main_process=(self.local_rank == 0))

    # TODO 当前只支持 Transformers 中的 model 和 tokenizer
    # TODO prior 保存最好的 n 个ckpt
    def save_optimal_model(self, model, configs: TrainConfig, tokenizer: PreTrainedTokenizer | None = None, silence=True):
        output_dir = Path(configs.save_dir, OPTIMAL_CHECKPOINT_NAME)
        if self.local_rank == 0:
            if output_dir.exists():
                shutil.rmtree(output_dir)
            output_dir.mkdir()
        if not silence and self.local_rank == 0:
            logger.debug("🚩 Saving optimal model weights ...")
            logger.debug(f"❔ The optimal model weights will be saved in {output_dir}.")
            # logger.debug(f"💾 Saving the optimal model and tokenizer to {output_dir} ...")

        # save model and tokenizer
        self.save_hf_model(configs, output_dir, model, tokenizer)

        # write performance
        if self.local_rank == 0:
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

        if not silence and self.local_rank == 0:
            logger.debug(f"✅ Save successfully.")

        # save watch dog(self)
        if self.local_rank == 0:
            self.save(output_dir)

    def save(self, save_dir: Path | str, json_file_name: str = WATCHDOG_DATA_NAME, silence=True, **kwargs):
        "Only master process will perform saving action."
        if self.local_rank == 0:
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
        # MetricDict.set_metric_for_compare(watch_dog.metric_for_compare)

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
        watch_dog = cls(attributes_dict["patience"], attributes_dict["metric_for_compare"], attributes_dict["record_cheat"])
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
                # attributes_dict[key]:MetricDict.save(save_dir=, file_name=)
        return json.dumps(attributes_dict, indent=2, sort_keys=False) + "\n"

    def to_dict(self) -> Dict[str, Any]:
        "Return the atrribute dictionary"
        output = copy.deepcopy(self.__dict__)
        return output

    @classmethod
    def metric_dicts_from_diff_seeds(
        cls, seeds_dir: Path | str, json_file_name: str = WATCHDOG_DATA_NAME, silence=False, load_cheat=False
    ) -> defaultdict[int, Dict[str, MetricDict]]:
        """
        Get a dict of validation metricdicts, test metricdicts and cheat metricdicts from different seed.\n
        return: `dict[seed][split] = MetricDict`
        """
        seeds_dir = Path(seeds_dir)
        assert seeds_dir.exists(), f"Seeds directory dose NOT exist: `{seeds_dir}`"
        seed_dirs = list(seeds_dir.iterdir())
        # seed_dirs = list(seeds_dir.glob("*"))
        success = 0
        ret = defaultdict(dict)
        for seed_dir in seed_dirs:
            watchdog_data_path = find_file(seed_dir, json_file_name)
            if watchdog_data_path and (watch_dog := cls.load(watchdog_data_path, silence=True)).is_finish():
                seed = int(Path(seed_dir).name)
                if watch_dog.optimal_val_metricdict is not None:
                    ret[seed]["val"] = watch_dog.optimal_val_metricdict
                if watch_dog.optimal_test_metricdict is not None:
                    ret[seed]["test"] = watch_dog.optimal_test_metricdict
                if load_cheat and watch_dog.cheat_test_metricdict is not None:
                    ret[seed]["cheat"] = watch_dog.cheat_test_metricdict
                success += 1
            else:
                logger.debug(f"❌ Failed: {seed_dir}")
        # print("xxxxxxx")
        if not silence:
            logger.info(f"success/total: {success}/{len(seed_dirs)}")
        return ret

    @staticmethod
    def _topk(metric_dicts: Dict[int, Dict[str, MetricDict]], top_k: int | None = None, base: str = "val"):
        if not metric_dicts:
            return dict(), dict()
        metric_dicts_topk = dict(nlargest(top_k, metric_dicts.items(), key=lambda item: item[1][base])) if top_k else metric_dicts
        for seed, dict_split in metric_dicts_topk.items():
            for split, metric_dict in dict_split.items():
                metric_dict.round(2)
        # best_seeds = list(metric_dicts_topk.keys())

        mean = dict(
            map(
                lambda item: (item[0], (item[1] / len(metric_dicts_topk)).round(2)),
                reduce(lambda x, y: {split: x[split] + y[split] for split in x.keys()}, metric_dicts_topk.values()).items(),
            )
        )

        return metric_dicts_topk, mean

    @staticmethod
    def topk(metric_dicts: Dict[int, Dict[str, MetricDict]], top_k: int | None = None, base: str | None = None, metric_for_compare: str | None = None):
        if metric_for_compare is not None:
            MetricDict.set_metric_for_compare(metric_for_compare)
        if base is None and metric_dicts:
            base = "test" if "test" in next(iter(metric_dicts.values())) else "val"
            if base == "test":
                metric_dicts = copy.deepcopy(metric_dicts)
                metric_dicts_cheat = dict()
                for seed, value in metric_dicts.items():
                    metric_dicts_cheat[seed] = {"cheat": value.pop("cheat", None)}
            metric_dicts_topk, mean = WatchDog._topk(metric_dicts, top_k, base)
            if base == "test" and next(iter(metric_dicts_cheat.values()))["cheat"]:
                _, mean_cheat = WatchDog._topk(metric_dicts_cheat, top_k, "cheat")
                mean["cheat"] = mean_cheat["cheat"]
        else:
            metric_dicts_topk, mean = WatchDog._topk(metric_dicts, top_k, base)

        return metric_dicts_topk, mean

    # @classmethod
    # def metric_dicts_from_diff_seeds(
    #     cls, seeds_dir: Path | str, json_file_name: str = WATCHDOG_DATA_NAME, silence=False
    # ) -> Dict[str, Dict[int, MetricDict]]:
    #     """
    #     Get a dict of validation metricdicts, test metricdicts and cheat metricdicts from different seed.\n
    #     return: `dict[split][seed] = MetricDict`
    #     """
    #     seeds_dir = Path(seeds_dir)
    #     assert seeds_dir.exists(), f"Seeds directory dose NOT exist: `{seeds_dir}`"
    #     seed_dirs = list(seeds_dir.iterdir())
    #     # seed_dirs = list(seeds_dir.glob("*"))
    #     success = 0
    #     val_metrics_dicts = dict()
    #     test_metrics_dicts = dict()
    #     cheat_metrics_dicts = dict()
    #     for seed_dir in seed_dirs:
    #         watchdog_data_path = find_file(seed_dir, json_file_name)
    #         if watchdog_data_path and (watch_dog := cls.load(watchdog_data_path, silence=True)).is_finish():
    #             seed = int(Path(seed_dir).name)
    #             if watch_dog.optimal_val_metricdict is not None:
    #                 val_metrics_dicts[seed] = watch_dog.optimal_val_metricdict
    #             if watch_dog.optimal_test_metricdict is not None:
    #                 test_metrics_dicts[seed] = watch_dog.optimal_test_metricdict
    #             if watch_dog.cheat_test_metricdict is not None:
    #                 cheat_metrics_dicts[seed] = watch_dog.cheat_test_metricdict
    #             success += 1
    #         else:
    #             logger.debug(f"❌ Failed: {seed_dir}")
    #     # print("xxxxxxx")
    #     if not silence:
    #         logger.info(f"success/total: {success}/{len(seed_dirs)}")
    #     return dict(val=val_metrics_dicts, test=test_metrics_dicts, cheat=cheat_metrics_dicts)

    # @staticmethod
    # def topk(metric_dicts_group: Dict[str, Dict[int, MetricDict]], top_k: int | None = None):
    #     metric_dicts = metric_dicts_group["test"] if metric_dicts_group["test"] else metric_dicts_group["val"]
    #     metric_dicts_topk = dict(nlargest(top_k, metric_dicts.items(), key=lambda item: item[1])) if top_k else metric_dicts
    #     best_seeds = list(metric_dicts_topk.keys())

    #     cheat_metric_dicts = metric_dicts_group["cheat"]
    #     cheat_metric_dicts_topk = dict(nlargest(top_k, cheat_metric_dicts.items(), key=lambda item: item[1])) if top_k else cheat_metric_dicts
    #     best_seeds_cheat = list(cheat_metric_dicts_topk.keys())

    #     ret = dict()
    #     for split, metric_dicts in metric_dicts_group.items():
    #         if metric_dicts:
    #             ret[split] = {seed: metric_dicts[seed].round(2) for seed in (best_seeds if split != "cheat" else best_seeds_cheat)}
    #     return ret

    # @staticmethod
    # def mean_topk(metric_dicts_group: Dict[str, Dict[int, MetricDict]], top_k: int | None = None) -> tuple[list[int], dict[str, MetricDict]]:
    #     metric_dicts_group = WatchDog.topk(metric_dicts_group, top_k)
    #     best_seeds = list(metric_dicts_group["val"].keys())

    #     ret = dict()
    #     for split, metric_dicts in metric_dicts_group.items():
    #         if metric_dicts:
    #             ret[split] = (reduce(lambda x, y: x + y, metric_dicts.values()) / len(best_seeds)).round(2)
    #     return best_seeds, ret
