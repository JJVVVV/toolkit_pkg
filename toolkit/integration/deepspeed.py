from collections import OrderedDict
from functools import partialmethod

from transformers import PretrainedConfig

from ..config import TrainConfig


class DeepspeedConfig:
    def __init__(self, ds_config: dict) -> None:
        self.ds_config = ds_config
        self.mismatches = []

    def find_config_node(self, ds_key_long: str):
        # find the config node of interest if it exists
        config = self.ds_config
        nodes = ds_key_long.split(".")
        ds_key = nodes.pop()
        for node in nodes:
            config = config.get(node)
            if config is None:
                return None, ds_key

        return config, ds_key

    def get_value(self, ds_key_long: str, default=None):
        """
        Returns the set value or `default` if no value is set
        """
        config, ds_key = self.find_config_node(ds_key_long)
        if config is None:
            return default
        return config.get(ds_key, default)

    def is_zero3(self):
        return self.get_value("zero_optimization.stage") == 3

    def is_auto(self, ds_key_long: str):
        val = self.get_value(ds_key_long)
        return val == "auto"

    def fill_match(self, ds_key_long, hf_val, hf_key=None, must_match=True):
        """
        A utility method that massages the config file and can optionally verify that the values match.

        1. Replace "auto" values with `TrainingArguments` value.

        2. If it wasn't "auto" and `must_match` is true, then check that DS config matches Trainer
        config values and if mismatched add the entry to `self.mismatched` - will assert during
        `trainer_config_finalize` for one or more mismatches.
        3. Do nothing if the ds_key_long doesn't exist.

        """
        config, ds_key = self.find_config_node(ds_key_long)
        if config is None:
            return

        if config.get(ds_key) == "auto":
            config[ds_key] = hf_val
            return

        if not must_match:
            return

        ds_val = config.get(ds_key)
        if ds_val is not None and ds_val != hf_val:
            self.mismatches.append(f"- ds {ds_key_long}={ds_val} vs hf {hf_key}={hf_val}")

    fill_only = partialmethod(fill_match, must_match=False)
    # fill_only = lambda ds_config, ds_key_long, hf_val: fill_match(ds_config, ds_key_long, hf_val, None, False)

    def fill_ds_config(self, train_config: TrainConfig, model_config: PretrainedConfig):
        """
        fill `auto` field of deepspeed config with trainer config
        """
        # half-precision
        self.fill_match("fp16.enabled", train_config.fp16, "fp16")
        self.fill_match("bf16.enabled", train_config.bf16, "bf16")
        # optimizer
        self.fill_match("optimizer.type", train_config.opt_type, "opt_type")
        self.fill_match("optimizer.params.betas", [train_config.opt_betas1, train_config.opt_betas2], "opt_betas1, opt_betas2")
        optimizer_params = ["lr", "eps", "weight_decay"]
        for key in optimizer_params:
            self.fill_match(f"optimizer.params.{key}", getattr(train_config, f"opt_{key}"), f"opt_{key}")
        # scheduler
        self.fill_match("scheduler.type", train_config.sch_type, "sch_type")
        scheduler_parmas = ["warmup_min_lr", "warmup_max_lr", "warmup_num_steps", "total_num_steps"]
        for key in scheduler_parmas:
            self.fill_match(f"scheduler.params.{key}", getattr(train_config, f"sch_{key}"), f"sch_{key}")
        # training hyper
        keys = [
            "gradient_accumulation_steps",
            "gradient_clipping",
            "train_batch_size",
            "train_micro_batch_size_per_gpu",
        ]  # "train_micro_batch_size_per_gpu"
        for key in keys:
            self.fill_match(key, getattr(train_config, key), key)

        if self.is_zero3():
            hidden_size_based_keys = [
                "zero_optimization.reduce_bucket_size",
                "zero_optimization.stage3_prefetch_bucket_size",
                "zero_optimization.stage3_param_persistence_threshold",
            ]
            hidden_size_auto_keys = [key for key in hidden_size_based_keys if self.is_auto(key)]
            if len(hidden_size_auto_keys) > 0:
                if hasattr(model_config, "hidden_size"):
                    hidden_size = model_config.hidden_size
                elif hasattr(model_config, "hidden_sizes"):
                    # if there are many hidden sizes pick the largest one
                    hidden_size = max(model_config.hidden_sizes)
                else:
                    raise ValueError(
                        "The model's config file has neither `hidden_size` nor `hidden_sizes` entry, "
                        "therefore it's not possible to automatically fill out the following `auto` entries "
                        f"in the DeepSpeed config file: {hidden_size_auto_keys}. You can fix that by replacing "
                        "`auto` values for these keys with an integer value of your choice."
                    )

                self.fill_only("zero_optimization.reduce_bucket_size", hidden_size * hidden_size)
                if self.is_zero3():
                    # automatically assign the optimal config values based on model config
                    self.fill_only("zero_optimization.stage3_prefetch_bucket_size", 0.9 * hidden_size * hidden_size)
                    self.fill_only("zero_optimization.stage3_param_persistence_threshold", 10 * hidden_size)

        assert len(self.mismatches) == 0, "\n" + "\n".join(self.mismatches)
