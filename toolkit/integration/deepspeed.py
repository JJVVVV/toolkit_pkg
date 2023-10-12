from collections import OrderedDict
from functools import partialmethod

from transformers import PretrainedConfig

from ..config import TrainConfig


def find_config_node(ds_config: OrderedDict, ds_key_long: str):
    config = ds_config

    # find the config node of interest if it exists
    nodes = ds_key_long.split(".")
    ds_key = nodes.pop()
    for node in nodes:
        config = config.get(node)
        if config is None:
            return None, ds_key

    return config, ds_key


def get_value(ds_config: OrderedDict, ds_key_long: str, default=None):
    """
    Returns the set value or `default` if no value is set
    """
    config, ds_key = find_config_node(ds_config, ds_key_long)
    if config is None:
        return default
    return config.get(ds_key, default)


def is_zero3(ds_config: OrderedDict):
    return get_value(ds_config, "zero_optimization.stage") == 3


def is_auto(ds_config: OrderedDict, ds_key_long: str):
    val = get_value(ds_config, ds_key_long)
    return val == "auto"


def fill_match(ds_config, ds_key_long, hf_val, hf_key=None, must_match=True):
    """
    A utility method that massages the config file and can optionally verify that the values match.

    1. Replace "auto" values with `TrainingArguments` value.

    2. If it wasn't "auto" and `must_match` is true, then check that DS config matches Trainer
    config values and if mismatched add the entry to `self.mismatched` - will assert during
    `trainer_config_finalize` for one or more mismatches.

    """
    mismatches = []
    config, ds_key = find_config_node(ds_config, ds_key_long)
    if config is None:
        return

    if config.get(ds_key) == "auto":
        config[ds_key] = hf_val
        return

    if not must_match:
        return

    ds_val = config.get(ds_key)
    if ds_val is not None and ds_val != hf_val:
        mismatches.append(f"- ds {ds_key_long}={ds_val} vs hf {hf_key}={hf_val}")


fill_only = partialmethod(fill_match, must_match=False)
fill_only = lambda ds_config, ds_key_long, hf_val, hf_key=None: fill_match(ds_config, ds_key_long, hf_val, hf_key, False)


def fill_ds_config(deepspeed_config: OrderedDict, train_config: TrainConfig, model_config: PretrainedConfig):
    """
    fill `auto` field of deepspeed config with trainer config
    """
    if (fp16 := deepspeed_config.get("fp16", None)) is not None:
        if fp16["enabled"] == "auto":
            fp16["enabled"] = train_config.fp16
    if (bf16 := deepspeed_config.get("bf16", None)) is not None:
        if bf16["enabled"] == "auto":
            bf16["enabled"] = train_config.bf16

    if (opt := deepspeed_config.get("optimizer", None)) is not None:
        if opt["type"] == "auto":
            opt["type"] = train_config.opt_type
        for key, value in opt["params"].items():
            if value == "auto":
                if key != "betas":
                    opt["params"][key] = getattr(train_config, f"opt_{key}")
                else:
                    opt["params"][key] = [train_config.opt_betas1, train_config.opt_betas2]
    if (sch := deepspeed_config.get("scheduler", None)) is not None:
        if sch["type"] == "auto":
            sch["type"] = train_config.sch_type
        for key, value in sch["params"].items():
            if value == "auto":
                sch["params"][key] = getattr(train_config, f"sch_{key}")

    keys = ["gradient_accumulation_steps", "gradient_clipping", "train_batch_size"]
    for key in keys:
        if (value := deepspeed_config.get(key, None)) is not None:
            deepspeed_config[key] = getattr(train_config, key) if value == "auto" else value

    if deepspeed_config["zero_optimization"]["stage"] == 3:
        # keys = ['reduce_bucket_size', 'stage3_prefetch_bucket_size', 'stage3_param_persistence_threshold']
        hidden_size_based_keys = [
            "zero_optimization.reduce_bucket_size",
            "zero_optimization.stage3_prefetch_bucket_size",
            "zero_optimization.stage3_param_persistence_threshold",
        ]
        hidden_size_auto_keys = [key for key in hidden_size_based_keys if is_auto(deepspeed_config, key)]
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

            fill_only(deepspeed_config, "zero_optimization.reduce_bucket_size", hidden_size * hidden_size)
            if is_zero3(deepspeed_config):
                # automatically assign the optimal config values based on model config
                fill_only(deepspeed_config, "zero_optimization.stage3_prefetch_bucket_size", 0.9 * hidden_size * hidden_size)
                fill_only(deepspeed_config, "zero_optimization.stage3_param_persistence_threshold", 10 * hidden_size)
