from collections import OrderedDict
from ..config import TrainConfig


def fill_ds_config(deepspeed_config: OrderedDict, config: TrainConfig):
    """
    fill `auto` field of deepspeed config with trainer config
    """
    if (fp16 := deepspeed_config.get("fp16", None)) is not None:
        if fp16["enabled"] == "auto":
            fp16["enabled"] = config.fp16
    if (bf16 := deepspeed_config.get("bf16", None)) is not None:
        if bf16["enabled"] == "auto":
            bf16["enabled"] = config.bf16

    if (opt := deepspeed_config.get("optimizer", None)) is not None:
        if opt["type"] == "auto":
            opt["type"] = config.opt_type
        for key, value in opt["params"].items():
            if value == "auto":
                if key != "betas":
                    opt["params"][key] = getattr(config, f"opt_{key}")
                else:
                    opt["params"][key] = [config.opt_betas1, config.opt_betas2]
    if (sch := deepspeed_config.get("scheduler", None)) is not None:
        if sch["type"] == "auto":
            sch["type"] = config.sch_type
        for key, value in sch["params"].items():
            if value == "auto":
                sch["params"][key] = getattr(config, f"sch_{key}")

    keys = ["gradient_accumulation_steps", "gradient_clipping", "train_batch_size"]
    for key in keys:
        if (value := deepspeed_config.get(key, None)) is not None:
            deepspeed_config[key] = getattr(config, key) if value == "auto" else value
