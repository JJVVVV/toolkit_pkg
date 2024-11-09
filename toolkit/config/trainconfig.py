# from collections import OrderedDict
from pathlib import Path
from typing import Self

from ..metric import MetricDict
from .config_base import ConfigBase, logger

# SILENCE = True
CONFIG_NAME = "train_config.json"


class TrainConfig(ConfigBase):
    def __init__(
        self,
        seed: int = 0,
        gpu: bool = True,
        problem_type: str = "",
        dataset_name: str = "",
        train_file_path: Path | str | None = None,
        val_file_path: Path | str | None = None,
        test_file_path: Path | str | None = None,
        model_type: str = "",
        model_name: str = "",
        model_dir: str | None = None,
        metric: str = "loss",
        epochs: int = 0,
        train_batch_size: int = 0,
        infer_batch_size: int = 0,
        opt_type: str | None = None,
        opt_lr: float = 1e-4,
        opt_betas1: float = 0.9,
        opt_betas2: float = 0.999,
        opt_eps: float = 1e-8,
        opt_weight_decay: float = 0.01,
        sch_type: str | None = None,
        sch_warmup_min_lr: float = 0,
        sch_warmup_max_lr: float = 0,
        sch_warmup_num_steps: int = -1,
        sch_warmup_ratio_steps: float = -1,
        sch_total_num_steps: int = -1,
        sch_num_cycles: float = -1,
        save_dir: Path | str | None = None,
        run_dir: Path | str | None = None,
        early_stop: bool = False,
        patience: int = -1,
        continue_train_more_patience: bool = False,
        eval_every_half_epoch: bool = False,
        eval_step: int = 0,
        save_ckpts: bool = True,
        save_all_ckpts: bool = False,
        save_last_ckpt: bool = False,
        max_optimal_ckpt_num: int = 1,
        cache_dataset: bool = False,
        gradient_accumulation_steps: int = 1,
        gradient_clipping: float = 1.0,
        parallel_mode: str | None = None,
        ddp_timeout: int = 1800,
        fp16: bool = False,
        bf16: bool = False,
        dashboard: str | None = None,
        shuffle: bool | None = None,
        logging_steps: int = 0,
        torch_dtype: str = "auto",
        cut_input_from_output: bool = False,
        use_deepspeed_ckpt: bool = False,
        show_lr: bool = False,
        show_step: bool = False,
        record_cheat: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # initialize
        self.seed = seed
        self.gpu = gpu

        # attributes related to the data
        self.problem_type = problem_type
        # allowed_problem_types = ("regression", "single_label_classification", "multi_label_classification")
        # if self.problem_type is not None and self.problem_type not in allowed_problem_types:
        #     raise ValueError(
        #         f"The config parameter `problem_type` was not understood: received {self.problem_type} "
        #         "but only 'regression', 'single_label_classification' and 'multi_label_classification' are valid."
        #     )
        self.dataset_name = dataset_name
        self.train_file_path = Path(train_file_path) if train_file_path is not None else None
        self.val_file_path = Path(val_file_path) if val_file_path is not None else None
        self.test_file_path = Path(test_file_path) if test_file_path is not None else None

        # attributes related to the model
        self.model_type = model_type
        self.model_name = model_name
        self.model_dir = model_dir

        # attributes related to the metric
        self.metric = metric

        # attributes related to training steps
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.infer_batch_size = infer_batch_size if infer_batch_size != 0 else train_batch_size

        # optimizer
        self.opt_type = opt_type
        self.opt_lr = opt_lr
        self.opt_betas1 = opt_betas1
        self.opt_betas2 = opt_betas2
        self.opt_eps = opt_eps
        self.opt_weight_decay = opt_weight_decay

        # scheduler
        self.sch_type = sch_type
        self.sch_warmup_min_lr = sch_warmup_min_lr
        self.sch_warmup_max_lr = sch_warmup_max_lr if sch_warmup_max_lr != 0 else opt_lr
        self.sch_warmup_num_steps = sch_warmup_num_steps
        self.sch_warmup_ratio_steps = sch_warmup_ratio_steps
        self.sch_total_num_steps = sch_total_num_steps
        self.sch_num_cycles = sch_num_cycles

        # outputs dir
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.run_dir = Path(run_dir) if run_dir is not None else None

        # during training
        self.early_stop = early_stop
        if self.early_stop:
            self.patience = patience
            self.continue_train_more_patience = continue_train_more_patience
        self.eval_every_half_epoch = eval_every_half_epoch
        self.eval_step = eval_step
        self.save_ckpts = save_ckpts
        self.save_all_ckpts = save_all_ckpts
        self.save_last_ckpt = save_last_ckpt if not save_all_ckpts else True
        self.gradient_clipping = gradient_clipping
        self.max_optimal_ckpt_num = max_optimal_ckpt_num

        # Optimization about load and memory
        self.cache_dataset = cache_dataset
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.parallel_mode = parallel_mode
        self.fp16 = fp16
        self.bf16 = bf16

        # 自动计算的一些值
        self.total_num_steps: int = kwargs.get("total_num_steps", -1)
        self.steps_per_epoch: int = kwargs.get("steps_per_epoch", -1)
        self.training_runtime: dict = kwargs.get("training_runtime", None)

        # 杂项
        self.dashboard = dashboard
        self.shuffle = shuffle
        self.ddp_timeout = ddp_timeout
        self.logging_steps = logging_steps
        self.torch_dtype = torch_dtype
        self.cut_input_from_output = cut_input_from_output
        self.use_deepspeed_ckpt = use_deepspeed_ckpt
        if not self.use_deepspeed_ckpt and self.parallel_mode == "deepspeed":
            logger.warning(f"⚠️  You are using deepspeed, but not save deepspeed checkpoint. Only model will be saved so you can not resume training.")
        self.show_lr = show_lr
        self.show_step = show_step
        self.record_cheat = record_cheat
        # self.warning_default()

        self.check()

    def save(self, save_directory: Path | str, json_file_name=CONFIG_NAME, silence=True, **kwargs):
        if not silence:
            logger.debug(f"💾 Saving training configuration ...")
        super().save(save_directory, json_file_name, True, **kwargs)
        if not silence:
            logger.debug(f"✔️  Saving successfully.")

    @classmethod
    def load(cls, load_dir_or_path: Path | str, json_file_name=CONFIG_NAME, silence=True, **kwargs) -> Self:
        if not silence:
            logger.debug(f"💾 Loading training configuration ...")
        return super().load(load_dir_or_path, json_file_name, silence, **kwargs)

    def check(self):
        """
        Check whether the data files exist.\\
        Check whether values of `metric`, `parallel_mode`, `fp16`, `bf16`, `dashboard` is valid.
        """
        if self.train_file_path is not None:
            assert self.train_file_path.exists(), f"Training file: {self.train_file_path} dose not exists"
        if self.val_file_path is not None:
            assert self.val_file_path.exists(), f"Validation file: {self.val_file_path} dose not exists"
        if self.test_file_path is not None:
            assert self.test_file_path.exists(), f"Test file: {self.test_file_path} dose not exists"
        if self.metric not in MetricDict.support_metrics():
            raise ValueError(
                f"The config parameter `metric` was not understood: received `{self.metric}` "
                f"but only {[key for key in  MetricDict.support_metrics()]} are valid."
            )
        assert self.parallel_mode in [None, "DDP", "deepspeed"], (
            f"[parallel_mode] Only `DDP` and `deepspeed` are supported, but got `{self.parallel_mode}`.\n"
            "if you do not need parallel, plase set it to `None`."
        )
        assert not (self.fp16 and self.bf16), f"❌  `fp16` and `bf16` cannot be enabled at the same time!"
        assert self.dashboard in ["wandb", "tensorboard", None], (
            f"Only `wandb` and `tensorboard` dashboards are supported, but got `{self.dashboard}`.\n"
            "if you do not need dashboard, plase set it to `None`."
        )

    # 未使用
    def warning_default(self):
        default = self.__class__()
        attri_to_check = ("metric", "seed", "epochs", "batch_size", "learning_rate", "fp16")
        for attri in attri_to_check:
            if getattr(self, attri) != getattr(default, attri):
                logger.warning(f"`{attri}` is not specified, default value: `{getattr(default, attri)}`")

    # def set_deepspeed(self, deepspeed_config: OrderedDict):
    #     """
    #     fill `auto` field of deepspeed config with trainer config
    #     """
    #     if (fp16 := deepspeed_config.get("fp16", None)) is not None:
    #         if fp16["enabled"] == "auto":
    #             fp16["enabled"] = self.fp16
    #     if (bf16 := deepspeed_config.get("bf16", None)) is not None:
    #         if bf16["enabled"] == "auto":
    #             bf16["enabled"] = self.bf16

    #     if (opt := deepspeed_config.get("optimizer", None)) is not None:
    #         if opt["type"] == "auto":
    #             opt["type"] = self.opt_type
    #         for key, value in opt["params"].items():
    #             if value == "auto":
    #                 if key != "betas":
    #                     opt["params"][key] = getattr(self, f"opt_{key}")
    #                 else:
    #                     opt["params"][key] = [self.opt_betas1, self.opt_betas2]
    #     if (sch := deepspeed_config.get("scheduler", None)) is not None:
    #         if sch["type"] == "auto":
    #             sch["type"] = self.sch_type
    #         for key, value in sch["params"].items():
    #             if value == "auto":
    #                 sch["params"][key] = getattr(self, f"sch_{key}")

    #     keys = ["gradient_accumulation_steps", "gradient_clipping", "train_batch_size"]
    #     for key in keys:
    #         if (value := deepspeed_config.get(key, None)) is not None:
    #             deepspeed_config[key] = getattr(self, key) if value == "auto" else value
