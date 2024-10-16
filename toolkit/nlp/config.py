import json
from pathlib import Path

from ..config.trainconfig import TrainConfig

# VALID_GEN_KWAG=set("max_length", )


class NLPTrainingConfig(TrainConfig):
    allowed_task_type = ("generate", "classify", "regress")

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
        save_last_ckpt: bool = True,
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
        logging_steps: int = 1,
        torch_dtype: str = "auto",
        cut_input_from_output: bool = False,
        use_deepspeed_ckpt: bool = False,
        show_lr: bool = False,
        show_step: bool = False,
        record_cheat: bool = True,
        max_length: int | None = None,
        max_length_input: int | None = None,
        max_length_label: int | None = None,
        padding_to_max_length: bool = False,
        gen_max_length: int | None = None,
        max_new_tokens: int | None = None,
        do_sample: bool = False,
        num_beams: int = 1,
        early_stopping: bool = False,
        use_cache: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 1.0,
        diversity_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        generation_config_file: Path | str | None = None,
        # hf_gen_config_file: Path | str | None = None,
        padding_side: str = "right",
        model_structure: str | None = None,
        task_type: str | None = None,
        activation_checkpointing: bool = False,
        **kwargs,
    ):
        """
        logging_steps: per logging_steps logging to consoles and tensorboard or wandb once.
        max_length: int | None = None, 限制整体最大长度, 该参数对于encoder结构的generate任务尤为有用.
        max_length_input: int | None = None, 限制输入的最大长度, 用于truncate.
        max_length_label: int | None = None, 限制标签的最大长度, 用于truncate.
        padding_to_max_length: bool = False, 是否padding到整个数据集的最大长度(*此处数据集指的是truncated后的), 即actual_max_length_input. 设置为True时可以用于测试是否能跑完整个数据集而不会OOV.
        generation_config_file 中的参数会覆盖传进来的参数
        """
        super().__init__(
            seed,
            gpu,
            problem_type,
            dataset_name,
            train_file_path,
            val_file_path,
            test_file_path,
            model_type,
            model_name,
            model_dir,
            metric,
            epochs,
            train_batch_size,
            infer_batch_size,
            opt_type,
            opt_lr,
            opt_betas1,
            opt_betas2,
            opt_eps,
            opt_weight_decay,
            sch_type,
            sch_warmup_min_lr,
            sch_warmup_max_lr,
            sch_warmup_num_steps,
            sch_warmup_ratio_steps,
            sch_total_num_steps,
            sch_num_cycles,
            save_dir,
            run_dir,
            early_stop,
            patience,
            continue_train_more_patience,
            eval_every_half_epoch,
            eval_step,
            save_ckpts,
            save_all_ckpts,
            save_last_ckpt,
            max_optimal_ckpt_num,
            cache_dataset,
            gradient_accumulation_steps,
            gradient_clipping,
            parallel_mode,
            ddp_timeout,
            fp16,
            bf16,
            dashboard,
            shuffle,
            logging_steps,
            torch_dtype,
            cut_input_from_output,
            use_deepspeed_ckpt,
            show_lr,
            show_step,
            record_cheat,
            **kwargs,
        )
        self.padding_side = padding_side
        self.max_length = max_length
        self.max_length_input = max_length_input
        self.max_length_label = max_length_label
        self.padding_to_max_length = padding_to_max_length
        self.model_structure = model_structure
        self.task_type = task_type
        self.activation_checkpointing = activation_checkpointing

        # generate config
        self.do_sample = do_sample
        self.num_beams = num_beams
        self.early_stopping = early_stopping
        self.use_cache = use_cache
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.diversity_penalty = diversity_penalty
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.max_new_tokens = max_new_tokens
        self.gen_max_length = gen_max_length
        self.generation_config_file = generation_config_file
        if generation_config_file is not None:
            self.generation_config_file = Path(generation_config_file)
            with self.generation_config_file.open() as f:
                self.generate_kwargs = json.load(f)
        # 如果没有设定gen_max_length,则使用训练时的max_lengthd
        self.gen_max_length = self.max_length if self.gen_max_length is None else self.gen_max_length
        # self.pretrained_model_path = pretrained_model_path

        def check():
            assert self.model_structure in ("encoder-decoder", "encoder", "decoder"), f"`model_structure` invalid value: {self.model_structure}"
            if self.task_type not in self.allowed_task_type:
                raise ValueError(
                    f"The parameter `task_type` was not understood: received `{self.task_type}` " f"but only {self.allowed_task_type} are valid."
                )

        if self.is_check:
            check()

    @property
    def generate_kwargs(self):
        ret = {
            "do_sample": self.do_sample,
            "num_beams": self.num_beams,
            "early_stopping": self.early_stopping,
            "use_cache": self.use_cache,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "diversity_penalty": self.diversity_penalty,
            "repetition_penalty": self.repetition_penalty,
            "length_penalty": self.length_penalty,
            "max_new_tokens": self.max_new_tokens,
            "max_length": self.gen_max_length,
        }

        # # 如果设置了`max_new_tokens`就不设置`max_length`, 防止 transformer 的 warning
        # if self.max_new_tokens is not None:
        #     ret["max_new_tokens"] = self.max_new_tokens
        # else:
        #     ret["max_length"] = self.max_length if self.max_length is not None else 20
        return ret

    # todo 当前只会覆盖 d 中出现的参数, 正常应该还要把 d 中没出现的参数置为默认值
    @generate_kwargs.setter
    def generate_kwargs(self, d: dict):
        # if set(d.keys()).issubset(set(self.generate_kwargs.keys())):
        if not isinstance(d, dict):
            raise ValueError(f"`generate_kwargs` must be a dict but got '{type(d)}'")
        # 区分训练的max_length和生成时的max_length
        if "max_length" in d:
            d["gen_max_length"] = d.pop("max_length")
        for key, value in d.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError(f"Invalid key for generate keyword arguments: `{key}`")

    # def print_some_info(self):
    #     logger.debug("***** Some training information *****")
    #     logger.debug(f"  Batch size = {self.batch_size}")
    #     logger.debug(f"  Total epochs = {self.epochs:d}")
    #     logger.debug(f"  Steps per epoch = {stepsPerEpoch:d}")
    #     logger.debug(f"  Total steps = {totalSteps:d}")
    #     if self.warmup:
    #         logger.debug(f"  Warmup steps = {warmupSteps:d}")
    #     logger.debug(f"  Model type = {self.model_type}")
    #     logger.debug(f"  fp16: {self.fp16}\n")
