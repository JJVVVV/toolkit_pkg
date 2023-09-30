from pathlib import Path

from ..config.trainconfig import TrainConfig


class NLPTrainingConfig(TrainConfig):
    def __init__(
        self,
        seed: int = 0,
        gpu: bool = True,
        problem_type: str | None = None,
        dataset_name: str = "",
        train_file_path: Path | str | None = None,
        val_file_path: Path | str | None = None,
        test_file_path: Path | str | None = None,
        model_type: str = "",
        model_name: str = "",
        model_dir: str | None = None,
        metric: str = "Loss",
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
        sch_warmup_num_steps: int = 0,
        sch_warmup_ratio_steps: float = 0,
        sch_total_num_steps: int = 0,
        save_dir: Path | str | None = None,
        run_dir: Path | str | None = None,
        early_stop: bool = False,
        patience: int = -1,
        continue_train_more_patience: bool = False,
        eval_every_half_epoch: bool = False,
        eval_step: int = 0,
        save_all_ckpts: bool = False,
        cache_dataset: bool = False,
        gradient_accumulation_steps: int = 1,
        gradient_clipping: float = 1.0,
        parallel_mode: str | None = None,
        fp16: bool = False,
        dashboard: str | None = None,
        shuffle: bool | None = None,
        max_length_input: int | None = None,
        max_length_label: int | None = None,
        max_length: int = 20,
        max_new_tokens: int | None = None,
        do_sample: bool = False,
        num_beams: int = 1,
        early_stopping: bool = False,
        use_cache: bool = True,
        top_k: int = 50,
        diversity_penalty: float = 0.0,
        length_penalty: float = 1.0,
        # pretrained_model_path: Path | str | None = None,
        padding_side: str = "right",
        **kwargs,
    ):
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
            save_dir,
            run_dir,
            early_stop,
            patience,
            continue_train_more_patience,
            eval_every_half_epoch,
            eval_step,
            save_all_ckpts,
            cache_dataset,
            gradient_accumulation_steps,
            gradient_clipping,
            parallel_mode,
            fp16,
            dashboard,
            shuffle,
            **kwargs,
        )
        self.padding_side = padding_side
        self.max_length_input = max_length_input
        self.max_length_label = max_length_label

        self.generate_kwargs = {
            "max_length": max_length,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_beams": num_beams,
            "early_stopping": early_stopping,
            "use_cache": use_cache,
            "top_k": top_k,
            "diversity_penalty": diversity_penalty,
            "length_penalty": length_penalty,
        }
        # self.pretrained_model_path = pretrained_model_path

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
