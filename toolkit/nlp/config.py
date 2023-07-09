from pathlib import Path

from ..config.trainconfig import TrainConfig, logger


class NLPTrainingConfig(TrainConfig):
    def __init__(
        self,
        dataset_name: str,
        train_file_path: Path | str,
        val_file_path: Path | str,
        metric: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        batch_size_infer: int = None,
        test_file_path: Path | str | None = None,
        model_type: str = "",
        model_name: str = "",
        problem_type: str | None = None,
        seed: int = 0,
        early_stop: bool = False,
        patience: int = 5,
        continue_train_more_patience: bool = False,
        warmup: bool = False,
        test_in_epoch: bool = False,
        weight_decay: float = 0.01,
        adam_epsilon: float = 1e-8,
        accumulate_step: int = 1,
        warmup_ratio: float = -1,
        fp16: bool = False,
        max_length_input: int | None = None,
        max_length_label: int | None = None,
        pretrained_model_path: Path | str = "",
        padding_side: str = "right",
        **kwargs,
    ):
        super().__init__(
            dataset_name,
            train_file_path,
            val_file_path,
            metric,
            epochs,
            batch_size,
            learning_rate,
            batch_size_infer,
            test_file_path,
            model_type,
            model_name,
            problem_type,
            seed,
            early_stop,
            patience,
            continue_train_more_patience,
            warmup,
            test_in_epoch,
            weight_decay,
            adam_epsilon,
            accumulate_step,
            warmup_ratio,
            fp16,
            **kwargs,
        )
        self.padding_side = padding_side
        self.max_length_input = max_length_input
        self.max_length_label = max_length_label
        self.pretrained_model_path = pretrained_model_path
        if kwargs:
            logger.info("Custom training parameters:")
            for key, value in kwargs.items():
                logger.info(f"{key}: {value}")

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
