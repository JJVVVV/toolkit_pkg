from pathlib import Path

from ..utils.trainconfig import TrainConfig


class NLPTrainingConfig(TrainConfig):
    def __init__(
        self,
        train_file_path: Path | str,
        val_file_path: Path | str,
        dataset_name: str,
        early_stop_metric: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
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
        test_file_path: Path | str | None = None,
        padding_side: str = "right",
        **kwargs,
    ):
        super().__init__(
            dataset_name,
            early_stop_metric,
            epochs,
            batch_size,
            learning_rate,
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
        self.train_file_path = train_file_path
        self.val_file_path = val_file_path
        self.test_file_path = test_file_path
