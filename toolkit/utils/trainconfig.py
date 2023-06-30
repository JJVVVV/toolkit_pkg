from pathlib import Path

from ..configuration import Config

CONFIG_NAME = "train_config.json"


class TrainConfig(Config):
    def __init__(
        self,
        dataset,
        early_stop_metric,
        epochs,
        batch_size,
        learning_rate,
        model_type="",
        model_name="",
        problem_type=None,
        seed=0,
        early_stop=False,
        patience=5,
        continue_train_more_patience=False,
        warmup=False,
        test_in_epoch=False,
        weight_decay=0.01,
        adam_epsilon=1e-8,
        accumulate_step=1,
        warmup_ratio=-1,
        fp16=False,
        **kwargs,
    ):
        super().__init__(model_type=model_type, model_name=model_name, **kwargs)
        # attributes related to the task
        self.dataset = dataset
        self.early_stop_metric = early_stop_metric
        self.problem_type = problem_type
        allowed_problem_types = ("regression", "single_label_classification", "multi_label_classification")
        if self.problem_type is not None and self.problem_type not in allowed_problem_types:
            raise ValueError(
                f"The config parameter `problem_type` was not understood: received {self.problem_type} "
                "but only 'regression', 'single_label_classification' and 'multi_label_classification' are valid."
            )

        # attributes related to training
        self.seed = seed
        self.early_stop = early_stop
        if self.early_stop:
            self.patience = patience
            self.continue_train_more_patience = continue_train_more_patience
        self.warmup = warmup
        self.test_in_epoch = test_in_epoch

        # optimization hyperparameter
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.adam_epsilon = adam_epsilon
        self.accumulate_step = accumulate_step
        self.warmup_ratio = warmup_ratio
        self.fp16 = fp16

    def save_pretrained(self, save_directory: Path | str, **kwargs):
        kwargs["config_file_name"] = CONFIG_NAME
        return super().save_pretrained(save_directory, **kwargs)
