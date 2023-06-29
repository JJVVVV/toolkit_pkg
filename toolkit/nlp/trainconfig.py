from pathlib import Path

from toolkit.configuration import Config

from ..configuration import Config

CONFIG_NAME = "train_config.json"


class TrainConfig(Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # attributes related to the task
        self.dataset = kwargs.pop("dataset")
        self.early_stop_metric = kwargs.pop("early_stop_metric")
        self.problem_type = kwargs.pop("problem_type", None)
        allowed_problem_types = ("regression", "single_label_classification", "multi_label_classification")
        if self.problem_type is not None and self.problem_type not in allowed_problem_types:
            raise ValueError(
                f"The config parameter `problem_type` was not understood: received {self.problem_type} "
                "but only 'regression', 'single_label_classification' and 'multi_label_classification' are valid."
            )

        # attributes related to the model
        self.model_type = kwargs.pop("model_type")
        self.model_name = kwargs.pop("model_name")

        # attributes related to training
        self.seed = kwargs.pop("seed", 0)
        self.early_stop = kwargs.pop("early_stop", False)
        if self.early_stop:
            self.patience = kwargs.pop("patience", 5)
            self.continue_train_more_patience = kwargs.pop("continue_train_more_patience", False)
        self.warmup = kwargs.pop("warmup", False)
        self.test_in_epoch = kwargs.pop("test_in_epoch", False)

        # optimization hyperparameter
        self.epochs = kwargs.pop("epochs")
        self.batch_size = kwargs.pop("batch_size")
        self.max_length_input = kwargs.pop("max_length_input", None)
        self.max_length_label = kwargs.pop("max_length_label", None)
        self.learning_rate = kwargs.pop("learning_rate")
        self.weight_decay = kwargs.pop("weight_decay", 1e-2)
        self.adam_epsilon = kwargs.pop("adam_epsilon", 1e-8)
        self.accumulate_step = kwargs.pop("accumulate_step", 1)
        self.warmup_ratio = kwargs.pop("warmup_ratio", (0.1 if self.warmup else -1))
        self.fp16 = kwargs.pop("fp16", False)

    def save_pretrained(self, save_directory: Path | str, **kwargs):
        kwargs["config_file_name"] = CONFIG_NAME
        return super().save_pretrained(save_directory, **kwargs)
