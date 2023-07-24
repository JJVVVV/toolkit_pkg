from pathlib import Path

from ..logger import _getLogger
from ..metric.metricdict import MetricDict
from .config_base import ConfigBase

logger = _getLogger("toolkit.TrainConfig")

CONFIG_NAME = "train_config.json"


class TrainConfig(ConfigBase):
    def __init__(
        self,
        train_file_path: Path | str,
        dataset_name: str = "",
        metric: str = "Loss",
        epochs: int = 3,
        batch_size: int = 16,
        optimizer: str | None = None,
        lr_scheduler: str | None = None,
        learning_rate: float = 1e-3,
        checkpoints_dir: Path | str | None = None,
        batch_size_infer: int = None,
        val_file_path: Path | str | None = None,
        test_file_path: Path | str | None = None,
        model_type: str = "",
        model_name: str = "",
        problem_type: str | None = None,
        seed: int = 0,
        early_stop: bool = False,
        patience: int = 5,
        continue_train_more_patience: bool = False,
        test_in_epoch: bool = False,
        weight_decay: float = 0.01,
        epsilon: float = 1e-8,
        accumulate_step: int = 1,
        warmup: bool = False,
        warmup_ratio: float = -1,
        fp16: bool = False,
        dashboard: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # attributes related to the task
        self.dataset_name = dataset_name
        self.train_file_path = Path(train_file_path)
        self.val_file_path = Path(val_file_path) if val_file_path is not None else None
        self.test_file_path = Path(test_file_path) if test_file_path is not None else None
        self.metric = metric
        if self.metric not in MetricDict.support_metrics():
            raise ValueError(
                f"The config parameter `metric` was not understood: received `{self.metric}` "
                f"but only {[key for key in  MetricDict.support_metrics()]} are valid."
            )
        self.problem_type = problem_type
        allowed_problem_types = ("regression", "single_label_classification", "multi_label_classification")
        if self.problem_type is not None and self.problem_type not in allowed_problem_types:
            raise ValueError(
                f"The config parameter `problem_type` was not understood: received {self.problem_type} "
                "but only 'regression', 'single_label_classification' and 'multi_label_classification' are valid."
            )

        # attributes related to the model
        self.model_type = model_type
        self.model_name = model_name

        # attributes related to training
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.checkpoints_dir = Path(checkpoints_dir) if checkpoints_dir is not None else None
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
        self.epsilon = epsilon
        self.accumulate_step = accumulate_step
        self.warmup_ratio = warmup_ratio
        self.fp16 = fp16

        # attributes related to validation and test
        self.batch_size_infer = batch_size_infer if batch_size_infer is not None else batch_size

        assert dashboard in ["wandb", "tensorboard", None], (
            f"Only `wandb` and `tensorboard` dashboards are supported, but got `{dashboard}`.\n"
            "if you do not need dashboard, plase set it to `None`."
        )
        self.dashboard = dashboard
        self.check_data_file()

    def save(self, save_directory: Path | str, silence=True, config_file_name=CONFIG_NAME):
        super().save(save_directory, silence=True, config_file_name=config_file_name)
        if not silence:
            logger.debug(f"Save training configuration successfully.")

    def check_data_file(self):
        """
        Check whether the data files exist.
        """
        assert self.train_file_path.exists(), f"Training file: {self.train_file_path} dose not exists"
        assert self.val_file_path.exists(), f"Development file: {self.val_file_path} dose not exists"
        if self.test_file_path is not None:
            assert self.test_file_path.exists(), f"Test file: {self.test_file_path} dose not exists"
