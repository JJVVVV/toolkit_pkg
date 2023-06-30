from ..utils.trainconfig import TrainConfig


class NLPTrainConfig(TrainConfig):
    def __init__(
        self,
        dataset,
        early_stop_metric,
        model_type,
        model_name,
        epochs,
        batch_size,
        learning_rate,
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
        max_length_input=None,
        max_length_label=None,
        **kwargs,
    ):
        super().__init__(
            dataset,
            early_stop_metric,
            model_type,
            model_name,
            epochs,
            batch_size,
            learning_rate,
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
        self.max_length_input = max_length_input
        self.max_length_label = max_length_label
