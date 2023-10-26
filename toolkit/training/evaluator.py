from typing import Callable, Self

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from ..config import TrainConfig
from ..enums import Split
from ..logger import _getLogger
from ..metric import MetricDict
from .dataloader import get_dataloader

logger = _getLogger("Evaluater")


class Evaluator:
    """
    If any of `model`, `dataset`, or `calculate_metric_callback` is `None`, then return `None`\n
    `task_type`: "generate", "classify", "regress"\n
    """

    def __new__(
        cls,
        task_type: str,
        split: Split,
        config: TrainConfig,
        model,
        dataset: Dataset,
        calculate_metric_callback: Callable[..., MetricDict],
        extral_args_evaluation: dict | None = None,
        tokenizer=None,
    ) -> Self:
        "if any of the following objects is `None`, then just return `None`"
        if model is None or dataset is None or calculate_metric_callback is None:
            return None
        return super().__new__(cls)

    def __init__(
        self,
        task_type: str,
        split: Split,
        config: TrainConfig,
        model,
        dataset: Dataset,
        calculate_metric_callback: Callable[..., MetricDict],
        extral_args_evaluation: dict | None = None,
        tokenizer=None,
    ) -> None:
        self.task_type = task_type
        self.split = split
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.calculate_metric_callback = calculate_metric_callback
        self.extral_args_evaluation = extral_args_evaluation if extral_args_evaluation is not None else dict()

    def eval(self, cuda_id=None) -> MetricDict:
        """
        if specify the `cuda_id`, the model will run in it, ohterwise, default
        """
        local_rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.dataloader = (
            get_dataloader(self.dataset, self.config, self.split, collate_fn=self.dataset.collate_fn)
            if not hasattr(self, "dataloader")
            else self.dataloader
        )

        all_losses = []
        all_labels = []
        all_logits = []

        if cuda_id is not None:
            self.config.gpu = True
            torch.cuda.set_device(cuda_id)
            self.model.cuda()
        self.model.eval()

        match self.task_type:
            case "generate":
                for batch in tqdm(self.dataloader, desc=self.split.name, colour="BLUE", unit="batch", smoothing=0.9):
                    with torch.no_grad():
                        labels = batch.pop("labels")
                        custom_inputs = batch.pop("custom_inputs", dict())
                        if self.config.gpu:
                            batch = {key: value.cuda() for key, value in batch.items()}
                        outputs = self.model.generate(**batch, **custom_inputs, **self.extral_args_evaluation, **self.config.generate_kwargs)
                        if self.config.cut_input_from_output:
                            texts = []
                            for idx, output in enumerate(outputs):
                                texts.append(self.tokenizer.decode(output[batch["input_ids"][idx].size(0) :], skip_special_tokens=True))
                        else:
                            texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                        all_losses.append(-1)
                        all_labels.extend(labels)
                        all_logits.extend(texts)
            case "classify" | "regress":
                for batch in tqdm(self.dataloader, desc=self.split.name, colour="BLUE", unit="batch", smoothing=0.9):
                    with torch.no_grad():
                        custom_inputs = batch.pop("custom_inputs", dict())
                        if self.config.gpu:
                            batch = {key: value.cuda() for key, value in batch.items()}
                        labels = batch["labels"]
                        outputs = self.model(**batch, **custom_inputs, **self.extral_args_evaluation)
                        loss, logits = outputs["loss"], outputs["logits"]
                        all_losses.append(loss.item())
                        all_labels.extend(labels.numpy(force=True).tolist())
                        all_logits.extend(logits.numpy(force=True).tolist())
        self.model.train()

        if world_size > 1:
            # logger.debug(
            #     f"local rank {local_rank}: num_labels: {len(all_labels)}, num_logits: {len(all_logits)}, num_batches: {len(self.dataloader)}"
            # )
            mean_loss = torch.tensor(all_losses, dtype=torch.float32).mean().cuda()

            labels_gather_list = [None for _ in range(world_size)]
            logits_gather_list = [None for _ in range(world_size)]
            loss_gather_list = [torch.zeros(1, dtype=torch.float32).cuda() for _ in range(world_size)]

            dist.all_gather_object(labels_gather_list, all_labels)
            dist.all_gather_object(logits_gather_list, all_logits)
            dist.all_gather(loss_gather_list, mean_loss)
            mean_loss = sum(loss_gather_list)

            # dist.gather_object(all_labels, labels_gather_list if local_rank == 0 else None, dst=0)
            # dist.gather_object(all_logits, logits_gather_list if local_rank == 0 else None, dst=0)
            # dist.reduce(mean_loss, dst=0, op=dist.ReduceOp.SUM, async_op=False)

            # if local_rank != 0:  # final result will be calculated on `local rank 0` process
            #     return None

            all_labels = sum(labels_gather_list, [])
            all_logits = sum(logits_gather_list, [])
            mean_loss = (mean_loss / world_size).item()
        else:
            mean_loss = sum(all_losses) / len(all_losses)

        return self.calculate_metric_callback(all_labels, all_logits, mean_loss)
