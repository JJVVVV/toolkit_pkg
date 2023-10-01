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
    `task_type`: "generate", "classify", "regress"\n
    """

    def __init__(self, task_type, config: TrainConfig, model, tokenizer, dataset: Dataset, calculate_metric_callback, extral_args_evaluation) -> None:
        self.task_type = task_type
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.calculate_metric_callback = calculate_metric_callback
        self.extral_args_evaluation = extral_args_evaluation

    def eval(self, split: Split = Split.VALIDATION, cuda_id=None) -> MetricDict | None:
        """
        if specify the `cuda_id`, the model will run in it, ohterwise, default
        """
        # todo prior deepspeed infer
        if self.config.parallel_mode == "deepspeed":
            raise NotImplementedError()
        elif self.config.parallel_mode == "DDP":
            local_rank = dist.get_rank() if dist.is_initialized() else 0
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            self.dataloader = (
                get_dataloader(self.dataset, self.config, split, collate_fn=self.dataset.collate_fn)
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
                    for batch in tqdm(self.dataloader, desc=split.name, colour="BLUE", unit="batch", smoothing=0.9):
                        with torch.no_grad():
                            labels = batch.pop("labels")
                            custom_inputs = batch.pop("custom_inputs", dict())
                            if self.config.gpu:
                                batch = {key: value.cuda() for key, value in batch.items()}
                            outputs = self.model.generate(**batch, **custom_inputs, **self.extral_args_evaluation, **self.config.generate_kwargs)
                            texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                            all_losses.append(-1)
                            all_labels.extend(labels)
                            all_logits.extend(texts)
                case "classify" | "regress":
                    for batch in tqdm(self.dataloader, desc=split.name, colour="BLUE", unit="batch", smoothing=0.9):
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
                logger.debug(f"local rank {local_rank}: num_labels: {len(all_labels)}, num_logits: {len(all_logits)}, num_batches: {len(all_losses)}")
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
