from typing import Callable, Literal, Self

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from pathlib import Path

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

    save_result = False

    def __new__(
        cls,
        task_type: Literal["generate", "classify", "regress"],
        split: Split | Literal["TRAINING", "VALIDATION", "TEST", "UNK"],
        config: TrainConfig,
        model,
        dataset: Dataset,
        extral_args_evaluation: dict | None = None,
        tokenizer=None,
    ) -> Self:
        "if any of the following objects is `None`, then just return `None`"
        if not isinstance(split, Split):
            split = Split[split]
        if model is None or dataset is None:
            return None
        return super().__new__(cls)

    def __init__(
        self,
        task_type: Literal["generate", "classify", "regress"],
        split: Split | Literal["TRAINING", "VALIDATION", "TEST", "UNK"],
        config: TrainConfig,
        model,
        dataset: Dataset,
        extral_args_evaluation: dict | None = None,
        tokenizer=None,
    ) -> None:
        if not isinstance(split, Split):
            split = Split[split]
        self.task_type = task_type
        self.split = split
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.extral_args_evaluation = extral_args_evaluation if extral_args_evaluation is not None else dict()

    def eval(self, cuda_id=None, print_head=False) -> MetricDict:
        """
        if specify the `cuda_id`, the model will run in it, ohterwise, default
        """
        local_rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.dataloader = (
            get_dataloader(self.dataset, self.config, self.split, shuffle=False, collate_fn=self.dataset.collate_fn)
            if not hasattr(self, "dataloader")
            else self.dataloader
        )
        # 当在训练集上进行推理时, 下面代码用来得到正确的dataloader, 而不是(dataloader, sampler)这个数组.
        if isinstance(self.dataloader, tuple):
            self.dataloader = self.dataloader[0]

        all_losses = []
        all_labels = []
        all_logits = []

        if cuda_id is not None:
            self.config.gpu = True
            torch.cuda.set_device(cuda_id)
        if self.model.device.type != "cuda":
            self.model.cuda()
        self.model.eval()

        # logger.debug(f"local rank {local_rank}: Start evaluating {self.split.name} dataset with {len(self.dataloader)} batches")
        match self.task_type:
            case "generate":
                for batch in tqdm(self.dataloader, desc=self.split.name.capitalize(), colour="BLUE", unit="batch", smoothing=0.9):
                    with torch.no_grad():
                        labels = batch.pop("labels")
                        custom_inputs = batch.pop("custom_inputs", dict())
                        if self.config.gpu:
                            batch = {key: value.cuda() for key, value in batch.items()}
                        if (
                            "generation_config" in self.extral_args_evaluation
                        ):  # 支持使用hugging face的GenerationConfig, 防止config.generation中的参数覆盖hf GenerationConfig中的参数
                            outputs = self.model.generate(**batch, **custom_inputs, **self.extral_args_evaluation)
                        else:
                            outputs = self.model.generate(
                                **batch,
                                **custom_inputs,
                                **self.extral_args_evaluation,
                                **self.config.generate_kwargs,
                                pad_token_id=self.tokenizer.pad_token_id,
                            )
                        if self.config.cut_input_from_output:
                            texts = self.tokenizer.batch_decode(outputs[..., batch["input_ids"].shape[-1] :], skip_special_tokens=True)
                            # texts = []
                            # for idx, output in enumerate(outputs):
                            #     texts.append(self.tokenizer.decode(output[batch["input_ids"][idx].size(0) :], skip_special_tokens=True))
                        else:
                            texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                        all_losses.append(-1)
                        if isinstance(labels, torch.Tensor):
                            labels = labels.numpy(force=True).tolist()
                        all_labels.extend(labels)
                        all_logits.extend(texts)
                        if print_head:
                            for i, o in zip(self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True), texts):
                                logger.debug(f"\n### Input ###\n{i}\n### Output ###\n{o}\n{'-'*60}")
                            print_head = False
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
                        all_logits.extend(logits.to(torch.float32).numpy(force=True).tolist())
        self.model.train()

        if world_size > 1:
            # logger.debug(
            #     f"local rank {local_rank}: num_labels: {len(all_labels)}, num_logits: {len(all_logits)}, num_batches: {len(self.dataloader)}"
            # )
            # logger.debug("Gathering all results from all processes...")
            mean_loss = torch.tensor(all_losses, dtype=torch.float32).mean().cuda()
            # logger.debug(f"local rank {local_rank}: mean_loss={mean_loss}, device={mean_loss.device}")

            labels_gather_list = [None for _ in range(world_size)]
            logits_gather_list = [None for _ in range(world_size)]
            loss_gather_list = [torch.zeros(1, dtype=torch.float32).cuda() for _ in range(world_size)]
            # logger.debug(f"local rank {local_rank}: loss_gather_list={loss_gather_list}, device={[l.device for l in loss_gather_list]}")

            logger.debug("Gathering labels ...")
            dist.all_gather_object(labels_gather_list, all_labels)
            logger.debug("Gathering logits ...")
            dist.all_gather_object(logits_gather_list, all_logits)
            logger.debug("Gathering loss ...")
            dist.all_gather(loss_gather_list, mean_loss)
            mean_loss = sum(loss_gather_list)
            logger.debug("Finish gather all data from all processes.")

            # dist.gather_object(all_labels, labels_gather_list if local_rank == 0 else None, dst=0)
            # dist.gather_object(all_logits, logits_gather_list if local_rank == 0 else None, dst=0)
            # dist.reduce(mean_loss, dst=0, op=dist.ReduceOp.SUM, async_op=False)

            # if local_rank != 0:  # final result will be calculated on `local rank 0` process
            #     return None
            # * 因为 dataloader 中的 sampler 的采用方式, labels_gather_list 中每一个列表的第 i 个元素的组合， 才对应第 i 个 batch 中的内容， 为了保证 all_labels
            all_labels = sum(zip(*labels_gather_list), ())
            all_logits = sum(zip(*logits_gather_list), ())
            mean_loss = (mean_loss / world_size).item()
        else:
            mean_loss = sum(all_losses) / len(all_losses)

        return self.calculate_metric_callback(all_labels, all_logits, mean_loss)

    def calculate_metric_callback(self, all_labels: list | tuple, all_logits: list | tuple, mean_loss: float) -> MetricDict:
        raise NotImplementedError("Please implement the function of calculate metrics.")

    def save_eval_result(self, all_labels: list | tuple, all_logits: list | tuple, output_path: str | None = None):
        """
        Save the evaluation result to the output path. If output_path is None, the result will be saved to the default path.\\
        Adapt to multi-gpu environment.
        """
        local_rank = dist.get_rank() if dist.is_initialized() else 0
        if local_rank != 0:
            return None
        if output_path is None:
            output_path = (
                self.config.save_dir
                / "evaluators"
                / self.__class__.__name__.lower()
                / self.split.name
                / f"epoch={self.config.training_runtime['cur_epoch']:03d}_step={self.config.training_runtime['cur_step']}.json"
            )
        output_path = Path(output_path)
        import pandas as pd

        df = pd.DataFrame.from_dict(
            dict(inputs=[a_sample.tolist() for a_sample in self.dataset.texts_input[: len(all_labels)]], preds=all_logits, labels=all_labels)
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(output_path, force_ascii=False, indent=2, orient="records")

    def calculate_metric_callback_rougel(
        self, all_labels: list | tuple, all_logits: list | tuple, mean_loss: float, language: Literal["en", "ch"] = "en"
    ) -> MetricDict:
        self.save_eval_result(all_labels, all_logits)
        from ..metric.similarity_metrics import rouge

        metric = (rouge(all_logits, all_labels, language, ("rougeL")) * 100).round(2)
        return metric
