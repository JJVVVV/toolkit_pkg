from math import ceil
import pathlib
from typing import Callable, Type, TypeVar

import torch
import torch.distributed as dist
import wandb
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.optim import AdamW, RMSprop
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, get_linear_schedule_with_warmup

from .. import toolkit_logger
from ..config import TrainConfig
from ..enums import Split
from ..logger import _getLogger
from ..metric import MetricDict
from ..nlp.config import NLPTrainingConfig
from .checkpoint_manager import CheckpointManager
from .components import Optimizer, Scaler, Scheduler, set_weight_decay
from .dataloader import get_dataloader, gradient_accumulate
from .watchdog import WatchDog

logger = _getLogger("Trainer")


map_str2optm = {"AdamW": AdamW, "RMSprop": RMSprop}
map_str2sche = {"LinearWarmup": get_linear_schedule_with_warmup}

OptimizerClass = TypeVar("OptimizerClass", bound=torch.optim.Optimizer)
SchedulerClass = TypeVar("SchedulerClass", bound=torch.optim.lr_scheduler.LRScheduler)

allowed_task_type = ("generate", "classify", "regress")


class Trainer:
    def __init__(
        self,
        task_type: str,
        evaluate_only: bool,
        config: TrainConfig | NLPTrainingConfig,
        model: torch.nn.Module,
        dataset_train: Dataset | None = None,
        dataset_val: Dataset | None = None,
        dataset_test: Dataset | None = None,
        calculate_metric_callback: Callable | None = None,
        optimizer: Type[OptimizerClass] | str | torch.optim.Optimizer | None = None,
        scheduler: Callable[..., torch.optim.lr_scheduler.LRScheduler] | str | torch.optim.lr_scheduler.LRScheduler | None = None,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        dashboard_writer: SummaryWriter | wandb.run.__class__ | None = None,
        project_name: str = "untitled",
    ) -> None:
        """
        `task_type`: "generate", "classify", "regress"\n
        `optimizer`: "AdamW", "RMSprop"\n
        `scheduler`: "LinearWarmup"\n
        """
        local_rank = dist.get_rank() if dist.is_initialized() else 0
        # world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.config = config
        # TODO: model to cuda
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test
        self.calculate_metric_callback = calculate_metric_callback
        if task_type not in allowed_task_type:
            raise ValueError(f"The parameter `task_type` was not understood: received `{task_type}` " f"but only {allowed_task_type} are valid.")
        self.task_type = task_type
        if evaluate_only:
            return

        if isinstance(optimizer, str):
            assert (
                optimizer in map_str2optm
            ), f"Only following optimizer can be mapped to the corresponding optimizer: {list(map_str2optm.keys())}, bug got {optimizer}"
            self.optimizer = map_str2optm[optimizer]
        else:
            self.optimizer = optimizer
        if isinstance(scheduler, str):
            assert (
                scheduler in map_str2sche
            ), f"Only following scheduler can be mapped to the corresponding scheduler: {list(map_str2sche.keys())}, bug got {scheduler}"
            self.scheduler = map_str2sche[scheduler]
        else:
            self.scheduler = scheduler
        self.scaler = GradScaler() if config.fp16 else None
        self.ckpt_manager = CheckpointManager(config.save_dir)
        # self.dashboard_writer = dashboard_writer
        if config.dashboard is not None:
            if dashboard_writer is not None:
                self.dashboard_writer = dashboard_writer
                if config.dashboard == "wandb":
                    assert self.dashboard_writer is wandb.run
            elif local_rank == 0:  # 未传入 dashboard 的 writer, 自动定义
                if config.dashboard == "tensorboard":
                    dataset_name = config.dataset_name if config.dataset_name else "unk_dataset"
                    model_type = (
                        (config.model_type[1:] if config.model_type.startswith("/") else config.model_type) if config.model_type else "unk_model_type"
                    )
                    model_name = config.model_name if config.model_name else "unk_model_name"
                    if config.model_dir is not None:
                        model_dir = config.model_dir[1:] if config.model_dir.startswith("/") else config.model_dir
                    else:
                        model_dir = ""
                    run_dir = pathlib.Path("tensorboard", dataset_name, model_type, model_dir, model_name)
                    run_dir.mkdir(parents=True, exist_ok=True)
                    self.dashboard_writer = SummaryWriter(comment="training", log_dir=run_dir)
                elif config.dashboard == "wandb":
                    self.dashboard_writer = wandb.init(
                        project=project_name,
                        config=config.to_dict(),
                        tags=[config.dataset_name, config.model_type, config.model_name],
                        # mode="disabled",
                    )
                    assert self.dashboard_writer is wandb.run
        else:
            self.dashboard_writer = None

    def __del__(self):
        if hasattr(self, "dashboard_writer") and self.dashboard_writer is not None:
            if self.config.dashboard == "wandb":
                self.dashboard_writer.finish()
            elif self.config.dashboard == "tensorboard":
                self.dashboard_writer.close()

    # TODO 自定义额外的模型输入, 如(is_train)
    def train(self) -> None:
        local_rank = dist.get_rank() if dist.is_initialized() else 0
        # world_size = dist.get_world_size() if dist.is_initialized() else 1

        # * Load training data, development data and test data
        # TODO: 通用性: collate_fn 并不一定需要
        dataloader_train, sampler = get_dataloader(self.dataset_train, self.config, Split.TRAINING, collate_fn=self.dataset_train.collate_fn)

        # * Define training parameters
        stepsPerEpoch = ceil(len(dataloader_train) / self.config.accumulate_step)
        totalSteps = stepsPerEpoch * self.config.epochs

        # * Initialize optimizer, scheduler, scaler
        if isinstance(self.optimizer, torch.optim.Optimizer):  # optimizer
            optimizer = self.optimizer
        else:  # optimizer class
            if self.optimizer in [AdamW, RMSprop]:
                optimizer_grouped_parameters = set_weight_decay(self.model, self.config.weight_decay)
                optimizer = self.optimizer(optimizer_grouped_parameters, lr=self.config.learning_rate, eps=self.config.epsilon)
            else:
                # optimizer_grouped_parameters = self.model.parameters()
                raise NotImplementedError(f"Initialization for {self.optimizer} have not been implemented.")
        self.optimizer = Optimizer(optimizer)
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.LRScheduler):
                scheduler = self.scheduler
            else:
                if self.scheduler is get_linear_schedule_with_warmup:
                    assert 1 >= self.config.warmup_ratio >= 0, f"`warmup_ratio` must be between 0 and 1"
                    warmupSteps = int(self.config.warmup_ratio * totalSteps)
                    scheduler = self.scheduler(self.optimizer.object_with_state_dict, warmupSteps, totalSteps)
                else:
                    raise NotImplementedError(f"Initialization for {self.scheduler} have not been implemented.")
        self.scheduler = Scheduler(scheduler)
        self.scaler = Scaler(self.scaler)

        # * Load optimizer_state_dict, scheduler_state_dict and scaler if possible
        if self.ckpt_manager.latest_dir.exists():
            self.optimizer.load(self.ckpt_manager.latest_dir, silence=False)
            if self.scheduler is not None:
                self.scheduler.load(self.ckpt_manager.latest_dir, silence=False)
            if self.scaler is not None:
                self.scaler.load(self.ckpt_manager.latest_dir, silence=False)

        # * Create or load watch dog
        if local_rank == 0:
            if self.ckpt_manager.latest_dir.exists():
                watch_dog = WatchDog.load(self.ckpt_manager.latest_dir, silence=False)
                # 如果因早停patience设置不合理导致训练不充分, 继续训练前: 需要重置WatchDog中的counter或增大patience
                if self.config.early_stop and self.config.continue_train_more_patience:
                    watch_dog.counter = 0
            else:
                watch_dog = WatchDog(patience=5 if self.config.early_stop else 2 * (self.config.epochs), metric=self.config.metric)

        # * Print some infomation for debug
        if local_rank == 0:
            logger.debug("===== 🔥 Start training 🔥 =====")
            logger.debug(f"  Batch size = {self.config.batch_size}")
            logger.debug(f"  Total epochs = {self.config.epochs:d}")
            logger.debug(f"  Steps per epoch = {stepsPerEpoch:d}")
            logger.debug(f"  Total steps = {totalSteps:d}")
            # if self.config.warmup_ratio >= 0:
            #     logger.debug(f"  Warmup steps = {warmupSteps:d}")
            logger.debug(f"  Model type = {self.config.model_type}")
            logger.debug(f"  fp16: {self.config.fp16}\n")
            logger.debug(f"  Start training from {self.ckpt_manager.latest_dir.name if self.ckpt_manager.latest_id>=0 else 'pretained model'}")

        self.ckpt_manager.next()
        curStepInGlobal = self.ckpt_manager.latest_id * stepsPerEpoch  # 总共已训练步数

        # log_losses = []
        # * ===========================================================训练===========================================================
        for epoch in range(self.ckpt_manager.latest_id, self.config.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            self.model.train()
            for curStepInEpoch, batch_in_accumulate in tqdm(
                enumerate(gradient_accumulate(dataloader_train, self.config.accumulate_step)),
                total=stepsPerEpoch,
                desc=f"{'training epoch':16}{epoch:#03d}",
                colour="GREEN",
                unit="batch",
                smoothing=0.8,
            ):
                # if curStepInEpoch < 3:
                #     if batch_in_accumulate[0]["input_ids"][0][0].numel() == 1:
                #         logger.debug(f'\n{tokenizer.decode(batch_in_accumulate[0]["input_ids"][0], skip_special_tokens=False)}\n')
                #     else:
                #         logger.debug(f'\n{tokenizer.decode(batch_in_accumulate[0]["input_ids"][0][0], skip_special_tokens=False)}\n')

                accumulate_loss = 0
                for batch in batch_in_accumulate:
                    # copy batch to GPU memory
                    custom_inputs = batch.pop("custom_inputs", dict())
                    batch = {key: value.cuda() for key, value in batch.items()}
                    if self.config.fp16:
                        # forward
                        with autocast(device_type="cuda", dtype=torch.float16):
                            outputs = self.model(**batch, **custom_inputs)
                            loss = outputs["loss"] / self.config.accumulate_step
                        # backward
                        self.scaler.scale(loss).backward()
                    else:
                        # forward
                        outputs = self.model(**batch, **custom_inputs)
                        loss = outputs["loss"] / self.config.accumulate_step
                        # backward
                        loss.backward()
                    accumulate_loss += loss.item()
                # logger.error(f"loss: {accumulate_loss}")
                if self.config.fp16:
                    # update parameters
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # update parameters
                    self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.optimizer.zero_grad()
                # # 梯度截断
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=15.0, norm_type=2.0)
                if local_rank == 0:
                    if curStepInGlobal & 15 == 0:
                        if self.config.dashboard == "wandb":
                            wandb.run.log(
                                {
                                    "training/loss": accumulate_loss,
                                    "training/learning_rate/downstream": self.optimizer.state_dict()["param_groups"][0]["lr"],
                                    "training/learning_rate/pretrain": self.optimizer.state_dict()["param_groups"][-1]["lr"],
                                },
                                step=curStepInGlobal,
                            )
                        elif self.config.dashboard == "tensorboard":
                            self.dashboard_writer.add_scalars(
                                "training/learning_rate",
                                {
                                    "downstream": self.optimizer.state_dict()["param_groups"][0]["lr"],
                                    "pretrain": self.optimizer.state_dict()["param_groups"][-1]["lr"],
                                },
                                curStepInGlobal,
                            )
                            self.dashboard_writer.add_scalar("training/loss", accumulate_loss, curStepInGlobal, new_style=True)
                # * Evaluate after each half epoch
                if self.config.test_in_epoch and curStepInEpoch == stepsPerEpoch >> 1:
                    val_metricdict = self.__evaluate(Split.VALIDATION, epoch, curStepInGlobal)
                    test_metricdict = self.__evaluate(Split.TEST, epoch, curStepInGlobal)
                    log_dict = dict()
                    log_dict[Split.VALIDATION.name] = dict(val_metricdict)
                    if test_metricdict is not None:
                        log_dict[Split.TEST.name] = dict(test_metricdict)
                    if self.config.dashboard == "wandb":
                        wandb.run.log(log_dict, step=curStepInGlobal)
                    elif self.config.dashboard == "tensorboard":
                        for split, metricdict in log_dict.items():
                            for metric, value in metricdict.items():
                                self.dashboard_writer.add_scalar(f"{split}/{metric}", value, curStepInGlobal, new_style=True)
                    if local_rank == 0:
                        watch_dog(
                            val_metricdict=val_metricdict,
                            test_metricdict=test_metricdict,
                            epoch=epoch,
                            step_global=curStepInGlobal,
                            model=self.model,
                            tokenizer=self.tokenizer,
                            configs=self.config,
                        )
                curStepInGlobal += 1
            # *----------------------------------one epoch finish-------------------------------------
            # * Evaluate after each epoch
            val_metricdict = self.__evaluate(Split.VALIDATION, epoch, curStepInGlobal)
            test_metricdict = self.__evaluate(Split.TEST, epoch, curStepInGlobal)
            log_dict = dict()
            log_dict[Split.VALIDATION.name] = dict(val_metricdict)
            if test_metricdict is not None:
                log_dict[Split.TEST.name] = dict(test_metricdict)
            if self.config.dashboard == "wandb":
                wandb.run.log(log_dict, step=curStepInGlobal)
            elif self.config.dashboard == "tensorboard":
                for split, metricdict in log_dict.items():
                    for metric, value in metricdict.items():
                        self.dashboard_writer.add_scalar(f"{split}/{metric}", value, curStepInGlobal, new_style=True)
            if local_rank == 0:
                watch_dog(
                    val_metricdict=val_metricdict,
                    test_metricdict=test_metricdict,
                    epoch=epoch,
                    step_global=curStepInGlobal,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    configs=self.config,
                )

            # # tensorboard 记录一个epoch中的平均loss
            # writer.add_scalars("loss/epoch", {"training": np.array(lossesInEpoch).mean(), "validation": devLoss}, epoch)
            if local_rank == 0:
                # * Save current checkpoint
                if epoch < self.config.epochs - 1:  # 当前设置为保存最后的checkpoint, 如果不需要, 则将configs.epochs改为configs.epochs - 1
                    logger.debug(f"🚩 Saving checkpoint: `{self.ckpt_manager.latest_dir.name}` ...")
                    self.ckpt_manager.latest_dir.mkdir()
                    logger.debug(f"❔ The checkpoint will be saved in {self.ckpt_manager.latest_dir}.")

                    logger.debug("💾 Saving model ...")
                    model_to_save = self.model.module if hasattr(self.model, "module") else self.model
                    model_to_save.save_pretrained(self.ckpt_manager.latest_dir)
                    if self.tokenizer is not None:
                        self.tokenizer.save_pretrained(self.ckpt_manager.latest_dir)
                    logger.debug("✔️  Save model successfully.")

                    self.config.save(self.ckpt_manager.latest_dir, silence=False)
                    watch_dog.save(self.ckpt_manager.latest_dir, silence=False)

                    self.optimizer.save(self.ckpt_manager.latest_dir, silence=False)
                    if self.scheduler is not None:
                        self.scheduler.save(self.ckpt_manager.latest_dir, silence=False)
                    if self.config.fp16:
                        self.scaler.save(self.ckpt_manager.latest_dir, silence=False)

                    logger.debug(f"✅ Save {self.ckpt_manager.latest_dir.name} successfully")

                # * delete last checkpoint
                if not self.config.save_all_ckpts:
                    self.ckpt_manager.delete_last_checkpoint()
                self.ckpt_manager.next()

                # * save WatchDog
                if epoch == self.config.epochs - 1:
                    watch_dog.finish()
                watch_dog.save(self.config.save_dir)

                # * Whether early stop is triggered
                if self.config.early_stop and watch_dog.need_to_stop:
                    break
            if dist.is_initialized():
                dist.barrier()
        # * ===========================================================训练结束===========================================================
        if local_rank == 0:
            # * Report the final information
            watch_dog.final_report(self.config)
            if self.config.dashboard == "wandb":
                wandb.run.summary.update(watch_dog.optimal_performance())
                wandb.run.finish()
            elif self.config.dashboard == "tensorboard":
                self.dashboard_writer.add_hparams(hparam_dict=self.config.to_dict(), metric_dict=watch_dog.optimal_performance())
                self.dashboard_writer.close()

    def __evaluate(self, split: Split, epoch: int, step_global: int) -> MetricDict | None:
        # if (split == Split.TEST and self.dataset_test is None) or (split == Split.VALIDATION and self.dataset_val is None):
        #     return None
        if split == Split.TEST and self.dataset_test is not None:
            if not hasattr(self, "dataloader_test"):
                self.dataloader_test = get_dataloader(self.dataset_test, self.config, Split.TEST, collate_fn=self.dataset_test.collate_fn)
        elif split == Split.VALIDATION and self.dataset_val is not None:
            if not hasattr(self, "dataloader_val"):
                self.dataloader_val = get_dataloader(self.dataset_val, self.config, Split.VALIDATION, collate_fn=self.dataset_val.collate_fn)
        else:
            return None
        logger.debug("")
        logger.debug(f"===== ❄️  Evaluate on {split.name} set ❄️ =====")
        logger.debug(f"===== epoch: {epoch:03d} step_global: {step_global:06d} =====")
        return self.evaluate(split)

    def evaluate(self, split: Split, cuda_id=None) -> MetricDict | None:
        local_rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        if split == Split.TEST and self.dataset_test is not None:
            dataloader = (
                get_dataloader(self.dataset_test, self.config, Split.TEST, collate_fn=self.dataset_test.collate_fn)
                if not hasattr(self, "dataloader_test")
                else self.dataloader_test
            )
        elif split == Split.VALIDATION and self.dataset_val is not None:
            dataloader = (
                get_dataloader(self.dataset_val, self.config, Split.VALIDATION, collate_fn=self.dataset_val.collate_fn)
                if not hasattr(self, "dataloader_val")
                else self.dataloader_val
            )
        else:
            return None

        all_losses = []
        all_labels = []
        all_logits = []
        if cuda_id is not None:
            torch.cuda.set_device(cuda_id)
            self.model.cuda()
        self.model.eval()
        match self.task_type:
            case "generate":
                for batch in tqdm(dataloader, desc=split.name, colour="BLUE", unit="batch", smoothing=0.9):
                    with torch.no_grad():
                        labels = batch.pop("labels")
                        batch = {key: value.cuda() for key, value in batch.items()}
                        outputs = self.model.generate(**batch, **self.config.generate_kwargs)
                        texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                        all_losses.append(-1)
                        all_labels.extend(labels)
                        all_logits.extend(texts)
            case "classify" | "regress":
                for batch in tqdm(dataloader, desc=split.name, colour="BLUE", unit="batch", smoothing=0.9):
                    batch = {key: value.cuda() for key, value in batch.items()}
                    labels = batch["labels"]
                    outputs = self.model(**batch)
                    loss, logits = outputs["loss"], outputs["logits"]
                    all_losses.append(loss.item())
                    all_labels.extend(labels.numpy(force=True).tolist())
                    all_logits.extend(logits.numpy(force=True).tolist())
        self.model.train()

        if world_size > 1:
            logger.debug(f"local rank {local_rank}: num_labels: {len(all_labels)}, num_logits: {len(all_logits)}, num_batches: {len(all_losses)}")

            labels_gather_list = [None for _ in range(world_size)]
            logits_gather_list = [None for _ in range(world_size)]
            mean_loss = torch.tensor(all_losses, dtype=torch.float32).mean().cuda()

            dist.gather_object(all_labels, labels_gather_list if local_rank == 0 else None, dst=0)
            dist.gather_object(all_logits, logits_gather_list if local_rank == 0 else None, dst=0)
            dist.reduce(mean_loss, dst=0, op=dist.ReduceOp.SUM, async_op=False)

            if local_rank != 0:  # final result will be calculated on `local rank 0` process
                return None

            all_labels = sum(labels_gather_list, [])
            all_logits = sum(logits_gather_list, [])
            mean_loss = (all_losses / world_size).item()
        else:
            mean_loss = sum(all_losses) / len(all_losses)

        return self.calculate_metric_callback(all_labels, all_logits, mean_loss)
