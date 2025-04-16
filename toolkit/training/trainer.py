import pathlib
from collections import OrderedDict, defaultdict
from math import ceil
from pathlib import Path
from typing import Callable, Literal, Type, TypeVar

import hjson
import torch
import torch.distributed as dist
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW, RMSprop
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import (  # PreTrainedModel,
    PretrainedConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

# from .. import toolkit_logger
from ..config import TrainConfig
from ..enums import Split
from ..integration.deepspeed import DeepspeedConfig
from ..logger import _getLogger
from ..metric import MetricDict
from ..nlp.config import NLPTrainingConfig
from .checkpoint_manager import CheckpointManager
from .components import Optimizer, Scaler, Scheduler, set_weight_decay
from .dataloader import get_dataloader, gradient_accumulate
from .evaluator import Evaluator
from .initializer import *
from .watchdog import WatchDog

logger = _getLogger("Trainer")
try:
    import wandb

    WandbWriter = wandb.run.__class__
except:
    logger.warning("Can not import wandb, so you shoud not set the `dashboard` to 'wandb'")
    WandbWriter = object

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    logger.warning("Can not import tensorboard, so you shoud not set the `dashboard` to 'tensorboard'")
    WandbWriter = object

map_str2optmClass = {"AdamW": AdamW, "RMSprop": RMSprop}
map_dsSchType2scheFnStr = {"WarmupLR": "linearWarmup", "WarmupDecayLR": "linearWarmupDecay"}
map_str2getScheFn = {
    "linearWarmup": get_constant_schedule_with_warmup,
    "linearWarmupDecay": get_linear_schedule_with_warmup,
    "cosineWarmupDecay": get_cosine_schedule_with_warmup,
    "cosineWarmupDecayRestart": get_cosine_with_hard_restarts_schedule_with_warmup,
}

OptimizerClass = TypeVar("OptimizerClass", bound=torch.optim.Optimizer)
SchedulerClass = TypeVar("SchedulerClass", bound=torch.optim.lr_scheduler.LRScheduler)

# allowed_task_type = ("generate", "classify", "regress")

# dschf: HfDeepSpeedConfig
# instructs transformers to partition the model directly over multiple gpus using
# deepspeed.zero.Init when model's `from_pretrained` method is called.
dschf = None


def sync():
    # * sync
    if dist.is_initialized():
        dist.barrier()


class Trainer:
    is_initialized = False

    def __init__(
        self,
        config: TrainConfig | NLPTrainingConfig,
        # model: torch.nn.Module | PreTrainedModel | None = None,
        model: torch.nn.Module | None = None,
        model_config: PretrainedConfig | None = None,
        dataset_train: Dataset | None = None,
        dataset_val: Dataset | None = None,
        dataset_test: Dataset | None = None,
        calculate_metric_callback: Callable | None = None,
        optimizer: Type[OptimizerClass] | str | torch.optim.Optimizer | None = None,
        scheduler: Callable[..., torch.optim.lr_scheduler.LRScheduler] | str | torch.optim.lr_scheduler.LRScheduler | None = None,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        # dashboard_writer: SummaryWriter | WandbWriter | None = None,
        dashboard_writer=None,
        project_name: str = "untitled",
        extral_args_training: dict | None = None,
        extral_args_evaluation: dict | None = None,
        extral_evaluators: list | None = None,
        callback_load_model: Callable = None,
    ) -> None:
        """
        `task_type`: "generate", "classify", "regress"\n
        `optimizer`: "AdamW", "RMSprop"\n
        `scheduler`: "linearWarmup", "linearWarmupDecay", "cosineWarmupDecay", "cosineWarmupDecayRestart"\n
        `calculate_metric_callback` will be called as `calculate_metric_callback(all_labels, all_logits, mean_loss)`
        """
        assert self.is_initialized, "❗ You must run `initialize` function before training."
        self.local_rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        # world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.task_type = config.task_type
        self.config = config
        self.model = model
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test
        self.calculate_metric_callback = calculate_metric_callback
        self.extral_args_training = extral_args_training or dict()
        self.extral_args_evaluation = extral_args_evaluation or dict()
        self.extral_evaluators = extral_evaluators or []
        self.callback_load_model = callback_load_model

        if isinstance(optimizer, str):
            assert (
                optimizer in map_str2optmClass
            ), f"Only following optimizer can be mapped to the corresponding optimizer class: {list(map_str2optmClass.keys())}, but got `{optimizer}`"
            self.optimizer = map_str2optmClass[optimizer]
        else:
            self.optimizer = optimizer
        # 使用deepspeed时忽略该参数, 因为当前只支持使用deepspeed提供的scheduler
        if config.parallel_mode == "deepspeed":
            scheduler = None
        if isinstance(scheduler, str):
            # 如果用的是deepspeed中scheduler的命名格式, 则进行映射
            if scheduler in map_dsSchType2scheFnStr:
                scheduler = map_dsSchType2scheFnStr[scheduler]
            assert (
                scheduler in map_str2getScheFn
            ), f"Only following scheduler can be mapped to the corresponding function that return a scheduler: {list(map_str2getScheFn.keys())}, but got `{scheduler}`"
        self.scheduler = scheduler
        self.scaler = GradScaler() if config.fp16 and config.parallel_mode != "deepspeed" else None
        self.ckpt_manager = CheckpointManager(config.save_dir)
        # self.dashboard_writer = dashboard_writer
        if config.dashboard is not None:
            if dashboard_writer is not None:
                self.dashboard_writer = dashboard_writer
                if config.dashboard == "wandb":
                    assert self.dashboard_writer is wandb.run
            elif self.local_rank == 0:  # 未传入 dashboard 的 writer, 自动定义
                if config.dashboard == "tensorboard":
                    from torch.utils.tensorboard import SummaryWriter

                    dataset_name = config.dataset_name if config.dataset_name else "unk_dataset"
                    model_type = (
                        (config.model_type[1:] if config.model_type.startswith("/") else config.model_type) if config.model_type else "unk_model_type"
                    )
                    model_name = config.model_name if config.model_name else "unk_model_name"
                    if config.model_dir is not None:
                        model_dir = config.model_dir[1:] if config.model_dir.startswith("/") else config.model_dir
                    else:
                        model_dir = ""
                    run_dir = pathlib.Path("runs", "tensorboard", dataset_name, model_type, model_dir, model_name)
                    run_dir.mkdir(parents=True, exist_ok=True)
                    self.dashboard_writer = SummaryWriter(comment="training", log_dir=run_dir)
                elif config.dashboard == "wandb":
                    self.dashboard_writer = wandb.init(
                        dir="./runs/wandb",
                        project=project_name,
                        config=config.to_dict(),
                        tags=[config.dataset_name, config.model_type, config.model_name],
                        # mode="disabled",
                    )
                    assert self.dashboard_writer is wandb.run
        else:
            self.dashboard_writer = None
        sync()

    def __del__(self):
        if hasattr(self, "dashboard_writer") and self.dashboard_writer is not None:
            if self.config.dashboard == "wandb":
                self.dashboard_writer.finish()
            elif self.config.dashboard == "tensorboard":
                self.dashboard_writer.close()

    @classmethod
    def initialize(cls, config: TrainConfig, allocate_memory: float | None = None, log_file_path: str | None = None):
        output_path_logger = log_file_path or Path(config.save_dir) / "report.log"
        set_file_logger(output_path_logger)

        setup_seed(config.seed)
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            cuda_device_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            cuda_device_ids = [cuda_device_id for cuda_device_id in cuda_device_ids if cuda_device_id]
            if len(cuda_device_ids) > 1:
                if config.parallel_mode == "DDP":
                    local_rank, world_size = setup_parallel_ddp(config.ddp_timeout)
                elif config.parallel_mode == "deepspeed":
                    local_rank, world_size = setup_parallel_deepspeed()
                else:
                    raise ValueError("You are using multi-gpu, and you must specify the `parallel_mode`")
            else:
                setup_single_gpu()
                local_rank, world_size = 0, 1
        else:
            if config.parallel_mode == "DDP":
                local_rank, world_size = setup_parallel_ddp(config.ddp_timeout)
            elif config.parallel_mode == "deepspeed":
                local_rank, world_size = setup_parallel_deepspeed()
            elif config.parallel_mode is None:
                setup_single_gpu()
                local_rank, world_size = 0, 1

        if allocate_memory is not None:
            allocate_gpu_memory(allocate_memory)
        cls.is_initialized = True
        return local_rank, world_size

    def train(self) -> None:
        self.config.training_runtime = dict()

        # * Initalize to gpu
        if self.config.parallel_mode == "deepspeed":
            # self.model.cuda()
            pass
        else:
            if self.config.gpu:
                self.model.cuda()

        # * Create or load watch dog
        if self.ckpt_manager.latest_dir.exists():
            watch_dog = WatchDog.load(self.ckpt_manager.latest_dir, silence=False)
            # 如果因早停patience设置不合理导致训练不充分, 继续训练前: 需要重置WatchDog中的counter或增大patience
            if self.config.early_stop and self.config.continue_train_more_patience:
                watch_dog.counter = 0
        else:
            watch_dog = WatchDog(patience=5 if self.config.early_stop else 2 * (self.config.epochs), record_cheat=self.config.record_cheat)
        sync()

        # # * Calculate some training parameters
        # self.set_training_steps_dataset()
        # self.set_sch_warmup()

        # # * wrap model
        # self.wrap_model()

        # # * Do some preliminary preparations
        # self.set_evaluator()

        # # * Load training data
        # if self.config.parallel_mode == "deepspeed":
        #     dataloader_train, sampler = self.training_dataloader, None
        # else:
        #     # TODO: 通用性: collate_fn 并不一定需要, nlp任务中使用collate_fn裁剪batch中样本的pad来加速训练，但其他任务可能不需要
        #     dataloader_train, sampler = get_dataloader(
        #         self.dataset_train, self.config, Split.TRAINING, collate_fn=self.dataset_train.collate_fn, shuffle=self.config.shuffle
        #     )

        # * Load training data
        # TODO: 通用性: collate_fn 并不一定需要, nlp任务中使用collate_fn裁剪batch中样本的pad来加速训练，但其他任务可能不需要
        dataloader_train, sampler = get_dataloader(
            self.dataset_train, self.config, Split.TRAINING, collate_fn=self.dataset_train.collate_fn, shuffle=self.config.shuffle
        )
        sync()

        # * Calculate some training parameters
        self.set_batches()
        self.set_training_steps(dataloader_train)
        self.set_sch_warmup()

        # todo 如果self.model==None, 尝试用ckpt_manager 加载模型进行继续训练。

        # * wrap model
        self.wrap_model()
        sync()

        # * Do some preliminary preparations
        self.set_evaluator()
        sync()

        # * Initialize optimizer, scheduler, scaler
        if self.config.parallel_mode == "deepspeed":
            # todo: 当前只支持用使用deepspeed自带的optimizer以及scheduler
            pass
        else:
            if isinstance(self.optimizer, torch.optim.Optimizer):  # optimizer
                optimizer = self.optimizer
            else:  # optimizer class
                if self.optimizer in [AdamW, RMSprop]:
                    optimizer_grouped_parameters = set_weight_decay(self.model, self.config.opt_weight_decay)
                    optimizer = self.optimizer(optimizer_grouped_parameters, lr=self.config.opt_lr, eps=self.config.opt_eps)
                else:
                    # optimizer_grouped_parameters = self.model.parameters()
                    raise NotImplementedError(f"Initialization for {self.optimizer} have not been implemented.")
            self.optimizer = Optimizer(optimizer)
            if self.scheduler is not None:  # scheduler
                if isinstance(self.scheduler, torch.optim.lr_scheduler.LRScheduler):
                    self.scheduler = self.scheduler
                else:  # a string which can be mapped to a function that return a scheduler with a given optimizer
                    get_scheduler_fn = map_str2getScheFn[self.scheduler]
                    if self.scheduler == "linearWarmup":
                        self.scheduler = get_scheduler_fn(self.optimizer.object_with_state_dict, self.config.sch_warmup_num_steps)
                    elif self.scheduler == "linearWarmupDecay":
                        self.scheduler = get_scheduler_fn(
                            self.optimizer.object_with_state_dict, self.config.sch_warmup_num_steps, self.config.total_num_steps
                        )
                    elif self.scheduler in ("cosineWarmupDecay", "cosineWarmupDecayRestart"):
                        if self.config.sch_num_cycles == -1:
                            self.scheduler = get_scheduler_fn(
                                self.optimizer.object_with_state_dict, self.config.sch_warmup_num_steps, self.config.total_num_steps
                            )
                        else:
                            self.scheduler = get_scheduler_fn(
                                self.optimizer.object_with_state_dict,
                                self.config.sch_warmup_num_steps,
                                self.config.total_num_steps,
                                self.config.sch_num_cycles,
                            )
                    else:
                        raise NotImplementedError(f"Initialization for {self.scheduler} have not been implemented.")
            self.scheduler = Scheduler(self.scheduler)
            self.scaler = Scaler(self.scaler)

            # * Load optimizer_state_dict, scheduler_state_dict and scaler if possible
            if self.ckpt_manager.latest_dir.exists():
                self.optimizer.load(self.ckpt_manager.latest_dir, silence=False)
                if self.scheduler is not None:
                    self.scheduler.load(self.ckpt_manager.latest_dir, silence=False)
                if self.scaler is not None:
                    self.scaler.load(self.ckpt_manager.latest_dir, silence=False)

        # * Print some infomation for debug
        if self.local_rank == 0:
            logger.debug("🔥 Start training 🔥")
            logger.debug(f"  Batch size = {self.config.train_batch_size}")
            logger.debug(f"  Total epochs = {self.config.epochs:d}")
            logger.debug(f"  Steps per epoch = {self.config.steps_per_epoch:d}")
            logger.debug(f"  Total steps = {self.config.total_num_steps:d}")
            logger.debug(f"  Model type = {self.config.model_type}")
            logger.debug(f"  fp16: {self.config.fp16}")
            logger.debug(f"  bf16: {self.config.bf16}")
            logger.debug(f"  Start training from {self.ckpt_manager.latest_dir.name if self.ckpt_manager.latest_id>=0 else 'pretained model'}")

        # * Enter into a new ckpt
        self.ckpt_manager.next()
        curStepInGlobal = self.ckpt_manager.latest_id * self.config.steps_per_epoch  # 总共已训练步数
        self.config.training_runtime["cur_step"] = curStepInGlobal
        if hasattr(self.model, "training_runtime"):
            self.model.training_runtime = self.config.training_runtime

        # log_losses = []
        # * ===========================================================训练===========================================================
        for epoch in range(self.ckpt_manager.latest_id, self.config.epochs):
            self.config.training_runtime["cur_epoch"] = epoch
            if sampler is not None:
                sampler.set_epoch(epoch)
            self.model.train()
            training_bar = tqdm(
                enumerate(gradient_accumulate(dataloader_train, self.config.gradient_accumulation_steps)),
                total=self.config.steps_per_epoch,
                desc=f"Training-{epoch:03d}",
                colour="GREEN",
                unit="batch",
                smoothing=0.8,
            )
            for curStepInEpoch, batch_in_accumulate in training_bar:
                # if curStepInEpoch < 3:
                #     if batch_in_accumulate[0]["input_ids"][0][0].numel() == 1:
                #         logger.debug(f'\n{self.tokenizer.batch_decode(batch_in_accumulate[0]["input_ids"], skip_special_tokens=True)}\n')
                #     else:
                #         logger.debug(f'\n{self.tokenizer.decode(batch_in_accumulate[0]["input_ids"][0], skip_special_tokens=True)}\n')

                # forward and backward
                accumulate_loss = 0
                loss_log = 0
                for batch in batch_in_accumulate:
                    custom_inputs = batch.pop("custom_inputs", dict())
                    # copy batch to GPU memory
                    if self.config.gpu:
                        batch = {key: value.cuda() for key, value in batch.items()}
                    # import pdb; pdb.set_trace()
                    # 如果使用deepspeed，无需手动累计梯度，DeepspeedEngine会实现梯度累计，但输入依旧是micro_batch而不是training_batch
                    if self.config.parallel_mode == "deepspeed":
                        # forward
                        outputs = self.model(**batch, **custom_inputs, **self.extral_args_training)
                        loss = outputs["loss"]
                        # backward
                        loss = self.model.backward(loss)
                        # update parameters
                        self.model.step()
                        # accumulate_loss += loss.item() / self.config.gradient_accumulation_steps
                        # accumulate_loss += loss.item()
                    else:
                        if self.config.fp16:
                            # forward
                            with autocast(device_type="cuda", dtype=torch.float16):
                                outputs = self.model(**batch, **custom_inputs, **self.extral_args_training)
                                loss = outputs["loss"] / self.config.gradient_accumulation_steps
                            # backward
                            self.scaler.scale(loss).backward()
                        elif self.config.bf16:
                            with autocast(device_type="cuda", dtype=torch.bfloat16):
                                outputs = self.model(**batch, **custom_inputs, **self.extral_args_training)
                                loss = outputs["loss"] / self.config.gradient_accumulation_steps
                            # backward
                            # self.scaler.scale(loss).backward()
                            loss.backward()
                        else:
                            # forward
                            outputs = self.model(**batch, **custom_inputs, **self.extral_args_training)
                            loss = outputs["loss"] / self.config.gradient_accumulation_steps
                            # backward
                            loss.backward()
                    accumulate_loss += loss.item()
                    if "loss_log" in outputs:
                        loss_log += outputs["loss_log"] / self.config.gradient_accumulation_steps
                    else:
                        loss_log += loss
                loss_log = loss_log.view(-1)

                # call step()
                if self.config.parallel_mode == "deepspeed":
                    # already called step() in accumulate loop
                    # self.model.step()
                    pass
                else:
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

                # * log loss and learning rate on consoles
                if self.config.logging_steps and ((curStepInGlobal + 1) % self.config.logging_steps) == 0:
                    if self.scheduler is None:
                        lr = self.config.opt_lr
                    else:
                        try:
                            lr = self.scheduler.get_last_lr()[0]
                        except:
                            lr = -1.0

                    info = OrderedDict()
                    # info["loss"] = f"{accumulate_loss:.3f}"
                    info["loss"] = f"{[round(l, 3) for l in loss_log.tolist()]}"
                    # logger.error(f"local_rank: {self.local_rank}, loss: {info['loss']}, step: {curStepInGlobal}")
                    if self.config.show_lr:
                        info["lr"] = f"{lr:.2e}"
                    if self.config.show_step:
                        info["step"] = f"{curStepInGlobal:<5d}"
                    training_bar.set_postfix(info)

                    # * log loss and learning rate on dashboard
                    # if curStepInGlobal & 15 == 0:
                    # if True:
                    if self.local_rank == 0:
                        if self.config.dashboard == "wandb":
                            wandb.run.log({"training/loss": accumulate_loss, "training/learning_rate": lr}, step=curStepInGlobal)
                        elif self.config.dashboard == "tensorboard":
                            self.dashboard_writer.add_scalar("training/loss", accumulate_loss, curStepInGlobal, new_style=True)
                            self.dashboard_writer.add_scalar("training/learning_rate", lr, curStepInGlobal, new_style=True)

                # * Evaluate after each half epoch or reach eval_step
                if (self.config.eval_every_half_epoch and curStepInEpoch == self.config.steps_per_epoch >> 1) or (
                    self.config.eval_step > 0 and (curStepInGlobal + 1) % self.config.eval_step == 0
                ):
                    val_metricdict = self.__evaluate(Split.VALIDATION, epoch, curStepInGlobal)
                    test_metricdict = self.__evaluate(Split.TEST, epoch, curStepInGlobal)
                    watch_dog(
                        val_metricdict=val_metricdict if val_metricdict is not None else MetricDict(loss=accumulate_loss),
                        test_metricdict=test_metricdict,
                        epoch=epoch,
                        step_global=curStepInGlobal,
                        model=self.model,
                        tokenizer=self.tokenizer,
                        configs=self.config,
                    )
                    self.dashboard_log_metrics(val_metricdict, test_metricdict, accumulate_loss, curStepInGlobal)
                curStepInGlobal += 1
                self.config.training_runtime["cur_step"] = curStepInGlobal
            training_bar.close()
            # *----------------------------------one epoch finish-------------------------------------
            # * sync
            sync()

            # * Evaluate after each epoch
            if not self.config.eval_only_after_last_epoch or epoch == self.config.epochs - 1:
                val_metricdict = self.__evaluate(Split.VALIDATION, epoch, curStepInGlobal)
                test_metricdict = self.__evaluate(Split.TEST, epoch, curStepInGlobal)
                watch_dog(
                    val_metricdict=val_metricdict if val_metricdict is not None else MetricDict(loss=accumulate_loss),
                    test_metricdict=test_metricdict,
                    epoch=epoch,
                    step_global=curStepInGlobal,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    configs=self.config,
                )
                self.dashboard_log_metrics(val_metricdict, test_metricdict, accumulate_loss, curStepInGlobal)

            # # tensorboard 记录一个epoch中的平均loss
            # writer.add_scalars("loss/epoch", {"training": np.array(lossesInEpoch).mean(), "validation": devLoss}, epoch)

            # Save checkpoint
            if self.config.save_ckpts:
                # TODO 保存最后 n 个ckpt
                if self.config.parallel_mode == "deepspeed":
                    # * Save current checkpoint
                    if self.local_rank == 0:
                        logger.debug(f"🚩 Saving checkpoint: `{self.ckpt_manager.latest_dir.name}` ...")
                        self.ckpt_manager.latest_dir.mkdir()
                        logger.debug(f"❔ The checkpoint will be saved in {self.ckpt_manager.latest_dir}.")

                    if self.config.use_deepspeed_ckpt:
                        if self.local_rank == 0:
                            logger.debug(f"💾 Saving model weights, optimizer and scheduler ...")
                        self.model.save_checkpoint(self.ckpt_manager.latest_dir)
                        if self.local_rank == 0 and self.tokenizer is not None:
                            self.tokenizer.save_pretrained(self.ckpt_manager.latest_dir)
                    else:
                        if self.local_rank == 0:
                            logger.debug(f"💾 Saving model weights ...")
                        watch_dog.save_hf_model(self.config, self.ckpt_manager.latest_dir, self.model, self.tokenizer)
                    if self.local_rank == 0:
                        logger.debug("✔️  Save successfully.")

                    if self.local_rank == 0:
                        self.config.save(self.ckpt_manager.latest_dir, silence=False)
                        watch_dog.save(self.ckpt_manager.latest_dir, silence=False)
                        logger.debug(f"✅ Save {self.ckpt_manager.latest_dir.name} successfully")
                    # * delete last checkpoint
                    if self.local_rank == 0:
                        if not self.config.save_all_ckpts:
                            self.ckpt_manager.delete_last_checkpoint()
                else:
                    if self.local_rank == 0:
                        # * Save current checkpoint
                        logger.debug(f"🚩 Saving checkpoint: `{self.ckpt_manager.latest_dir.name}` ...")
                        self.ckpt_manager.latest_dir.mkdir()
                        logger.debug(f"❔ The checkpoint will be saved in {self.ckpt_manager.latest_dir}.")

                        logger.debug("💾 Saving model weights ...")
                        watch_dog.save_hf_model(self.config, self.ckpt_manager.latest_dir, self.model, self.tokenizer)
                        logger.debug("✔️  Save model successfully.")
                        # model_to_save = self.model.module if hasattr(self.model, "module") else self.model
                        # model_to_save.save_pretrained(self.ckpt_manager.latest_dir)
                        # if self.tokenizer is not None:
                        #     self.tokenizer.save_pretrained(self.ckpt_manager.latest_dir)

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

            # * save WatchDog to root save_dir
            watch_dog.save(self.config.save_dir)

            # * Whether early stop is triggered
            if self.config.early_stop and watch_dog.need_to_stop:
                break

            # * next ckpt
            self.ckpt_manager.next()

            # * sync
            sync()
        # * ===========================================================训练结束===========================================================
        if self.local_rank == 0:
            # * Report the final information
            watch_dog.final_report(self.config)
            if self.config.dashboard == "wandb":
                wandb.run.summary.update(watch_dog.optimal_performance())
                wandb.run.finish()
            elif self.config.dashboard == "tensorboard":
                self.dashboard_writer.add_hparams(hparam_dict=self.config.to_dict(flat=True), metric_dict=watch_dog.optimal_performance())
                self.dashboard_writer.close()

    def __evaluate(self, split: Split | Literal["TRAINING", "VALIDATION", "TEST", "UNK"], epoch: int, step_global: int) -> MetricDict | None:
        "Return metricdict if there are evaluators corresponding to the split, otherwise None will be returned."
        if not isinstance(split, Split):
            split = Split[split]
        evaluators = self.evaluators[split]
        if len(evaluators) == 0:
            return None
        if self.local_rank == 0:
            # logger.debug("")
            logger.debug(f"❄️  Evaluate on {split.name} set ❄️")
            logger.debug(f"epoch: {epoch:03d} step_global: {step_global:06d}")
        metricdict = MetricDict(metric_for_compare=self.config.metric)
        for evaluator in evaluators:
            metricdict.update(evaluator.eval())
            sync()
        return metricdict

    # def __evaluate(self, split: Split, epoch: int, step_global: int) -> MetricDict | None:
    #     # if (split == Split.TEST and self.dataset_test is None) or (split == Split.VALIDATION and self.dataset_val is None):
    #     #     return None
    #     if split == Split.TEST and self.evaluator_test is not None:
    #         evaluater = self.evaluator_test
    #     elif split == Split.VALIDATION and self.evaluator_val is not None:
    #         evaluater = self.evaluator_val
    #     else:
    #         return None
    #     if self.local_rank == 0:
    #         logger.debug("")
    #         logger.debug(f"===== ❄️  Evaluate on {split.name} set ❄️ =====")
    #         logger.debug(f"===== epoch: {epoch:03d} step_global: {step_global:06d} =====")
    #     return evaluater.eval()

    def dashboard_log_metrics(self, val_metricdict, test_metricdict, loss, curStepInGlobal):
        """
        log metrics to dashboard if dashboard is not None.
        if no evaluation, just log the loss of current step.
        """
        if self.local_rank == 0:
            log_dict = dict()
            if val_metricdict is not None:
                log_dict[Split.VALIDATION.name] = dict(val_metricdict)
            if test_metricdict is not None:
                log_dict[Split.TEST.name] = dict(test_metricdict)
            if len(log_dict) == 0:
                log_dict["Training"] = {"cur_step_loss": loss}
            if self.config.dashboard == "wandb":
                wandb.run.log(log_dict, step=curStepInGlobal)
            elif self.config.dashboard == "tensorboard":
                for split, metricdict in log_dict.items():
                    for metric, value in metricdict.items():
                        self.dashboard_writer.add_scalar(f"{split}/{metric}", value, curStepInGlobal, new_style=True)

    def set_batches(self):
        self.config.train_micro_batch_size_per_gpu = int(self.config.train_batch_size / self.config.gradient_accumulation_steps / self.world_size)

    def set_training_steps(self, dataloader):
        """
        calculate the training steps per epoch and the total steps after dataloader initialized.
        """
        # 使用DDP或deepspeed时， dataloader中的 batch 为某卡的某一个累计的 micro_batch,
        # 因此在该卡上一个epoch的step数为 dataloader的长度除梯度累计步数，
        # 因为是数据并行，每个卡上一个epoch的step数都相等且等于实际step数
        stepsPerEpoch = ceil(len(dataloader) / self.config.gradient_accumulation_steps)
        totalSteps = stepsPerEpoch * self.config.epochs
        self.config.total_num_steps = totalSteps
        self.config.steps_per_epoch = stepsPerEpoch

    def set_training_steps_dataset(self):
        """
        calculate the training steps per epoch and the total steps after dataset initialized.
        """
        # 使用DDP或deepspeed时， dataloader中的 batch 为某卡的某一个累计的 micro_batch,
        # 因此在该卡上一个epoch的step数为 dataloader的长度除梯度累计步数，
        # 因为是数据并行，每个卡上一个epoch的step数都相等且等于实际step数
        stepsPerEpoch = ceil(len(self.dataset_train) / self.config.train_batch_size)
        totalSteps = stepsPerEpoch * self.config.epochs
        self.config.total_num_steps = totalSteps
        self.config.steps_per_epoch = stepsPerEpoch

    def set_sch_warmup(self):
        """
        calculate the warmup steps and total steps in scheduler,
        if `sch_warmup_ratio_steps` is set after `set_training_steps` is called.
        """
        self.config.sch_total_num_steps = self.config.total_num_steps
        if self.config.sch_warmup_ratio_steps != -1 and self.config.sch_warmup_num_steps != -1:
            raise ValueError("❌ `sch_warmup_num_steps` and `sch_warmup_ratio_steps` cannot be set simultaneously.")
        elif self.config.sch_warmup_num_steps == -1 and self.config.sch_warmup_ratio_steps != -1:
            assert (
                1 >= self.config.sch_warmup_ratio_steps >= 0
            ), f"`warmup_ratio` must be between 0 and 1, but got {self.config.sch_warmup_ratio_steps}"
            self.config.sch_warmup_num_steps = round(self.config.sch_total_num_steps * self.config.sch_warmup_ratio_steps)
        elif self.config.sch_warmup_num_steps != -1 and self.config.sch_warmup_ratio_steps == -1:
            pass
        else:
            assert (
                self.scheduler is None
            ), "Neither `sch_warmup_num_steps` nor `sch_warmup_ratio_steps` is set, so the scheduler must be set to `None`"
        logger.debug(f"warmup steps: {self.config.sch_warmup_num_steps}/{self.config.total_num_steps}")

    def wrap_model(self):
        """
        warp model with deepspeed engine or DDP if necessary
        """
        if self.model is not None and self.config.activation_checkpointing:
            self.model.gradient_checkpointing_enable()
        if self.config.parallel_mode is None:
            return
        logger.debug("Wrapping the model ...")
        if self.config.parallel_mode == "DDP":
            self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=False)
        elif self.config.parallel_mode == "deepspeed":
            with open(self.config.deepspeed_config, "r") as ds_config_file:
                deepspeed_config = hjson.load(ds_config_file)
            deepspeed_config = DeepspeedConfig(deepspeed_config)
            # if self.model is not None and isinstance(self.model, PreTrainedModel):
            if self.model is not None:
                logger.warning(
                    (
                        "⚠️  You loaded a model with `from_pretrained` before setting deepspeed config, "
                        "so the model will be loaded to cpu memory before loaded to GPU."
                        "It is less efficient and when there is little CPU RAM may fail."
                        "We recommend you just offer the `model_config` and `callback_load_model`."
                        "Then we will load the model directly to GPU."
                    )
                )
                deepspeed_config.fill_ds_config(self.config, self.model.config)
            else:
                deepspeed_config.fill_ds_config(self.config, self.model_config)
                global dschf
                from transformers.integrations import HfDeepSpeedConfig

                dschf = HfDeepSpeedConfig(deepspeed_config.ds_config)
                self.model = self.callback_load_model(self.tokenizer)
                # todo: 当前只支持transformers模型
                if self.config.activation_checkpointing:
                    self.model.gradient_checkpointing_enable()
            # todo prior: 使用deepspeed的dataloader
            import deepspeed

            self.model, self.optimizer, self.training_dataloader, self.scheduler = deepspeed.initialize(
                model=self.model, config=deepspeed_config.ds_config, training_data=self.dataset_train, collate_fn=self.dataset_train.collate_fn
            )
            # deepspeed.DeepSpeedEngine.save_16bit_model()

    def set_evaluator(self):
        """
        Important: must initialize self.model(call wrap_model) before call this function.
        """
        # all_evaluator_class = [(Evaluator, self.calculate_metric_callback)] + self.extral_evaluators
        all_evaluator_class = self.extral_evaluators
        self.evaluators: defaultdict[Split, list[Evaluator]] = defaultdict(list)
        for split in (Split.VALIDATION, Split.TEST):
            # for evaluator_class, calculate_metric_callback in all_evaluator_class:
            for evaluator_class in all_evaluator_class:
                evaluator = evaluator_class(
                    task_type=self.task_type,
                    split=split,
                    config=self.config,
                    model=self.model.module if hasattr(self.model, "module") else self.model,
                    dataset=self.dataset_val if split == Split.VALIDATION else self.dataset_test,
                    # calculate_metric_callback=calculate_metric_callback,
                    extral_args_evaluation=self.extral_args_evaluation,
                    tokenizer=self.tokenizer,
                )
                if evaluator is not None:
                    self.evaluators[split].append(evaluator)
        if self.local_rank == 0:
            for key, value in self.evaluators.items():
                logger.debug(f"{key.name}: {[e.__class__.__name__ for e in value]}")
        # self.evaluator_val = Evaluator(
        #     self.task_type,
        #     Split.VALIDATION,
        #     self.config,
        #     self.model.module if hasattr(self.model, "module") else self.model,
        #     self.dataset_val,
        #     self.calculate_metric_callback,
        #     self.extral_args_evaluation,
        #     self.tokenizer,
        # )

        # self.evaluator_test = Evaluator(
        #     self.task_type,
        #     Split.TEST,
        #     self.config,
        #     self.model.module if hasattr(self.model, "module") else self.model,
        #     self.dataset_test,
        #     self.calculate_metric_callback,
        #     self.extral_args_evaluation,
        #     self.tokenizer,
        # )

    # ! deprecate
    # def evaluate(self, split: Split, cuda_id=None) -> MetricDict | None:
    #     local_rank = dist.get_rank() if dist.is_initialized() else 0
    #     world_size = dist.get_world_size() if dist.is_initialized() else 1

    #     if split == Split.TEST and self.dataset_test is not None:
    #         dataloader = (
    #             get_dataloader(self.dataset_test, self.config, Split.TEST, collate_fn=self.dataset_test.collate_fn)
    #             if not hasattr(self, "dataloader_test")
    #             else self.dataloader_test
    #         )
    #     elif split == Split.VALIDATION and self.dataset_val is not None:
    #         dataloader = (
    #             get_dataloader(self.dataset_val, self.config, Split.VALIDATION, collate_fn=self.dataset_val.collate_fn)
    #             if not hasattr(self, "dataloader_val")
    #             else self.dataloader_val
    #         )
    #     else:
    #         return None

    #     all_losses = []
    #     all_labels = []
    #     all_logits = []
    #     if cuda_id is not None:
    #         torch.cuda.set_device(cuda_id)
    #         self.model.cuda()
    #     self.model.eval()
    #     match self.task_type:
    #         case "generate":
    #             for batch in tqdm(dataloader, desc=split.name, colour="BLUE", unit="batch", smoothing=0.9):
    #                 with torch.no_grad():
    #                     labels = batch.pop("labels")
    #                     custom_inputs = batch.pop("custom_inputs", dict())
    #                     batch = {key: value.cuda() for key, value in batch.items()}
    #                     outputs = self.model.generate(**batch, **custom_inputs, **self.extral_args_evaluation, **self.config.generate_kwargs)
    #                     texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    #                     all_losses.append(-1)
    #                     all_labels.extend(labels)
    #                     all_logits.extend(texts)
    #         case "classify" | "regress":
    #             for batch in tqdm(dataloader, desc=split.name, colour="BLUE", unit="batch", smoothing=0.9):
    #                 with torch.no_grad():
    #                     custom_inputs = batch.pop("custom_inputs", dict())
    #                     batch = {key: value.cuda() for key, value in batch.items()}
    #                     labels = batch["labels"]
    #                     outputs = self.model(**batch, **custom_inputs, **self.extral_args_evaluation)
    #                     loss, logits = outputs["loss"], outputs["logits"]
    #                     all_losses.append(loss.item())
    #                     all_labels.extend(labels.numpy(force=True).tolist())
    #                     all_logits.extend(logits.numpy(force=True).tolist())
    #     self.model.train()

    #     if world_size > 1:
    #         logger.debug(f"local rank {local_rank}: num_labels: {len(all_labels)}, num_logits: {len(all_logits)}, num_batches: {len(all_losses)}")

    #         labels_gather_list = [None for _ in range(world_size)]
    #         logits_gather_list = [None for _ in range(world_size)]
    #         mean_loss = torch.tensor(all_losses, dtype=torch.float32).mean().cuda()

    #         dist.gather_object(all_labels, labels_gather_list if local_rank == 0 else None, dst=0)
    #         dist.gather_object(all_logits, logits_gather_list if local_rank == 0 else None, dst=0)
    #         dist.reduce(mean_loss, dst=0, op=dist.ReduceOp.SUM, async_op=False)

    #         if local_rank != 0:  # final result will be calculated on `local rank 0` process
    #             return None

    #         all_labels = sum(labels_gather_list, [])
    #         all_logits = sum(logits_gather_list, [])
    #         mean_loss = (mean_loss / world_size).item()
    #     else:
    #         mean_loss = sum(all_losses) / len(all_losses)

    #     return self.calculate_metric_callback(all_labels, all_logits, mean_loss)

    # if self.local_rank == 0:
    #     if self.config.parallel_mode == "deepspeed":
    #         if self.config.dashboard == "wandb":
    #             wandb.run.log({"training/loss": accumulate_loss}, step=curStepInGlobal)
    #         elif self.config.dashboard == "tensorboard":
    #             self.dashboard_writer.add_scalar("training/loss", accumulate_loss, curStepInGlobal, new_style=True)
    #     else:
    #         if self.config.dashboard == "wandb":
    #             wandb.run.log(
    #                 {
    #                     "training/loss": accumulate_loss,
    #                     "training/learning_rate/downstream": self.scheduler.get_last_lr()[0],
    #                     "training/learning_rate/pretrain": self.scheduler.get_last_lr()[0],
    #                 },
    #                 step=curStepInGlobal,
    #             )
    #         elif self.config.dashboard == "tensorboard":
    #             self.dashboard_writer.add_scalars(
    #                 "training/learning_rate",
    #                 {"downstream": self.scheduler.get_last_lr()[0], "pretrain": self.scheduler.get_last_lr()[0]},
    #                 curStepInGlobal,
    #             )
    #             self.dashboard_writer.add_scalar("training/loss", accumulate_loss, curStepInGlobal, new_style=True)
