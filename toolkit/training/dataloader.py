import itertools
from typing import Dict, Generator, List, Literal, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset

from ..config.trainconfig import TrainConfig
from ..enums import Split
from ..logger import _getLogger

logger = _getLogger(__name__)


def gradient_accumulate(dataloader: DataLoader, accumulate_step: int) -> Generator[List[Dict], None, None]:
    """Get a generator used for gradient accumulate. \n
    yield a `list` of batches where the batches will be used for gradient accumulate"""
    batch_in_accumulate = []
    for batch in dataloader:
        batch_in_accumulate.append(batch)
        if len(batch_in_accumulate) == accumulate_step:
            yield batch_in_accumulate
            batch_in_accumulate.clear()
    if len(batch_in_accumulate) > 0:
        yield batch_in_accumulate


def get_dataloader(
    dataset: Dataset, configs: TrainConfig, split: Split | Literal["TRAINING", "VALIDATION", "TEST", "UNK"], shuffle=None, **dataloader_kwargs
) -> Tuple[DataLoader, DistributedSampler] | DataLoader:
    """Getting the dataloader when using multiple GPUs, which is also compatible with a single GPU"""
    if not isinstance(split, Split):
        split = Split[split]
    local_rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    def split_batch(x, n):
        quotient, remainder = divmod(x, n)  # 计算每一份的基础值和剩余的单位数
        return [(quotient + 1 if i < remainder else quotient) for i in range(n)]

    # 训练
    if split == Split.TRAINING:
        batch_size_per_prog = split_batch(configs.train_batch_size, world_size)
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=False, seed=configs.seed) if world_size != 1 else None
        g = torch.Generator()
        g.manual_seed(configs.seed)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size_per_prog[local_rank] // configs.gradient_accumulation_steps,
            shuffle=(sampler is None) if shuffle is None else shuffle,
            pin_memory=True,
            #   worker_init_fn=seed_worker,
            generator=g,
            sampler=sampler,
            **dataloader_kwargs,
        )
    # 推理
    else:
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=True) if world_size != 1 else None
        dataloader = DataLoader(
            dataset=dataset, batch_size=configs.infer_batch_size // world_size, shuffle=False, pin_memory=True, sampler=sampler, **dataloader_kwargs
        )
        # logger.debug(f"local rank {local_rank}: {len(dataset)}, {len(dataloader.sampler)}, {len(dataloader)}")
    # logger.debug(f'\n{tokenizer.decode(dataset.tokenized_dict["input_ids"][0][0], skip_special_tokens=False)}\n')

    # * If there is a tail in development dataset, concatenate it. (Max length of tail: world_size.)
    if split != Split.TRAINING and len(dataloader.sampler) * world_size < len(dataset):
        logger.debug(f"local rank {local_rank}: Concatenate tail of dataloader, length: {len(dataset) - len(dataloader.sampler) * world_size}")
        # * if deepspeed is not used, that means infer in DDP or single GPU, only need one GPU to deal with the tail
        # * otherwise, in model parallel(MP, ZERO3), the tail must be passed to all GPU.
        # todo 当前做法是丢弃最后的 tail, 如过不丢弃, 应该为 `if local_rank == 0 or configs.parallel_mode == "deepspeed":`
        if local_rank == 0 and configs.parallel_mode != "deepspeed":
            dataset_tail = Subset(dataset, range(len(dataloader.sampler) * world_size, len(dataset)))
            dataloader_tail = DataLoader(dataset=dataset_tail, batch_size=configs.infer_batch_size // world_size, shuffle=False, **dataloader_kwargs)
            logger.debug(f"Tail batch num: {len(dataloader_tail)}")
            dataloader = DataLoader(
                list(itertools.chain(dataloader, dataloader_tail)), batch_size=None, batch_sampler=None, shuffle=False, pin_memory=True
            )

    # * warning about accumulate
    if split == split.TRAINING:
        if (tail_batch_num := len(dataloader) % configs.gradient_accumulation_steps) != 0 and local_rank == 0:
            logger.warning(
                (
                    # "The last batch in training data will be discarded! "
                    "The last batch in training is Not strictly batch gradient descent! "
                    "Because gradient accumulation is enabled, and the last few micro batches are less than the accumulate step: "
                    f"{tail_batch_num} < {configs.gradient_accumulation_steps}"
                )
            )
    return (dataloader, sampler) if split == Split.TRAINING else dataloader
