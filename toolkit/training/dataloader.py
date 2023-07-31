import itertools
from typing import Dict, Generator, List, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset

from ..config.trainconfig import TrainConfig
from ..enums import Split
from ..logger import _getLogger

logger = _getLogger(__name__)


# TODO: If the batches in one epoch is not divisible by accumulate_step, the last few splited batches will be discarded.
def gradient_accumulate(dataloader: DataLoader, accumulate_step: int) -> Generator[List[Dict], None, None]:
    """Get a generator used for gradient accumulate. \n
    yield a `list` of batches where the batches will be used for gradient accumulate"""
    batch_in_accumulate = []
    for batch in dataloader:
        batch_in_accumulate.append(batch)
        if len(batch_in_accumulate) == accumulate_step:
            yield batch_in_accumulate
            batch_in_accumulate.clear()
    yield batch_in_accumulate


def get_dataloader(dataset: Dataset, configs: TrainConfig, split: Split, **dataloader_kwargs) -> Tuple[DataLoader, DistributedSampler] | DataLoader:
    """Getting the dataloader when using multiple GPUs, which is also compatible with a single GPU"""
    local_rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    def split_batch(x, n):
        quotient, remainder = divmod(x, n)  # 计算每一份的基础值和剩余的单位数
        return [(quotient + 1 if i < remainder else quotient) for i in range(n)]

    batch_size_per_prog = split_batch(configs.batch_size, world_size)

    if split == Split.TRAINING:
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=False, seed=configs.seed) if world_size != 1 else None
        g = torch.Generator()
        g.manual_seed(configs.seed)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size_per_prog[local_rank] // configs.accumulate_step,
            shuffle=(sampler is None),
            pin_memory=True,
            #   worker_init_fn=seed_worker,
            generator=g,
            sampler=sampler,
            **dataloader_kwargs,
        )
    else:
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=True) if world_size != 1 else None
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=configs.batch_size_infer // world_size // configs.accumulate_step,
            shuffle=False,
            pin_memory=True,
            sampler=sampler,
            **dataloader_kwargs,
        )
    # logger.debug(f'\n{tokenizer.decode(dataset.tokenized_dict["input_ids"][0][0], skip_special_tokens=False)}\n')

    # * If there is a tail in development dataset, concatenate it. (Max length of tail: world_size.)
    if local_rank == 0 and not split == Split.TRAINING and len(dataloader.sampler) * world_size < len(dataset):
        dataset_tail = Subset(dataset, range(len(dataloader.sampler) * world_size, len(dataset)))
        dataloader_tail = DataLoader(
            dataset=dataset_tail, batch_size=configs.batch_size_infer // world_size // configs.accumulate_step, shuffle=False, **dataloader_kwargs
        )
        logger.debug(f"Tail batch num: {len(dataloader_tail)}")
        dataloader = DataLoader(
            list(itertools.chain(dataloader, dataloader_tail)), batch_size=None, batch_sampler=None, shuffle=False, pin_memory=True
        )

    # * warning about accumulate
    if split == split.TRAINING:
        if (tail_batch_num := len(dataloader) % configs.accumulate_step) != 0 and local_rank == 0:
            logger.warning(
                (
                    # "The last batch in training data will be discarded! "
                    "The last batch in training is Not strictly batch gradient descent! "
                    "Because gradient accumulation is enabled. And the last few split batches are less than the accumulate step: "
                    f"{tail_batch_num} < {configs.accumulate_step}"
                )
            )
    return (dataloader, sampler) if split == Split.TRAINING else dataloader
