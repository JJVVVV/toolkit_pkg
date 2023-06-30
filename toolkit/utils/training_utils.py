import itertools
import os
from typing import Generator, Tuple

import torch
from torch import Generator
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset
from trainconfig import TrainConfig

from ..logger import _getLogger

logger = _getLogger(__name__)


# TODO: If the batches in one epoch is not divisible by accumulate_step, the last few batches will be discarded.
def accumulate_dataloader(dataloader: DataLoader, accumulate_step: int) -> Generator[list, None, None]:
    """Get a generator used for gradient accumulate. \n
    yield a `list` of batches where the batches will be used for gradient accumulate"""
    batch_in_accumulate = []
    for batch in dataloader:
        batch_in_accumulate.append(batch)
        if len(batch_in_accumulate) == accumulate_step:
            yield batch_in_accumulate
            batch_in_accumulate.clear()


def get_dataloader(
    dataset: Dataset, train_config: TrainConfig, local_rank=0, world_size=1, is_train: bool = True, desc: str = "training", **dataloader_kwargs
) -> tuple[DataLoader, DistributedSampler] | DataLoader:
    """Getting the dataloader when using multiple GPUs, which is also compatible with a single GPU"""

    def split_batch(x, n):
        quotient, remainder = divmod(x, n)  # 计算每一份的基础值和剩余的单位数
        return [(quotient + 1 if i < remainder else quotient) for i in range(n)]

    batch_size_per_prog = split_batch(train_config.batch_size, world_size)

    if is_train:
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=False, seed=train_config.seed)
        g = Generator()
        g.manual_seed(train_config.seed)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size_per_prog[local_rank] // train_config.accumulate_step,
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
            batch_size=batch_size_per_prog[local_rank] // train_config.accumulate_step,
            shuffle=False,
            pin_memory=True,
            sampler=sampler,
            **dataloader_kwargs,
        )
    # logger.debug(f'\n{tokenizer.decode(dataset.tokenized_dict["input_ids"][0][0], skip_special_tokens=False)}\n')

    # * If there is a tail in development dataset, concatenate it. (Max length of tail: world_size.)
    if local_rank == 0 and not is_train and len(dataloader.sampler) * world_size < len(dataset):
        dataset_tail = Subset(dataset, range(len(dataloader.sampler) * world_size, len(dataset)))
        dataloader_tail = DataLoader(
            dataset=dataset_tail, batch_size=batch_size_per_prog[local_rank] // train_config.accumulate_step, shuffle=False, **dataloader_kwargs
        )
        logger.debug(f"Tail batch num: {len(dataloader_tail)}")
        dataloader = DataLoader(
            list(itertools.chain(dataloader, dataloader_tail)), batch_size=None, batch_sampler=None, shuffle=False, pin_memory=True
        )
        # for batch in dataloader:
        #     print(batch)
    return (dataloader, sampler) if is_train else dataloader


# torch申请显存
def check_mem(cuda_device_id: int) -> tuple[int, int]:
    """Get total and used memory (unit: `MB`) of GPU with the corresponding ID."""
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device_id)].split(",")
    return int(total), int(used)


def allocate_gpu_memory(ratio=0.8, local_rank=0):
    """Allocate GPU memory.\n
    Support multiple GPUs, but the GPU used by the current process must be specified by `torch.cuda.set_device(local_rank)`"""
    cuda_device_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    cuda_device_ids = [cuda_device_id for cuda_device_id in cuda_device_ids if cuda_device_id]
    local_cuda_device_id = cuda_device_ids[local_rank]
    total, used = check_mem(local_cuda_device_id)
    block_mem = int((int(total) - int(used)) * ratio)
    x = torch.cuda.FloatTensor(256, 1024, block_mem)
    del x
