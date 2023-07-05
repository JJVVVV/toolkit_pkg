import datetime
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist

from ..logger import _getLogger
from .misc import type_to_str

logger = _getLogger(__name__)


# torch申请显存
def check_mem(cuda_device_id: int) -> Tuple[int, int]:
    """Get total and used memory (unit: `MB`) of GPU with the corresponding ID."""
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device_id)].split(",")
    return int(total), int(used)


def allocate_gpu_memory(ratio=0.8) -> None:
    """Allocate GPU memory.\n
    Support multiple GPUs, but the GPU used by the current process must be specified by `torch.cuda.set_device(local_rank)`"""
    try:
        local_rank = dist.get_rank()
    except:
        local_rank = 0

    cuda_device_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    cuda_device_ids = [cuda_device_id for cuda_device_id in cuda_device_ids if cuda_device_id]
    local_cuda_device_id = cuda_device_ids[local_rank]
    total, used = check_mem(local_cuda_device_id)
    block_mem = int((int(total) - int(used)) * ratio)
    x = torch.cuda.FloatTensor(256, 1024, block_mem)
    del x


def setup_seed(seed: int) -> None:
    """Set random seed"""
    # 如果读取数据的过程采用了随机预处理(如RandomCrop、RandomHorizontalFlip等)，那么对Python、Numpy的随机数生成器也需要设置种子。
    random.seed(seed)
    np.random.seed(seed)
    # 为了禁止hash随机化，使得实验可复现
    os.environ["PYTHONHASHSEED"] = str(seed)
    # torch中的随机种子
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    # if you are using multi-GPU. 为所有GPU设置随机种子
    torch.cuda.manual_seed_all(seed)


def setup_parallel() -> Tuple[int, int]:
    """Initial parallel backend"""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    try:
        logger.debug(f"NCCL_BLOCKING_WAIT={os.environ['NCCL_BLOCKING_WAIT']}")
    except:
        pass
    try:
        logger.debug(f"NCCL_ASYNC_ERROR_HANDLING={os.environ['NCCL_ASYNC_ERROR_HANDLING']}")
    except:
        pass
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=7200))
    torch.cuda.set_device(local_rank)

    return local_rank, world_size


def set_weight_decay(model: torch.nn.Module, weight_decay: float) -> List[Dict[str, Any]]:
    """Get optimizer grouped parameters with setting weight decay of some parameters (i.e. `bias`, `LayerNorm.weight`) to `0` and others to `weight_dacay: float`"""
    names_str = " ".join([name for name, para in model.named_parameters()])
    no_decay = ["bias"]
    if re.search(r"LayerNorm.weight", names_str):
        no_decay.append("LayerNorm.weight")
    if re.search(r"layer_norm.weight", names_str):
        no_decay.append("layer_norm.weight")
    logger.debug(f"no_dacay: {no_decay}")
    optimizer_grouped_parameters = [
        {"params": [param for name, param in model.named_parameters() if not any(nd in name for nd in no_decay)], "weight_decay": weight_decay},
        {"params": [param for name, param in model.named_parameters() if any(nd in name for nd in no_decay)], "weight_decay": 0.0},
    ]
    return optimizer_grouped_parameters


# TODO 多卡的兼容性
class StateDictMixin:
    default_file_name: str = ""

    def __init__(self, object_with_state_dict) -> None:
        self.object_with_state_dict = object_with_state_dict
        try:
            self.local_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        except:
            self.local_rank = 0
            self.world_size = 1
        logger.debug(f"{f'local rank {self.local_rank}: ' if self.world_size!=1 else ''}Initialize {type_to_str(self)} successfully.")

    def __getattr__(self, name):
        return getattr(self.object_with_state_dict, name)

    # TODO 多卡并行时, 不同卡上的的state dict可能需要分别存??
    def save(self, file_dir_or_path: Path | str, file_name: str | None = None, silence=True) -> None:
        if file_name is None:
            file_name = self.default_file_name
        if isinstance(file_dir_or_path, str):
            file_dir_or_path = Path(file_dir_or_path)

        if file_dir_or_path.is_file():
            file_path = file_dir_or_path
        else:
            file_path = file_dir_or_path / file_name
        try:
            torch.save(self.object_with_state_dict.state_dict(), file_path)
            if not silence:
                # logger.debug(
                #     f"{f'local rank {self.local_rank}: ' if self.world_size!=1 else ''}Save {file_dir_or_path if file_dir_or_path.is_file() else file_name} successfully."
                # )
                logger.debug(f"Save {type_to_str(self).split('.')[-1]} successfully.")

        except RuntimeError as e:
            logger.error(f"Failed to save {type_to_str(self).split('.')[-1]}. {e}")
            exit(1)

    def load(self, file_dir_or_path: Path | str, file_name: str | None = None, silence=True) -> None:
        if file_name is None:
            file_name = self.default_file_name
        if isinstance(file_dir_or_path, str):
            file_dir_or_path = Path(file_dir_or_path)

        if file_dir_or_path.is_file():
            file_path = file_dir_or_path
        else:
            file_path = file_dir_or_path / file_name
        try:
            self.object_with_state_dict.load_state_dict(torch.load(file_path))
            if not silence:
                # logger.debug(
                #     f"{f'local rank {self.local_rank}: ' if self.world_size!=1 else ''}Load {file_dir_or_path if file_dir_or_path.is_file() else file_name} successfully."
                # )
                logger.debug(f"Load {type_to_str(self).split('.')[-1]} successfully.")
        except FileNotFoundError:
            logger.warning(f"Failed to load {type_to_str(self).split('.')[-1]}. {file_path} dose not exist! ")
            exit(1)


class Optimizer(StateDictMixin):
    default_file_name = "state_dict_optimizer.pt"


class Scheduler(StateDictMixin):
    default_file_name = "state_dict_scheduler.pt"


class Scaler(StateDictMixin):
    default_file_name = "state_dict_gradient_scaler.pt"
