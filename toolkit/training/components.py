import re
from pathlib import Path
from typing import Any, Dict, List, Self

import torch
import torch.distributed as dist

from ..logger import _getLogger
from ..utils.misc import type_to_str

# from .trainer import logger

logger = _getLogger("Components")


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

    def __new__(cls, object_with_state_dict) -> Self:
        if object_with_state_dict is None:
            return None
        return super().__new__(cls)

    def __init__(self, object_with_state_dict) -> None:
        self.object_with_state_dict = object_with_state_dict
        self.local_rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        logger.debug(f"{f'local rank {self.local_rank}: ' if self.world_size!=1 else ''}Initialize {type_to_str(self).split('.')[-1]} successfully.")

    def __getattr__(self, name):
        return getattr(self.object_with_state_dict, name)

    # TODO 多卡并行时, 不同卡上的的state dict可能需要分别存??
    def save(self, file_dir_or_path: Path | str, file_name: str | None = None, silence=True) -> None:
        if not silence:
            logger.debug(f"💾 Saving {type_to_str(self).split('.')[-1]} state ...")
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
                logger.debug(f"✔️ Save successfully.")

        except RuntimeError as e:
            logger.warning(f"⚠️ Failed to save {type_to_str(self).split('.')[-1]}. {e}")
            # exit(1)

    def load(self, file_dir_or_path: Path | str, file_name: str | None = None, silence=True) -> None:
        if not silence:
            logger.debug(f"💾 Loading {type_to_str(self).split('.')[-1]} state ...")
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
                logger.debug(f"✔️ Load successfully.")
        except FileNotFoundError:
            logger.error(f"❌ Failed to load {type_to_str(self).split('.')[-1]}. {file_path} dose not exist! ")
            exit(1)


class Optimizer(StateDictMixin):
    default_file_name = "state_dict_optimizer.pt"


class Scheduler(StateDictMixin):
    default_file_name = "state_dict_scheduler.pt"


class Scaler(StateDictMixin):
    default_file_name = "state_dict_gradient_scaler.pt"
