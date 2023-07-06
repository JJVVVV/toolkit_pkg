from pathlib import Path
from shutil import rmtree

from ..logger import _getLogger

logger = _getLogger(__name__)


def epoch2checkpoint(epoch: int) -> str:
    return f"checkpoint-{epoch:03d}"


class CheckpointManager:
    def __init__(self, checkpoints_dir: Path | str) -> None:
        self.checkpoints_dir = Path(checkpoints_dir)
        self.latest_checkpoint_id = -1
        self.cur_checkpoint_id = -1
        self.latest_checkpoint = None
        self.cur_checkpoint = None

    def id2checkpoint(self, checkpoint_id: int) -> Path:
        return self.checkpoints_dir / f"checkpoint-{checkpoint_id:03d}"

    def search_latest_checkpoint(self):
        for checkpoint_dir in self.checkpoints_dir.glob("checkpoint-*"):
            if self.latest_checkpoint_id < (id_searched := int(checkpoint_dir.name.split("-")[-1])):
                self.latest_checkpoint_id = id_searched
        self.latest_checkpoint = self.id2checkpoint(self.latest_checkpoint_id)

        if self.latest_checkpoint_id > -1:
            logger.debug(f"Find `{self.latest_checkpoint.name}` successfully!")
        else:
            logger.debug("There is no checkpoint.")

    def next(self):
        self.latest_checkpoint_id += 1
        self.latest_checkpoint = self.id2checkpoint(self.latest_checkpoint_id)

    def set_checkpoint(self, checkpoint_id: int):
        self.cur_checkpoint_id = checkpoint_id
        self.cur_checkpoint = self.id2checkpoint(checkpoint_id)

    def delete_last_checkpoint(self):
        if self.latest_checkpoint_id > 0:
            last_checkpoint = self.id2checkpoint(self.latest_checkpoint_id - 1)
            rmtree(last_checkpoint)
            logger.debug(f"Delete last checkpoint: `{last_checkpoint.name}` successfully")
