from pathlib import Path
from shutil import rmtree

from ..logger import _getLogger

logger = _getLogger("CheckpointManager")


def epoch2checkpoint(epoch: int) -> str:
    return f"checkpoint-{epoch:03d}"


class CheckpointManager:
    def __init__(self, checkpoints_dir: Path | str) -> None:
        self.checkpoints_dir = Path(checkpoints_dir)
        self.id_latest_dir = -1
        self.latest_dir = None
        self.search()

    def id2dir(self, checkpoint_id: int) -> Path:
        "Convert a id (int) to a checkpoint dir"
        return self.checkpoints_dir / f"checkpoint-{checkpoint_id:03d}"

    @property
    def latest_dir(self):
        return self.latest_dir

    @latest_dir.setter
    def latest_dir(self):
        raise AttributeError("Can not set latest checkpoint dir directly.")

    @property
    def latest_id(self):
        return self.id_latest_dir

    @latest_id.setter
    def latest_id(self):
        raise AttributeError("Can not set id of the latest checkpoint directly.")

    def search(self):
        "Search in checkpoints dir, get the latest checkpoint dir"
        for checkpoint_dir in self.checkpoints_dir.glob("checkpoint-*"):
            if self.id_latest_dir < (id_searched := int(checkpoint_dir.name.split("-")[-1])):
                self.id_latest_dir = id_searched
        self.latest_dir = self.id2dir(self.id_latest_dir)

        if self.id_latest_dir > -1:
            logger.debug(f"Find `{self.latest_dir.name}` successfully!")
        else:
            logger.debug("There is no checkpoint.")

    def next(self):
        "Get a new checkpoint dir"
        self.id_latest_dir += 1
        self.latest_dir = self.id2dir(self.id_latest_dir)

    def delete_last_checkpoint(self):
        "Deletes the previous checkpoint of the latest checkpoint"
        if self.id_latest_dir > 0:
            last_checkpoint = self.id2dir(self.id_latest_dir - 1)
            rmtree(last_checkpoint)
            logger.debug(f"Delete last checkpoint: `{last_checkpoint.name}` successfully")

    # def set_checkpoint(self, checkpoint_id: int):
    #     self.cur_checkpoint_id = checkpoint_id
    #     self.cur_checkpoint = self.id2checkpoint(checkpoint_id)
