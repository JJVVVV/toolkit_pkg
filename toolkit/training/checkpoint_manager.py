from pathlib import Path
from shutil import rmtree

from ..logger import _getLogger

logger = _getLogger("CheckpointManager")


# def epoch2checkpoint(epoch: int) -> str:
#     return f"checkpoint-{epoch:03d}"


class CheckpointManager:
    def __init__(self, checkpoints_dir: Path | str) -> None:
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoints directory: `{self.checkpoints_dir}`")
        self.__id_latest_dir = -1
        self.__latest_dir = None
        self.search()

    def id2dir(self, checkpoint_id: int) -> Path:
        "Convert a id (int) to a checkpoint dir"
        return self.checkpoints_dir / f"checkpoint-{checkpoint_id:03d}"

    @property
    def latest_dir(self):
        return self.__latest_dir

    @latest_dir.setter
    def latest_dir(self):
        raise AttributeError("Can not set latest checkpoint dir directly.")

    @property
    def latest_id(self):
        return self.__id_latest_dir

    @latest_id.setter
    def latest_id(self):
        raise AttributeError("Can not set id of the latest checkpoint directly.")

    def search(self):
        "Search in checkpoints dir, get the latest checkpoint dir"
        for checkpoint_dir in self.checkpoints_dir.glob("checkpoint-*"):
            if self.__id_latest_dir < (id_searched := int(checkpoint_dir.name.split("-")[-1])):
                self.__id_latest_dir = id_searched
        self.__latest_dir = self.id2dir(self.__id_latest_dir)

        if self.__id_latest_dir > -1:
            logger.debug(f"Find `{self.__latest_dir.name}` successfully!")
        else:
            logger.debug("There is no checkpoint.")

    def next(self):
        "Get a new checkpoint dir"
        self.__id_latest_dir += 1
        self.__latest_dir = self.id2dir(self.__id_latest_dir)

    def delete_last_checkpoint(self):
        "Deletes the previous checkpoint of the latest checkpoint"
        if self.__id_latest_dir > 0:
            last_checkpoint = self.id2dir(self.__id_latest_dir - 1)
            rmtree(last_checkpoint)
            logger.debug(f"Delete last checkpoint: `{last_checkpoint.name}` successfully")

    # def set_checkpoint(self, checkpoint_id: int):
    #     self.cur_checkpoint_id = checkpoint_id
    #     self.cur_checkpoint = self.id2checkpoint(checkpoint_id)
