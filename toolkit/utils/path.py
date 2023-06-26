import os


class Path(os.PathLike):
    def __init__(self, path):
        self.__path = path

    def __fspath__(self):
        return self.__path.replace("/", os.path.sep)
