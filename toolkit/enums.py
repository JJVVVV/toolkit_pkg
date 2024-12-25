from enum import Enum, auto


class Split(Enum):
    TRAINING = auto()
    VALIDATION = auto()
    TEST = auto()
    UNK = auto()


class ConversationStyle(Enum):
    SINGLE = auto()
    INSTRUCTION = auto()
    BLANK = auto()
