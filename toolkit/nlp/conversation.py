import dataclasses
from typing import Optional

from ..enums import ConversationStyle


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    roles: tuple[str, ...]
    messages: list[list[str, str | list[tuple[bool, str]]]]
    offset: int
    conversation_style: ConversationStyle = ConversationStyle.SINGLE
    sep: str = "###"
    sep2: Optional[str] = None

    conv_id: Optional[str] = None
    model_name: Optional[str] = None
    system: str = ""

    def clear(self):
        self.messages.clear()

    def append_message(self, role: str, message: str):
        self.messages.append([role, message])

    def append_message_(self, role: str, message: list[tuple[bool, str]]):
        self.messages.append([role, message])

    def get_prompt(self):
        match self.conversation_style:
            case ConversationStyle.SINGLE:
                ret = self.system
                for role, message in self.messages:
                    if message:
                        ret += self.sep + " " + role + ": " + message
                    else:
                        ret += self.sep + " " + role + ":"
                return ret
            case ConversationStyle.INSTRUCTION:
                seps = [self.sep, self.sep2]
                ret = self.system
                for i, (role, message) in enumerate(self.messages):
                    if message:
                        ret += role + ":\n" + message + seps[i % 2]
                        if i % 2 == 1:
                            ret += "\n\n"
                    else:
                        ret += role + ":\n"
                return ret
            case ConversationStyle.BLANK:
                seps = [self.sep, self.sep2]
                ret = self.system
                for i, (role, message) in enumerate(self.messages):
                    if message:
                        ret += message + seps[i % 2]
                    else:
                        pass
                return ret
            case _:
                raise ValueError(f"Invalid style: {self.conversation_style}")

    def get_prompt_(self) -> list[tuple[bool, str]]:
        match self.conversation_style:
            # TODO
            case ConversationStyle.SINGLE:
                ret = self.system
                for role, message in self.messages:
                    if message:
                        ret += self.sep + " " + role + ": " + message
                    else:
                        ret += self.sep + " " + role + ":"
                return ret
            # TODO
            case ConversationStyle.INSTRUCTION:
                seps = [self.sep, self.sep2]
                ret = self.system
                for i, (role, message) in enumerate(self.messages):
                    if message:
                        ret += role + ":\n" + message + seps[i % 2]
                        if i % 2 == 1:
                            ret += "\n\n"
                    else:
                        ret += role + ":\n"
                return ret
            case ConversationStyle.BLANK:
                seps = [self.sep, self.sep2]
                ret = [(False, self.system)]
                for i, (role, message) in enumerate(self.messages):
                    if message:
                        ret.extend(message)
                        ret.append((False, seps[i % 2]))
                    else:
                        pass
                return ret
            case _:
                raise ValueError(f"Invalid style: {self.conversation_style}")

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            conversation_style=self.conversation_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id,
            model_name=self.model_name,
        )

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
            "model_name": self.model_name,
        }


conv_instruct = Conversation(
    system="Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n",
    roles=("### Instruction", "### Response"),
    messages=[],
    offset=0,
    conversation_style=ConversationStyle.INSTRUCTION,
    sep="\n\n",
    sep2="### End",
)

conv_blank = Conversation(system="", roles=("", ""), messages=[], offset=0, conversation_style=ConversationStyle.BLANK, sep=" ", sep2=r"<\s>")

default_conv = {ConversationStyle.INSTRUCTION: conv_instruct, ConversationStyle.BLANK: conv_blank}
