from collections import defaultdict
from pathlib import Path
from types import NoneType
from typing import Callable, Dict, List, Tuple

import torch
from torch.utils.data import Dataset, default_collate
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from ..logger import _getLogger

Tokens = list[int]
BatchTokens = list[Tokens]
ModelInput = dict[str, Tokens]
ModelInputSplited = dict[str, list[Tokens]]
BatchModelInput = dict[str, BatchTokens]
BoolStrings = Tuple[Tuple[bool, str], ...]
TextPair = Tuple[str, str | None]
ClassificationID = List[int]

logger = _getLogger(__name__)


class TextDataset(Dataset):
    """
    A demo of get_data_from_file:
    ```
    def get_data_from_file(data_file_path, model_type, tokenizer, is_train, **kargs):
        special_tokens_map = tokenizer.special_tokens_map
        BOS = special_tokens_map["bos_token"] if "bos_token" in special_tokens_map.keys() else None
        EOS = special_tokens_map["eos_token"] if "eos_token" in special_tokens_map.keys() else None
        SEP = special_tokens_map["sep_token"] if "sep_token" in special_tokens_map.keys() else None
        MASK = special_tokens_map["mask_token"] if "mask_token" in special_tokens_map.keys() else None
        CLS = special_tokens_map["cls_token"] if "cls_token" in special_tokens_map.keys() else None
        sep_num = 1

        with jsonlines.open(data_file_path, "r") as jlReader:
            dict_objs = list(jlReader)
            if isinstance(dict_objs[0], str):
                dict_objs = dict_objs[1:]

        inputs = []
        labels = []
        for dict_obj in dict_objs:
            inputs.append((dict_obj["question1"], dict_obj["question2"]))
            # input_texts.append(((False, CLS), (True, dict_obj["question1"]), (False, SEP), (True, dict_obj["question2"]), (False, SEP)))
            labels.append([dict_obj["label"]])
            # labels.append(((False, CLS), (True, dict_obj["question1"])))
            # labels.append((dict_obj["question1"], None))
            # labels.append(dict_obj["question1"])
        return inputs, labels
    ```
    """

    def __init__(
        self,
        data_file_path: Path | str,
        model_type: str,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        get_data_from_file: Callable[
            [str, str, PreTrainedTokenizer | PreTrainedTokenizerFast, bool],
            Tuple[List[BoolStrings] | list[TextPair], List[BoolStrings] | list[TextPair] | List[ClassificationID]],
        ],
        padding_side: str = "right",
        max_length_input: int | None = None,
        max_length_label: int | None = None,
        is_train: bool = True,
        **kargs,
    ) -> None:
        super().__init__()
        max_length_input = tokenizer.model_max_length if max_length_input is None else max_length_input
        max_length_label = tokenizer.model_max_length if max_length_label is None else max_length_label

        # get input and label texts
        self.padding_side = padding_side
        self.splited_texts_input, self.splited_texts_label = get_data_from_file(
            data_file_path=data_file_path, model_type=model_type, tokenizer=tokenizer, is_train=is_train, **kargs
        )

        # tokenize input texts
        tokenizer.padding_side = padding_side
        if (
            isinstance(self.splited_texts_input[0], tuple)
            and isinstance(self.splited_texts_input[0][0], str)
            and isinstance(self.splited_texts_input[0][1], str | NoneType)
        ):  # if the input type is `TextPair`, i.e. `tuple[str, str | None]`
            self.batch_model_input = self.transformers_tokenizer_tqdm(
                tokenizer, self.splited_texts_input, max_length_input, desc="Tokenize input texts"
            )
        elif (
            isinstance(self.splited_texts_input[0], tuple)
            and isinstance(self.splited_texts_input[0][0], tuple)
            and isinstance(self.splited_texts_input[0][0][0], bool)
            and isinstance(self.splited_texts_input[0][0][1], str)
        ):  # if the input type is `BoolStrings`, i.e. `tuple[tuple[bool,str], ...]`
            self.batch_model_input = self.__tokenize(self.splited_texts_input, tokenizer, max_length_input, desc="Tokenize input texts")
        else:
            raise ValueError("The input type must be `TextPair (Tuple[str, str | None])` or `BoolStrings (Tuple[Tuple[bool, str], ...])`")
        self.batch_model_input = {key: torch.tensor(value) for key, value in self.batch_model_input.items()}
        if self.padding_side == "right":
            self.first_pad_indexes_input = torch.argmax(torch.eq(self.batch_model_input["input_ids"], tokenizer.pad_token_id).int(), dim=-1)
            self.first_pad_indexes_input[self.first_pad_indexes_input == 0] = max_length_input
            self.max_length_inputs = torch.max(self.first_pad_indexes_input).item()
            self.batch_model_input = {key: value[..., : self.max_length_inputs] for key, value in self.batch_model_input.items()}
        elif self.padding_side == "left":
            self.first_not_pad_indexes_input = torch.argmax(torch.ne(self.batch_model_input["input_ids"], tokenizer.pad_token_id).int(), dim=-1)
            self.max_length_inputs = max_length_input - torch.min(self.first_not_pad_indexes_input).item()
            self.batch_model_input = {key: value[..., -self.max_length_inputs :] for key, value in self.batch_model_input.items()}
            self.first_not_pad_indexes_input = torch.argmax(torch.ne(self.batch_model_input["input_ids"], tokenizer.pad_token_id).int(), dim=-1)

        # tokenize label texts
        tokenizer.padding_side = "right"
        if (
            isinstance(self.splited_texts_label[0], tuple)
            and isinstance(self.splited_texts_label[0][0], str)
            and isinstance(self.splited_texts_label[0][1], str | NoneType)
        ):  # if the label type is `TextPair`, i.e. `tuple[str, str | None]`
            self.tokens_labels = self.transformers_tokenizer_tqdm(tokenizer, self.splited_texts_label, max_length_label, desc="Tokenize label texts")[
                "input_ids"
            ]
            self.tokens_labels = torch.tensor(self.tokens_labels)
            self.first_pad_indexes_label = torch.argmax(torch.eq(self.tokens_labels, tokenizer.pad_token_id).int(), dim=-1)
            self.first_pad_indexes_label[self.first_pad_indexes_label == 0] = max_length_label
            self.max_length_labels = torch.max(self.first_pad_indexes_label).item()
            self.tokens_labels = torch.narrow(self.tokens_labels, -1, 0, self.max_length_labels)
            self.tokens_labels[self.tokens_labels == tokenizer.pad_token_id] = -100
        elif (
            isinstance(self.splited_texts_label[0], tuple)
            and isinstance(self.splited_texts_label[0][0], tuple)
            and isinstance(self.splited_texts_label[0][0][0], bool)
            and isinstance(self.splited_texts_label[0][0][1], str)
        ):  # if the label type is `BoolStrings`, i.e. `tuple[tuple[bool,str], ...]`
            self.tokens_labels = self.__tokenize(self.splited_texts_label, tokenizer, max_length_label, desc="Tokenize label texts")["input_ids"]
            self.tokens_labels = torch.tensor(self.tokens_labels)
            self.first_pad_indexes_label = torch.argmax(torch.eq(self.tokens_labels, tokenizer.pad_token_id).int(), dim=-1)
            self.first_pad_indexes_label[self.first_pad_indexes_label == 0] = max_length_label
            self.max_length_labels = torch.max(self.first_pad_indexes_label).item()
            self.tokens_labels = torch.narrow(self.tokens_labels, -1, 0, self.max_length_labels)
            self.tokens_labels[self.tokens_labels == tokenizer.pad_token_id] = -100
        elif isinstance(self.splited_texts_label[0], str):  # if the label type is `str`
            self.tokens_labels = self.splited_texts_label
            self.max_length_labels = -1
        elif isinstance(self.splited_texts_label[0], list):  # if the label type is `ClassificationID`, i.e. `List[int]`
            self.tokens_labels = torch.tensor(self.splited_texts_label, dtype=torch.int)
            self.max_length_labels = self.tokens_labels.shape[-1]
        else:
            raise ValueError(
                (
                    "If the label is text, it must be `BoolStrings (Tuple[Tuple[bool, str], ...])` or `TextPair (Tuple[str, str | None])` or `str`, "
                    "if the label is classification, it must be `ClassificationID (List[int])`"
                )
            )

        # if "roberta" in self.model_type:
        #     inputs_ids = self.tokenized_dict["input_ids"]
        #     self.cls_sep_indexes = (
        #         ((inputs_ids == tokenizer.cls_token_id) | (inputs_ids == tokenizer.sep_token_id)).nonzero()[:, 1].reshape(inputs_ids.shape[0], -1)
        #     )

    def __getitem__(self, item: int) -> dict:
        ret_dict = dict()
        ret_dict["model_input"] = {key: value[item] for key, value in self.batch_model_input.items()}
        if self.padding_side == "right":
            ret_dict["first_pad_index_input"] = self.first_pad_indexes_input[item]
        else:
            ret_dict["first_not_pad_index_input"] = self.first_not_pad_indexes_input[item]

        if self.tokens_labels is not None:
            ret_dict["labels"] = self.tokens_labels[item]
        return ret_dict

    def __len__(self):
        return len(self.splited_texts_input)

    @staticmethod
    def __truncate(model_input_splited: ModelInputSplited, waiting_to_trunc_idxs: list[int], num_tokens_to_remove: int) -> None:
        lengths = [len(value_part) for value_part in model_input_splited["input_ids"]]
        lengths_waiting_to_trunc = [lengths[i] for i in waiting_to_trunc_idxs]
        if (all_tokens := sum(lengths_waiting_to_trunc)) < num_tokens_to_remove:
            raise ValueError(
                f"The number of tokens need to be truncated is greater than the total number of tokens: {num_tokens_to_remove}>{all_tokens}"
            )
        for _ in range(num_tokens_to_remove):
            index_max = lengths_waiting_to_trunc.index(max(lengths_waiting_to_trunc))
            lengths_waiting_to_trunc[index_max] -= 1
        for i, j in zip(waiting_to_trunc_idxs, range(len(lengths_waiting_to_trunc))):
            lengths[i] = lengths_waiting_to_trunc[j]
        for value in model_input_splited.values():
            for i in waiting_to_trunc_idxs:
                value[i] = value[i][: lengths[i]]

    @staticmethod
    def __pad(
        model_input: ModelInput,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        max_length: int
        # pad_to_multiple_of: Optional[int] = None,
    ) -> None:
        # if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        #     max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
        difference = max_length - len(model_input["input_ids"])

        if tokenizer.padding_side == "right":
            if "attention_mask" in model_input:
                model_input["attention_mask"].extend([0] * difference)
            if "token_type_ids" in model_input:
                model_input["token_type_ids"].extend([tokenizer.pad_token_type_id] * difference)
            if "special_tokens_mask" in model_input:
                model_input["special_tokens_mask"].extend([1] * difference)
            model_input["input_ids"].extend([tokenizer.pad_token_id] * difference)
        elif tokenizer.padding_side == "left":
            if "attention_mask" in model_input:
                model_input["attention_mask"] = [0] * difference + model_input["attention_mask"]
            if "token_type_ids" in model_input:
                model_input["token_type_ids"] = [tokenizer.pad_token_type_id] * difference + model_input["token_type_ids"]
            if "special_tokens_mask" in model_input:
                model_input["special_tokens_mask"] = [1] * difference + model_input["special_tokens_mask"]
            model_input["input_ids"] = [tokenizer.pad_token_id] * difference + model_input["input_ids"]
        else:
            raise ValueError("Invalid padding strategy:" + str(tokenizer.padding_side))

    @classmethod
    def __tokenize(cls, istrunc_texts: List[BoolStrings], tokenizer: PreTrainedTokenizer, max_length: int, desc: str, **kargs) -> BatchModelInput:
        # TODO: bug: token_type_ids全为0
        if "token_type_ids" in tokenizer.model_input_names:
            logger.warning(f"model input include 'token_type_ids'. There is a bug causing all the token_type_ids to be `0`")
        tokenized_dict = defaultdict(list)
        waiting_to_trunc_idxs = [idx for idx in range(len(istrunc_texts[0])) if istrunc_texts[0][idx][0]]
        for text in tqdm(istrunc_texts, desc=desc, colour="RED"):
            # text: BoolStrings = tuple[tuple[bool, str], ...]
            cur_dict: ModelInputSplited = defaultdict(list)
            origin_length = 0
            for _, text_part in text:
                cur_dict_ = tokenizer(text=text_part, padding=False, truncation=False, max_length=None, add_special_tokens=False, **kargs)
                origin_length += len(cur_dict_["input_ids"])
                for key, value in cur_dict_.items():
                    cur_dict[key].append(value)
            num_tokens_to_remove = origin_length - max_length
            if num_tokens_to_remove > 0:
                cls.__truncate(cur_dict, waiting_to_trunc_idxs, num_tokens_to_remove)
            cur_dict: ModelInput = {key: sum(value, []) for key, value in cur_dict.items()}
            # tokenizer.pad(cur_dict, padding="max_length", max_length=model_max_length)
            cls.__pad(cur_dict, tokenizer, max_length)
            for key, value in cur_dict.items():
                tokenized_dict[key].append(value)
        return tokenized_dict

    @staticmethod
    def transformers_tokenizer_tqdm(
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, text_pairs: List[TextPair], max_length: int, desc: str
    ) -> BatchModelInput:
        # print(text_pairs)
        batch_model_input = defaultdict(list)
        for text1, text2 in tqdm(text_pairs, desc=desc, colour="RED"):
            for key, value in tokenizer(text=text1, text_pair=text2, padding="max_length", truncation="longest_first", max_length=max_length).items():
                batch_model_input[key].append(value)
        return batch_model_input

    @staticmethod
    def stack_tensor_in_dicts(batch: list[dict]):
        ret_dict = dict()
        for key, value in batch[0].items():
            if isinstance(value, torch.Tensor):
                ret_dict[key] = torch.stack([it_dict[key] for it_dict in batch])
            elif isinstance(value, dict):
                ret_dict[key] = TextDataset.stack_tensor_in_dicts([it_dict[key] for it_dict in batch])
            else:
                raise Exception(f"Data type in batch must be Tensor or Dict, but got {type(batch[0][key])}")
        return ret_dict

    # TODO clip the label
    @staticmethod
    def collate_fn_padding_right(batch: list[dict]):
        # batch = MyDataset.stack_tensor_in_dicts(batch)
        batch = default_collate(batch)
        first_pad_index_input = batch.pop("first_pad_index_input")
        batch_max_length = torch.max(first_pad_index_input).item()
        model_input = batch.pop("model_input")
        batch.update({key: value[..., :batch_max_length] for key, value in model_input.items()})
        return batch

    @staticmethod
    def collate_fn_padding_left(batch: list[dict]):
        # batch = MyDataset.stack_tensor_in_dicts(batch)
        batch = default_collate(batch)
        first_not_pad_index_input = batch.pop("first_not_pad_index_input")
        min_start = torch.min(first_not_pad_index_input).item()
        model_input = batch.pop("model_input")
        batch.update({key: value[..., min_start:] for key, value in model_input.items()})
        return batch

    # # ? 递归改循环, 貌似对速度没影响?
    # @staticmethod
    # def stack_tensor_in_dicts(batch: list[dict]):
    #     ret_dict = dict()
    #     input_stack = [batch]
    #     output_stack = [ret_dict]
    #     while input_stack:
    #         batch = input_stack.pop()
    #         output = output_stack.pop()
    #         for key, value in batch[0].items():
    #             if isinstance(value, torch.Tensor):
    #                 output[key] = torch.stack([it_dict[key] for it_dict in batch])
    #             elif isinstance(value, dict):
    #                 input_stack.append([dict_it[key] for dict_it in batch])
    #                 output[key] = dict()
    #                 output_stack.append(output[key])
    #             else:
    #                 raise Exception(f"data type in batch must be Tensor or Dict, but got {type(batch[0][key])}")
    #     return ret_dict


# def cut_text(pair, max_word_num):
#     """Cutting a pair of texts"""
#     pair[0] = pair[0].split()
#     pair[1] = pair[1].split()
#     n = len(pair[0] + pair[1]) - max_word_num
#     if n > 0:
#         if (len(pair[0]) - len(pair[1])) > 0:
#             long, short = 0, 1
#         else:
#             long, short = 1, 0
#         dif = abs(len(pair[0]) - len(pair[1]))
#         if dif >= n:
#             pair[long] = pair[long][:-n]
#         else:
#             n = n - dif
#             pair[long] = pair[long][: -(n // 2 + dif)]
#             pair[short] = pair[short][: -(n - (n // 2))]
#     pair[0] = " ".join(pair[0])
#     pair[1] = " ".join(pair[1])
#     return pair


# Demo
# def myfun(file_path, tokenizer, model_type, **kargs):
#     special_tokens_map = tokenizer.special_tokens_map
#     BOS = special_tokens_map["bos_token"] if "bos_token" in special_tokens_map.keys() else None
#     EOS = special_tokens_map["eos_token"] if "eos_token" in special_tokens_map.keys() else None
#     SEP = special_tokens_map["sep_token"] if "sep_token" in special_tokens_map.keys() else None
#     MASK = special_tokens_map["mask_token"] if "mask_token" in special_tokens_map.keys() else None
#     CLS = special_tokens_map["cls_token"] if "cls_token" in special_tokens_map.keys() else None
#     sep_num = 1

#     with jsonlines.open(file_path, "r") as jlReader:
#         dict_objs = list(jlReader)
#         if isinstance(dict_objs[0], str):
#             dict_objs = dict_objs[1:]

#     input_texts = []
#     labels = []
#     for dict_obj in dict_objs:
#         # input_texts.append(f"{BOS}{dict_obj['question1']}{SEP*sep_num}{dict_obj['question2']}{EOS}")
#         input_texts.append(((False, CLS), (True, dict_obj["question1"]), (False, SEP), (True, dict_obj["question2"]), (False, SEP)))
#         labels.append([dict_obj["label"]])
#         # labels.append(((False, CLS), (True, dict_obj["question1"])))
#         # labels.append(dict_obj["question1"])
#     return input_texts, labels
