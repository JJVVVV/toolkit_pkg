import fcntl
import mmap
import pickle
import time
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Literal, Self, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, default_collate
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from .. import toolkit_logger

# from ..utils.misc import get_data_types
from ..enums import Split
from ..logger import _getLogger
from ..utils.misc import max_len_nest_list
from .config import NLPTrainingConfig

# import os
# from collections import defaultdict
# from math import ceil
# from types import NoneType
# from .. import file_logger_path

Tokens = List[int]
BatchTokens = List[Tokens]
ModelInput = Dict[str, Tokens | List[Tokens]]
ModelInputSplited = Dict[str, List[Tokens]]
BatchModelInput = Dict[str, BatchTokens]

ClassificationID = List[int]
INFINITE = 1000000000000000019884624838656
CACHE_DIR = "./.cache/dataset/"
logger = _getLogger("TextDataset")
# logger = getLogger("TextDataset", file_logger_path)


class PairedText:
    def __init__(self, first_text: str | Iterable[str], second_text: str | Iterable[str] | None = None) -> None:
        if second_text is not None:
            assert (isinstance(first_text, str) and isinstance(second_text, str)) or (
                isinstance(first_text, Iterable) and isinstance(second_text, Iterable)
            ), f"Different type for text pair: {type(first_text), type(second_text)}"
        self.first_text = first_text
        self.second_text = second_text

    def __getitem__(self, index):
        if index == 0:
            return self.first_text
        elif index == 1:
            return self.second_text
        else:
            raise IndexError(f"{type(self)} only have two item. Valid indexs are `0` and `1`, but got index `{index}`")

    def tolist(self):
        return [self.first_text, self.second_text] if self.second_text is not None else [self.first_text]

    def __repr__(self) -> str:
        return f"Pair 1️⃣: \n{self.first_text}\nPair 2️⃣: \n{self.second_text}" if self.second_text is not None else self.first_text


class FinelyControlledText:
    """
    Texts with a flag indicating whether it will be truncated.

    Example:
    ```
    a_sample = FinelyControlledText((False, CLS), (True, dict_obj["question1"]), (False, SEP), (True, dict_obj["question2"]), (False, SEP))
    ```
    `False` indicate the corresponding text can not be truncated.

    `True` indicate the corresponding text can be truncated if necessary.
    """

    def __init__(self, *texts: Tuple[bool, str]) -> None:
        self.texts = texts

    def __getitem__(self, index):
        return self.texts[index]

    def __len__(self):
        return len(self.texts)

    def __iter__(self):
        yield from self.texts


class ClassificationLabel(list):
    def __init__(self, *values: int):
        super().__init__(values)


class RegressionLabel(list):
    def __init__(self, *values: float):
        super().__init__(values)


# bug(已解决, 使用方案1): 缓存数据集存在问题, 当前版本直接在tokenize数据集时进行了truncation操作, 导致缓存的数据集是truncated后的
# 解决方案1: 将truncation操作分离出来, 每次加载数据集时在进行truncation
# 解放方案2: 将max_length, max_length_input, max_length_label加入到缓存路径中, 来区分不同的长度的缓存
class TextDataset(Dataset):
    """
    A demo of get_data_from_file:
    ```
    from pathlib import Path

    from toolkit.enums import Split
    from toolkit.nlp.data import ClassificationLabel, FinelyControlledText, PairedText, RegressionLabel
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


    def load_data_fn(data_file_path: Path | str, model_type: str, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, split: Split, **kwargs):
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
        customs = []
        for dict_obj in dict_objs:
            # Single
            a_sample = (dict_obj["question1"], None)
            a_sample = PairedText(dict_obj["question1"])

            # Pair
            a_sample = (dict_obj["question1"], dict_obj["question2"])
            a_sample = PairedText(dict_obj["question1"], dict_obj["question2"])
            a_sample = (
                [dict_obj["question1"], dict_obj["question1"], dict_obj["rephrase1"], dict_obj["rephrase1"]],
                [dict_obj["question2"], dict_obj["rephrase2"], dict_obj["question2"], dict_obj["rephrase2"]],
            )
            a_sample = PairedText(
                [dict_obj["question1"], dict_obj["question1"], dict_obj["rephrase1"], dict_obj["rephrase1"]],
                [dict_obj["question2"], dict_obj["rephrase2"], dict_obj["question2"], dict_obj["rephrase2"]],
            )

            # Finely controll
            a_sample = ((False, CLS), (True, dict_obj["question1"]), (False, SEP), (True, dict_obj["question2"]), (False, SEP))
            a_sample = FinelyControlledText((False, CLS), (True, dict_obj["question1"]), (False, SEP), (True, dict_obj["question2"]), (False, SEP))

            # label
            a_label = [dict_obj["label"]]  # List[int]
            a_label = ClassificationLabel(dict_obj["label"])  # ClassificationLabel
            a_label = [dict_obj["label"]]  # List[float]
            a_label = RegressionLabel(dict_obj["label"])  # RegressionLabel
            a_label = dict_obj["question1"]  # str
            a_label = PairedText(dict_obj["question1"])  # paired text
            a_label = FinelyControlledText((False, CLS), (True, dict_obj["question1"]))  # finely controlled text

            # custom
            a_custom_dict = {}
            a_custom_dict[arg_name1] = XXX
            a_custom_dict[arg_name2] = XXX

            inputs.append(a_sample)
            labels.append(a_label)
            customs.append(a_cumstom_dict)

        return inputs, labels, customs

    ```
    """

    def __init__(
        self,
        data_file_path: Path | str,
        task_type: str,
        model_structure: str,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        load_data_fn: Callable[
            [Path | str, str, PreTrainedTokenizer | PreTrainedTokenizerFast, Split],
            Tuple[List[FinelyControlledText] | list[PairedText], List[FinelyControlledText] | list[PairedText] | List[ClassificationID] | List[str]],
        ],
        split: Split | Literal["TRAINING", "VALIDATION", "TEST", "UNK"] = Split.UNK,
        padding_side: str = "right",
        **kwargs_load_data,
    ) -> None:
        super().__init__()
        local_rank = dist.get_rank() if dist.is_initialized() else 0
        if local_rank == 0:
            logger.debug(f"Model max length: {tokenizer.model_max_length if tokenizer.model_max_length != INFINITE else 'INFINITE'}")
        if not isinstance(split, Split):
            split = Split[split]
        assert model_structure in ("encoder-decoder", "encoder", "decoder"), f"`model_structure` invalid value: {model_structure}"
        assert task_type in ("generate", "classify", "regress"), f"`task_type` invalid value: {task_type}"
        # config padding settings
        assert padding_side in ("left", "right"), f"`padding_side={padding_side}` is invalid, only `left` and `right` are valid values"
        if padding_side == "right" and model_structure == "decoder":
            logger.warning(
                f"\n{'⚠️   '*10}\nDetect the `padding_side=right` and the `model_structure=decoder`, We strongly recommend that padding_side be set to `left` for decoder-only models.\n{'⚠️   '*10}"
            )
        self.model_structure = model_structure
        self.padding_side = padding_side
        self.task_type = task_type
        self.split = split
        self.inputkey2padid = {
            "input_ids": tokenizer.pad_token_id,
            "token_type_ids": tokenizer.pad_token_type_id,
            "attention_mask": 0,
            "special_tokens_mask": 1,
            "labels": -100,
        }
        assert tokenizer.pad_token_id is not None, "Pad token must be defined for batch training and inference."

        # get input and label texts
        self.texts_input, self.texts_label, *custom_args = load_data_fn(
            data_file_path=data_file_path, tokenizer=tokenizer, split=split, **kwargs_load_data
        )
        if len(custom_args) > 0:
            dicts_custom_inputs = custom_args[0]
            if dicts_custom_inputs:
                assert isinstance(dicts_custom_inputs, list) and isinstance(
                    dicts_custom_inputs[0], dict
                ), "Custom inputs of a sample must be a `Dict` and all `Dict` must in a `List`"
                self.dicts_custom_inputs: List[Dict] = dicts_custom_inputs

        # tokenize input texts
        if isinstance(self.texts_input[0], PairedText):  # if the input type is `PairedText`
            self.batch_model_input, self.dataset_max_length_input = self.transformers_tokenizer(
                tokenizer, self.texts_input, desc=f"Tokenize {split.name} input texts", is_label=False
            )
        # todo 关于 FinelyControlledText 代码未更新
        # elif isinstance(self.texts_input[0], FinelyControlledText):  # if the input type is `FinelyControlledText`
        #     self.batch_model_input = self.__tokenize(
        #         self.texts_input, tokenizer, tokenizer.model_max_length, desc=f"Tokenize {split.name} input texts"
        #     )
        else:
            raise ValueError("The input type must be `PairedText` or `FinelyControlledText`")

        # tokenize label texts
        # ! 不再使用 tokenizer 做 padding
        # tokenizer.padding_side = "right"
        self.truncate_pad_label = False  # 用于控制是否对label进行truncate和pad。只有生成任务的训练才需要设为 True
        self.custom_label = False
        if self.texts_label is None:
            self.tokens_labels = None
        else:
            if isinstance(self.texts_label[0], PairedText):  # if the label type is `PairedText`
                self.tokens_labels, self.dataset_max_length_label = self.transformers_tokenizer(
                    tokenizer, self.texts_label, desc=f"Tokenize {split.name} label texts", is_label=True
                )
                # self.tokens_labels = self.tokens_labels["input_ids"]
                self.truncate_pad_label = True
            # todo 关于 FinelyControlledText 代码未更新
            # elif isinstance(self.texts_label[0], FinelyControlledText):  # if the label type is `FinelyControlledText`
            #     self.tokens_labels = self.__tokenize(
            #         self.texts_label, tokenizer, tokenizer.model_max_length, desc=f"Tokenize {split.name} label texts"
            #     )["input_ids"]
            #     self.truncate_pad_label = True
            elif isinstance(self.texts_label[0], str):  # if the label type is  `str`
                self.tokens_labels = self.texts_label
                self.dataset_max_length_label = -1
            elif (isinstance(self.texts_label[0], list) and isinstance(self.texts_label[0][0], int)) or isinstance(
                self.texts_label[0], ClassificationLabel
            ):  # if the label type is `ClassificationID`, i.e. `List[int]`
                self.tokens_labels = torch.tensor(self.texts_label, dtype=torch.long)
                self.dataset_max_length_label = self.tokens_labels.shape[-1]
            elif (isinstance(self.texts_label[0], list) and isinstance(self.texts_label[0][0], float)) or isinstance(
                self.texts_label[0], RegressionLabel
            ):  # if the label type is `RegressionValue`, i.e. `List[float]`
                self.tokens_labels = torch.tensor(self.texts_label, dtype=torch.float32)
                self.dataset_max_length_label = self.tokens_labels.shape[-1]
            elif isinstance(self.texts_label[0], dict | list | tuple):
                logger.debug("Using custom labels ...")
                self.custom_label = True
                self.tokens_labels = self.texts_label
                self.dataset_max_length_label = -1
            else:
                raise ValueError(
                    (
                        "If the label is text, it must be `FinelyControlledText` or `PairedText` or `str`, "
                        "if the label is classification, it must be `ClassificationID (List[int])`",
                        "if the label is regression value, ti must be `RegressionValue (List[float])`",
                        "if the label is custom value, ti must be `dcit|list|tuple`",
                    )
                )

    def __getitem__(self, item: int) -> dict:
        ret_dict = dict()
        ret_dict["model_inputs"] = self.batch_model_input[item]
        ret_dict["labels"] = self.tokens_labels[item]
        if hasattr(self, "dicts_custom_inputs"):
            ret_dict["custom_inputs"] = self.dicts_custom_inputs[item]
        return ret_dict

    def __len__(self):
        return len(self.batch_model_input)

    def report(self):
        "Log some information of dataset (before being truncated)."
        toolkit_logger.info(f"Total {self.split.name} data: {len(self)}")
        toolkit_logger.info(f"Max length of input tokens: {self.dataset_max_length_input}")
        toolkit_logger.info(f"Max length of label tokens: {self.dataset_max_length_label}")

    def __truncate_one(self, l: list[int] | list[list], max_length: int):
        if not isinstance(l[0], list):
            return l[:max_length]
        return [self.__truncate_one(item, max_length) for item in l]

    def _truncate(
        self,
        tokens_inputs: List[ModelInput],
        tokens_labels: List[Tokens | ClassificationLabel | RegressionLabel | str],
        model_max_length: int,
        max_length: int | None = None,
        max_length_input: int | None = None,
        max_length_label: int | None = None,
    ) -> Tuple[List | int]:
        "返回的`max_length_input_after_trunc`是经过裁切后的<数据集>的最大长度, 不是<设置>的最大长度 !!!"
        "bug!!! 当decoder的generate任务时, 训练集中的inputs['input_ids']: list[list[int]]时无法裁切"
        cnt = 0
        max_length_input_after_trunc = 0
        max_length_label_after_trunc = 0
        # 对于decoder结构generate任务, 应该只通过`max_length`来控制裁切长度
        if self.task_type == "generate" and self.model_structure == "decoder":
            assert max_length_input is None and max_length_label is None, (
                "You should use `max_length` to control total truncation length instead of "
                f"using `max_length_input: {max_length_input}` and `max_length_label: {max_length_label}` to control the length of the input and label respectively."
            )
            max_length = max_length or model_max_length
            # 对于"decoder"的"generate"任务, 训练时需要对input和label进一步处理
            if self.split == Split.TRAINING:
                for idx, (inputs, labels) in enumerate(zip(tokens_inputs, tokens_labels)):
                    inputs_len = len(inputs["input_ids"])
                    inputs["input_ids"] = inputs["input_ids"] + labels
                    if len(inputs["input_ids"]) > max_length:
                        cnt += 1
                    inputs["input_ids"] = inputs["input_ids"][:max_length]
                    inputs["attention_mask"] = (inputs["attention_mask"] + [1] * len(labels))[:max_length]
                    tokens_labels[idx] = ([self.inputkey2padid["labels"]] * inputs_len + labels)[:max_length]
                    max_length_input_after_trunc = max_length_label_after_trunc = max(max_length_input_after_trunc, len(inputs["input_ids"]))
            else:
                for idx, (inputs, labels) in enumerate(zip(tokens_inputs, tokens_labels)):
                    if max_len_nest_list(inputs["input_ids"]) > max_length:
                        cnt += 1
                    for key in inputs.keys():
                        # inputs[key] = inputs[key][:max_length]
                        inputs[key] = self.__truncate_one(inputs[key], max_length)
                    if self.truncate_pad_label:
                        # labels = labels[:max_length]
                        labels = self.__truncate_one(labels, max_length)
                        tokens_labels[idx] = labels
                    max_length_input_after_trunc = max(max_length_input_after_trunc, max_len_nest_list(inputs["input_ids"]))
                    if self.truncate_pad_label:
                        max_length_label_after_trunc = max(max_length_label_after_trunc, max_len_nest_list(labels))
        else:
            max_length_input = max_length_input or model_max_length
            max_length_label = max_length_label or model_max_length
            for idx, (inputs, labels) in enumerate(zip(tokens_inputs, tokens_labels)):
                if max_len_nest_list(inputs["input_ids"]) > max_length_input:
                    cnt += 1
                for key in inputs.keys():
                    # inputs[key] = inputs[key][:max_length_input]
                    inputs[key] = self.__truncate_one(inputs[key], max_length_input)
                if self.truncate_pad_label:
                    # labels = labels[:max_length_label]
                    labels = self.__truncate_one(labels, max_length_label)
                    tokens_labels[idx] = labels
                max_length_input_after_trunc = max(max_length_input_after_trunc, max_len_nest_list(inputs["input_ids"]))
                if self.truncate_pad_label:
                    max_length_label_after_trunc = max(max_length_label_after_trunc, max_len_nest_list(labels))
        return tokens_inputs, tokens_labels, cnt, max_length_input_after_trunc, max_length_label_after_trunc or self.dataset_max_length_label

    def __pad_one(self, model_input: ModelInput, max_length: int):
        def helper(l: List, inputkey: str):
            nonlocal max_length
            if not isinstance(l[0], List):
                diff = max_length - len(l)
                # if diff <= 0:
                #     return l
                # 生成任务时，如果模型结构是encoder-decoder, 则labels的pad应该始终在右边 (而对于decoder模型，input和label的pad方向应一致, 因为模型最终的输入和标签都是input+label)
                if (inputkey == "labels" and self.model_structure == "encoder-decoder") or self.padding_side == "right":
                    return l + [self.inputkey2padid[inputkey]] * diff
                else:
                    return [self.inputkey2padid[inputkey]] * diff + l
            ret = []
            for l_ in l:
                ret.append(helper(l_, inputkey))
            return ret

        # return {key: helper(value, key) for key, value in model_input.items()}
        return {key: torch.tensor(helper(value, key), dtype=torch.long) for key, value in model_input.items()}

    def __pad_batch(self, batch: list[ModelInput], max_length: int | None = None):
        """
        如果 max_length = None, 则 pad 到 batch 内样本的最大长度
        """
        if max_length is None:
            if "input_ids" in batch[0]:
                max_length = max_len_nest_list([sample["input_ids"] for sample in batch])
            else:
                max_length = max_len_nest_list([sample["labels"] for sample in batch])
        return [self.__pad_one(model_input, max_length) for model_input in batch]

    def __pad_one_return_list(self, model_input: ModelInput, max_length: int):
        def helper(l: List, inputkey: str):
            nonlocal max_length
            if not isinstance(l[0], List):
                diff = max_length - len(l)
                # if diff <= 0:
                #     return l
                # 生成任务时，如果模型结构是encoder-decoder, 则labels的pad应该始终在右边 (而对于decoder模型，input和label的pad方向应一致, 因为模型最终的输入和标签都是input+label)
                if (inputkey == "labels" and self.model_structure == "encoder-decoder") or self.padding_side == "right":
                    return l + [self.inputkey2padid[inputkey]] * diff
                else:
                    return [self.inputkey2padid[inputkey]] * diff + l
            ret = []
            for l_ in l:
                ret.append(helper(l_, inputkey))
            return ret

        return {key: helper(value, key) for key, value in model_input.items()}

    def __pad_batch_return_list(self, batch: list[ModelInput], max_length: int | None = None):
        """
        如果 max_length = None, 则 pad 到 batch 内样本的最大长度
        """
        if max_length is None:
            if "input_ids" in batch[0]:
                max_length = max_len_nest_list([sample["input_ids"] for sample in batch])
            else:
                max_length = max_len_nest_list([sample["labels"] for sample in batch])
        return [self.__pad_one_return_list(model_input, max_length) for model_input in batch]

    # @staticmethod
    def transformers_tokenizer(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        text_pairs: List[PairedText],
        desc: str,
        is_label: bool,
        progress_bar: bool = True,
    ) -> Tuple[List[ModelInput | Tokens], int]:
        # // """此函数同时完成truncation操作"""
        # print(text_pairs)
        batch_model_input = []
        longest = 0
        iterator = tqdm(text_pairs, desc=desc, colour="RED", smoothing=0.99) if progress_bar else text_pairs
        for text1, text2 in iterator:
            # # 为 decoder-only 模型的 label 加上 eos
            # if is_label and self.model_structure == "decoder":
            #     text1 += tokenizer.eos_token
            # * 为了当一个样本是一个列表时，能允许列表中部分是单个输入，部分是成对输入（即text_pair=None）
            if not isinstance(text1, str):
                a_sample = dict()
                for t1, t2 in zip(text1, text2):
                    part_a_sample = tokenizer(
                        text=t1,
                        text_pair=t2,
                        padding=False,
                        truncation=False,
                        max_length=None,
                        add_special_tokens=not (is_label and self.model_structure == "decoder"),
                        # add_special_tokens=self.model_structure != "decoder",
                        return_attention_mask=not is_label,
                    )
                    for k, v in part_a_sample.items():
                        if k in a_sample:
                            a_sample[k].append(v)
                        else:
                            a_sample[k] = [v]
            else:
                a_sample = tokenizer(
                    text=text1,
                    text_pair=text2,
                    padding=False,
                    truncation=False,
                    max_length=None,
                    add_special_tokens=not (is_label and self.model_structure == "decoder"),
                    # add_special_tokens=self.model_structure != "decoder",
                    return_attention_mask=not is_label,
                )
            if is_label:
                batch_model_input.append(a_sample["input_ids"])
            else:
                batch_model_input.append(a_sample)
            input_ids = a_sample["input_ids"]
            # if isinstance(input_ids[0], list):
            #     longest = max(longest, max([len(value_) for value_ in input_ids]))
            # else:
            #     longest = max(longest, len(input_ids))
            longest = max(longest, max_len_nest_list(input_ids))
        return batch_model_input, longest

    def collate_fn(self, batch: list[dict]):
        batch_model_inputs = self.__pad_batch([item.pop("model_inputs") for item in batch], None)
        # import pdb; pdb.set_trace()
        ret: dict = default_collate(batch_model_inputs)
        if self.truncate_pad_label:
            batch_labels = self.__pad_batch([{"labels": item.pop("labels")} for item in batch], None)
            ret.update(default_collate(batch_labels))
        if self.custom_label:
            batch_labels = [item.pop("labels") for item in batch]
            ret["labels"] = batch_labels
        ret.update(default_collate(batch))
        # import pdb; pdb.set_trace()
        # print(ret["input_ids"].shape)
        return ret

    @classmethod
    def from_file(
        cls,
        *,
        data_file_path: Path | str | None = None,
        task_type: str | None = None,
        model_structure: str | None = None,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        load_data_fn: Callable[
            [str, str, PreTrainedTokenizer | PreTrainedTokenizerFast, bool],
            Tuple[List[FinelyControlledText] | list[PairedText], List[FinelyControlledText] | list[PairedText] | List[ClassificationID]],
        ],
        split: Split | Literal["TRAINING", "VALIDATION", "TEST", "UNK"],
        padding_side: str | None = None,
        use_cache: bool | None = None,
        max_length: int | None = None,
        max_length_input: int | None = None,
        max_length_label: int | None = None,
        padding_to_max_length: bool | None = None,
        configs: NLPTrainingConfig | None = None,
        **kwargs_load_data,
    ) -> Self | None:
        """
        Load dataset from file with the given `NLPTrainingConfig`.
        """
        local_rank = dist.get_rank() if dist.is_initialized() else 0

        if not isinstance(split, Split):
            split = Split[split]

        if configs is None:
            configs = NLPTrainingConfig(
                task_type=task_type,
                cache_dataset=use_cache,
                model_structure=model_structure,
                padding_side=padding_side,
                max_length=max_length,
                max_length_input=max_length_input,
                max_length_label=max_length_label,
                padding_to_max_length=padding_to_max_length,
            )

        # 如果未指定 data_file_path, 则尝试根据 split 从 configs 中找对应的文件
        if data_file_path is None:
            match split:
                case Split.TRAINING:
                    data_file_path = configs.train_file_path
                case Split.VALIDATION:
                    data_file_path = configs.val_file_path
                case Split.TEST:
                    data_file_path = configs.test_file_path

        if data_file_path is None:
            if local_rank == 0:
                logger.warning(f"⚠️  Fail to load {split.name} data. The data file path is not specified (received `NoneType`).")
            return None
            # raise TypeError(f"❌ Fail to load {split.name} data. The data file path is not specified (received `NoneType`).")
        if isinstance(data_file_path, str):
            data_file_path = Path(data_file_path)
        if isinstance(data_file_path, Path) and not data_file_path.exists():
            if local_rank == 0:
                raise FileNotFoundError(f"❌ Fail to load test data. {data_file_path} does not exists.")

        start = time.time()
        if local_rank == 0:
            logger.debug(f"⏳ Loading {split.name} dataset ...")
        if use_cache is None:  # use_cache will cover the config.cache_dataset
            use_cache = configs.cache_dataset
        if use_cache:
            dataset = cls.from_cache(data_file_path, repr(tokenizer.__class__)[8:-2], **kwargs_load_data)
        else:
            dataset = None
        if dataset is None:
            dataset = cls(
                data_file_path=data_file_path,
                task_type=configs.task_type,
                model_structure=configs.model_structure,
                tokenizer=tokenizer,
                load_data_fn=load_data_fn,
                split=split,
                padding_side=configs.padding_side,
                **kwargs_load_data,
            )
            if use_cache:
                try:
                    dataset.cache(data_file_path, repr(tokenizer.__class__)[8:-2], **kwargs_load_data)
                except:
                    logger.error(f"❌ Fail to cache dataset!")

        end = time.time()
        if local_rank == 0:
            logger.debug(f"⌛ Loading {split.name} data takes {end - start:.2f} sec.")
            cls.report(dataset)

        # truncate dataset
        _, _, cnt, max_length_input_after_trunc, max_length_label_after_trunc = dataset._truncate(
            dataset.batch_model_input,
            dataset.tokens_labels,
            tokenizer.model_max_length,
            configs.max_length,
            configs.max_length_input,
            configs.max_length_label,
        )
        if local_rank == 0:
            toolkit_logger.info(
                f"✂️  Truncating {split.name} data: cnt={cnt}, input_len={max_length_input_after_trunc}, label_len={max_length_label_after_trunc}."
            )
        dataset.max_length_input_after_trunc = max_length_input_after_trunc
        dataset.max_length_label_after_trunc = max_length_label_after_trunc

        # pad dataset
        if configs.padding_to_max_length:
            dataset.batch_model_input = dataset.__pad_batch_return_list(dataset.batch_model_input, dataset.max_length_input_after_trunc)
            if dataset.truncate_pad_label:
                dataset.tokens_labels = dataset.__pad_batch_return_list(dataset.tokens_labels, dataset.max_length_label_after_trunc)
            if local_rank == 0:
                toolkit_logger.info(f"🧷 Padding {split.name} to max length of dataset.")

        # ? 此段代码只是为了测试固定长度的输入所需的显存(为了项目 memcal), 正常训练无需设置这个参数, 因为除了浪费算力外没有任何意义
        if hasattr(configs, "padding_to_configed_max_length") and configs.padding_to_configed_max_length:
            dataset.max_length_input_after_trunc = configs.max_length_input or configs.max_length
            dataset.max_length_label_after_trunc = configs.max_length_label or configs.max_length
            print(dataset.max_length_input_after_trunc)
            print(dataset.max_length_label_after_trunc)
            dataset.batch_model_input = dataset.__pad_batch_return_list(dataset.batch_model_input, dataset.max_length_input_after_trunc)
            if dataset.truncate_pad_label:
                dataset.tokens_labels = dataset.__pad_batch_return_list(dataset.tokens_labels, dataset.max_length_label_after_trunc)
            if local_rank == 0:
                toolkit_logger.info(f"🧷 Padding {split.name} to the config length.")

        return dataset

    @staticmethod
    def cache_path(origin_data_path: Path, tokenizer_type: str, **kwargs_load_data) -> Path:
        "Convert the original data file path to a path where dataset will be cached in."
        absolute_path_data = origin_data_path.resolve()
        resolved_kwargs_list = []
        for key, value in kwargs_load_data.items():
            if issubclass(value.__class__, Enum):
                value = value.name
            resolved_kwargs_list.append(f"{key}={value}")
        cache_path = Path(
            CACHE_DIR,
            tokenizer_type,
            "/".join(resolved_kwargs_list),
            str(absolute_path_data)[1:] if str(absolute_path_data).startswith("/") else str(absolute_path_data),
        )
        cache_path = cache_path.with_suffix(".pkl")
        return cache_path

    def cache(self, origin_data_path: Path, tokenizer_type: str = "unknown_tokenizer", **kwargs_load_data):
        "Cache tokenized dataset. Compatible with DDP and deepspeed."
        local_rank = dist.get_rank() if dist.is_initialized() else 0
        if local_rank == 0:
            logger.debug(f"💿 Caching dataset from {origin_data_path} ...")
            cache_path = self.cache_path(origin_data_path, tokenizer_type, **kwargs_load_data)
            logger.debug(f"❔ Cache file will be saved in {cache_path}")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            # if cache_path.exists():
            #     return
            try:
                f = cache_path.open("wb")
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                data = pickle.dumps(self)
                f.write(data)
                f.flush()
                fcntl.flock(f, fcntl.LOCK_UN)
                logger.debug("✔️  Cache successfully.")
            except IOError:
                logger.debug("⚠️ Skip this operation because other programs are writing files ...")
            finally:
                f.close()

    @classmethod
    def from_cache(cls, cached_dataset_or_origin_data_path: Path | str, tokenizer_type: str = "unknown_tokenizer", **kwargs_load_data) -> Self | None:
        "Try to load dataset from cache. If there if no cache, `None` will be returned."
        local_rank = dist.get_rank() if dist.is_initialized() else 0
        cached_dataset_or_origin_data_path = Path(cached_dataset_or_origin_data_path)
        if cached_dataset_or_origin_data_path.suffix == ".pkl":
            cached_dataset_path = cached_dataset_or_origin_data_path
            logger.warning("You are loading data from a `.pkl` file, the file will be considered as a dataset cache file.")
            logger.warning("If you are sure that the file is a raw data file, rename it to a non-PKL suffix.")
        else:
            cached_dataset_path = cls.cache_path(cached_dataset_or_origin_data_path, tokenizer_type, **kwargs_load_data)
            # print(cached_dataset_path)
        if not cached_dataset_path.exists():
            if local_rank == 0:
                logger.debug("❕ There is no cache.")
            return None
        try:
            f = cached_dataset_path.open("rb")
            if local_rank == 0:
                logger.debug(f"🔒 Applying for read lock ...")
            fcntl.flock(f, fcntl.LOCK_SH)
            if local_rank == 0:
                logger.debug(f"💿 Loading dataset from cache ...")
            dataset = pickle.load(f)
            fcntl.flock(f, fcntl.LOCK_UN)
            if local_rank == 0:
                logger.debug("✔️  Load successfully.")
        except IOError:
            if local_rank == 0:
                logger.debug("⚠️ Fail to load cache. Maybe the file is being written.")
            dataset = None
        except Exception as e:
            if local_rank == 0:
                logger.debug("⚠️ Fail to load cache.")
                logger.debug(str(e))
            dataset = None
        finally:
            f.close()
        return dataset

    # @staticmethod
    # def __truncate(model_input_splited: ModelInputSplited, waiting_to_trunc_idxs: list[int], num_tokens_to_remove: int) -> None:
    #     lengths = [len(value_part) for value_part in model_input_splited["input_ids"]]
    #     lengths_waiting_to_trunc = [lengths[i] for i in waiting_to_trunc_idxs]
    #     if (all_tokens := sum(lengths_waiting_to_trunc)) < num_tokens_to_remove:
    #         raise ValueError(
    #             f"The number of tokens need to be truncated is greater than the total number of tokens: {num_tokens_to_remove}>{all_tokens}"
    #         )
    #     for _ in range(num_tokens_to_remove):
    #         index_max = lengths_waiting_to_trunc.index(max(lengths_waiting_to_trunc))
    #         lengths_waiting_to_trunc[index_max] -= 1
    #     for i, j in zip(waiting_to_trunc_idxs, range(len(lengths_waiting_to_trunc))):
    #         lengths[i] = lengths_waiting_to_trunc[j]
    #     for value in model_input_splited.values():
    #         for i in waiting_to_trunc_idxs:
    #             value[i] = value[i][: lengths[i]]

    # @classmethod
    # def __tokenize(
    #     cls, finely_controlled_text_list: List[FinelyControlledText], tokenizer: PreTrainedTokenizer, max_length: int, desc: str, **kargs
    # ) -> BatchModelInput:
    #     # TODO: 新版本的data.py暂未适配 self.__tokenize
    #     raise NotImplemented
    #     # TODO: bug: token_type_ids全为0
    #     # TODO: bug: 对于可接受无限长输入的模型, 因为其没有 max length, 因此无法pad
    #     if max_length == INFINITE:
    #         raise NotImplementedError("TODO")

    #     if "token_type_ids" in tokenizer.model_input_names:
    #         logger.warning(f" model input include 'token_type_ids'. There is a bug causing all the token_type_ids to be `0`")
    #     tokenized_dict = defaultdict(list)
    #     waiting_to_trunc_idxs = [idx for idx in range(len(finely_controlled_text_list[0])) if finely_controlled_text_list[0][idx][0]]
    #     for finely_controlled_text in tqdm(finely_controlled_text_list, desc=desc, colour="RED", smoothing=0.99):
    #         # text: FinelyControlledText = tuple[tuple[bool, str], ...]
    #         cur_dict: ModelInputSplited = defaultdict(list)
    #         origin_length = 0
    #         for _, part_text in finely_controlled_text:
    #             cur_dict_ = tokenizer(text=part_text, padding=False, truncation=False, max_length=None, add_special_tokens=False, **kargs)
    #             origin_length += len(cur_dict_["input_ids"])
    #             for key, value in cur_dict_.items():
    #                 cur_dict[key].append(value)
    #         num_tokens_to_remove = (origin_length - max_length) if max_length != INFINITE else 0
    #         if num_tokens_to_remove > 0:
    #             cls.__truncate(cur_dict, waiting_to_trunc_idxs, num_tokens_to_remove)
    #         cur_dict: ModelInput = {key: sum(value, []) for key, value in cur_dict.items()}
    #         # tokenizer.pad(cur_dict, padding="max_length", max_length=model_max_length)
    #         cls.__pad_one(cur_dict, tokenizer, max_length)
    #         for key, value in cur_dict.items():
    #             tokenized_dict[key].append(value)
    #     return tokenized_dict


# todo 添加控制是否启用 offsets cache 的参数
class LazyTextDataset(TextDataset):
    def __init__(
        self,
        data_file_path,
        task_type,
        model_structure,
        tokenizer,
        parse_item_fn,
        build_offsets_fn=None,
        split=Split.UNK,
        padding_side="right",
        max_length: int | None = None,
        max_length_input: int | None = None,
        max_length_label: int | None = None,
        **kwargs_parse_item,
    ):
        """
        parse_item_fn: 用于解析 item, 例如从中解析出 input 和 label.
        build_offsets_fn: 用于划分 item, 默认情况下按行划分.
        """
        self.model_structure = model_structure
        self.padding_side = padding_side
        self.task_type = task_type
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_length_input = max_length_input
        self.max_length_label = max_length_label
        self.kwargs_parse_item = kwargs_parse_item
        self.inputkey2padid = {
            "input_ids": tokenizer.pad_token_id,
            "token_type_ids": tokenizer.pad_token_type_id,
            "attention_mask": 0,
            "special_tokens_mask": 1,
            "labels": -100,
        }

        self.file_path = data_file_path
        self.file = open(data_file_path, "rb")
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

        # 尝试加载预存的偏移量，否则重新计算
        self._build_offsets = build_offsets_fn or self._build_offsets
        offsets_path = Path(data_file_path).with_suffix(".offsets.npy")
        if offsets_path.exists():
            self.offsets = np.load(offsets_path).tolist()
        else:
            self.offsets = self._build_offsets()
            np.save(offsets_path, np.array(self.offsets))
        self.parse_item_fn = parse_item_fn
        self.dataset_max_length_input = -1  # 由于不直接读取全部数据，该值无法统计
        self.dataset_max_length_label = -1  # 由于不直接读取全部数据，该值无法统计

    def __del__(self):
        self.mm.close()
        self.file.close()

    def _build_offsets(self):
        offsets = []
        self.mm.seek(0)
        while True:
            pos = self.mm.tell()
            line = self.mm.readline()
            if not line:  # 遇到EOF时退出循环
                break
            # if line.strip():  # 可选：忽略纯空白行（如只有\n）
            offsets.append(pos)
        return offsets

    def __getitem__(self, idx):
        ret_dict = dict()
        self.mm.seek(self.offsets[idx])
        item = self.mm.readline().decode("utf-8").strip()
        ori_input, ori_label, *custom_args = self.parse_item_fn(item, self.tokenizer, self.split, **self.kwargs_parse_item)

        # custom inputs
        if len(custom_args) > 0:
            dicts_custom_inputs: Dict = custom_args[0]
            assert isinstance(dicts_custom_inputs, dict), "Custom inputs of a sample must be a `Dict`."
            ret_dict["custom_inputs"] = dicts_custom_inputs

        # tokenize input texts
        if isinstance(ori_input, PairedText):  # if the input type is `PairedText`
            batch_model_input, num_tokens = self.transformers_tokenizer(self.tokenizer, (ori_input,), desc=None, is_label=False, progress_bar=False)
            ret_dict["model_inputs"] = batch_model_input[0]
        else:
            raise ValueError("The input type must be `PairedText` or `FinelyControlledText`")

        # tokenize label texts
        self.truncate_pad_label = False  # 用于控制是否对label进行truncate和pad。只有生成任务的训练才需要设为 True
        self.custom_label = False
        if isinstance(ori_label, PairedText):  # if the label type is `PairedText`
            batch_input_ids, num_tokens = self.transformers_tokenizer(self.tokenizer, (ori_label,), desc=None, is_label=True, progress_bar=False)
            ret_dict["labels"] = batch_input_ids[0]
            self.truncate_pad_label = True
        elif isinstance(ori_label, str):  # if the label type is  `str`
            ret_dict["labels"] = ori_label
        elif (isinstance(ori_label, list) and isinstance(ori_label[0], int)) or isinstance(
            ori_label, ClassificationLabel
        ):  # if the label type is `ClassificationID`, i.e. `List[int]`
            ret_dict["labels"] = torch.tensor(ori_label, dtype=torch.long)
        elif (isinstance(ori_label, list) and isinstance(ori_label[0], float)) or isinstance(
            ori_label, RegressionLabel
        ):  # if the label type is `RegressionValue`, i.e. `List[float]`
            ret_dict["labels"] = torch.tensor(ori_label, dtype=torch.float32)
        elif isinstance(ori_label, dict | list | tuple):
            logger.debug("Using custom labels ...")
            self.custom_label = True
            ret_dict["labels"] = ori_label
        else:
            raise ValueError(
                (
                    "If the label is text, it must be `FinelyControlledText` or `PairedText` or `str`, "
                    "if the label is classification, it must be `ClassificationID (List[int])`",
                    "if the label is regression value, ti must be `RegressionValue (List[float])`",
                    "if the label is custom value, ti must be `dcit|list|tuple`",
                )
            )

        # model_inputs_truncated, labels_truncated, _, _, _ = self._truncate(
        #     [ret_dict["model_inputs"]],
        #     [ret_dict["labels"]],
        #     self.tokenizer.model_max_length,
        #     self.max_length,
        #     self.max_length_input,
        #     self.max_length_label,
        # )
        # ret_dict["model_inputs"], ret_dict["labels"] = model_inputs_truncated[0], labels_truncated[0]

        return ret_dict

    def __len__(self):
        return len(self.offsets)

    def collate_fn(self, batch):
        list_model_inputs, list_labels_tokens, _, _, _ = self._truncate(
            [item["model_inputs"] for item in batch],
            [item["labels"] for item in batch],
            self.tokenizer.model_max_length,
            self.max_length,
            self.max_length_input,
            self.max_length_label,
        )
        for item, model_input, label in zip(batch, list_model_inputs, list_labels_tokens):
            item["model_inputs"] = model_input
            item["labels"] = label
        return super().collate_fn(batch)

    @classmethod
    def from_file(
        cls,
        *,
        data_file_path: Path | str | None = None,
        task_type: str | None = None,
        model_structure: str | None = None,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        parse_item_fn,
        build_offsets_fn=None,
        split: Split | Literal["TRAINING", "VALIDATION", "TEST", "UNK"],
        padding_side: str | None = None,
        use_cache: bool | None = None,
        max_length: int | None = None,
        max_length_input: int | None = None,
        max_length_label: int | None = None,
        padding_to_max_length: bool | None = None,
        configs: NLPTrainingConfig | None = None,
        **kwargs_parse_item,
    ) -> Self | None:
        """
        Load dataset from file with the given `NLPTrainingConfig`.
        """
        local_rank = dist.get_rank() if dist.is_initialized() else 0

        if not isinstance(split, Split):
            split = Split[split]

        if configs is None:
            configs = NLPTrainingConfig(
                task_type=task_type,
                cache_dataset=use_cache,
                model_structure=model_structure,
                padding_side=padding_side,
                max_length=max_length,
                max_length_input=max_length_input,
                max_length_label=max_length_label,
                padding_to_max_length=padding_to_max_length,
            )

        # 如果未指定 data_file_path, 则尝试根据 split 从 configs 中找对应的文件
        if data_file_path is None:
            match split:
                case Split.TRAINING:
                    data_file_path = configs.train_file_path
                case Split.VALIDATION:
                    data_file_path = configs.val_file_path
                case Split.TEST:
                    data_file_path = configs.test_file_path

        if data_file_path is None:
            if local_rank == 0:
                logger.warning(f"⚠️  Fail to load {split.name} data. The data file path is not specified (received `NoneType`).")
            return None
            # raise TypeError(f"❌ Fail to load {split.name} data. The data file path is not specified (received `NoneType`).")
        if isinstance(data_file_path, str):
            data_file_path = Path(data_file_path)
        if isinstance(data_file_path, Path) and not data_file_path.exists():
            if local_rank == 0:
                raise FileNotFoundError(f"❌ Fail to load test data. {data_file_path} does not exists.")

        start = time.time()
        if local_rank == 0:
            logger.debug(f"⏳ Loading {split.name} dataset ...")

        dataset = cls(
            data_file_path=data_file_path,
            task_type=configs.task_type,
            model_structure=configs.model_structure,
            tokenizer=tokenizer,
            parse_item_fn=parse_item_fn,
            build_offsets_fn=build_offsets_fn,
            split=split,
            padding_side=configs.padding_side,
            max_length=configs.max_length,
            max_length_input=configs.max_length_input,
            max_length_label=configs.max_length_label,
            **kwargs_parse_item,
        )
        end = time.time()
        if local_rank == 0:
            logger.debug(f"⌛ Loading {split.name} data takes {end - start:.2f} sec.")
            cls.report(dataset)

        return dataset


def show_model_inputs_case(dataset, tokenizer, is_decode_label=True):
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    YELLOW = "\033[33m"
    ORANGE_256 = "\033[38;5;208m"
    RESET = "\033[0m"  # 重置为默认颜色
    print(f"{'-'*100}\n{BOLD + YELLOW}### Special tokens map:{RESET} \n{tokenizer.special_tokens_map}\n{'-'*100}")
    dataloader = DataLoader(dataset, collate_fn=dataset.collate_fn)
    a_batch = next(iter(dataloader))
    print(f"{BOLD + RED}### Token ids:{RESET} \n{a_batch}\n{'-'*100}")
    print(f"{BOLD + GREEN}### Decoded input ids:{RESET} \n{tokenizer.batch_decode(a_batch['input_ids'], skip_special_tokens=False)[0]}\n{'-'*100}")
    if is_decode_label:
        a_batch["labels"] = torch.where(a_batch["labels"] != -100, a_batch["labels"], tokenizer.pad_token_id)
        print(f"{BOLD + BLUE}### Decoded label ids:{RESET} \n{tokenizer.batch_decode(a_batch['labels'], skip_special_tokens=False)[0]}\n{'-'*100}")


# show_model_inputs_case(dataset, tokenizer)
