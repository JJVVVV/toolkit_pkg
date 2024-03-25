import fcntl
import os
import pickle
import time
from collections import defaultdict
from enum import Enum
from math import ceil
from pathlib import Path
from types import NoneType
from typing import Callable, Dict, Iterable, List, Self, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, default_collate
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

# from ..utils.misc import get_data_types
from ..enums import Split
from ..logger import _getLogger
from ..utils.misc import max_len_nest_list
from .config import NLPTrainingConfig

Tokens = List[int]
BatchTokens = List[Tokens]
ModelInput = Dict[str, Tokens | List[Tokens]]
ModelInputSplited = Dict[str, List[Tokens]]
BatchModelInput = Dict[str, BatchTokens]

ClassificationID = List[int]
INFINITE = 1000000000000000019884624838656
CACHE_DIR = "./.cache/dataset/"
logger = _getLogger("TextDataset")


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

    def __str__(self) -> str:
        return str(self.tolist())


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


# bug: 缓存数据集存在问题, 当前版本直接在tokenize数据集时进行了truncation操作, 导致缓存的数据集是truncated后的
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
        model_type: str,
        model_structure: str,
        task_type: str,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        load_data_fn: Callable[
            [Path | str, str, PreTrainedTokenizer | PreTrainedTokenizerFast, Split],
            Tuple[List[FinelyControlledText] | list[PairedText], List[FinelyControlledText] | list[PairedText] | List[ClassificationID] | List[str]],
        ],
        padding_side: str = "right",
        max_length: int | None = None,
        max_length_input: int | None = None,
        max_length_label: int | None = None,
        padding_to_max_length: bool = False,
        split: Split = Split.ANY,
        **kwargs_load_data,
    ) -> None:
        """
        max_length: int | None = None, 限制整体最大长度, 该参数对于encoder结构的generate任务尤为有用.
        max_length_input: int | None = None, 限制输入的最大长度, 用于truncate.
        max_length_label: int | None = None, 限制标签的最大长度, 用于truncate.
        padding_to_max_length: bool = False, 是否padding到整个数据集的最大长度(*此处数据集指的是truncated后的), 即actual_max_length_input.
        """
        super().__init__()
        local_rank = dist.get_rank() if dist.is_initialized() else 0
        if local_rank == 0:
            logger.debug(f"Model max length: {tokenizer.model_max_length if tokenizer.model_input_names != INFINITE else 'INFINITE'}")
        assert model_structure in ("encoder-decoder", "encoder", "decoder"), f"`model_structure` invalid value: {model_structure}"
        self.model_structure = model_structure
        assert task_type in ("generate", "classify", "regress"), f"`task_type` invalid value: {task_type}"
        self.task_type = task_type
        # 对于decoder结构generate任务, 应该只通过`max_length`来控制裁切长度
        if self.task_type == "generate" and self.model_structure == "decoder":
            assert max_length_input is None and max_length_label is None, (
                "You should use `max_length` to control total truncation length instead of "
                f"using `max_length_input: {max_length_input}` and `max_length_label: {max_length_label}` to control the length of the input and label respectively."
            )
        max_length_input = tokenizer.model_max_length if max_length_input is None else max_length_input
        max_length_label = tokenizer.model_max_length if max_length_label is None else max_length_label
        max_length = tokenizer.model_max_length if max_length is None else max_length
        self.padding_to_max_length = padding_to_max_length
        self.split = split
        # self.tokenizer = tokenizer
        self.inputkey2padid = {
            "input_ids": tokenizer.pad_token_id,
            "token_type_ids": tokenizer.pad_token_type_id,
            "attention_mask": 0,
            "special_tokens_mask": 1,
            "labels": -100,
        }

        # get input and label texts
        assert padding_side in ("left", "right"), f"`padding_side={padding_side}` is not understood, only `left` and `right` are valid values"
        self.padding_side = padding_side
        self.texts_input, self.texts_label, *custom_args = load_data_fn(
            data_file_path=data_file_path, model_type=model_type, tokenizer=tokenizer, split=split, **kwargs_load_data
        )
        if len(custom_args) > 0:
            dicts_custom_inputs = custom_args[0]
            if dicts_custom_inputs:
                assert isinstance(dicts_custom_inputs, list) and isinstance(
                    dicts_custom_inputs[0], dict
                ), "Custom inputs of a sample must be a `Dict` and all `Dict` must in a `List`"
                self.dicts_custom_inputs: List[Dict] = dicts_custom_inputs

        # tokenize input texts
        tokenizer.padding_side = padding_side
        if isinstance(self.texts_input[0], PairedText):  # if the input type is `PairedText`
            self.batch_model_input, self.actual_max_length_input = self.transformers_tokenizer_tqdm(
                tokenizer, self.texts_input, max_length_input, desc=f"Tokenize {split.name} input texts", is_label=False
            )
        elif isinstance(self.texts_input[0], FinelyControlledText):  # if the input type is `FinelyControlledText`
            self.batch_model_input = self.__tokenize(self.texts_input, tokenizer, max_length_input, desc=f"Tokenize {split.name} input texts")
        else:
            raise ValueError("The input type must be `PairedText` or `FinelyControlledText`")

        # tokenize label texts
        self.padding_label = False
        tokenizer.padding_side = "right"
        if isinstance(self.texts_label[0], PairedText):  # if the label type is `PairedText`
            self.tokens_labels, self.actual_max_length_label = self.transformers_tokenizer_tqdm(
                tokenizer, self.texts_label, max_length_label, desc=f"Tokenize {split.name} label texts", is_label=True
            )
            # self.tokens_labels = self.tokens_labels["input_ids"]
            self.padding_label = True
        elif isinstance(self.texts_label[0], FinelyControlledText):  # if the label type is `FinelyControlledText`
            self.tokens_labels = self.__tokenize(self.texts_label, tokenizer, max_length_label, desc=f"Tokenize {split.name} label texts")[
                "input_ids"
            ]
            self.padding_label = True

        elif (isinstance(self.texts_label[0], list) and isinstance(self.texts_label[0][0], str)) or isinstance(
            self.texts_label[0], str
        ):  # if the label type is `List[str]` or `str`
            self.tokens_labels = self.texts_label
            self.actual_max_length_label = -1
        elif (isinstance(self.texts_label[0], list) and isinstance(self.texts_label[0][0], int)) or isinstance(
            self.texts_label[0], ClassificationLabel
        ):  # if the label type is `ClassificationID`, i.e. `List[int]`
            self.tokens_labels = torch.tensor(self.texts_label, dtype=torch.long)
            self.actual_max_length_label = self.tokens_labels.shape[-1]
        elif (isinstance(self.texts_label[0], list) and isinstance(self.texts_label[0][0], float)) or isinstance(
            self.texts_label[0], RegressionLabel
        ):  # if the label type is `RegressionValue`, i.e. `List[float]`
            self.tokens_labels = torch.tensor(self.texts_label, dtype=torch.float32)
            self.actual_max_length_label = self.tokens_labels.shape[-1]
        else:
            raise ValueError(
                (
                    "If the label is text, it must be `FinelyControlledText` or `PairedText` or `str`, "
                    "if the label is classification, it must be `ClassificationID (List[int])`",
                    "if the label is regression value, ti must be `RegressionValue (List[float])`",
                )
            )

        # 对于"decoder"的"generate"任务, 训练时需要对input和label进一步处理
        if self.split == Split.TRAINING and self.task_type == "generate" and self.model_structure == "decoder":
            new_labels = []
            for inputs, labels in zip(self.batch_model_input, self.tokens_labels):
                inputs_len = len(inputs["input_ids"])
                inputs["input_ids"] = (inputs["input_ids"] + labels)[:max_length]
                new_labels.append(([self.inputkey2padid["labels"]] * inputs_len + labels)[:max_length])
            self.tokens_labels = new_labels

    def __getitem__(self, item: int) -> dict:
        ret_dict = dict()
        ret_dict["model_inputs"] = self.batch_model_input[item]
        if self.tokens_labels is not None:
            ret_dict["labels"] = self.tokens_labels[item]
        if hasattr(self, "dicts_custom_inputs"):
            ret_dict["custom_inputs"] = self.dicts_custom_inputs[item]
        return ret_dict

    def __len__(self):
        return len(self.batch_model_input)

    def report(self):
        "Log some information of dataset."
        logger.info(f"Total data: {len(self)}")
        logger.info(f"Max length of input: {self.actual_max_length_input}")
        logger.info(f"Max length of label: {self.actual_max_length_label}")

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

    def __pad_one(self, model_input: ModelInput, max_length: int):
        def helper(l: List, inputkey: str):
            nonlocal max_length
            if not isinstance(l[0], List):
                diff = max_length - len(l)
                # 生成任务时，labels的pad应该始终在右边
                if inputkey == "labels" or self.padding_side == "right":
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
        if max_length is None:
            if "input_ids" in batch[0]:
                max_length = max_len_nest_list([sample["input_ids"] for sample in batch])
            else:
                max_length = max_len_nest_list([sample["labels"] for sample in batch])
        return [self.__pad_one(model_input, max_length) for model_input in batch]

    @classmethod
    def __tokenize(
        cls, finely_controlled_text_list: List[FinelyControlledText], tokenizer: PreTrainedTokenizer, max_length: int, desc: str, **kargs
    ) -> BatchModelInput:
        # TODO: 新版本的data.py暂未适配 self.__tokenize
        raise NotImplemented
        # TODO: bug: token_type_ids全为0
        # TODO: bug: 对于可接受无限长输入的模型, 因为其没有 max length, 因此无法pad
        if max_length == INFINITE:
            raise NotImplementedError("TODO")

        if "token_type_ids" in tokenizer.model_input_names:
            logger.warning(f" model input include 'token_type_ids'. There is a bug causing all the token_type_ids to be `0`")
        tokenized_dict = defaultdict(list)
        waiting_to_trunc_idxs = [idx for idx in range(len(finely_controlled_text_list[0])) if finely_controlled_text_list[0][idx][0]]
        for finely_controlled_text in tqdm(finely_controlled_text_list, desc=desc, colour="RED", smoothing=0.99):
            # text: FinelyControlledText = tuple[tuple[bool, str], ...]
            cur_dict: ModelInputSplited = defaultdict(list)
            origin_length = 0
            for _, part_text in finely_controlled_text:
                cur_dict_ = tokenizer(text=part_text, padding=False, truncation=False, max_length=None, add_special_tokens=False, **kargs)
                origin_length += len(cur_dict_["input_ids"])
                for key, value in cur_dict_.items():
                    cur_dict[key].append(value)
            num_tokens_to_remove = (origin_length - max_length) if max_length != INFINITE else 0
            if num_tokens_to_remove > 0:
                cls.__truncate(cur_dict, waiting_to_trunc_idxs, num_tokens_to_remove)
            cur_dict: ModelInput = {key: sum(value, []) for key, value in cur_dict.items()}
            # tokenizer.pad(cur_dict, padding="max_length", max_length=model_max_length)
            cls.__pad_one(cur_dict, tokenizer, max_length)
            for key, value in cur_dict.items():
                tokenized_dict[key].append(value)
        return tokenized_dict

    # @staticmethod
    def transformers_tokenizer_tqdm(
        self, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, text_pairs: List[PairedText], max_length: int, desc: str, is_label: bool
    ) -> Tuple[List[ModelInput], int]:
        """此函数同时完成truncation操作"""
        # print(text_pairs)
        batch_model_input = []
        longest = 0
        for text1, text2 in tqdm(text_pairs, desc=desc, colour="RED", smoothing=0.99):
            a_sample = tokenizer(
                text=text1,
                text_pair=text2,
                padding=False,
                truncation="longest_first" if max_length != INFINITE else False,
                max_length=max_length if max_length != INFINITE else None,
                add_special_tokens=not (is_label and self.model_structure == "decoder"),
                return_attention_mask=not is_label,
            )
            if is_label:
                batch_model_input.append(a_sample["input_ids"])
            else:
                batch_model_input.append(a_sample)
            input_ids = a_sample["input_ids"]
            if isinstance(input_ids[0], list):
                longest = max(longest, max([len(value_) for value_ in input_ids]))
            else:
                longest = max(longest, len(input_ids))
        return batch_model_input, longest

    def collate_fn(self, batch: list[dict]):
        batch_model_inputs = self.__pad_batch(
            [item.pop("model_inputs") for item in batch], self.actual_max_length_input if self.padding_to_max_length else None
        )
        ret: dict = default_collate(batch_model_inputs)
        if self.padding_label:
            batch_labels = self.__pad_batch(
                [{"labels": item.pop("labels")} for item in batch], self.actual_max_length_label if self.padding_to_max_length else None
            )
            ret.update(default_collate(batch_labels))
        ret.update(default_collate(batch))
        return ret

    # def collate_fn(self, batch: list[dict]):
    #     batch_model_inputs = default_collate(
    #         self.__pad_batch([item.pop("model_inputs") for item in batch], self.actual_max_length_input if self.padding_to_max_length else None)
    #     )
    #     batch_model_inputs.update(default_collate(batch))
    #     return batch_model_inputs

    @classmethod
    def from_file(
        cls,
        data_file_path: Path | str,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        split: Split,
        configs: NLPTrainingConfig,
        load_data_fn: Callable[
            [str, str, PreTrainedTokenizer | PreTrainedTokenizerFast, bool],
            Tuple[List[FinelyControlledText] | list[PairedText], List[FinelyControlledText] | list[PairedText] | List[ClassificationID]],
        ],
        use_cache: bool | None = None,
        **kwargs_load_data,
    ) -> Self:
        """Load dataset from file with the given `NLPTrainingConfig`."""
        if data_file_path is None:
            logger.warning(f"⚠️  Fail to load {split.name} data. The data file path is not specified (received `NoneType`).")
            return None
            # raise TypeError(f"❌ Fail to load {split.name} data. The data file path is not specified (received `NoneType`).")
        if isinstance(data_file_path, str):
            data_file_path = Path(data_file_path)
        if isinstance(data_file_path, Path) and not data_file_path.exists():
            raise FileNotFoundError(f"❌ Fail to load test data. {data_file_path} does not exists.")

        local_rank = dist.get_rank() if dist.is_initialized() else 0

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
                model_type=configs.model_type,
                tokenizer=tokenizer,
                load_data_fn=load_data_fn,
                model_structure=configs.model_structure,
                task_type=configs.task_type,
                padding_side=configs.padding_side,
                max_length=configs.max_length,
                max_length_input=configs.max_length_input,
                max_length_label=configs.max_length_label,
                padding_to_max_length=configs.padding_to_max_length,
                split=split,
                **kwargs_load_data,
            )
            if use_cache:
                dataset.cache(data_file_path, repr(tokenizer.__class__)[8:-2], **kwargs_load_data)

        end = time.time()
        if local_rank == 0:
            logger.debug(f"⌛ Loading {split.name} data takes {end - start:.2f} sec.")
            cls.report(dataset)

        # 确保无论是从cache加载或是重新初始化, 下面这些参数的值始终是configs中设置的值
        dataset.padding_to_max_length = configs.padding_to_max_length
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
