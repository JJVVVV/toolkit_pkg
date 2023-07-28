import pickle
import time
from collections import defaultdict
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
from .config import NLPTrainingConfig

Tokens = List[int]
BatchTokens = List[Tokens]
ModelInput = Dict[str, Tokens]
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


class TextDataset(Dataset):
    """
    A demo of get_data_from_file:
    ```
    from toolkit.nlp.data import PairedText, FinelyControlledText
    def load_data_fn(data_file_path: Path | str, model_type: str, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, split: Split, **kargs):
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


            a_label = [dict_obj["label"]] # List[int]
            a_label = dict_obj["question1"] # str
            a_label = PairedText(dict_obj["question1"]) # paired text
            a_label = FinelyControlledText((False, CLS), (True, dict_obj["question1"])) # finely controlled text


            inputs.append(a_sample)
            labels.append(a_label)

        return inputs, labels
    ```
    """

    def __init__(
        self,
        data_file_path: Path | str,
        model_type: str,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        load_data_fn: Callable[
            [Path | str, str, PreTrainedTokenizer | PreTrainedTokenizerFast, Split],
            Tuple[List[FinelyControlledText] | list[PairedText], List[FinelyControlledText] | list[PairedText] | List[ClassificationID]],
        ],
        padding_side: str = "right",
        max_length_input: int | None = None,
        max_length_label: int | None = None,
        split: Split = Split.ANY,
        **kwargs_load_data,
    ) -> None:
        super().__init__()
        max_length_input = tokenizer.model_max_length if max_length_input is None else max_length_input
        max_length_label = tokenizer.model_max_length if max_length_label is None else max_length_label
        self.split = split

        # get input and label texts
        assert padding_side in ("left", "right"), f"`padding_side={padding_side}` is not understood, only `left` and `right` are valid values"
        self.padding_side = padding_side
        self.splited_texts_input, self.splited_texts_label = load_data_fn(
            data_file_path=data_file_path, model_type=model_type, tokenizer=tokenizer, split=split, **kwargs_load_data
        )

        # tokenize input texts
        tokenizer.padding_side = padding_side
        if (
            isinstance(self.splited_texts_input[0], tuple)
            and isinstance(self.splited_texts_input[0][0], list | str)
            and isinstance(self.splited_texts_input[0][1], list | str | NoneType)
        ) or isinstance(
            self.splited_texts_input[0], PairedText
        ):  # if the input type is `PairedText`
            self.batch_model_input = self.transformers_tokenizer_tqdm(
                tokenizer, self.splited_texts_input, max_length_input, desc="Tokenize input texts"
            )
        elif (
            isinstance(self.splited_texts_input[0], tuple)
            and isinstance(self.splited_texts_input[0][0], tuple)
            and isinstance(self.splited_texts_input[0][0][0], bool)
            and isinstance(self.splited_texts_input[0][0][1], str)
        ) or isinstance(
            self.splited_texts_input[0], FinelyControlledText
        ):  # if the input type is `FinelyControlledText`
            self.batch_model_input = self.__tokenize(self.splited_texts_input, tokenizer, max_length_input, desc="Tokenize input texts")
        else:
            raise ValueError("The input type must be `PairedText` or `FinelyControlledText`")
        self.batch_model_input = {key: torch.tensor(value) for key, value in self.batch_model_input.items()}
        max_length_input = self.batch_model_input["input_ids"].shape[1] if max_length_input == INFINITE else max_length_input
        if self.padding_side == "right":
            self.first_pad_indexes_input = torch.argmax(torch.eq(self.batch_model_input["input_ids"], tokenizer.pad_token_id).int(), dim=-1)
            self.first_pad_indexes_input[self.first_pad_indexes_input == 0] = max_length_input
            self.max_length_input = torch.max(self.first_pad_indexes_input).item()
            self.batch_model_input = {key: value[..., : self.max_length_input] for key, value in self.batch_model_input.items()}
        elif self.padding_side == "left":
            self.first_not_pad_indexes_input = torch.argmax(torch.ne(self.batch_model_input["input_ids"], tokenizer.pad_token_id).int(), dim=-1)
            self.max_length_input = max_length_input - torch.min(self.first_not_pad_indexes_input).item()
            self.batch_model_input = {key: value[..., -self.max_length_input :] for key, value in self.batch_model_input.items()}
            self.first_not_pad_indexes_input = torch.argmax(torch.ne(self.batch_model_input["input_ids"], tokenizer.pad_token_id).int(), dim=-1)

        # tokenize label texts
        tokenizer.padding_side = "right"
        if (
            isinstance(self.splited_texts_label[0], tuple)
            and isinstance(self.splited_texts_label[0][0], str)
            and isinstance(self.splited_texts_label[0][1], str | NoneType)
        ) or isinstance(
            self.splited_texts_label[0], PairedText
        ):  # if the label type is `PairedText`
            self.tokens_labels = self.transformers_tokenizer_tqdm(tokenizer, self.splited_texts_label, max_length_label, desc="Tokenize label texts")[
                "input_ids"
            ]
            max_length_label = self.tokens_labels.shape[1] if max_length_label == INFINITE else max_length_label
            self.tokens_labels = torch.tensor(self.tokens_labels)
            self.first_pad_indexes_label = torch.argmax(torch.eq(self.tokens_labels, tokenizer.pad_token_id).int(), dim=-1)
            self.first_pad_indexes_label[self.first_pad_indexes_label == 0] = max_length_label
            self.max_length_label = torch.max(self.first_pad_indexes_label).item()
            self.tokens_labels = torch.narrow(self.tokens_labels, -1, 0, self.max_length_label)
            self.tokens_labels[self.tokens_labels == tokenizer.pad_token_id] = -100
        elif (
            isinstance(self.splited_texts_label[0], tuple)
            and isinstance(self.splited_texts_label[0][0], tuple)
            and isinstance(self.splited_texts_label[0][0][0], bool)
            and isinstance(self.splited_texts_label[0][0][1], str)
        ) or isinstance(
            self.splited_texts_label[0], FinelyControlledText
        ):  # if the label type is `FinelyControlledText`
            self.tokens_labels = self.__tokenize(self.splited_texts_label, tokenizer, max_length_label, desc="Tokenize label texts")["input_ids"]
            self.tokens_labels = torch.tensor(self.tokens_labels)
            self.first_pad_indexes_label = torch.argmax(torch.eq(self.tokens_labels, tokenizer.pad_token_id).int(), dim=-1)
            self.first_pad_indexes_label[self.first_pad_indexes_label == 0] = max_length_label
            self.max_length_label = torch.max(self.first_pad_indexes_label).item()
            self.tokens_labels = torch.narrow(self.tokens_labels, -1, 0, self.max_length_label)
            self.tokens_labels[self.tokens_labels == tokenizer.pad_token_id] = -100
        elif (isinstance(self.splited_texts_label[0], list) and isinstance(self.splited_texts_label[0][0], str)) or isinstance(
            self.splited_texts_label[0], str
        ):  # if the label type is `List[str]` or `str`
            self.tokens_labels = self.splited_texts_label
            self.max_length_label = -1
        elif isinstance(self.splited_texts_label[0], list) and isinstance(
            self.splited_texts_label[0][0], int
        ):  # if the label type is `ClassificationID`, i.e. `List[int]`
            self.tokens_labels = torch.tensor(self.splited_texts_label, dtype=torch.int)
            self.max_length_label = self.tokens_labels.shape[-1]
        elif isinstance(self.splited_texts_label[0], list) and isinstance(
            self.splited_texts_label[0][0], float
        ):  # if the label type is `RegressionValue`, i.e. `List[float]`
            self.tokens_labels = torch.tensor(self.splited_texts_label, dtype=torch.float32)
            self.max_length_label = self.tokens_labels.shape[-1]
        else:
            raise ValueError(
                (
                    "If the label is text, it must be `FinelyControlledText` or `PairedText` or `str`, "
                    "if the label is classification, it must be `ClassificationID (List[int])`",
                    "if the label is regression value, ti must be `RegressionValue (List[float])`",
                )
            )

        # if "roberta" in self.model_type:
        #     inputs_ids = self.tokenized_dict["input_ids"]
        #     self.cls_sep_indexes = (
        #         ((inputs_ids == tokenizer.cls_token_id) | (inputs_ids == tokenizer.sep_token_id)).nonzero()[:, 1].reshape(inputs_ids.shape[0], -1)
        #     )

    @property
    def collate_fn(self):
        return self.collate_fn_padding_left if self.padding_side == "left" else self.collate_fn_padding_right

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
        # return len(self.splited_texts_input)
        return self.batch_model_input["input_ids"].shape[0]

    def report(self):
        "Log some information of dataset."
        logger.info(f"Total data: {len(self)}")
        logger.info(f"Max length of input: {self.max_length_input}")
        logger.info(f"Max length of label: {self.max_length_label}")

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
    def __tokenize(
        cls, finely_controlled_text_list: List[FinelyControlledText], tokenizer: PreTrainedTokenizer, max_length: int, desc: str, **kargs
    ) -> BatchModelInput:
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
            cls.__pad(cur_dict, tokenizer, max_length)
            for key, value in cur_dict.items():
                tokenized_dict[key].append(value)
        return tokenized_dict

    @staticmethod
    def transformers_tokenizer_tqdm(
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, text_pairs: List[PairedText], max_length: int, desc: str
    ) -> BatchModelInput:
        # print(text_pairs)
        batch_model_input = defaultdict(list)
        # longest = 0
        for text1, text2 in tqdm(text_pairs, desc=desc, colour="RED", smoothing=0.99):
            for key, value in tokenizer(
                text=text1,
                text_pair=text2,
                padding="max_length" if max_length != INFINITE else False,
                truncation="longest_first" if max_length != INFINITE else False,
                max_length=max_length if max_length != INFINITE else None,
            ).items():
                batch_model_input[key].append(value)
            # longest = max(longest, len(value))
        if max_length == INFINITE:
            batch_model_input = tokenizer.pad(batch_model_input, padding="longest")
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

    # TODO: clip the label
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
        use_cache: bool = True,
        **kwargs_load_data,
    ) -> Self:
        """Load dataset from file with the given `NLPTrainingConfig`."""
        if data_file_path is None:
            raise TypeError(f"❌ Fail to load {split.name} data. The data file path is not specified (received NoneType).")
        if isinstance(data_file_path, str):
            data_file_path = Path(data_file_path)
        if not data_file_path.exists():
            raise FileNotFoundError(f"❌ Fail to load test data. {data_file_path} does not exists.")

        local_rank = dist.get_rank() if dist.is_initialized() else 0

        start = time.time()
        if local_rank == 0:
            logger.debug(f"⏳ Loading {split.name} dataset ...")
        if use_cache:
            dataset = cls.from_cache(data_file_path)
        else:
            dataset = None
        if dataset is None:
            dataset = cls(
                data_file_path=data_file_path,
                model_type=configs.model_type,
                tokenizer=tokenizer,
                load_data_fn=load_data_fn,
                padding_side=configs.padding_side,
                max_length_input=configs.max_length_input,
                max_length_label=configs.max_length_label,
                split=split,
                **kwargs_load_data,
            )
            if use_cache:
                dataset.cache(data_file_path)

        end = time.time()
        if local_rank == 0:
            logger.debug(f"⌛ Loading {split.name} data takes {end - start:.2f} sec.")
            cls.report(dataset)
        return dataset

    @staticmethod
    def cache_path(origin_data_path: Path) -> Path:
        "Convert the original data file path to a path where dataset will be cached in."
        absolute_path = origin_data_path.resolve()
        cache_path = Path(CACHE_DIR, str(absolute_path)[1:] if str(absolute_path).startswith("/") else str(absolute_path))
        cache_path = cache_path.with_suffix(".pkl")
        return cache_path

    def cache(self, origin_data_path: Path):
        "Cache tokenized dataset."
        local_rank = dist.get_rank() if dist.is_initialized() else 0
        if local_rank == 0:
            logger.debug(f"💿 Caching dataset from {origin_data_path} ...")
            cache_path = self.cache_path(origin_data_path)
            cache_path.parent.mkdir(parents=True)
            with cache_path.open("wb") as f:
                pickle.dump(self, f)
            logger.debug("✔️ Cache successfully.")

    @classmethod
    def from_cache(cls, cached_dataset_or_origin_data_path: Path | str) -> Self | None:
        "Try to load dataset from cache. If there if no cache, `None` will be returned."
        local_rank = dist.get_rank() if dist.is_initialized() else 0
        if local_rank == 0:
            logger.debug(f"💿 Loading dataset from cache ...")
        cached_dataset_or_origin_data_path = Path(cached_dataset_or_origin_data_path)
        if cached_dataset_or_origin_data_path.suffix == ".pkl":
            cached_dataset_path = cached_dataset_or_origin_data_path
        else:
            cached_dataset_path = cls.cache_path(cached_dataset_or_origin_data_path)
        try:
            with cached_dataset_path.open("rb") as f:
                dataset = pickle.load(f)
            if local_rank == 0:
                logger.debug("✔️ Load successfully.")
        except FileNotFoundError as e:
            if local_rank == 0:
                logger.debug(" ❕ There is no cache.")
                dataset = None
        return dataset

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
