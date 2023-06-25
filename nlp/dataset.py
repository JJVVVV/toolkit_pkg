from collections import defaultdict
from enum import Enum, auto
from typing import Callable, Optional

import jsonlines
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

Tokens = list[int]
BatchTokens = list[Tokens]
ModelInput = dict[str, Tokens]
ModelInputSplited = dict[str, list[Tokens]]
BatchModelInput = dict[str, BatchTokens]


class MyDataset(Dataset):
    def __init__(
        self,
        data_file_path: str,
        model_type: str,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        input_text_format_func: Callable[[str, str, PreTrainedTokenizer | PreTrainedTokenizerFast, bool], tuple[list, list]],
        padding_side: str = "right",
        max_input_length: int = 0,
        max_label_length: int = 0,
        is_train: bool = True,
        **kargs,
    ) -> None:
        super().__init__()
        max_input_length = tokenizer.model_max_length if max_input_length == 0 else max_input_length
        max_label_length = tokenizer.model_max_length if max_label_length == 0 else max_label_length

        # get input and label texts
        self.padding_side = padding_side
        self.input_texts, self.label_texts = input_text_format_func(
            data_file_path, model_type=model_type, tokenizer=tokenizer, is_train=is_train, **kargs
        )  # * kargs: text_type, is_train, max_word_num

        # tokenize input texts
        tokenizer.padding_side = padding_side
        self.tokenized_dict_inputs = self.__tokenize(self.input_texts, tokenizer, max_input_length, desc="Tokenize input texts")
        # print(self.tokenized_dict_inputs)
        self.tokenized_dict_inputs = {key: torch.tensor(value) for key, value in self.tokenized_dict_inputs.items()}
        if self.padding_side == "right":
            self.first_pad_indexes_input = torch.argmax(torch.eq(self.tokenized_dict_inputs["input_ids"], tokenizer.pad_token_id).int(), dim=-1)
            self.first_pad_indexes_input[self.first_pad_indexes_input == 0] = max_input_length
            self.max_length_inputs = torch.max(self.first_pad_indexes_input).item()
            self.tokenized_dict_inputs = {key: value[..., : self.max_length_inputs] for key, value in self.tokenized_dict_inputs.items()}
        elif self.padding_side == "left":
            self.first_not_pad_indexes_input = torch.argmax(torch.ne(self.tokenized_dict_inputs["input_ids"], tokenizer.pad_token_id).int(), dim=-1)
            self.max_length_inputs = max_input_length - torch.min(self.first_not_pad_indexes_input).item()
            self.tokenized_dict_inputs = {key: value[..., -self.max_length_inputs - 1 :] for key, value in self.tokenized_dict_inputs.items()}
            self.first_not_pad_indexes_input = torch.argmax(torch.ne(self.tokenized_dict_inputs["input_ids"], tokenizer.pad_token_id).int(), dim=-1)

        # tokenize label texts
        tokenizer.padding_side = "right"
        if is_train and isinstance(self.label_texts[0], tuple):
            self.tokenized_labels = self.__tokenize(self.label_texts, tokenizer, max_label_length, desc="Tokenize label texts")["input_ids"]
            self.tokenized_labels = torch.tensor(self.tokenized_labels)
            self.first_pad_indexes_label = torch.argmax(torch.eq(self.tokenized_labels, tokenizer.pad_token_id).int(), dim=-1)
            self.first_pad_indexes_label[self.first_pad_indexes_label == 0] = max_label_length
            self.max_length_labels = torch.max(self.first_pad_indexes_label).item()
            self.tokenized_labels = torch.narrow(self.tokenized_labels, -1, 0, self.max_length_labels)
            self.tokenized_labels[self.tokenized_labels == tokenizer.pad_token_id] = -100
        # TODO 如果原任务本来就是生成任务, label 无法转化成 tensor
        else:
            self.tokenized_labels = torch.tensor(self.label_texts, dtype=torch.int)
            self.max_length_labels = 1
        # if "roberta" in self.model_type:
        #     inputs_ids = self.tokenized_dict["input_ids"]
        #     self.cls_sep_indexes = (
        #         ((inputs_ids == tokenizer.cls_token_id) | (inputs_ids == tokenizer.sep_token_id)).nonzero()[:, 1].reshape(inputs_ids.shape[0], -1)
        #     )

    def __getitem__(self, item: int) -> dict:
        ret_dict = dict()
        ret_dict["tokenized_dict_inputs"] = {key: value[item] for key, value in self.tokenized_dict_inputs.items()}
        if self.padding_side == "right":
            ret_dict["first_pad_index_input"] = self.first_pad_indexes_input[item]
        else:
            ret_dict["first_not_pad_index_input"] = self.first_not_pad_indexes_input[item]

        if self.label_texts:
            ret_dict["labels"] = self.tokenized_labels[item]
        return ret_dict

    def __len__(self):
        return len(self.input_texts)

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

    @staticmethod
    def __tokenize(istrunc_texts: list[tuple[tuple[bool, str]]], tokenizer: PreTrainedTokenizer, max_length, desc="", **kargs) -> BatchModelInput:
        tokenized_dict = defaultdict(list)
        waiting_to_trunc_idxs = [idx for idx in range(len(istrunc_texts[0])) if istrunc_texts[0][idx][0]]
        for text in tqdm(istrunc_texts, desc=desc, colour="RED"):
            # text: tuple[tuple[bool, str]]
            cur_dict: ModelInputSplited = defaultdict(list)
            origin_length = 0
            for _, text_part in text:
                # TODO: bug: token_type_ids全为0
                cur_dict_ = tokenizer(text=text_part, padding=False, truncation=False, max_length=None, add_special_tokens=False, **kargs)
                origin_length += len(cur_dict_["input_ids"])
                for key, value in cur_dict_.items():
                    cur_dict[key].append(value)
            num_tokens_to_remove = origin_length - max_length
            if num_tokens_to_remove > 0:
                MyDataset.__truncate(cur_dict, waiting_to_trunc_idxs, num_tokens_to_remove)
            cur_dict: ModelInput = {key: sum(value, []) for key, value in cur_dict.items()}
            # tokenizer.pad(cur_dict, padding="max_length", max_length=model_max_length)
            MyDataset.__pad(cur_dict, tokenizer, max_length)
            for key, value in cur_dict.items():
                tokenized_dict[key].append(value)
        return tokenized_dict

    @staticmethod
    def stack_tensor_in_dicts(batch: list[dict]):
        ret_dict = dict()
        for key, value in batch[0].items():
            if isinstance(value, torch.Tensor):
                ret_dict[key] = torch.stack([it_dict[key] for it_dict in batch])
            elif isinstance(value, dict):
                ret_dict[key] = MyDataset.stack_tensor_in_dicts([it_dict[key] for it_dict in batch])
            else:
                raise Exception(f"data type in batch must be Tensor or Dict, but got {type(batch[0][key])}")
        return ret_dict

    @staticmethod
    def collate_fn_padding_right(batch: list[dict]):
        batch = MyDataset.stack_tensor_in_dicts(batch)
        first_pad_index_input = batch.pop("first_pad_index_input")
        batch_max_length = torch.max(first_pad_index_input).item()
        tokenized_dict_inputs = batch.pop("tokenized_dict_inputs")
        batch.update({key: value[..., :batch_max_length] for key, value in tokenized_dict_inputs.items()})
        return batch

    @staticmethod
    def collate_fn_padding_left(batch: list[dict]):
        batch = MyDataset.stack_tensor_in_dicts(batch)
        first_not_pad_index_input = batch.pop("first_not_pad_index_input")
        min_start = torch.min(first_not_pad_index_input).item()
        tokenized_dict_inputs = batch.pop("tokenized_dict_inputs")
        batch.update({key: value[..., min_start:] for key, value in tokenized_dict_inputs.items()})
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

    # def __tokenize(self, texts: list[str] | list[list[tuple[bool, str]]], tokenizer: PreTrainedTokenizer, max_length, desc="", **kargs):
    #     tokenized_dict = defaultdict(list)
    #     model_max_length = tokenizer.model_max_length
    #     if not isinstance(texts[0], str):
    #         waiting_to_trunc_idxs = [idx for idx in range(len(texts[0])) if texts[0][idx][0]]
    #     for text in tqdm(texts, desc=desc, colour="RED"):
    #         if isinstance(text, str):
    #             cur_dict = tokenizer(text=text, padding="max_length", truncation=True, max_length=max_length, **kargs)
    #             for key, value in cur_dict.items():
    #                 tokenized_dict[key].append(value)
    #         else:  # input_text: list[tuple[bool, str]]
    #             cur_dict = defaultdict(list)
    #             for truncation, text_part in text:
    #                 cur_dict_ = tokenizer(text=text_part, padding=False, truncation=False, max_length=None, add_special_tokens=False, **kargs)
    #                 for key, value in cur_dict_.items():
    #                     cur_dict[key].append(value)
    #             origin_length = sum([len(ids_part) for ids_part in cur_dict["input_ids"]])
    #             num_tokens_to_remove = origin_length - max_length
    #             for key, value in cur_dict.items():
    #                 # if num_tokens_to_remove:
    #                 #     print(origin_length, num_tokens_to_remove)
    #                 #     print(value)
    #                 ids, ids_pair, _ = tokenizer.truncate_sequences(
    #                     ids=value[waiting_to_trunc_idxs[0]],
    #                     pair_ids=value[waiting_to_trunc_idxs[1]] if len(waiting_to_trunc_idxs) == 2 else None,
    #                     num_tokens_to_remove=num_tokens_to_remove,
    #                     truncation_strategy="longest_first",
    #                 )
    #                 cur_dict[key][waiting_to_trunc_idxs[0]] = ids
    #                 if len(waiting_to_trunc_idxs) == 2:
    #                     cur_dict[key][waiting_to_trunc_idxs[1]] = ids_pair
    #                 # if num_tokens_to_remove:
    #                 #     print(value)
    #             cur_dict = {key: sum(value, []) for key, value in cur_dict.items()}
    #             tokenizer.pad(cur_dict, padding="max_length", max_length=model_max_length)
    #             # print(cur_dict)
    #             for key, value in cur_dict.items():
    #                 tokenized_dict[key].append(value)
    #     return tokenized_dict
