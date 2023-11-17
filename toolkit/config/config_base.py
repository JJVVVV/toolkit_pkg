import copy
import json
from pathlib import Path
from typing import Any, Dict, Self

import hjson

from ..logger import _getLogger

CONFIG_NAME = "config.json"
logger = _getLogger("Configuration")


class ConfigBase:
    # model_type: str = ""
    attribute_alias_map: Dict[str, str] = dict()

    def __setattr__(self, key, value):
        if key in super().__getattribute__("attribute_alias_map"):
            key = super().__getattribute__("attribute_alias_map")[key]
        super().__setattr__(key, value)

    def __getattribute__(self, key):
        if key != "attribute_alias_map" and key in super().__getattribute__("attribute_alias_map"):
            key = super().__getattribute__("attribute_alias_map")[key]
        return super().__getattribute__(key)

    def __init__(self, **kwargs):
        # Attributes with defaults
        # self.model_type = kwargs.pop("model_type", "")
        # self.model_name = kwargs.pop("model_name", "")

        # Name or path to the pretrained checkpoint
        self._name_or_path = str(kwargs.pop("name_or_path", ""))

        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err
        # todo 并行时会重复log
        if kwargs:
            logger.info("👻 Custom attributes:")
            for key, value in kwargs.items():
                logger.info(f"   {key}={value}")

    @property
    def name_or_path(self) -> str:
        return getattr(self, "_name_or_path", None)

    @name_or_path.setter
    def name_or_path(self, value):
        self._name_or_path = str(value)  # Make sure that name_or_path is a string (for JSON encoding)

    # @property
    # def num_labels(self) -> int:
    #     """
    #     `int`: The number of labels for classification models.
    #     """
    #     return len(self.id2label)

    # @num_labels.setter
    # def num_labels(self, num_labels: int):
    #     if not hasattr(self, "id2label") or self.id2label is None or len(self.id2label) != num_labels:
    #         self.id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
    #         self.label2id = dict(zip(self.id2label.values(), self.id2label.keys()))

    def save(self, save_directory: Path | str, json_file_name=CONFIG_NAME, silence=True, **kwargs):
        if isinstance(save_directory, str):
            save_directory = Path(save_directory)
        if save_directory.is_file():
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")
        save_directory.mkdir(parents=True, exist_ok=True)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file_path = save_directory / json_file_name

        self.to_json_file(output_config_file_path, only_diff=True)
        if not silence:
            logger.debug(f"✔️  Save configuration file in `{output_config_file_path}` successfully.")

    @classmethod
    def load(cls, load_dir_or_path: Path | str, json_file_name=CONFIG_NAME, silence=True, **kwargs) -> Self:
        if isinstance(load_dir_or_path, str):
            load_dir_or_path = Path(load_dir_or_path)
        if load_dir_or_path.is_file():
            load_path = load_dir_or_path
        else:
            load_path = load_dir_or_path / json_file_name
        assert load_path.exists()

        config_dict = cls.get_config_dict(load_path, silence=silence, **kwargs)
        # if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
        #     logger.warning(
        #         f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
        #         f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
        #     )
        config = cls.from_dict(config_dict)
        return config

    @classmethod
    def get_config_dict(cls, load_dir_or_path: Path | str, json_file_name=CONFIG_NAME, silence=True, **kwargs) -> Dict[str, Any]:
        """
        From a `load_dir_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        [`PretrainedConfig`] using `from_dict`.

        Parameters:
            load_dir_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

        Returns:
            `Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the configuration object.

        """
        if isinstance(load_dir_or_path, str):
            load_dir_or_path = Path(load_dir_or_path)
        if load_dir_or_path.is_file():
            load_path = load_dir_or_path
        else:
            load_path = load_dir_or_path / json_file_name

        original_kwargs = copy.deepcopy(kwargs)
        # Get config dict associated with the base config file
        config_dict = cls._get_config_dict(load_path, silence, **kwargs)

        # That config file may point us toward another config file to use.
        if "configuration_files" in config_dict:
            # The another config file must be a path or be in the same folder as the first
            config_dict = cls._get_config_dict(config_dict["configuration_files"], silence, **original_kwargs)

        return config_dict

    @classmethod
    def _get_config_dict(cls, load_path: Path | str, silence, **kwargs) -> Dict[str, Any]:
        if isinstance(load_path, str):
            load_path = Path(load_path)

        # Load config dict
        try:
            config_dict = cls._dict_from_json_file(load_path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise EnvironmentError(f"It looks like the config file at `{load_path}` is not a valid JSON file.")

        if not silence:
            logger.debug(f"✔️  Load configuration file from `{load_path}` successfully.")

        config_dict.update(kwargs)
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> Self:
        config = cls(**config_dict)
        return config

    @classmethod
    def from_json_file(cls, json_file: Path | str) -> Self:
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    @staticmethod
    def _dict_from_json_file(json_file: Path | str) -> Dict:
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)
        # with open(json_file, "r", encoding="utf-8") as reader:
        #     return hjson.load(reader)

    @staticmethod
    def dict_torch_dtype_to_str(d: Dict[str, Any]) -> None:
        """
        Checks whether the passed dictionary and its nested dicts have a `torch_dtype` key and if it's not None,
        converts torch.dtype to a string of just the type. For example, `torch.float32` get converted into `float32`
        string, which can then be stored in the json format.
        """
        if d.get("torch_dtype", None) is not None and not isinstance(d["torch_dtype"], str):
            d["torch_dtype"] = str(d["torch_dtype"]).split(".")[1]
        for value in d.values():
            if isinstance(value, dict):
                ConfigBase.dict_torch_dtype_to_str(value)

    @staticmethod
    def _convert_objects(d: Dict):
        """
        Convert objects' type to the types that can be encoded by json.
        For example: `Path` -> `str`
        """
        for key, value in d.items():
            if isinstance(value, Path):
                d[key] = str(value)

    @staticmethod
    def _convert_and_flat_objects(d: Dict):
        """
        Convert objects' type to the types that can be encoded by json.
        For example: `Path` -> `str`\n
        And flat the nest dict.
        """
        dict_need2parse = []
        for key, value in d.items():
            if isinstance(value, Path):
                d[key] = str(value)
            if isinstance(value, dict):
                dict_need2parse.append((key, value))
        for key, value in dict_need2parse:
            d.pop(key)
            d.update(value)

    def __eq__(self, other):
        return isinstance(other, ConfigBase) and (self.__dict__ == other.__dict__)

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_dict(self, flat=False) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.\n
        flat: Whether to flat the nest dict.
        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)

        # * ***************************************************
        # if hasattr(self, "metric"):
        #     output["metric"] = output["metric"].name
        # if not hasattr(self, "model_type") and hasattr(self.__class__, "model_type"):
        # if hasattr(self.__class__, "model_type"):
        #     output["model_type"] = self.__class__.model_type
        self.dict_torch_dtype_to_str(output)
        if flat:
            self._convert_and_flat_objects(output)
        else:
            self._convert_objects(output)
        return output

    def to_diff_dict(self) -> Dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()
        # print(config_dict)

        # get the default config dict of ConfigBase
        default_values_baseclass = ConfigBase().to_dict()
        # print(f"base:        {base_config_dict}")
        # get class specific config dict (including attributes defined in subclass)
        default_values_subclass = self.__class__().to_dict()
        # print(f"class specific: {class_config_dict}")
        # serializable_config_dict = {}

        # only serialize values that differ from the default config
        serializable_config_dict = {}
        for key, value in config_dict.items():
            if (
                key not in default_values_subclass
                or value != default_values_subclass[key]
                # not (key in base_config_dict and value != base_config_dict[key])
                # or (key in class_config_dict and value != class_config_dict[key])
                # or key not in class_config_dict
                # or (key in class_config_dict and key not in base_config_dict)
            ):
                serializable_config_dict[key] = value

        return serializable_config_dict

    def to_json_string(self, only_diff: bool = True) -> str:
        if only_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        # return hjson.dumps(config_dict) + "\n"
        return json.dumps(config_dict, indent=2, sort_keys=False) + "\n"

    def to_json_file(self, json_file_path: Path | str, only_diff: bool = True):
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string(only_diff=only_diff))

    def update(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            setattr(self, key, value)

    def update_from_string(self, update_str: str):
        """
        Updates attributes of this class with attributes from `update_str`.

        The expected format is ints, floats and strings as is, and for booleans use `true` or `false`. For example:
        "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        """

        d = dict(x.strip().split("=") for x in update_str.split(","))
        for k, v in d.items():
            if not hasattr(self, k):
                raise ValueError(f"key {k} isn't in the original config dict")
            old_v = getattr(self, k)
            if isinstance(old_v, bool):
                if v.lower() in ["true", "1", "y", "yes"]:
                    v = True
                elif v.lower() in ["false", "0", "n", "no"]:
                    v = False
                else:
                    raise ValueError(f"can't derive true or false from {v} (key {k})")
            elif isinstance(old_v, int):
                v = int(v)
            elif isinstance(old_v, float):
                v = float(v)
            elif not isinstance(old_v, str):
                raise ValueError(f"You can only update int, float, bool or string values in the config, got {v} for key {k}")

            setattr(self, k, v)
