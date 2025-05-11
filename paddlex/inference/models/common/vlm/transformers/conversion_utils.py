# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, TypeVar

import numpy as np
import paddle
from numpy import ndarray, transpose

from ......utils import logging
from ..distributed import distributed_allgather, distributed_gather
from ..utils import device_guard, get_env_device

if TYPE_CHECKING:
    from .configuration_utils import PretrainedConfig

# the type hinting for pytorch model & layer & tensor
Module = TypeVar("Module")
PytorchTensor = TypeVar("PytorchTensor")


@dataclass
class StateDictNameMapping:
    """NameMapping of StateDict between two models"""

    source_name: str
    target_name: str = None

    action: Optional[str] = None  # the value can be: transpose, merge_last_two_dim
    index: Optional[int] = None

    slots: list[str] = None

    def __post_init__(self):
        self.target_name = self.target_name or self.source_name

    def should_transpose(self) -> bool:
        return self.action == "transpose"

    def should_merge_last_two_dim(self) -> bool:
        """check that whether merge last two dim"""
        return self.action == "merge_last_two_dim"

    def run(self, state_dict: dict[str, ndarray], name: str) -> ndarray:
        """run some custom operation on ndarray, eg: transpose, merge_last_two_dim

        Args:
            tensor (ndarray): the source of the tensor data

        Returns:
            ndarray: the final tensor
        """
        tensor = state_dict.pop(name)
        if callable(self.action):
            return self.action(tensor)
        if self.action == "transpose":
            return transpose(tensor, [1, 0])
        if self.action == "merge_last_two_dim":
            shape = tensor.shape
            assert len(shape) == 3
            return np.reshape(tensor, [shape[0], -1])
        if self.action == "split":
            assert (
                self.index is not None
            ), "when action is `split`, index field is required."
            # FIXME if the order of split starts from index=2, no tensor left.
            if self.index < 2:
                state_dict[name] = tensor
            # qkv is stored in same tensor, so it should be split into 3 arr
            tensors = np.split(tensor, 3, axis=-1)
            return tensors[self.index]

        return tensor

    def matched(self, text: str) -> bool:
        """check whether the layer_name match the current pattern

        Args:
            text (str): the name of layer

        Returns:
            bool: whether the
        """
        if text == self.source_name:
            return True

        if not self.slots:
            return False


class ConversionMixin:
    @classmethod
    def support_conversion(cls, config: PretrainedConfig) -> bool:
        """check whether the model support conversion"""
        try:
            # try to get the name-mapping info
            _ = cls._get_name_mappings(config)
        except NotImplementedError:
            return False
        finally:
            return True

    @classmethod
    def _get_name_mappings(cls, config: PretrainedConfig) -> List[StateDictNameMapping]:
        """get name mapping of PretrainedModel

        Args:
            config (PretrainedConfig): the configuration of name-mapping

        Raises:
            NotImplementedError:

        Returns:
            List[StateDictNameMapping]: the name-mappings of pretrained model
        """
        raise NotImplementedError

    @classmethod
    def get_tensor_parallel_convert_actions(
        cls,
        config: PretrainedConfig,
        loaded_state_dict_keys,
        is_split=True,
        ignore_error=False,
    ):
        name_action_mappings = cls._get_tensor_parallel_mappings(
            config, is_split=is_split
        )
        state_keys_map = cls._resolve_prefix_keys(
            name_action_mappings.keys(), loaded_state_dict_keys, ignore_error
        )
        for k, v in state_keys_map.items():
            name_action_mappings[v] = name_action_mappings.pop(k)
        return name_action_mappings

    @classmethod
    def convert_tensor_parallel(
        cls,
        weight_file: str,
        config: PretrainedConfig,
        state_dict=None,
        ignore_error=False,
    ) -> None:
        """the entry of converting config and converting model file

        Args:
            weight_file (str | None): the weight file path of `model_state.pdparams` file
            config (PretrainedConfig): the PretrainedConfig instance of model
        """

        name_action_mappings = cls._get_tensor_parallel_mappings(config)
        if state_dict is None:
            with device_guard("cpu"):
                state_dict = paddle.load(weight_file, return_numpy=False)
            logging.info(
                "Starting to convert original state_dict to tensor parallel state_dict."
            )

        state_keys_map = cls._resolve_prefix_keys(
            name_action_mappings.keys(), state_dict.keys(), ignore_error
        )

        for k, v in state_keys_map.items():
            name_action_mappings[v] = name_action_mappings.pop(k)

        for name, action in name_action_mappings.items():
            if name not in state_dict:
                if not ignore_error:
                    logging.warning(f"Key <{name}> not in the model state weight file.")
                continue
            tensor = state_dict.pop(name)
            new_tensor = action(tensor)
            with device_guard("cpu"):
                state_dict[name] = paddle.Tensor(new_tensor, zero_copy=True)

        return state_dict

    @classmethod
    def merge_tensor_parallel(cls, state_dict, config) -> None:
        """the entry of converting config and converting model file

        Args:
            input_dir (str | None): the input dir which contains `pytorch_model.bin` and `config.json` file
            config (PretrainedConfig): the PretrainedConfig instance of model
        """
        name_action_mappings = cls._get_tensor_parallel_mappings(config, is_split=False)
        state_keys_map = cls._resolve_prefix_keys(
            name_action_mappings.keys(), state_dict.keys()
        )

        for k, v in state_keys_map.items():
            name_action_mappings[v] = name_action_mappings.pop(k)

        state_dict_to_save = {}

        hcg = paddle.distributed.fleet.get_hybrid_communicate_group()
        mp_group = hcg.get_model_parallel_group()
        is_dst = paddle.distributed.get_rank(mp_group) == 0

        for key in state_dict.keys():
            tensor = state_dict[key]
            if key in name_action_mappings:
                if get_env_device() == "xpu":
                    ret = distributed_allgather(tensor, group=mp_group, offload=True)
                else:
                    ret = distributed_gather(tensor, group=mp_group, offload=True)
                action = name_action_mappings.pop(key)
                tensor = action(ret) if is_dst else None
            else:
                tensor = tensor.cpu().numpy() if is_dst else None

            # keep state dict use paddle.tensor
            if isinstance(tensor, np.ndarray):
                with device_guard("cpu"):
                    tensor = paddle.Tensor(tensor, zero_copy=True)

            state_dict_to_save[key] = tensor

        if len(name_action_mappings) > 0:
            for x in name_action_mappings.keys():
                logging.debug(
                    f"key <{x}> need to merge tensor parallel but we can't find in model state."
                )

        return state_dict_to_save

    @classmethod
    def _get_tensor_parallel_mappings(
        cls, config: PretrainedConfig, is_split=True
    ) -> List[StateDictNameMapping]:
        """get name mapping of PretrainedModel

        Args:
            config (PretrainedConfig): the configuration of name-mapping

        Raises:
            NotImplementedError:

        Returns:
            List[StateDictNameMapping]: the name-mappings for tensor_parallel
        """
        raise NotImplementedError

    @staticmethod
    def _resolve_prefix_keys(state_keys_base, state_keys_real, ignore_error=False):
        # state_keys_map base to real
        state_keys_map = {}

        # sorted by length，match from long to short for A.key B.key ...
        state_keys_base = sorted(state_keys_base, key=lambda x: len(x), reverse=True)
        state_keys_real = set(state_keys_real)

        for key in state_keys_base:
            for x in state_keys_real:
                if x.endswith(key):
                    state_keys_map[key] = x
                    break
            if key not in state_keys_map:
                if not ignore_error:
                    logging.debug(
                        f"tensor parallel conversion: could not find name {key} in loaded state dict!"
                    )
            else:
                state_keys_real.remove(state_keys_map[key])

        return state_keys_map

    @classmethod
    def convert_fuse_and_split(
        cls, config: PretrainedConfig, state_dict, tp_actions=None
    ):
        loaded_keys = state_dict.keys()
        # collect and convert fuse/split action
        fused_and_split_keys = []
        convert_with_same_keys = []
        fuse_actions, resume_keys = cls.get_fuse_or_split_param_convert_actions(
            config, loaded_keys, is_fuse=True
        )
        for keys, action in fuse_actions.items():
            if keys[-1] in keys[:-1]:
                assert len(keys) == 2, "only 2 keys can be converted with the same name"
                convert_with_same_keys.append(keys[-1])
            origin_states = [state_dict.pop(key) for key in keys[:-1]]
            state_dict[keys[-1]] = action(origin_states)
            fused_and_split_keys.append(keys[-1])
            logging.debug(f"Fusing parameter: {keys[:-1]} into {keys[-1]}")

        split_actions, _ = cls.get_fuse_or_split_param_convert_actions(
            config, loaded_keys, is_fuse=False
        )
        for keys, action in split_actions.items():
            if keys[-1] in keys[:-1]:
                assert len(keys) == 2, "only 2 keys can be converted with the same name"
                convert_with_same_keys.append(keys[-1])
            origin_state = state_dict.pop(keys[-1])
            split_states = action(origin_state)
            for key_idx, key in enumerate(keys[:-1]):
                state_dict[key] = split_states[key_idx]
                fused_and_split_keys.append(key)
            logging.debug(f"Splitting parameter: {keys[-1]} into {keys[:-1]}")

        if tp_actions is not None:
            for key in fused_and_split_keys:
                if key in convert_with_same_keys:
                    continue

                for name in tp_actions.keys():
                    if key.endswith(name):
                        with device_guard():
                            state_dict[key] = paddle.Tensor(
                                tp_actions[name](state_dict.pop(key)), zero_copy=True
                            )
                        break

        resume_state_dict = {k: state_dict[k] for k in resume_keys if k in state_dict}
        return state_dict, resume_state_dict

    @classmethod
    def get_fuse_or_split_param_convert_actions(
        cls,
        config: PretrainedConfig,
        loaded_state_dict_keys,
        is_fuse=True,
        ignore_error=False,
    ):
        name_action_mappings = cls._get_fuse_or_split_param_mappings(config, is_fuse)
        state_keys_map = cls._resolve_prefix_keys_for_fuse_and_split(
            name_action_mappings.keys(), loaded_state_dict_keys, ignore_error, is_fuse
        )
        for k, v in state_keys_map.items():
            name_action_mappings[v] = name_action_mappings.pop(k)

        filter_name_action = {}
        resume_keys = []
        if is_fuse:
            for k, v in name_action_mappings.items():
                cond = True
                if not all(item in loaded_state_dict_keys for item in k[:-1]):
                    # resume keys for next fuse
                    resume_keys += k[:-1]
                    cond = False
                if cond:
                    filter_name_action[k] = v
        else:
            for k, v in name_action_mappings.items():
                if k[-1] in loaded_state_dict_keys:
                    filter_name_action[k] = v

        return filter_name_action, resume_keys

    @classmethod
    def _get_fuse_or_split_param_mappings(
        cls, config: PretrainedConfig, is_fuse=True
    ) -> List[StateDictNameMapping]:
        """get fused parameter mapping of PretrainedModel

        Args:
            config (PretrainedConfig): the configuration of name-mapping

        Raises:
            NotImplementedError:

        Returns:
            List[StateDictNameMapping]: the name-mappings for tensor_parallel
        """

        return {}

    @staticmethod
    def _resolve_prefix_keys_for_fuse_and_split(
        state_keys_base, state_keys_real, ignore_error=False, is_fuse=True
    ):
        state_keys_map = {}

        for keys in state_keys_base:
            prefix = ""
            if is_fuse:
                for x in state_keys_real:
                    for base_key in keys[:-1]:
                        if x.endswith(base_key):
                            prefix = x.replace(base_key, "")
                            break
                    if prefix != "":
                        break
            else:
                base_key = keys[-1]
                for x in state_keys_real:
                    if x.endswith(base_key):
                        prefix = x.replace(base_key, "")
                        break

            new_keys = tuple([prefix + key for key in keys])
            state_keys_map[keys] = new_keys

        return state_keys_map
