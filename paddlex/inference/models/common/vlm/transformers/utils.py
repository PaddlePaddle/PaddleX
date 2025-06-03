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

import importlib
import inspect
import os
from contextlib import ExitStack
from typing import ContextManager, List

import paddle

from ..utils import device_guard


class ContextManagers:
    """
    Wrapper for `contextlib.ExitStack` which enters a collection of context managers. Adaptation of `ContextManagers`
    in the `fastcore` library.
    """

    def __init__(self, context_managers: List[ContextManager]):
        self.context_managers = context_managers
        self.stack = ExitStack()

    def __enter__(self):
        for context_manager in self.context_managers:
            self.stack.enter_context(context_manager)

    def __exit__(self, *args, **kwargs):
        self.stack.__exit__(*args, **kwargs)


def fn_args_to_dict(func, *args, **kwargs):
    """
    Inspect function `func` and its arguments for running, and extract a
    dict mapping between argument names and keys.
    """
    if hasattr(inspect, "getfullargspec"):
        (spec_args, spec_varargs, spec_varkw, spec_defaults, _, _, _) = (
            inspect.getfullargspec(func)
        )
    else:
        (spec_args, spec_varargs, spec_varkw, spec_defaults) = inspect.getargspec(func)
    # add positional argument values
    init_dict = dict(zip(spec_args, args))
    # add default argument values
    kwargs_dict = (
        dict(zip(spec_args[-len(spec_defaults) :], spec_defaults))
        if spec_defaults
        else {}
    )
    for k in list(kwargs_dict.keys()):
        if k in init_dict:
            kwargs_dict.pop(k)
    kwargs_dict.update(kwargs)
    init_dict.update(kwargs_dict)
    return init_dict


def get_checkpoint_shard_files(
    pretrained_model_name_or_path,
    index_filename,
    cache_dir=None,
    subfolder="",
    **kwargs,
):
    """
    For a given model:
    - download and cache all the shards of a sharded checkpoint if `pretrained_model_name_or_path` is a model ID on the
        Hub
    - returns the list of paths to all the shards, as well as some metadata.
    For the description of each arg, see [`PretrainedModel.from_pretrained`]. `index_filename` is the full path to the
    index (downloaded and cached if `pretrained_model_name_or_path` is a model ID on the Hub).
    """

    import json

    if not os.path.isfile(index_filename):
        raise ValueError(
            f"Can't find a checkpoint index ({index_filename}) in {pretrained_model_name_or_path}."
        )

    with open(index_filename, "r") as f:
        index = json.loads(f.read())

    shard_filenames = sorted(set(index["weight_map"].values()))
    sharded_metadata = index["metadata"]
    sharded_metadata["all_checkpoint_keys"] = list(index["weight_map"].keys())
    sharded_metadata["weight_map"] = index["weight_map"].copy()

    file_map = {file: set() for file in shard_filenames}
    for weight, file in index["weight_map"].items():
        file_map[file].add(weight)

    sharded_metadata["file_map"] = file_map

    # Adapt for PaddleX
    assert os.path.isdir(pretrained_model_name_or_path)
    shard_filenames = [
        os.path.join(pretrained_model_name_or_path, subfolder, f)
        for f in shard_filenames
    ]
    return shard_filenames, sharded_metadata


def is_paddle_support_lazy_init():
    return hasattr(paddle, "LazyGuard")


def is_safetensors_available():
    return importlib.util.find_spec("safetensors") is not None


def paddlenlp_load(path, map_location="cpu"):
    assert map_location in ["cpu", "gpu", "xpu", "npu", "numpy", "np"]
    if map_location in ["numpy", "np"]:
        return paddle.load(path, return_numpy=True)
    else:
        with device_guard(map_location):
            return paddle.load(path)
            # TODO(zhonghui03): the following code has problems when hot start optimizer checkpoint.
            if map_location == "cpu":
                from paddle.framework.io import (
                    _parse_every_object,
                    _to_LodTensor,
                    _transformed_from_lodtensor,
                )

                def _ndarray_to_tensor(obj, return_numpy=False):
                    if return_numpy:
                        return obj
                    if paddle.in_dynamic_mode():
                        return paddle.Tensor(obj, zero_copy=True)
                    else:
                        return _to_LodTensor(obj)

                state_dict = paddle.load(path, return_numpy=True)
                # Hack for zero copy for saving loading time. for paddle.load there need copy to create paddle.Tensor
                return _parse_every_object(
                    state_dict, _transformed_from_lodtensor, _ndarray_to_tensor
                )

            else:
                return paddle.load(path)


def use_hybrid_parallel():
    try:
        from paddle.distributed import fleet

        hcg = fleet.get_hybrid_communicate_group()
        return hcg
    except:
        return None


def weight_name_suffix():
    hcg = use_hybrid_parallel()
    if hcg is not None:
        name = []
        if hcg.get_model_parallel_world_size() > 1:
            name.append(f"tp{hcg.get_model_parallel_rank():0>2d}")
        if hcg.get_pipe_parallel_world_size() > 1:
            name.append(f"pp{hcg.get_stage_id():0>2d}")
        return "_".join(name)
    else:
        return None
