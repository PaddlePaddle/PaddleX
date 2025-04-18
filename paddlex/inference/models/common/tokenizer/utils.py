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

import inspect
from typing import List

__all__ = [
    "convert_to_dict_message",
    "fn_args_to_dict",
]


def convert_to_dict_message(conversation: List[List[str]]):
    """Convert the list of chat messages to a role dictionary chat messages."""
    conversations = []
    for index, item in enumerate(conversation):
        assert (
            1 <= len(item) <= 2
        ), "Each Rounds in conversation should have 1 or 2 elements."
        if isinstance(item[0], str):
            conversations.append({"role": "user", "content": item[0]})
            if len(item) == 2 and isinstance(item[1], str):
                conversations.append({"role": "assistant", "content": item[1]})
            else:
                # If there is only one element in item, it must be the last round.
                # If it is not the last round, it must be an error.
                if index != len(conversation) - 1:
                    raise ValueError(f"Round {index} has error round")
        else:
            raise ValueError("Each round in list should be string")
    return conversations


def fn_args_to_dict(func, *args, **kwargs):
    """
    Inspect function `func` and its arguments for running, and extract a
    dict mapping between argument names and keys.
    """
    (spec_args, spec_varargs, spec_varkw, spec_defaults, _, _, _) = (
        inspect.getfullargspec(func)
    )
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
