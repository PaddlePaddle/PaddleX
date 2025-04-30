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

import os

import joblib

from . import logging
from .flags import EXP_USE_PARALLEL_COMPUTING

__all__ = [
    "set_global_parallel_computing_executor",
    "get_global_parallel_computing_executor",
    "maybe_parallelize",
]

_executor = None


def _get_default_num_jobs():
    return min(32, os.cpu_count() + 4)


def set_global_parallel_computing_executor(executor):
    global _executor
    if _executor is not None:
        logging.warning("The old executor will be replaced.")
    old_executor = _executor
    _executor = executor
    return old_executor


def get_global_parallel_computing_executor():
    return _executor


def maybe_parallelize(func, /, *iterables, executor=None):
    if EXP_USE_PARALLEL_COMPUTING:
        should_parallelize = True
        if iterables:
            try:
                size = len(iterables[0])
            except TypeError:
                size = None
            if size == 1:
                should_parallelize = False
    else:
        should_parallelize = False
    if should_parallelize:
        if executor is None:
            executor = _executor
        if executor is None:
            executor = joblib.Parallel(n_jobs=_get_default_num_jobs(), prefer="threads")
        return executor(joblib.delayed(func)(*x) for x in zip(*iterables))
    else:
        return [func(*x) for x in zip(*iterables)]
