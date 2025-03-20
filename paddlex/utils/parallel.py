# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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

from .flags import EXP_USE_PARALLEL_COMPUTING
from . import logging

__all__ = [
    "set_default_parallel_computing_executor",
    "get_default_parallel_computing_executor",
    "maybe_parallelize",
]

_executor = None


def _get_default_num_jobs():
    return min(32, os.cpu_count() + 4)


def set_default_parallel_computing_executor(executor):
    global _executor
    if _executor is not None:
        logging.warning("The old executor will be replaced.")
    old_executor = _executor
    _executor = executor
    return old_executor


def get_default_parallel_computing_executor():
    return _executor


def maybe_parallelize(func, iterable, /, *, executor=None):
    if not EXP_USE_PARALLEL_COMPUTING:
        return [func(x) for x in iterable]
    if executor is None:
        executor = _executor
    if executor is None:
        executor = joblib.Parallel(n_jobs=_get_default_num_jobs(), prefer="threads")
    return executor(joblib.delayed(func)(x) for x in iterable)
