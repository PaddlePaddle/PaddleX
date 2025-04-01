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

import copy
import csv
import functools
import inspect
import time
import uuid
from pathlib import Path
from types import GeneratorType

import numpy as np
from prettytable import PrettyTable

from ...utils import logging
from ...utils.flags import (
    INFER_BENCHMARK,
    INFER_BENCHMARK_OUTPUT_DIR,
    INFER_BENCHMARK_USE_CACHE_FOR_READ,
)

ENTRY_POINT_NAME = "_entry_point_"

# XXX: Global mutable state
_inference_operations = []

_is_measuring_time = False


class Benchmark:
    def __init__(self, enabled):
        self._enabled = enabled
        self._elapses = {}
        self._warmup = False

    def timeit_with_options(self, name=None, is_read_operation=False):
        # TODO: Refactor
        def _deco(func_or_cls):
            if not self._enabled:
                return func_or_cls

            nonlocal name
            if name is None:
                name = func_or_cls.__qualname__

            if isinstance(func_or_cls, type):
                if not hasattr(func_or_cls, "__call__"):
                    raise TypeError
                func = func_or_cls.__call__
            else:
                if not callable(func_or_cls):
                    raise TypeError
                func = func_or_cls

            try:
                source_file = inspect.getsourcefile(func)
                source_line = inspect.getsourcelines(func)[1]
                location = f"{source_file}:{source_line}"
            except (TypeError, OSError) as e:
                location = uuid.uuid4().hex
                logging.debug(
                    f"Benchmark: failed to get source file and line number: {e}"
                )

            use_cache = is_read_operation and INFER_BENCHMARK_USE_CACHE_FOR_READ
            if use_cache:
                if inspect.isgeneratorfunction(func):
                    raise RuntimeError(
                        "When `is_read_operation` is `True`, the wrapped function should not be a generator."
                    )

                func = functools.lru_cache(maxsize=128)(func)

                @functools.wraps(func)
                def _wrapper(*args, **kwargs):
                    args = tuple(
                        tuple(arg) if isinstance(arg, list) else arg for arg in args
                    )
                    kwargs = {
                        k: tuple(v) if isinstance(v, list) else v
                        for k, v in kwargs.items()
                    }
                    output = func(*args, **kwargs)
                    output = copy.deepcopy(output)
                    return output

            else:

                @functools.wraps(func)
                def _wrapper(*args, **kwargs):
                    global _is_measuring_time
                    operation_name = f"{name}@{location}"
                    if _is_measuring_time:
                        raise RuntimeError(
                            "Nested calls detected: Check the timed modules and exclude nested calls to prevent double-counting."
                        )
                    if not operation_name.startswith(f"{ENTRY_POINT_NAME}@"):
                        _is_measuring_time = True
                    tic = time.perf_counter()
                    try:
                        output = func(*args, **kwargs)
                    finally:
                        if not operation_name.startswith(f"{ENTRY_POINT_NAME}@"):
                            _is_measuring_time = False
                    if isinstance(output, GeneratorType):
                        return self.watch_generator(output, operation_name)
                    else:
                        self._update(time.perf_counter() - tic, operation_name)
                        return output

            if isinstance(func_or_cls, type):
                func_or_cls.__call__ = _wrapper
                return func_or_cls
            else:
                return _wrapper

        return _deco

    def timeit(self, func_or_cls):
        return self.timeit_with_options()(func_or_cls)

    def watch_generator(self, generator, name):
        @functools.wraps(generator)
        def wrapper():
            global _is_measuring_time
            while True:
                try:
                    if _is_measuring_time:
                        raise RuntimeError(
                            "Nested calls detected: Check the timed modules and exclude nested calls to prevent double-counting."
                        )
                    if not name.startswith(f"{ENTRY_POINT_NAME}@"):
                        _is_measuring_time = True
                    tic = time.perf_counter()
                    try:
                        item = next(generator)
                    finally:
                        if not name.startswith(f"{ENTRY_POINT_NAME}@"):
                            _is_measuring_time = False
                    self._update(time.perf_counter() - tic, name)
                    yield item
                except StopIteration:
                    break

        return wrapper()

    def reset(self):
        self._elapses = {}

    def _update(self, elapse, name):
        elapse = elapse * 1000
        if name in self._elapses:
            self._elapses[name].append(elapse)
        else:
            self._elapses[name] = [elapse]

    @property
    def logs(self):
        return self._elapses

    def start_timing(self):
        self._enabled = True

    def stop_timing(self):
        self._enabled = False

    def start_warmup(self):
        self._warmup = True

    def stop_warmup(self):
        self._warmup = False
        self.reset()

    def gather(self, batch_size):
        # NOTE: The gathering logic here is based on the following assumptions:
        # 1. The operations are performed sequentially.
        # 2. An operation is performed only once at each iteration.
        # 3. Operations do not nest, except that the entry point operation
        #    contains all other operations.
        # 4. The input batch size for each operation is `batch_size`.
        # 5. Preprocessing operations are always performed before inference
        #    operations, and inference operations are completed before
        #    postprocessing operations. There is no interleaving among these
        #    stages.

        logs = {k: v for k, v in self.logs.items()}

        summary = {"preprocessing": 0, "inference": 0, "postprocessing": 0}
        for key in logs:
            if key.startswith(f"{ENTRY_POINT_NAME}@"):
                base_predictor_time_list = logs.pop(key)
                break
        iters = len(base_predictor_time_list)
        instances = iters * batch_size
        summary["end_to_end"] = np.mean(base_predictor_time_list)
        detail_list = []
        operation_list = []
        op_tag = "preprocessing"

        for name, time_list in logs.items():
            assert len(time_list) == iters
            avg = np.mean(time_list)
            operation_name = name.split("@")[0]
            location = name.split("@")[1]
            if ":" not in location:
                location = "Unknown"
            detail_list.append(
                (iters, batch_size, instances, operation_name, avg, avg / batch_size)
            )
            operation_list.append((operation_name, location))

            if operation_name in _inference_operations:
                summary["inference"] += avg
                op_tag = "postprocessing"
            else:
                summary[op_tag] += avg

        summary["core"] = (
            summary["preprocessing"] + summary["inference"] + summary["postprocessing"]
        )

        summary["other"] = summary["end_to_end"] - summary["core"]

        summary_list = [
            (
                iters,
                batch_size,
                instances,
                "Preprocessing",
                summary["preprocessing"],
                summary["preprocessing"] / batch_size,
            ),
            (
                iters,
                batch_size,
                instances,
                "Inference",
                summary["inference"],
                summary["inference"] / batch_size,
            ),
            (
                iters,
                batch_size,
                instances,
                "Postprocessing",
                summary["postprocessing"],
                summary["postprocessing"] / batch_size,
            ),
            (
                iters,
                batch_size,
                instances,
                "Core",
                summary["core"],
                summary["core"] / batch_size,
            ),
            (
                iters,
                batch_size,
                instances,
                "Other",
                summary["other"],
                summary["other"] / batch_size,
            ),
            (
                iters,
                batch_size,
                instances,
                "End-to-End",
                summary["end_to_end"],
                summary["end_to_end"] / batch_size,
            ),
        ]

        return detail_list, summary_list, operation_list

    def collect(self, batch_size):
        detail_list, summary_list, operation_list = self.gather(batch_size)

        if self._warmup:
            summary_head = [
                "Iters",
                "Batch Size",
                "Instances",
                "Type",
                "Avg Time Per Iter (ms)",
                "Avg Time Per Instance (ms)",
            ]
            table = PrettyTable(summary_head)
            summary_list = [
                i[:4] + (f"{i[4]:.8f}", f"{i[5]:.8f}") for i in summary_list
            ]
            table.add_rows(summary_list)
            table_title = "Warmup Data".center(len(str(table).split("\n")[0]), " ")
            logging.info(table_title)
            logging.info(table)

        else:
            operation_head = [
                "Operation",
                "Source Code Location",
            ]
            table = PrettyTable(operation_head)
            table.add_rows(operation_list)
            table_title = "Operation Info".center(len(str(table).split("\n")[0]), " ")
            logging.info(table_title)
            logging.info(table)

            detail_head = [
                "Iters",
                "Batch Size",
                "Instances",
                "Operation",
                "Avg Time Per Iter (ms)",
                "Avg Time Per Instance (ms)",
            ]
            table = PrettyTable(detail_head)
            detail_list = [i[:4] + (f"{i[4]:.8f}", f"{i[5]:.8f}") for i in detail_list]
            table.add_rows(detail_list)
            table_title = "Detail Data".center(len(str(table).split("\n")[0]), " ")
            logging.info(table_title)
            logging.info(table)

            summary_head = [
                "Iters",
                "Batch Size",
                "Instances",
                "Type",
                "Avg Time Per Iter (ms)",
                "Avg Time Per Instance (ms)",
            ]
            table = PrettyTable(summary_head)
            summary_list = [
                i[:4] + (f"{i[4]:.8f}", f"{i[5]:.8f}") for i in summary_list
            ]
            table.add_rows(summary_list)
            table_title = "Summary Data".center(len(str(table).split("\n")[0]), " ")
            logging.info(table_title)
            logging.info(table)

            if INFER_BENCHMARK_OUTPUT_DIR:
                save_dir = Path(INFER_BENCHMARK_OUTPUT_DIR)
                save_dir.mkdir(parents=True, exist_ok=True)
                csv_data = [detail_head, *detail_list]
                with open(Path(save_dir) / "detail.csv", "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows(csv_data)

                csv_data = [summary_head, *summary_list]
                with open(Path(save_dir) / "summary.csv", "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows(csv_data)


def get_inference_operations():
    return _inference_operations


def set_inference_operations(val):
    global _inference_operations
    _inference_operations = val


if INFER_BENCHMARK:
    benchmark = Benchmark(enabled=True)
else:
    benchmark = Benchmark(enabled=False)
