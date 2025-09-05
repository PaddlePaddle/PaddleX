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
    PIPELINE_BENCHMARK,
)

ENTRY_POINT_NAME = "_entry_point_"

# XXX: Global mutable state
_inference_operations = []

_is_measuring_time = False

PIPELINE_FUNC_BLACK_LIST = ["inintial_predictor"]
_step = 0
_level = 0
_top_func = None


class Benchmark:
    def __init__(self, enabled):
        self._enabled = enabled
        self._elapses = {}
        self._warmup = False
        self._detail_list = []
        self._summary_list = []
        self._operation_list = []

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
                if INFER_BENCHMARK:

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

                elif PIPELINE_BENCHMARK:

                    @functools.wraps(func)
                    def _wrapper(*args, **kwargs):
                        global _step, _level, _top_func

                        _step += 1
                        _level += 1

                        if _level == 1:
                            if _top_func is None:
                                _top_func = f"{name}@{location}"
                            elif _top_func != f"{name}@{location}":
                                raise RuntimeError(
                                    f"Multiple top-level function calls detected:\n"
                                    f"  Function 1: {_top_func.split('@')[0]}\n"
                                    f"    Location: {_top_func.split('@')[1]}\n"
                                    f"  Function 2: {name}\n"
                                    f"    Location: {location}\n"
                                    "Only one top-level function can be tracked at a time.\n"
                                    "Please call 'benchmark.reset()' between top-level function calls."
                                )

                        operation_name = f"{_step}@{_level}@{name}@{location}"

                        tic = time.perf_counter()
                        output = func(*args, **kwargs)
                        if isinstance(output, GeneratorType):
                            return self.watch_generator_simple(output, operation_name)
                        else:
                            self._update(time.perf_counter() - tic, operation_name)
                            _level -= 1

                        return output

            if isinstance(func_or_cls, type):
                func_or_cls.__call__ = _wrapper
                return func_or_cls
            else:
                return _wrapper

        return _deco

    def timeit(self, func_or_cls):
        return self.timeit_with_options()(func_or_cls)

    def _is_public_method(self, name):
        return not name.startswith("_")

    def time_methods(self, cls):
        for name, func in cls.__dict__.items():
            if (
                callable(func)
                and self._is_public_method(name)
                and not name.startswith("__")
                and name not in PIPELINE_FUNC_BLACK_LIST
            ):
                setattr(cls, name, self.timeit(func))
        return cls

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

    def watch_generator_simple(self, generator, name):
        @functools.wraps(generator)
        def wrapper():
            global _level

            try:
                while True:
                    tic = time.perf_counter()
                    try:
                        item = next(generator)
                    except StopIteration:
                        break
                    self._update(time.perf_counter() - tic, name)
                    yield item
            finally:
                _level -= 1

        return wrapper()

    def reset(self):
        global _step, _level, _top_func

        _step = 0
        _level = 0
        _top_func = None
        self._elapses = {}
        self._detail_list = []
        self._summary_list = []
        self._operation_list = []

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

    def gather_pipeline(self):
        info_list = []
        detail_list = []
        operation_list = set()
        summary_list = []
        max_level = 0
        loop_num = 0

        for name, time_list in self.logs.items():
            op_time = np.sum(time_list)

            parts = name.split("@")
            step = int(parts[0])
            level = int(parts[1])
            operation_name = parts[2]
            location = parts[3]
            if ":" not in location:
                location = "Unknown"

            operation_list.add((operation_name, location))
            max_level = max(level, max_level)

            if level == 1:
                loop_num += 1
                format_operation_name = operation_name
            else:
                format_operation_name = "    " * int(level - 1) + "-> " + operation_name
            info_list.append(
                (step, level, operation_name, format_operation_name, op_time)
            )

        operation_list = list(operation_list)
        info_list.sort(key=lambda x: x[0])
        step_num = int(len(info_list) / loop_num)
        for idx in range(step_num):
            step = info_list[idx][0]
            format_operation_name = info_list[idx][3]
            op_time = (
                np.sum(
                    [info_list[pos][4] for pos in range(idx, len(info_list), step_num)]
                )
                / loop_num
            )
            detail_list.append([step, format_operation_name, op_time])

        level_time_list = [[0] for _ in range(max_level)]
        for idx, info in enumerate(info_list):
            step = info[0]
            level = info[1]
            operation_name = info[2]
            op_time = info[4]

            # The total time consumed by all operations on this layer
            if level > info_list[idx - 1][1]:
                level_time_list[level - 1].append(info_list[idx - 1][4])

            # The total time consumed by each operation on this layer
            while len(summary_list) < level:
                summary_list.append([len(summary_list) + 1, {}])
            if summary_list[level - 1][1].get(operation_name, None) is None:
                summary_list[level - 1][1][operation_name] = [op_time]
            else:
                summary_list[level - 1][1][operation_name].append(op_time)

        new_summary_list = []
        for i in range(len(summary_list)):
            level = summary_list[i][0]
            op_dict = summary_list[i][1]

            ops_all_time = 0.0
            op_info_list = []
            for idx, (name, time_list) in enumerate(op_dict.items()):
                op_all_time = np.sum(time_list) / loop_num
                op_info_list.append([level if i + idx == 0 else "", name, op_all_time])
                ops_all_time += op_all_time

            if i > 0:
                new_summary_list.append(["", "", ""])
                new_summary_list.append(
                    [level, "Layer", np.sum(level_time_list[i]) / loop_num]
                )
                new_summary_list.append(["", "Core", ops_all_time])
                new_summary_list.append(
                    ["", "Other", np.sum(level_time_list[i]) / loop_num - ops_all_time]
                )
            new_summary_list += op_info_list

        return detail_list, new_summary_list, operation_list

    def _initialize_pipeline_data(self):
        if not (self._operation_list and self._detail_list and self._summary_list):
            self._detail_list, self._summary_list, self._operation_list = (
                self.gather_pipeline()
            )

    def print_pipeline_data(self):
        self._initialize_pipeline_data()
        self.print_operation_info()
        self.print_detail_data()
        self.print_summary_data()

    def print_operation_info(self):
        self._initialize_pipeline_data()
        operation_head = [
            "Operation",
            "Source Code Location",
        ]
        table = PrettyTable(operation_head)
        table.add_rows(self._operation_list)
        table_title = "Operation Info".center(len(str(table).split("\n")[0]), " ")
        logging.info(table_title)
        logging.info(table)

    def print_detail_data(self):
        self._initialize_pipeline_data()
        detail_head = [
            "Step",
            "Operation",
            "Time (ms)",
        ]
        table = PrettyTable(detail_head)
        table.add_rows(self._detail_list)
        table_title = "Detail Data".center(len(str(table).split("\n")[0]), " ")
        table.align["Operation"] = "l"
        table.align["Time (ms)"] = "l"
        logging.info(table_title)
        logging.info(table)

    def print_summary_data(self):
        self._initialize_pipeline_data()
        summary_head = [
            "Level",
            "Operation",
            "Time (ms)",
        ]
        table = PrettyTable(summary_head)
        table.add_rows(self._summary_list)
        table_title = "Summary Data".center(len(str(table).split("\n")[0]), " ")
        table.align["Operation"] = "l"
        table.align["Time (ms)"] = "l"
        logging.info(table_title)
        logging.info(table)

    def save_pipeline_data(self, save_path):
        self._initialize_pipeline_data()
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        detail_head = [
            "Step",
            "Operation",
            "Time (ms)",
        ]
        csv_data = [detail_head, *self._detail_list]
        with open(Path(save_dir) / "detail.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(csv_data)

        summary_head = [
            "Level",
            "Operation",
            "Time (ms)",
        ]
        csv_data = [summary_head, *self._summary_list]
        with open(Path(save_dir) / "summary.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(csv_data)


def get_inference_operations():
    return _inference_operations


def set_inference_operations(val):
    global _inference_operations
    _inference_operations = val


if INFER_BENCHMARK or PIPELINE_BENCHMARK:
    benchmark = Benchmark(enabled=True)
else:
    benchmark = Benchmark(enabled=False)
