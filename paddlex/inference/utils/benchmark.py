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

import csv
import functools
from types import GeneratorType
import time
from pathlib import Path
import inspect
import numpy as np
from prettytable import PrettyTable

from ...utils.flags import INFER_BENCHMARK, INFER_BENCHMARK_OUTPUT_DIR
from ...utils import logging

ENTRY_POINT_NAME = "_entry_point_"

# XXX: Global mutable state
_inference_operations = []


class Benchmark:
    def __init__(self, enabled):
        self._enabled = enabled
        self._elapses = {}
        self._warmup = False

    def timeit_with_name(self, name=None):
        # TODO: Refactor
        def _deco(func_or_cls):
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

            location = None

            @functools.wraps(func)
            def _wrapper(*args, **kwargs):
                nonlocal location

                if not self._enabled:
                    return func(*args, **kwargs)

                if location is None:
                    try:
                        source_file = inspect.getsourcefile(func)
                        source_line = inspect.getsourcelines(func)[1]
                        location = f"{source_file}:{source_line}"
                    except (TypeError, OSError) as e:
                        location = "Unknown"
                        logging.debug(
                            f"Benchmark: failed to get source file and line number: {e}"
                        )

                tic = time.perf_counter()
                output = func(*args, **kwargs)
                if isinstance(output, GeneratorType):
                    return self.watch_generator(output, f"{name}@{location}")
                else:
                    self._update(time.perf_counter() - tic, f"{name}@{location}")
                    return output

            if isinstance(func_or_cls, type):
                func_or_cls.__call__ = _wrapper
                return func_or_cls
            else:
                return _wrapper

        return _deco

    def timeit(self, func_or_cls):
        return self.timeit_with_name(None)(func_or_cls)

    def watch_generator(self, generator, name):
        @functools.wraps(generator)
        def wrapper():
            while True:
                try:
                    tic = time.perf_counter()
                    item = next(generator)
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
        # 5. Inference operations are always performed, while preprocessing and
        #    postprocessing operations are optional.
        # 6. If present, preprocessing operations are always performed before
        #    inference operations, and inference operations are completed before
        #    any postprocessing operations. There is no interleaving among these
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
            table_name = "WarmUp Data".center(len(str(table).split("\n")[0]), " ")
            logging.info(table_name)
            logging.info(table)

        else:
            operation_head = [
                "Operation",
                "Source Code Location",
            ]
            table = PrettyTable(operation_head)
            table.add_rows(operation_list)
            table_name = "Operation Info".center(len(str(table).split("\n")[0]), " ")
            logging.info(table_name)
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
            table_name = "Detail Data".center(len(str(table).split("\n")[0]), " ")
            logging.info(table_name)
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
            table_name = "Summary Data".center(len(str(table).split("\n")[0]), " ")
            logging.info(table_name)
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
