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
import numpy as np
from prettytable import PrettyTable

from ...utils.flags import INFER_BENCHMARK, INFER_BENCHMARK_OUTPUT
from ...utils import logging

# XXX: Global mutable state
_inference_operations = []


class Benchmark:
    def __init__(self, enabled):
        self._enabled = enabled
        self._elapses = {}
        self._warmup = False

    def timeit(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self._enabled:
                return func(*args, **kwargs)

            name = func.__qualname__

            tic = time.time()
            output = func(*args, **kwargs)
            if isinstance(output, GeneratorType):
                return self.watch_generator(output, name)
            else:
                self._update(time.time() - tic, name)
                return output

        return wrapper

    def watch_generator(self, generator, name):
        @functools.wraps(generator)
        def wrapper():
            while True:
                try:
                    tic = time.time()
                    item = next(generator)
                    self._update(time.time() - tic, name)
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
        logs = {k.split(".")[0]: v for k, v in self.logs.items()}

        detail_list = []
        summary = {"preprocessing": 0, "inference": 0, "postprocessing": 0}
        op_tag = "preprocessing"

        for name, time_list in logs.items():
            iters = len(time_list)
            instances = iters * batch_size
            avg = np.mean(time_list)
            detail_list.append(
                (iters, batch_size, instances, name, avg, avg / batch_size)
            )

            if name in _inference_operations:
                summary["inference"] += avg
                op_tag = "postprocessing"
            else:
                summary[op_tag] += avg

        summary["end_to_end"] = (
            summary["preprocessing"] + summary["inference"] + summary["postprocessing"]
        )
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
                "End-to-End",
                summary["end_to_end"],
                summary["end_to_end"] / batch_size,
            ),
        ]

        return detail_list, summary_list

    def collect(self, batch_size):
        detail_list, summary_list = self.gather(batch_size)

        if self._warmup:
            summary_head = [
                "Iters",
                "Batch Size",
                "Instances",
                "Stage",
                "Avg Time Per Iter (ms)",
                "Avg Time Per Instance (ms)",
            ]
            table = PrettyTable(summary_head)
            summary_list = [
                i[:4] + (f"{i[4]:.8f}", f"{i[5]:.8f}") for i in summary_list
            ]
            table.add_rows(summary_list)
            header = "WarmUp Data".center(len(str(table).split("\n")[0]), " ")
            logging.info(header)
            logging.info(table)

        else:
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
            header = "Detail Data".center(len(str(table).split("\n")[0]), " ")
            logging.info(header)
            logging.info(table)

            summary_head = [
                "Iters",
                "Batch Size",
                "Instances",
                "Stage",
                "Avg Time Per Iter (ms)",
                "Avg Time Per Instance (ms)",
            ]
            table = PrettyTable(summary_head)
            summary_list = [
                i[:4] + (f"{i[4]:.8f}", f"{i[5]:.8f}") for i in summary_list
            ]
            table.add_rows(summary_list)
            header = "Summary Data".center(len(str(table).split("\n")[0]), " ")
            logging.info(header)
            logging.info(table)

            if INFER_BENCHMARK_OUTPUT:
                save_dir = Path(INFER_BENCHMARK_OUTPUT)
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
