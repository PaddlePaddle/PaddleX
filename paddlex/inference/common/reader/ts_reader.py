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

import pandas as pd

from ...utils.benchmark import benchmark
from ...utils.io import CSVReader


@benchmark.timeit_with_options(name=None, is_read_operation=True)
class ReadTS:

    def __init__(self):
        super().__init__()
        self._reader = CSVReader(backend="pandas")

    def __call__(self, ts_list):
        """apply"""
        return [self.read(ts) for ts in ts_list]

    def read(self, ts):
        if isinstance(ts, pd.DataFrame):
            return ts
        elif isinstance(ts, str):
            ts_data = self._reader.read(ts)
            if ts_data is None:
                raise Exception(f"TS read Error: {ts}")
            return ts_data
        else:
            raise TypeError(
                f"ReadTS only supports the following types:\n"
                f"1. str, indicating a CSV file path or a directory containing CSV files.\n"
                f"2. pandas.DataFrame.\n"
                f"However, got type: {type(ts).__name__}."
            )
