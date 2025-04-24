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
from pathlib import Path

import pandas as pd

from ....utils import logging
from ....utils.cache import CACHE_DIR
from ....utils.download import download
from .base_batch_sampler import BaseBatchSampler, Batch


class TSBatchSampler(BaseBatchSampler):
    """Batch sampler for time series data, supporting CSV file inputs."""

    SUFFIX = ["csv"]

    def _download_from_url(self, in_path: str) -> str:
        """Download a file from a URL to a cache directory.

        Args:
            in_path (str): URL of the file to be downloaded.

        Returns:
            str: Path to the downloaded file.
        """
        file_name = Path(in_path).name
        save_path = Path(CACHE_DIR) / "predict_input" / file_name
        download(in_path, save_path, overwrite=True)
        return save_path.as_posix()

    def _get_files_list(self, fp: str) -> list:
        """Get a list of CSV files from a directory or a single file path.

        Args:
            fp (str): Path to a directory or a single CSV file.

        Returns:
            list: Sorted list of CSV file paths.

        Raises:
            Exception: If no CSV file is found in the path.
        """
        file_list = []
        if fp is None or not os.path.exists(fp):
            raise Exception(f"Not found any csv file in path: {fp}")

        if os.path.isfile(fp) and fp.split(".")[-1] in self.SUFFIX:
            file_list.append(fp)
        elif os.path.isdir(fp):
            for root, dirs, files in os.walk(fp):
                for single_file in files:
                    if single_file.split(".")[-1] in self.SUFFIX:
                        file_list.append(os.path.join(root, single_file))
        if len(file_list) == 0:
            raise Exception("Not found any file in {}".format(fp))
        file_list = sorted(file_list)
        return file_list

    def sample(self, inputs: list) -> list:
        """Generate batches of data from inputs, which can be DataFrames or file paths.

        Args:
            inputs (list): List of DataFrames or file paths.

        Yields:
            list: A batch of data which is either DataFrames or file paths.
        """
        if not isinstance(inputs, list):
            inputs = [inputs]

        batch = Batch()
        for input in inputs:
            if isinstance(input, pd.DataFrame):
                batch.append(input, None)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = Batch()
            elif isinstance(input, str):
                file_path = (
                    self._download_from_url(input)
                    if input.startswith("http")
                    else input
                )
                file_list = self._get_files_list(file_path)
                for file_path in file_list:
                    batch.append(file_path, file_path)
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = Batch()
            else:
                logging.warning(
                    f"Not supported input data type! Only `pd.DataFrame` and `str` are supported! So has been ignored: {input}."
                )
        if len(batch) > 0:
            yield batch
