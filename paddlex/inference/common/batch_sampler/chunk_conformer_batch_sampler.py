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
from pathlib import Path
import numpy as np

from ....utils import logging
from ....utils.download import download
from ....utils.cache import CACHE_DIR
from .base_batch_sampler import BaseBatchSampler


class ChunkConformerBatchSampler(BaseBatchSampler):
    def __init__(self):
        """Initializes the ChunkConformerBatchSampler"""
        super().__init__()
        self.batch_size = 1

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

    def sample(self, inputs: str) -> Iterator[List[str]]:
        """Generate list of input file paths.

        Args:
            inputs (str): file path.

        Yields:
            list: list of file paths.
        """
        if isinstance(inputs, str):
            if inputs.startswith("http"):
                inputs = self._download_from_url(inputs)
            yield [inputs]
        else:
            logging.warning(
                f"Not supported input data type! Only `str` are supported, but got: {type(inputs)}."
            )

    @BaseBatchSampler.batch_size.setter
    def batch_size(self, batch_size: int) -> None:
        """Sets the batch size with validation"""
        if batch_size != 1:
            logging.warning(
                f"ChunkConformer sampler only supports batch size 1, but got {batch_size}."
            )
        else:
            self._batch_size = batch_size
