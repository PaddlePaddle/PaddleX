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

import numpy as np

from ....utils import logging
from ....utils.cache import CACHE_DIR
from ....utils.download import download
from ...utils.io import PDFReader
from .base_batch_sampler import BaseBatchSampler, Batch


class ImgBatch(Batch):
    def __init__(self):
        super().__init__()
        self.page_indexes = []

    def append(self, instance, input_path, page_index):
        super().append(instance, input_path)
        self.page_indexes.append(page_index)

    def reset(self):
        super().reset()
        self.page_indexes = []


class ImageBatchSampler(BaseBatchSampler):

    IMG_SUFFIX = ["jpg", "png", "jpeg", "bmp"]
    PDF_SUFFIX = ["pdf"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pdf_reader = PDFReader()

    # XXX: auto download for url
    def _download_from_url(self, in_path):
        file_name = Path(in_path).name
        save_path = Path(CACHE_DIR) / "predict_input" / file_name
        download(in_path, save_path, overwrite=True)
        return save_path.as_posix()

    def _get_files_list(self, fp):
        if fp is None or not os.path.exists(fp):
            raise Exception(f"Not found any files in path: {fp}")
        if os.path.isfile(fp):
            return [fp]

        file_list = []
        if os.path.isdir(fp):
            for root, dirs, files in os.walk(fp):
                for single_file in files:
                    if (
                        single_file.split(".")[-1].lower()
                        in self.IMG_SUFFIX + self.PDF_SUFFIX
                    ):
                        file_list.append(os.path.join(root, single_file))
        if len(file_list) == 0:
            raise Exception("Not found any file in {}".format(fp))
        file_list = sorted(file_list)
        return file_list

    def sample(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]

        batch = ImgBatch()
        for input in inputs:
            if isinstance(input, np.ndarray):
                batch.append(input, None, None)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = ImgBatch()
            elif isinstance(input, str):
                suffix = input.split(".")[-1].lower()
                if suffix in self.PDF_SUFFIX:
                    file_path = (
                        self._download_from_url(input)
                        if input.startswith("http")
                        else input
                    )
                    for page_idx, page_img in enumerate(
                        self.pdf_reader.read(file_path)
                    ):
                        batch.append(page_img, file_path, page_idx)
                        if len(batch) == self.batch_size:
                            yield batch
                            batch = ImgBatch()
                elif suffix in self.IMG_SUFFIX:
                    file_path = (
                        self._download_from_url(input)
                        if input.startswith("http")
                        else input
                    )
                    batch.append(file_path, file_path, None)
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = ImgBatch()
                else:
                    file_list = self._get_files_list(input)
                    yield from self.sample(file_list)
            else:
                logging.warning(
                    f"Not supported input data type! Only `numpy.ndarray` and `str` are supported! So has been ignored: {input}."
                )
        if len(batch) > 0:
            yield batch
