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

from ....utils.download import download
from ....utils.cache import CACHE_DIR
from ..utils.mixin import BatchSizeMixin
from ..base import BaseComponent


class _BaseRead(BaseComponent, BatchSizeMixin):
    """Load image from the file."""

    SUFFIX = []

    def __init__(self, batch_size=1):
        super().__init__()
        BatchSizeMixin.__init__(self, batch_size)

    # XXX: auto download for url
    def _download_from_url(self, in_path):
        if in_path.startswith("http"):
            file_name = Path(in_path).name
            save_path = Path(CACHE_DIR) / "predict_input" / file_name
            download(in_path, save_path, overwrite=True)
            return save_path.as_posix()
        return in_path

    def _get_files_list(self, fp):
        file_list = []
        if fp is None or not os.path.exists(fp):
            raise Exception(f"Not found any img file in path: {fp}")

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
