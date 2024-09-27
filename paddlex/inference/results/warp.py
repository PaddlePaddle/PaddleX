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

import numpy as np
import copy
import json

from ...utils import logging
from .base import BaseResult


class DocTrResult(BaseResult):
    def __init__(self, data):
        super().__init__(data)
        # We use opencv backend to save both numpy arrays
        self._img_writer.set_backend("opencv")

    def _get_res_img(self):
        doctr_img = np.array(self["doctr_img"])
        return doctr_img

    def print(self, json_format=True, indent=4, ensure_ascii=False):
        str_ = copy.deepcopy(self)
        del str_["doctr_img"]
        if json_format:
            str_ = json.dumps(str_, indent=indent, ensure_ascii=ensure_ascii)
        logging.info(str_)
