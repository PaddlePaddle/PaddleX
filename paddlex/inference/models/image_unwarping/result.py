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

import numpy as np

from ...common.result import BaseCVResult, JsonMixin


class DocTrResult(BaseCVResult):
    """
    Result class for DocTr, encapsulating the output of a document image processing task.

    Attributes:
        (inherited from BaseCVResult): Any attributes defined in the base class.

    Methods:
        _to_img(self) -> np.ndarray:
            Converts the stored image result to a numpy array.
    """

    def _to_img(self) -> np.ndarray:
        result = np.array(self["doctr_img"])
        return {"res": result}

    def _to_str(self, *args, **kwargs):
        data = copy.deepcopy(self)
        data.pop("input_img")
        data["doctr_img"] = "..."
        return JsonMixin._to_str(data, *args, **kwargs)

    def _to_json(self, *args, **kwargs):
        data = copy.deepcopy(self)
        data.pop("input_img")
        return JsonMixin._to_json(data, *args, **kwargs)
