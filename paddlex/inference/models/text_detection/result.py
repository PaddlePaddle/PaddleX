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

from ....utils.deps import class_requires_deps, is_dep_available
from ...common.result import BaseCVResult, JsonMixin

if is_dep_available("opencv-contrib-python"):
    import cv2


@class_requires_deps("opencv-contrib-python")
class TextDetResult(BaseCVResult):

    def _to_img(self):
        """draw rectangle"""
        boxes = self["dt_polys"]
        image = self["input_img"]
        for box in boxes:
            box = np.reshape(np.array(box).astype(int), [-1, 1, 2]).astype(np.int64)
            cv2.polylines(image, [box], True, (0, 0, 255), 2)
        return {"res": image[:, :, ::-1]}

    def _to_str(self, *args, **kwargs):
        data = copy.deepcopy(self)
        data.pop("input_img")
        return JsonMixin._to_str(data, *args, **kwargs)

    def _to_json(self, *args, **kwargs):
        data = copy.deepcopy(self)
        data.pop("input_img")
        return JsonMixin._to_json(data, *args, **kwargs)
