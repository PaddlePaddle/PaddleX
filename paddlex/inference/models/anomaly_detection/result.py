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
from PIL import Image

from ...common.result import BaseCVResult, JsonMixin


class UadResult(BaseCVResult):
    """Save Result Transform"""

    def _to_img(self):
        """apply"""
        seg_map = self["pred"]
        pc_map = self.get_pseudo_color_map(seg_map[0])
        return {"res": pc_map}

    def get_pseudo_color_map(self, pred):
        """get_pseudo_color_map"""
        if pred.min() < 0 or pred.max() > 255:
            raise ValueError("`pred` cannot be cast to uint8.")
        pred = pred.astype(np.uint8)
        pred_mask = Image.fromarray(pred, mode="P")
        color_map = self._get_color_map_list(256)
        pred_mask.putpalette(color_map)
        return pred_mask

    @staticmethod
    def _get_color_map_list(num_classes, custom_color=None):
        """_get_color_map_list"""
        num_classes += 1
        color_map = num_classes * [0, 0, 0]
        for i in range(0, num_classes):
            j = 0
            lab = i
            while lab:
                color_map[i * 3] |= ((lab >> 0) & 1) << (7 - j)
                color_map[i * 3 + 1] |= ((lab >> 1) & 1) << (7 - j)
                color_map[i * 3 + 2] |= ((lab >> 2) & 1) << (7 - j)
                j += 1
                lab >>= 3
        color_map = color_map[3:]

        if custom_color:
            color_map[: len(custom_color)] = custom_color
        return color_map

    def _to_str(self, *args, **kwargs):
        data = copy.deepcopy(self)
        data.pop("input_img")
        data["pred"] = "..."
        return JsonMixin._to_str(data, *args, **kwargs)

    def _to_json(self, *args, **kwargs):
        data = copy.deepcopy(self)
        data.pop("input_img")
        return JsonMixin._to_json(data, *args, **kwargs)
