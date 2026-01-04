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

from ...common.result import BaseCVResult, JsonMixin
from ..pp_shitu_v2.result import draw_box


class FaceRecResult(BaseCVResult):

    def _to_img(self):
        """apply"""
        boxes = [
            {
                "coordinate": box["coordinate"],
                "label": box["labels"][0] if box["labels"] is not None else "Unknown",
                "score": box["rec_scores"][0] if box["rec_scores"] is not None else 0,
            }
            for box in self["boxes"]
        ]
        image = draw_box(self["input_img"][..., ::-1], boxes)
        return {"res": image}

    def _to_str(self, *args, **kwargs):
        data = copy.deepcopy(self)
        data.pop("input_img")
        return JsonMixin._to_str(data, *args, **kwargs)

    def _to_json(self, *args, **kwargs):
        data = copy.deepcopy(self)
        data.pop("input_img")
        return JsonMixin._to_json(data, *args, **kwargs)
