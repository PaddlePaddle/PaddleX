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
from .base import CVResult
from .det import draw_box


class FaceRecResult(CVResult):

    def _to_img(self):
        """apply"""
        image = self._img_reader.read(self["input_path"])
        boxes = [
            {
                "coordinate": box["coordinate"],
                "label": box["labels"][0] if box["labels"] is not None else "Unknown",
                "score": box["rec_scores"][0] if box["rec_scores"] is not None else 0,
                "cls_id": box["rec_ids"][0] if box["rec_ids"] is not None else 0 # rec ids start from 1
            }
            for box in self["boxes"]
        ]
        image = draw_box(image, boxes)
        return image
