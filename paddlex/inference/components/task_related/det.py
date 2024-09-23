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

from ....utils import logging
from ...utils.io import ImageReader
from ..base import BaseComponent


class DetPostProcess(BaseComponent):
    """Save Result Transform"""

    INPUT_KEYS = ["img_path", "boxes"]
    OUTPUT_KEYS = ["boxes", "labels"]
    DEAULT_INPUTS = {"boxes": "boxes"}
    DEAULT_OUTPUTS = {
        "boxes": "boxes",
        "labels": "labels",
    }

    def __init__(self, threshold=0.5, labels=None):
        super().__init__()
        self.threshold = threshold
        self.labels = labels

    def apply(self, boxes):
        """apply"""
        expect_boxes = (boxes[:, 1] > self.threshold) & (boxes[:, 0] > -1)
        boxes = boxes[expect_boxes, :]
        result = {"boxes": boxes, "labels": self.labels}

        return result


class CropByBoxes(BaseComponent):
    """Crop Image by Box"""

    INPUT_KEYS = ["img_path", "boxes", "labels"]
    OUTPUT_KEYS = ["img", "box", "label"]
    DEAULT_INPUTS = {"img_path": "img_path", "boxes": "boxes", "labels": "labels"}
    DEAULT_OUTPUTS = {"img": "img", "box": "box", "label": "label"}

    def __init__(self):
        super().__init__()
        self._reader = ImageReader(backend="opencv")

    def apply(self, img_path, boxes, labels=None):
        output_list = []
        img = self._reader.read(img_path)
        for bbox in boxes:
            label_id = int(bbox[0])
            box = bbox[2:]
            if labels is not None:
                label = labels[label_id]
            else:
                label = label_id
            xmin, ymin, xmax, ymax = [int(i) for i in box]
            img_crop = img[ymin:ymax, xmin:xmax]
            output_list.append({"img": img_crop, "box": box, "label": label})

        return output_list
