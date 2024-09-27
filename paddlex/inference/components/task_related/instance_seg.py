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

import numpy as np
from ....utils import logging
from ..base import BaseComponent
from .det import restructured_boxes


def extract_masks_from_boxes(boxes, masks):
    """
    Extracts the portion of each mask that is within the corresponding box.
    """
    new_masks = []

    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = box["coordinate"]
        x_min, y_min, x_max, y_max = map(
            lambda x: int(round(x)), [x_min, y_min, x_max, y_max]
        )

        cropped_mask = masks[i][y_min:y_max, x_min:x_max]
        new_masks.append(cropped_mask)

    return new_masks


class InstanceSegPostProcess(BaseComponent):
    """Save Result Transform"""

    INPUT_KEYS = ["boxes", "masks", "img_size"]
    OUTPUT_KEYS = ["img_path", "boxes", "masks"]
    DEAULT_INPUTS = {"boxes": "boxes", "masks": "masks", "img_size": "ori_img_size"}
    DEAULT_OUTPUTS = {
        "boxes": "boxes",
        "masks": "masks",
    }

    def __init__(self, threshold=0.5, labels=None):
        super().__init__()
        self.threshold = threshold
        self.labels = labels

    def apply(self, boxes, masks, img_size):
        """apply"""
        expect_boxes = (boxes[:, 1] > self.threshold) & (boxes[:, 0] > -1)
        boxes = boxes[expect_boxes, :]
        boxes = restructured_boxes(boxes, self.labels, img_size)
        masks = masks[expect_boxes, :, :]
        masks = extract_masks_from_boxes(boxes, masks)
        result = {"boxes": boxes, "masks": masks}

        return result
