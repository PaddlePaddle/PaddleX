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

import numpy as np

from ...utils.benchmark import benchmark
from ..common.vision import F


@benchmark.timeit
class Crop:
    """Crop region from the image."""

    def __init__(self, crop_size, mode="C"):
        """
        Initialize the instance.

        Args:
            crop_size (list|tuple|int): Width and height of the region to crop.
            mode (str, optional): 'C' for cropping the center part and 'TL' for
                cropping the top left part. Default: 'C'.
        """
        super().__init__()
        if isinstance(crop_size, int):
            crop_size = [crop_size, crop_size]
        F.check_image_size(crop_size)

        self.crop_size = crop_size

        if mode not in ("C", "TL"):
            raise ValueError("Unsupported interpolation method")
        self.mode = mode

    def __call__(self, imgs):
        """apply"""
        return [self.crop(img) for img in imgs]

    def crop(self, img):
        h, w = img.shape[:2]
        cw, ch = self.crop_size
        if self.mode == "C":
            x1 = max(0, (w - cw) // 2)
            y1 = max(0, (h - ch) // 2)
        elif self.mode == "TL":
            x1, y1 = 0, 0
        x2 = min(w, x1 + cw)
        y2 = min(h, y1 + ch)
        coords = (x1, y1, x2, y2)
        if w < cw or h < ch:
            raise ValueError(
                f"Input image ({w}, {h}) smaller than the target size ({cw}, {ch})."
            )
        img = F.slice(img, coords=coords)
        return img


@benchmark.timeit
class Topk:
    """Topk Transform"""

    def __init__(self, class_ids=None):
        super().__init__()
        self.class_id_map = self._parse_class_id_map(class_ids)

    def _parse_class_id_map(self, class_ids):
        """parse class id to label map file"""
        if class_ids is None:
            return None
        class_id_map = {id: str(lb) for id, lb in enumerate(class_ids)}
        return class_id_map

    def __call__(self, preds, topk=5):
        indexes = preds[0].argsort(axis=1)[:, -topk:][:, ::-1].astype("int32")
        scores = [
            np.around(pred[index], decimals=5) for pred, index in zip(preds[0], indexes)
        ]
        label_names = [[self.class_id_map[i] for i in index] for index in indexes]
        return indexes, scores, label_names
