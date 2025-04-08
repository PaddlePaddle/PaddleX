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


from typing import Dict, List, Tuple

import numpy as np

from .....utils.deps import class_requires_deps, is_dep_available

if is_dep_available("opencv-contrib-python"):
    import cv2


@class_requires_deps("opencv-contrib-python")
class LetterResize(object):
    def __init__(
        self,
        scale=[640, 640],
        pad_val=144,
        use_mini_pad=False,
        stretch_only=False,
        allow_scale_up=False,
    ):
        super(LetterResize, self).__init__()
        self.scale = scale
        self.pad_val = pad_val

        self.use_mini_pad = use_mini_pad
        self.stretch_only = stretch_only
        self.allow_scale_up = allow_scale_up

    def _resize_img(self, image: np.ndarray) -> Dict:

        scale = self.scale
        image_shape = image.shape[:2]

        ratio = min(scale[0] / image_shape[0], scale[1] / image_shape[1])

        if not self.allow_scale_up:
            ratio = min(ratio, 1.0)
        ratio = [ratio, ratio]

        no_pad_shape = (
            int(round(image_shape[0] * ratio[0])),
            int(round(image_shape[1] * ratio[1])),
        )
        padding_h, padding_w = [scale[0] - no_pad_shape[0], scale[1] - no_pad_shape[1]]
        if self.use_mini_pad:
            padding_w, padding_h = np.mod(padding_w, 32), np.mod(padding_h, 32)
        elif self.stretch_only:
            padding_h, padding_w = 0.0, 0.0
            no_pad_shape = (scale[0], scale[1])
            ratio = [scale[0] / image_shape[0], scale[1] / image_shape[1]]

        if image_shape != no_pad_shape:
            image = cv2.resize(
                image,
                (no_pad_shape[1], no_pad_shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        scale_factor = (
            no_pad_shape[1] / image_shape[1],
            no_pad_shape[0] / image_shape[0],
        )

        top_padding, left_padding = int(round(padding_h // 2 - 0.1)), int(
            round(padding_w // 2 - 0.1)
        )
        bottom_padding = padding_h - top_padding
        right_padding = padding_w - left_padding

        padding_list = [top_padding, bottom_padding, left_padding, right_padding]
        if (
            top_padding != 0
            or bottom_padding != 0
            or left_padding != 0
            or right_padding != 0
        ):
            pad_val = self.pad_val
            if isinstance(pad_val, int) and image.ndim == 3:
                pad_val = tuple(pad_val for _ in range(image.shape[2]))
            top, bottom, left, right = padding_list
            image = cv2.copyMakeBorder(
                image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_val
            )

        result = dict()
        result["image"] = image
        result["scale_factor"] = np.array(scale_factor, dtype=np.float32)
        result["pad_param"] = np.array(padding_list, dtype=np.float32)

        return result

    def __call__(self, images: List[np.ndarray]) -> List[Dict]:

        if not isinstance(images, (List, Tuple)):
            images = [images]

        rst_images = [self._resize_img(image) for image in images]

        return rst_images
