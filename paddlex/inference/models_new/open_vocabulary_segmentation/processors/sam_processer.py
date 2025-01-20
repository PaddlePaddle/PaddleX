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
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import PIL
from copy import deepcopy

from .....utils.lazy_loader import LazyLoader

# NOTE: LazyLoader is used to avoid conflicts between ultra-infer and Paddle
paddle = LazyLoader("lazy_paddle", globals(), "paddle")
T = LazyLoader("T", globals(), "paddle.vision.transforms")
F = LazyLoader("F", globals(), "paddle.nn.functional")


def _get_preprocess_shape(
    oldh: int, oldw: int, long_side_length: int
) -> Tuple[int, int]:
    """Compute the output size given input size and target long side length."""
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


class SAMProcessor(object):

    def __init__(
        self,
        size: Optional[Union[List[int], int]] = None,
        image_mean: Union[float, List[float]] = [123.675, 116.28, 103.53],
        image_std: Union[float, List[float]] = [58.395, 57.12, 57.375],
        **kwargs,
    ) -> None:

        size = size if size is not None else 1024
        self.size = size

        if isinstance(image_mean, float):
            image_mean = [image_mean] * 3
        if isinstance(image_std, float):
            image_std = [image_std] * 3

        self.image_mean = image_mean
        self.image_std = image_std

        self.image_processor = SamImageProcessor(
            self.size, self.image_mean, self.image_std
        )
        self.prompt_processor = SamPromptProcessor(self.size)

    def preprocess(
        self,
        images,
        *,
        point_prompt=None,
        box_prompt=None,
        **kwargs,
    ):

        if point_prompt is not None and box_prompt is not None:
            raise ValueError(
                "SAM can only use either points or boxes as prompt, not both at the same time."
            )
        if point_prompt is None and box_prompt is None:
            raise ValueError(
                "SAM must use either points or boxes as prompt, now both is None."
            )

        point_prompt = (
            np.array(point_prompt).reshape(-1, 2) if point_prompt is not None else None
        )
        box_prompt = (
            np.array(box_prompt).reshape(-1, 4) if box_prompt is not None else None
        )

        if point_prompt is not None and point_prompt.size > 2:
            raise ValueError(
                "SAM now only support one point for using point promot, your input format should be like [[x, y]] only."
            )

        image_seg = self.image_processor(images)
        self.original_size = self.image_processor.original_size
        self.input_size = self.image_processor.input_size
        prompt = self.prompt_processor(
            self.original_size,
            point_coords=point_prompt,
            box=box_prompt,
        )

        return image_seg, prompt

    def postprocess(self, low_res_masks, mask_threshold: float = 0.0):

        if isinstance(low_res_masks, list):
            assert len(low_res_masks) == 1
            low_res_masks = low_res_masks[0]

        masks = F.interpolate(
            paddle.to_tensor(low_res_masks),
            (self.size, self.size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : self.input_size[0], : self.input_size[1]]
        masks = F.interpolate(
            masks, self.original_size, mode="bilinear", align_corners=False
        )
        masks = (masks > mask_threshold).numpy().astype(np.int8)

        return [masks]


class SamPromptProcessor(object):
    """Constructs a Sam prompt processor."""

    def __init__(
        self,
        size: int = 1024,
    ):
        self.size = size

    def apply_coords(
        self, coords: np.ndarray, original_size: Tuple[int, ...]
    ) -> np.ndarray:
        """Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = _get_preprocess_shape(
            original_size[0], original_size[1], self.size
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(
        self, boxes: np.ndarray, original_size: Tuple[int, ...]
    ) -> np.ndarray:
        """Expects a numpy array shape Nx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape([-1, 2, 2]), original_size)
        return boxes.reshape([-1, 4])

    def __call__(
        self,
        original_size,
        point_coords=None,
        box=None,
        **kwargs,
    ):
        if point_coords is not None and box is not None:
            raise ValueError(
                "SAM can only use either points or boxes as prompt, not both at the same time."
            )
        if point_coords is not None:
            point_coords = self.apply_coords(point_coords, original_size)
            point_coords = point_coords[None, ...]
            return point_coords.astype(np.float32)

        if box is not None:
            box = self.apply_boxes(box, original_size)
            return box.astype(np.float32)


class SamImageProcessor(object):
    """Constructs a Sam image processor."""

    def __init__(
        self,
        size: Union[List[int], int] = None,
        image_mean: Union[float, List[float]] = [0.5, 0.5, 0.5],
        image_std: Union[float, List[float]] = [0.5, 0.5, 0.5],
        **kwargs,
    ) -> None:

        size = size if size is not None else 1024
        self.size = size

        if isinstance(image_mean, float):
            image_mean = [image_mean] * 3
        if isinstance(image_std, float):
            image_std = [image_std] * 3

        self.image_mean = image_mean
        self.image_std = image_std

        self.original_size = None
        self.input_size = None

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """Expects a numpy array with shape HxWxC in uint8 format."""
        target_size = _get_preprocess_shape(image.shape[0], image.shape[1], self.size)
        if isinstance(image, np.ndarray):
            image = PIL.Image.fromarray(image)

        return np.array(T.resize(image, target_size))

    def __call__(self, images, **kwargs):
        if not isinstance(images, (list, tuple)):
            images = [images]
        return self.preprocess(images)

    def preprocess(
        self,
        images,
    ):
        """Preprocess an image or a batch of images with a same shape."""

        size = self.size

        input_image = [self.apply_image(image) for image in images]

        input_image_paddle = paddle.to_tensor(input_image).cast("int32")

        input_image_paddle = input_image_paddle.transpose([0, 3, 1, 2])

        original_image_size = images[0].shape[:2]

        self.original_size = original_image_size
        self.input_size = tuple(input_image_paddle.shape[-2:])

        mean = paddle.to_tensor(self.image_mean).reshape([-1, 1, 1])
        std = paddle.to_tensor(self.image_std).reshape([-1, 1, 1])
        input_image_paddle = (input_image_paddle.astype(std.dtype) - mean) / std

        h, w = input_image_paddle.shape[-2:]
        padh = self.size - h
        padw = self.size - w
        input_image = F.pad(input_image_paddle, (0, padw, 0, padh))

        return input_image.numpy()
