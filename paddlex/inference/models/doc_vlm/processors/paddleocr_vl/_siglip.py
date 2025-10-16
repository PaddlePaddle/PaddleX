# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

# This file is based on https://github.com/Kwai-Keye/Keye/blob/main/keye-vl-8b-preview/image_processing_keye.py
# Original header:
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Image processor class for Keye."""

# TODO: Support videos

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from ......utils import logging
from ..common import (
    BatchFeature,
    convert_to_rgb,
    make_batched_images,
    make_list_of_images,
    to_numpy_array,
)

_OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def adjust_size(size, patch_size):
    num_patches = size // patch_size
    if num_patches % 2 != 0:
        num_patches -= 1
    return num_patches * patch_size


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 28 * 28 * 130,
    max_pixels: int = 28 * 28 * 1280,
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    # if height < factor or width < factor:
    #    raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    # if int(height < factor//4) + int(width < factor//4):
    #     raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor//4}")

    if height < factor:
        logging.debug(
            f"smart_resize: height={height} < factor={factor}, reset height=factor"
        )
        width = round((width * factor) / height)
        height = factor

    if width < factor:
        logging.debug(
            f"smart_resize: width={width} < factor={factor}, reset width=factor"
        )
        height = round((height * factor) / width)
        width = factor

    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class SiglipImageProcessor(object):
    model_input_names = [
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
    ]

    def __init__(
        self,
        do_resize: bool = True,
        resample: int = 3,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        min_pixels: int = 147384,
        max_pixels: int = 28 * 28 * 3600,
        patch_size: int = 14,
        temporal_patch_size: int = 1,
        merge_size: int = 2,
        **kwargs,
    ) -> None:
        super().__init__()
        self.do_resize = do_resize
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else _OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else _OPENAI_CLIP_STD
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.size = {"min_pixels": min_pixels, "max_pixels": max_pixels}  # not used
        self.do_convert_rgb = do_convert_rgb

    @classmethod
    def from_pretrained(cls, pretrained_model_dir):
        pretrained_model_dir = Path(pretrained_model_dir)
        image_processor_config_path = pretrained_model_dir / "preprocessor_config.json"
        with open(image_processor_config_path, "r", encoding="utf-8") as f:
            image_processor_config = json.load(f)
        return cls(**image_processor_config)

    def _preprocess(
        self,
        images,
        do_resize: Optional[bool] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: Optional[bool] = None,
    ):
        images = make_list_of_images(images)

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        width, height = images[0].size
        resized_height, resized_width = height, width
        processed_images = []

        for image in images:
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=self.patch_size * self.merge_size,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                )

                image = image.resize(
                    (resized_width, resized_height), resample=self.resample
                )

            image = to_numpy_array(image)

            if do_rescale:
                image = (image * rescale_factor).astype(np.float32)

            if do_normalize:
                image = image.astype(np.float32)
                image -= np.array(image_mean, dtype=np.float32)
                image /= np.array(image_std, dtype=np.float32)

            processed_images.append(image)

        patches = np.array(processed_images)
        patches = patches.transpose(0, 3, 1, 2)
        if patches.shape[0] == 1:
            patches = np.tile(patches, (self.temporal_patch_size, 1, 1, 1))
        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h, grid_w = (
            resized_height // self.patch_size,
            resized_width // self.patch_size,
        )

        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h,
            self.patch_size,
            grid_w,
            self.patch_size,
        )
        patches = patches.transpose(0, 3, 5, 2, 1, 4, 6)
        assert self.temporal_patch_size == 1
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel, self.patch_size, self.patch_size
        )
        return flatten_patches, (grid_t, grid_h, grid_w)

    def __call__(
        self,
        images,
        videos=None,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: Optional[bool] = None,
        return_tensors=None,
    ):
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = (
            rescale_factor if rescale_factor is not None else self.rescale_factor
        )
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = (
            do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        )

        if images is not None:
            images = make_batched_images(images)
        if videos is not None:
            raise NotImplementedError("Videos are not yet supported")

        if images is not None:
            pixel_values, vision_grid_thws = [], []
            for image in images:
                patches, image_grid_thw = self._preprocess(
                    image,
                    do_resize=do_resize,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    do_convert_rgb=do_convert_rgb,
                )
                pixel_values.extend(patches)
                vision_grid_thws.append(image_grid_thw)
            pixel_values = np.array(pixel_values)
            vision_grid_thws = np.array(vision_grid_thws)
            data = {"pixel_values": pixel_values, "image_grid_thw": vision_grid_thws}

        return BatchFeature(data=data, tensor_type=return_tensors)
