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

import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
from PIL.Image import Image

from ......utils.deps import function_requires_deps, is_dep_available
from ....infra import utils as serving_utils
from ....infra.storage import Storage, SupportsGetURL

if is_dep_available("opencv-contrib-python"):
    import cv2


def prune_result(result: dict) -> dict:
    KEYS_TO_REMOVE = ["input_path", "page_index"]

    def _process_obj(obj):
        if isinstance(obj, dict):
            return {
                k: _process_obj(v) for k, v in obj.items() if k not in KEYS_TO_REMOVE
            }
        elif isinstance(obj, list):
            return [_process_obj(item) for item in obj]
        else:
            return obj

    return _process_obj(result)


@function_requires_deps("opencv-contrib-python")
def postprocess_image(
    image: np.ndarray,
    log_id: str,
    filename: str,
    *,
    file_storage: Optional[Storage] = None,
    return_url: bool = False,
    max_img_size: Optional[Tuple[int, int]] = None,
) -> str:
    if return_url:
        if not file_storage:
            raise ValueError(
                "`file_storage` must not be None when URLs need to be returned."
            )
        if not isinstance(file_storage, SupportsGetURL):
            raise TypeError("The provided storage does not support getting URLs.")

    key = f"{log_id}/{filename}"
    ext = os.path.splitext(filename)[1]
    h, w = image.shape[0:2]
    if max_img_size is not None:
        if w > max_img_size[1] or h > max_img_size[0]:
            if w / h > max_img_size[0] / max_img_size[1]:
                factor = max_img_size[0] / w
            else:
                factor = max_img_size[1] / h
            image = cv2.resize(image, (int(factor * w), int(factor * h)))
    img_bytes = serving_utils.image_array_to_bytes(image, ext=ext)
    if file_storage is not None:
        file_storage.set(key, img_bytes)
        if return_url:
            assert isinstance(file_storage, SupportsGetURL)
            return file_storage.get_url(key)
    return serving_utils.base64_encode(img_bytes)


def postprocess_images(
    images: Dict[str, Union[Image, np.ndarray]],
    log_id: str,
    filename_template: str = "{key}.jpg",
    file_storage: Optional[Storage] = None,
    return_urls: bool = False,
    max_img_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, str]:
    output_images: Dict[str, str] = {}
    for key, img in images.items():
        output_images[key] = postprocess_image(
            (
                cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
                if isinstance(img, Image)
                else img
            ),
            log_id=log_id,
            filename=filename_template.format(key=key),
            file_storage=file_storage,
            return_url=return_urls,
            max_img_size=max_img_size,
        )
    return output_images
