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
from typing import Optional, Tuple

import cv2
import numpy as np
from numpy.typing import ArrayLike

from ... import utils as serving_utils
from ...storage import Storage, SupportsGetURL


def postprocess_image(
    image: ArrayLike,
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
    image = np.asarray(image)
    h, w = image.shape[0:2]
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
