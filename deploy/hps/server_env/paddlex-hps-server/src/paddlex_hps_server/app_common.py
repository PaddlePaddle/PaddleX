import os
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
from PIL.Image import Image

from . import utils
from .storage import Storage, SupportsGetURL


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


def postprocess_image(
    image: np.array,
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
    img_bytes = utils.image_array_to_bytes(image, ext=ext)
    if file_storage is not None:
        file_storage.set(key, img_bytes)
        if return_url:
            assert isinstance(file_storage, SupportsGetURL)
            return file_storage.get_url(key)
    return utils.base64_encode(img_bytes)


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
