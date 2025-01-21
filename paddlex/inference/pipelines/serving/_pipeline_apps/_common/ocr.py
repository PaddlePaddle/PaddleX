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

import asyncio
from typing import Awaitable, Final, List, Literal, Optional, Tuple

import numpy as np
from fastapi import HTTPException
from numpy.typing import ArrayLike
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypeAlias, assert_never

from ......utils import logging
from ... import utils as serving_utils
from .cv import postprocess_image
from ...models import DataInfo
from ...storage import create_storage, SupportsGetURL
from ...app import AppContext

DEFAULT_MAX_NUM_INPUT_IMGS: Final[int] = 10
DEFAULT_MAX_OUTPUT_IMG_SIZE: Final[Tuple[int, int]] = (2000, 2000)

FileType: TypeAlias = Literal[0, 1]


class InferenceParams(BaseModel):
    maxLongSide: Optional[Annotated[int, Field(gt=0)]] = None


class InferRequest(BaseModel):
    file: str
    fileType: Optional[FileType] = None
    inferenceParams: Optional[InferenceParams] = None


def update_app_context(app_context: AppContext) -> None:
    extra_cfg = app_context.config.extra or {}
    app_context.extra["file_storage"] = None
    if "file_storage" in extra_cfg:
        app_context.extra["file_storage"] = create_storage(extra_cfg["file_storage"])
    app_context.extra["return_img_urls"] = extra_cfg.get("return_img_urls", False)
    if app_context.extra["return_img_urls"]:
        file_storage = app_context.extra["file_storage"]
        if not file_storage:
            raise ValueError(
                "The file storage must be properly configured when URLs need to be returned."
            )
        if not isinstance(file_storage, SupportsGetURL):
            raise TypeError(
                f"`{type(file_storage).__name__}` does not support getting URLs."
            )
    app_context.extra["max_num_input_imgs"] = extra_cfg.get(
        "max_num_input_imgs", DEFAULT_MAX_NUM_INPUT_IMGS
    )
    app_context.extra["max_output_img_size"] = extra_cfg.get(
        "max_output_img_size", DEFAULT_MAX_OUTPUT_IMG_SIZE
    )


def get_file_type(request: InferRequest) -> Literal["IMAGE", "PDF"]:
    if request.fileType is None:
        if serving_utils.is_url(request.file):
            try:
                file_type = serving_utils.infer_file_type(request.file)
            except Exception:
                logging.exception("Failed to infer the file type")
                raise HTTPException(
                    status_code=422,
                    detail="The file type cannot be inferred from the URL. Please specify the file type explicitly.",
                )
        else:
            raise HTTPException(status_code=422, detail="Unknown file type")
    else:
        file_type = "PDF" if request.fileType == 0 else "IMAGE"

    return file_type


async def get_images(
    request: InferRequest, app_context: AppContext
) -> Tuple[List[np.ndarray], DataInfo]:
    file_type = get_file_type(request)
    # XXX: Currently, we use 500 for consistency. However, 422 may be more
    # appropriate.
    try:
        file_bytes = await serving_utils.get_raw_bytes(
            request.file,
            app_context.aiohttp_session,
        )
        images, data_info = await serving_utils.call_async(
            serving_utils.file_to_images,
            file_bytes,
            file_type,
            max_num_imgs=app_context.extra["max_num_input_imgs"],
        )
    except Exception:
        logging.exception("Unexpected exception")
        raise HTTPException(status_code=500, detail="Internal server error")

    if file_type == "IMAGE":
        return images, DataInfo(image=data_info)
    elif file_type == "PDF":
        return images, DataInfo(pdf=data_info)
    else:
        assert_never()


async def postprocess_images(
    *,
    log_id: str,
    index: str,
    app_context: AppContext,
    input_image: Optional[ArrayLike] = None,
    layout_image: Optional[ArrayLike] = None,
    ocr_image: Optional[ArrayLike] = None,
) -> List[str]:
    if input_image is None and layout_image is None and ocr_image is None:
        raise ValueError("At least one of the images must be provided.")
    file_storage = app_context.extra["file_storage"]
    return_img_urls = app_context.extra["return_img_urls"]
    max_img_size = app_context.extra["max_output_img_size"]
    futures: List[Awaitable] = []
    if input_image is not None:
        future = serving_utils.call_async(
            postprocess_image,
            input_image,
            log_id=log_id,
            filename=f"input_image_{index}.jpg",
            file_storage=file_storage,
            return_url=return_img_urls,
            max_img_size=max_img_size,
        )
        futures.append(future)
    if layout_image is not None:
        future = serving_utils.call_async(
            postprocess_image,
            layout_image,
            log_id=log_id,
            filename=f"layout_image_{index}.jpg",
            file_storage=file_storage,
            return_url=return_img_urls,
            max_img_size=max_img_size,
        )
        futures.append(future)
    if ocr_image is not None:
        future = serving_utils.call_async(
            postprocess_image,
            ocr_image,
            log_id=log_id,
            filename=f"ocr_image_{index}.jpg",
            file_storage=file_storage,
            max_img_size=max_img_size,
        )
        futures.append(future)
    return await asyncio.gather(*futures)
