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

from typing import Final, List, Tuple, Union

import numpy as np
from typing_extensions import Literal

from ......utils.deps import function_requires_deps, is_dep_available
from ....infra import utils as serving_utils
from ....infra.models import ImageInfo, PDFInfo
from ....infra.storage import SupportsGetURL, create_storage
from ....schemas.shared.ocr import BaseInferRequest
from ..._app import AppContext

if is_dep_available("fastapi"):
    from fastapi import HTTPException


DEFAULT_MAX_NUM_INPUT_IMGS: Final[int] = 10
DEFAULT_MAX_OUTPUT_IMG_SIZE: Final[Tuple[int, int]] = (2000, 2000)


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


@function_requires_deps("fastapi")
def get_file_type(request: BaseInferRequest) -> Literal["PDF", "IMAGE"]:
    if request.fileType is None:
        if serving_utils.is_url(request.file):
            maybe_file_type = serving_utils.infer_file_type(request.file)
            if maybe_file_type is None or not (
                maybe_file_type == "PDF" or maybe_file_type == "IMAGE"
            ):
                raise HTTPException(status_code=422, detail="Unsupported file type")
            file_type = maybe_file_type
        else:
            raise HTTPException(
                status_code=422, detail="File type cannot be determined"
            )
    else:
        file_type = "PDF" if request.fileType == 0 else "IMAGE"
    return file_type


async def get_images(
    request: BaseInferRequest, app_context: AppContext
) -> Tuple[List[np.ndarray], Union[ImageInfo, PDFInfo]]:
    file_type = get_file_type(request)
    # XXX: Should we return 422?

    file_bytes = await serving_utils.get_raw_bytes_async(
        request.file,
        app_context.aiohttp_session,
    )
    images, data_info = await serving_utils.call_async(
        serving_utils.file_to_images,
        file_bytes,
        file_type,
        max_num_imgs=app_context.extra["max_num_input_imgs"],
    )

    return images, data_info
