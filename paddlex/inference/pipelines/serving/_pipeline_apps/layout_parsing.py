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
import re
import uuid
from typing import Final, List, Literal, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from numpy.typing import ArrayLike
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypeAlias, assert_never

from .....utils import logging
from ...layout_parsing import LayoutParsingPipeline
from .. import file_storage
from .. import utils as serving_utils
from ..app import AppConfig, create_app
from ..models import Response, ResultResponse

_DEFAULT_MAX_IMG_SIZE: Final[Tuple[int, int]] = (2000, 2000)
_DEFAULT_MAX_NUM_IMGS: Final[int] = 10


FileType: TypeAlias = Literal[0, 1]


class InferenceParams(BaseModel):
    maxLongSide: Optional[Annotated[int, Field(gt=0)]] = None


class InferRequest(BaseModel):
    file: str
    fileType: Optional[FileType] = None
    useImgOrientationCls: bool = True
    useImgUnwrapping: bool = True
    useSealTextDet: bool = True
    inferenceParams: Optional[InferenceParams] = None


BoundingBox: TypeAlias = Annotated[List[float], Field(min_length=4, max_length=4)]


class LayoutElement(BaseModel):
    bbox: BoundingBox
    label: str
    text: str
    layoutType: Literal["single", "double"]
    image: Optional[str] = None


class LayoutParsingResult(BaseModel):
    layoutElements: List[LayoutElement]


class InferResult(BaseModel):
    layoutParsingResults: List[LayoutParsingResult]


def _generate_request_id() -> str:
    return str(uuid.uuid4())


def _infer_file_type(url: str) -> FileType:
    # Is it more reliable to guess the file type based on the response headers?
    SUPPORTED_IMG_EXTS: Final[List[str]] = [".jpg", ".jpeg", ".png"]

    url_parts = urlparse(url)
    ext = os.path.splitext(url_parts.path)[1]
    # HACK: The support for BOS URLs with query params is implementation-based,
    # not interface-based.
    is_bos_url = (
        re.fullmatch(r"(?:bj|bd|su|gz|cd|hkg|fwh|fsh)\.bcebos\.com", url_parts.netloc)
        is not None
    )
    if is_bos_url and url_parts.query:
        params = parse_qs(url_parts.query)
        if (
            "responseContentDisposition" not in params
            or len(params["responseContentDisposition"]) != 1
        ):
            raise ValueError("`responseContentDisposition` not found")
        match_ = re.match(
            r"attachment;filename=(.*)", params["responseContentDisposition"][0]
        )
        if not match_ or not match_.groups()[0] is not None:
            raise ValueError(
                "Failed to extract the filename from `responseContentDisposition`"
            )
        ext = os.path.splitext(match_.groups()[0])[1]
    ext = ext.lower()
    if ext == ".pdf":
        return 0
    elif ext in SUPPORTED_IMG_EXTS:
        return 1
    else:
        raise ValueError("Unsupported file type")


def _bytes_to_arrays(
    file_bytes: bytes,
    file_type: FileType,
    *,
    max_img_size: Tuple[int, int],
    max_num_imgs: int,
) -> List[np.ndarray]:
    if file_type == 0:
        images = serving_utils.read_pdf(
            file_bytes, resize=True, max_num_imgs=max_num_imgs
        )
    elif file_type == 1:
        images = [serving_utils.image_bytes_to_array(file_bytes)]
    else:
        assert_never(file_type)
    h, w = images[0].shape[0:2]
    if w > max_img_size[1] or h > max_img_size[0]:
        if w / h > max_img_size[0] / max_img_size[1]:
            factor = max_img_size[0] / w
        else:
            factor = max_img_size[1] / h
        images = [cv2.resize(img, (int(factor * w), int(factor * h))) for img in images]
    return images


def _postprocess_image(
    img: ArrayLike,
    request_id: str,
    filename: str,
    file_storage_config: file_storage.FileStorageConfig,
) -> str:
    key = f"{request_id}/{filename}"
    ext = os.path.splitext(filename)[1]
    img = np.asarray(img)
    _, encoded_img = cv2.imencode(ext, img)
    encoded_img = encoded_img.tobytes()
    return file_storage.postprocess_file(
        encoded_img, config=file_storage_config, key=key
    )


def create_pipeline_app(
    pipeline: LayoutParsingPipeline, app_config: AppConfig
) -> FastAPI:
    app, ctx = create_app(
        pipeline=pipeline, app_config=app_config, app_aiohttp_session=True
    )

    if "file_storage_config" in ctx.extra:
        ctx.extra["file_storage_config"] = file_storage.parse_file_storage_config(
            ctx.extra["file_storage_config"]
        )
    else:
        ctx.extra["file_storage_config"] = file_storage.InMemoryStorageConfig()
    ctx.extra.setdefault("max_img_size", _DEFAULT_MAX_IMG_SIZE)
    ctx.extra.setdefault("max_num_imgs", _DEFAULT_MAX_NUM_IMGS)

    @app.post(
        "/layout-parsing",
        operation_id="infer",
        responses={422: {"model": Response}},
        response_model_exclude_none=True,
    )
    async def _infer(
        request: InferRequest,
    ) -> ResultResponse[InferResult]:
        pipeline = ctx.pipeline
        aiohttp_session = ctx.aiohttp_session

        request_id = _generate_request_id()

        if request.fileType is None:
            if serving_utils.is_url(request.file):
                try:
                    file_type = _infer_file_type(request.file)
                except Exception as e:
                    logging.exception(e)
                    raise HTTPException(
                        status_code=422,
                        detail="The file type cannot be inferred from the URL. Please specify the file type explicitly.",
                    )
            else:
                raise HTTPException(status_code=422, detail="Unknown file type")
        else:
            file_type = request.fileType

        if request.inferenceParams:
            max_long_side = request.inferenceParams.maxLongSide
            if max_long_side:
                raise HTTPException(
                    status_code=422,
                    detail="`max_long_side` is currently not supported.",
                )

        try:
            file_bytes = await serving_utils.get_raw_bytes(
                request.file, aiohttp_session
            )
            images = await serving_utils.call_async(
                _bytes_to_arrays,
                file_bytes,
                file_type,
                max_img_size=ctx.extra["max_img_size"],
                max_num_imgs=ctx.extra["max_num_imgs"],
            )

            result = await pipeline.infer(
                images,
                use_doc_image_ori_cls_model=request.useImgOrientationCls,
                use_doc_image_unwarp_model=request.useImgUnwrapping,
                use_seal_text_det_model=request.useSealTextDet,
            )

            layout_parsing_results: List[LayoutParsingResult] = []
            for i, item in enumerate(result):
                layout_elements: List[LayoutElement] = []
                for j, subitem in enumerate(
                    item["layout_parsing_result"]["parsing_result"]
                ):
                    dyn_keys = subitem.keys() - {"input_path", "layout_bbox", "layout"}
                    if len(dyn_keys) != 1:
                        raise RuntimeError(f"Unexpected result: {subitem}")
                    label = next(iter(dyn_keys))
                    if label in ("image", "figure", "img", "fig"):
                        image_ = await serving_utils.call_async(
                            _postprocess_image,
                            subitem[label]["img"],
                            request_id=request_id,
                            filename=f"image_{i}_{j}.jpg",
                            file_storage_config=ctx.extra["file_storage_config"],
                        )
                        text = subitem[label]["image_text"]
                    else:
                        image_ = None
                        text = subitem[label]
                    layout_elements.append(
                        LayoutElement(
                            bbox=subitem["layout_bbox"],
                            label=label,
                            text=text,
                            layoutType=subitem["layout"],
                            image=image_,
                        )
                    )
                layout_parsing_results.append(
                    LayoutParsingResult(layoutElements=layout_elements)
                )

            return ResultResponse(
                logId=serving_utils.generate_log_id(),
                errorCode=0,
                errorMsg="Success",
                result=InferResult(
                    layoutParsingResults=layout_parsing_results,
                ),
            )

        except Exception as e:
            logging.exception(e)
            raise HTTPException(status_code=500, detail="Internal server error")

    return app
