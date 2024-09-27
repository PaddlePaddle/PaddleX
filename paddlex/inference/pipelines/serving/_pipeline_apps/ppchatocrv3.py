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
import os
import re
import uuid
from typing import List, Optional, Literal, Final, Tuple
from functools import partial
from urllib.parse import urlparse, parse_qs

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypeAlias, assert_never

from .. import utils as serving_utils
from ..app import AppConfig, create_app
from ..models import Response, ResultResponse
from .. import file_storage
from ...table_recognition import TableRecPipeline
from .....utils import logging

_DEFAULT_MAX_IMG_SIZE: Final[Tuple[int, int]] = (2000, 2000)
_DEFAULT_MAX_NUM_IMGS: Final[int] = 10


FileType: TypeAlias = Literal[0, 1]


class InferenceParams(BaseModel):
    maxLongSide: Optional[Annotated[int, Field(gt=0)]] = None


class InferVisualRequest(BaseModel):
    file: str
    fileType: Optional[FileType] = None
    aistudioToken: Optional[str] = None
    ernieModel: Optional[str] = None
    useOricls: Optional[bool] = None
    useUvdoc: Optional[bool] = None
    inferenceParams: Optional[InferenceParams] = None


class InferVisualResult(BaseModel):
    tableOcrResult: dict
    documentType: str
    ocrAllResult: str
    rawImages: List[str]
    visualOcr: List[str]
    visualLayout: List[str]
    htmlResult: List[str]
    layoutAllResult: dict


class InferLLMRequest(BaseModel):
    queryKey: str
    tableOcrResult: dict
    documentType: str
    ocrAllResult: str
    aistudioToken: Optional[str] = None
    ernieModel: Optional[str] = None
    rules: Optional[str] = None
    fewShot: Optional[str] = None
    taskDescription: Optional[str] = None


class InferLLMResult(BaseModel):
    llmResult: str


def _generate_request_id() -> str:
    return str(uuid.uuid4())


def _infer_file_type(url: str) -> FileType:
    # Is it more reliable to guess the file type based on the response headers?
    SUPPORTED_IMG_EXTS: Final[List[str]] = [".jpg", ".jpeg", ".png"]

    url_parts = urlparse(url)
    ext = os.path.splitext(url_parts.path)[1]
    # HACK: The support for BOS URLs with query params is implementation-based,
    # not interface-based.
    is_bos_url = url_parts.netloc == "bj.bcebos.com"
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
        images = [
            cv2.imdecode(np.frombuffer(file_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        ]
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


async def _postprocess_image(
    img: np.ndarray,
    request_id: str,
    filename: str,
    file_storage_config: file_storage.FileStorageConfig,
) -> None:
    key = f"{request_id}/{filename}"
    ext = os.path.splitext(filename)[1]
    _, encoded_img = cv2.imencode(ext, img)
    encoded_img = encoded_img.tobytes()
    file_storage.postprocess_file(encoded_img, config=file_storage_config, key=key)


def create_pipeline_app(pipeline: TableRecPipeline, app_config: AppConfig) -> FastAPI:
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
        "/chatocr_visual", operation_id="analyze", responses={422: {"model": Response}}
    )
    async def _analyze_image(
        request: InferVisualRequest,
    ) -> ResultResponse[InferVisualResult]:
        pipeline = ctx.pipeline
        aiohttp_session = ctx.aiohttp_session

        loop = asyncio.get_running_loop()

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

        try:
            inputs = []

            # images
            file_bytes = await serving_utils.get_raw_bytes(
                request.file, aiohttp_session
            )
            images = await loop.run_in_executor(
                None,
                partial(
                    _bytes_to_arrays,
                    file_bytes,
                    file_type,
                    max_img_size=ctx.extra["max_img_size"],
                    max_num_imgs=ctx.extra["max_num_imgs"],
                ),
            )
            input_ = create_triton_input(
                np.array([images]), model_info["input_names"][0], model_info["inputs"]
            )
            inputs.append(input_)

            # optional_params
            optional_params: Dict[str, Any] = {}
            ernie_config = {}
            if request.aistudioToken is not None:
                ernie_config["aistudio_access_token"] = request.aistudioToken
            if request.ernieModel is not None:
                ernie_config["ernie_model"] = request.ernieModel
            if ernie_config:
                optional_params["ernie_config"] = ernie_config
            if request.useOricls is not None:
                optional_params["use_oricls"] = request.useOricls
            if request.useUvdoc is not None:
                optional_params["use_uvdoc"] = request.useUvdoc
            if request.inferenceParams is not None:
                inference_params = {}
                if request.inferenceParams.maxLongSide is not None:
                    inference_params["max_long_side"] = (
                        request.inferenceParams.maxLongSide
                    )
                optional_params["inference_params"] = inference_params
            optional_params_bytes = json.dumps(optional_params).encode("utf-8")
            input_ = create_triton_input(
                np.array([[optional_params_bytes]]),
                model_info["input_names"][1],
                model_info["inputs"],
            )
            inputs.append(input_)

            outputs = [
                triton_grpc.InferRequestedOutput(name)
                for name in model_info["output_names"]
            ]

            results = await triton_client.infer(
                model_name=MODEL_NAME,
                model_version=MODEL_VERSION,
                inputs=inputs,
                outputs=outputs,
            )
            results = {
                name: results.as_numpy(name) for name in model_info["output_names"]
            }

            table_ocr_result = results["table_ocr_result"][0][0].decode("utf-8")
            table_ocr_result = json.loads(table_ocr_result)
            document_type = results["document_type"][0][0].decode("utf-8")
            ocr_all_result = results["ocr_all_result"][0][0].decode("utf-8")
            raw_images = await asyncio.gather(
                *(
                    loop.run_in_executor(
                        partial(
                            _postprocess_image,
                            img,
                            request_id=request_id,
                            filename=f"raw_images_{i}.jpg",
                            file_storage_config=ctx.extra["file_storage_config"],
                        )
                    )
                    for i, img in enumerate(results["raw_images"][0])
                )
            )
            visual_ocr = await asyncio.gather(
                *(
                    loop.run_in_executor(
                        partial(
                            _postprocess_image,
                            img,
                            request_id=request_id,
                            filename=f"visual_ocr_{i}.jpg",
                            file_storage_config=ctx.extra["file_storage_config"],
                        )
                    )
                    for i, img in enumerate(results["visual_ocr"][0])
                )
            )
            visual_layout = await asyncio.gather(
                *(
                    loop.run_in_executor(
                        partial(
                            _postprocess_image,
                            img,
                            request_id=request_id,
                            filename=f"visual_layout_{i}.jpg",
                            file_storage_config=ctx.extra["file_storage_config"],
                        )
                    )
                    for i, img in enumerate(results["visual_layout"][0])
                )
            )
            html_result = [item.decode("utf-8") for item in results["html_result"][0]]
            layout_all_result = results["layout_all_result"][0][0].decode("utf-8")
            layout_all_result = json.loads(layout_all_result)
            return ResultResponse(
                logId=serving_utils.generate_log_id(),
                errorCode=0,
                errorMsg="Success",
                result=InferVisualResult(
                    tableOcrResult=table_ocr_result,
                    documentType=document_type,
                    ocrAllResult=ocr_all_result,
                    rawImages=raw_images,
                    visualOcr=visual_ocr,
                    visualLayout=visual_layout,
                    htmlResult=html_result,
                    layoutAllResult=layout_all_result,
                ),
            )

        except Exception as e:
            logging.exception(e)
            raise HTTPException(status_code=500, detail="Internal server error")

    return app
