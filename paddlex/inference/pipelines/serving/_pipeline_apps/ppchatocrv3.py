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
from typing import Awaitable, Final, List, Literal, Optional, Tuple, Union
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from numpy.typing import ArrayLike
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypeAlias, assert_never

from .....utils import logging
from ...ppchatocrv3 import PPChatOCRPipeline
from .. import file_storage
from .. import utils as serving_utils
from ..app import AppConfig, create_app
from ..models import Response, ResultResponse

_DEFAULT_MAX_IMG_SIZE: Final[Tuple[int, int]] = (2000, 2000)
_DEFAULT_MAX_NUM_IMGS: Final[int] = 10


FileType: TypeAlias = Literal[0, 1]


class InferenceParams(BaseModel):
    maxLongSide: Optional[Annotated[int, Field(gt=0)]] = None


class AnalyzeImageRequest(BaseModel):
    file: str
    fileType: Optional[FileType] = None
    useOricls: bool = True
    useCurve: bool = True
    useUvdoc: bool = True
    inferenceParams: Optional[InferenceParams] = None


BoundingBox: TypeAlias = Annotated[List[float], Field(min_length=4, max_length=4)]


class Text(BaseModel):
    bbox: BoundingBox
    text: str
    score: float


class Table(BaseModel):
    bbox: BoundingBox
    html: str


class VisionResult(BaseModel):
    texts: List[Text]
    tables: List[Table]
    inputImage: str
    ocrImage: str
    layoutImage: str


class AnalyzeImageResult(BaseModel):
    visionResults: List[VisionResult]
    visionInfo: dict


class AIStudioParams(BaseModel):
    accessToken: str
    apiType: Literal["aistudio"] = "aistudio"


class QianfanParams(BaseModel):
    apiKey: str
    secretKey: str
    apiType: Literal["qianfan"] = "qianfan"


LLMName: TypeAlias = Literal[
    "ernie-3.5",
    "ernie-3.5-8k",
    "ernie-lite",
    "ernie-4.0",
    "ernie-4.0-turbo-8k",
    "ernie-speed",
    "ernie-speed-128k",
    "ernie-tiny-8k",
    "ernie-char-8k",
]
LLMParams: TypeAlias = Union[AIStudioParams, QianfanParams]


class BuildVectorStoreRequest(BaseModel):
    visionInfo: dict
    minChars: Optional[int] = None
    llmRequestInterval: Optional[int] = None
    llmName: Optional[LLMName] = None
    llmParams: Optional[Annotated[LLMParams, Field(discriminator="apiType")]] = None


class BuildVectorStoreResult(BaseModel):
    vectorStore: dict


class RetrieveKnowledgeRequest(BaseModel):
    keys: List[str]
    vectorStore: dict
    visionInfo: dict
    llmName: Optional[LLMName] = None
    llmParams: Optional[Annotated[LLMParams, Field(discriminator="apiType")]] = None


class RetrieveKnowledgeResult(BaseModel):
    retrievalResult: str


class ChatRequest(BaseModel):
    keys: List[str]
    visionInfo: dict
    taskDescription: Optional[str] = None
    rules: Optional[str] = None
    fewShot: Optional[str] = None
    useVectorStore: bool = True
    vectorStore: Optional[dict] = None
    retrievalResult: Optional[str] = None
    returnPrompt: bool = True
    llmName: Optional[LLMName] = None
    llmParams: Optional[Annotated[LLMParams, Field(discriminator="apiType")]] = None


class Prompts(BaseModel):
    ocr: str
    table: str
    html: str


class ChatResult(BaseModel):
    chatResult: str
    prompts: Optional[Prompts] = None


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


def _llm_params_to_dict(llm_params: LLMParams) -> dict:
    if llm_params.apiType == "aistudio":
        return {"api_type": "aistudio", "access_token": llm_params.accessToken}
    elif llm_params.apiType == "qianfan":
        return {
            "api_type": "qianfan",
            "ak": llm_params.apiKey,
            "sk": llm_params.secretKey,
        }
    else:
        assert_never(llm_params.apiType)


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


def create_pipeline_app(pipeline: PPChatOCRPipeline, app_config: AppConfig) -> FastAPI:
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
        "/chatocr-vision",
        operation_id="analyzeImage",
        responses={422: {"model": Response}},
    )
    async def _analyze_image(
        request: AnalyzeImageRequest,
    ) -> ResultResponse[AnalyzeImageResult]:
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
                use_oricls=request.useOricls,
                use_curve=request.useCurve,
                use_uvdoc=request.useUvdoc,
            )

            vision_results: List[VisionResult] = []
            for i, (img, item) in enumerate(zip(images, result["visual_result"])):
                pp_img_futures: List[Awaitable] = []
                future = serving_utils.call_async(
                    _postprocess_image,
                    img,
                    request_id=request_id,
                    filename=f"input_image_{i}.jpg",
                    file_storage_config=ctx.extra["file_storage_config"],
                )
                pp_img_futures.append(future)
                future = serving_utils.call_async(
                    _postprocess_image,
                    item["ocr_result"].img,
                    request_id=request_id,
                    filename=f"ocr_image_{i}.jpg",
                    file_storage_config=ctx.extra["file_storage_config"],
                )
                pp_img_futures.append(future)
                future = serving_utils.call_async(
                    _postprocess_image,
                    item["layout_result"].img,
                    request_id=request_id,
                    filename=f"layout_image_{i}.jpg",
                    file_storage_config=ctx.extra["file_storage_config"],
                )
                pp_img_futures.append(future)
                texts: List[Text] = []
                for bbox, text, score in zip(
                    item["ocr_result"]["dt_polys"],
                    item["ocr_result"]["rec_text"],
                    item["ocr_result"]["rec_score"],
                ):
                    texts.append(Text(bbox=bbox, text=text, score=score))
                tables = [
                    Table(bbox=r["layout_bbox"], html=r["html"])
                    for r in result["table_result"]
                ]
                input_img, ocr_img, layout_img = await asyncio.gather(*pp_img_futures)
                vision_result = VisionResult(
                    texts=texts,
                    tables=tables,
                    inputImage=input_img,
                    ocrImage=ocr_img,
                    layoutImage=layout_img,
                )
                vision_results.append(vision_result)

            return ResultResponse(
                logId=serving_utils.generate_log_id(),
                errorCode=0,
                errorMsg="Success",
                result=AnalyzeImageResult(
                    visionResults=vision_results,
                    visionInfo=result["visual_info"],
                ),
            )

        except Exception as e:
            logging.exception(e)
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post(
        "/chatocr-vector",
        operation_id="buildVectorStore",
        responses={422: {"model": Response}},
    )
    async def _build_vector_store(
        request: BuildVectorStoreRequest,
    ) -> ResultResponse[BuildVectorStoreResult]:
        pipeline = ctx.pipeline

        try:
            kwargs = {"visual_info": request.visionInfo}
            if request.minChars is not None:
                kwargs["min_characters"] = request.minChars
            if request.llmRequestInterval is not None:
                kwargs["llm_request_interval"] = request.llmRequestInterval
            if request.llmName is not None:
                kwargs["llm_name"] = request.llmName
            if request.llmParams is not None:
                kwargs["llm_params"] = _llm_params_to_dict(request.llmParams)

            result = await serving_utils.call_async(
                pipeline.pipeline.get_vector_text, **kwargs
            )

            return ResultResponse(
                logId=serving_utils.generate_log_id(),
                errorCode=0,
                errorMsg="Success",
                result=BuildVectorStoreResult(vectorStore=result),
            )

        except Exception as e:
            logging.exception(e)
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post(
        "/chatocr-retrieval",
        operation_id="retrieveKnowledge",
        responses={422: {"model": Response}},
    )
    async def _retrieve_knowledge(
        request: RetrieveKnowledgeRequest,
    ) -> ResultResponse[RetrieveKnowledgeResult]:
        pipeline = ctx.pipeline

        try:
            kwargs = {
                "key_list": request.keys,
                "vector": request.vectorStore,
                "visual_info": request.visionInfo,
            }
            if request.llmName is not None:
                kwargs["llm_name"] = request.llmName
            if request.llmParams is not None:
                kwargs["llm_params"] = _llm_params_to_dict(request.llmParams)

            result = await serving_utils.call_async(
                pipeline.pipeline.get_retrieval_text, **kwargs
            )

            return ResultResponse(
                logId=serving_utils.generate_log_id(),
                errorCode=0,
                errorMsg="Success",
                result=RetrieveKnowledgeResult(
                    retrievalResult=result["retrieval_result"]
                ),
            )

        except Exception as e:
            logging.exception(e)
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post(
        "/chatocr-chat", operation_id="chat", responses={422: {"model": Response}}
    )
    async def _chat(
        request: ChatRequest,
    ) -> ResultResponse[ChatResult]:
        pipeline = ctx.pipeline

        try:
            kwargs = {
                "key_list": request.keys,
                "visual_info": request.visionInfo,
            }
            if request.taskDescription is not None:
                kwargs["user_task_description"] = request.taskDescription
            if request.rules is not None:
                kwargs["rules"] = request.rules
            if request.fewShot is not None:
                kwargs["few_shot"] = request.fewShot
            kwargs["use_vector"] = request.useVectorStore
            if request.vectorStore is not None:
                kwargs["vector_store"] = request.vectorStore
            if request.retrievalResult is not None:
                kwargs["retrieval_result"] = request.retrievalResult
            kwargs["save_prompt"] = request.returnPrompt
            if request.llmName is not None:
                kwargs["llm_name"] = request.llmName
            if request.llmParams is not None:
                kwargs["llm_params"] = _llm_params_to_dict(request.llmParams)

            result = await serving_utils.call_async(pipeline.pipeline.chat, **kwargs)

            if result["prompt"]:
                prompts = Prompts(
                    ocr=result["prompt"]["ocr_prompt"],
                    table=result["prompt"]["table_prompt"],
                    html=result["prompt"]["html_prompt"],
                )
                chat_result = ChatResult(
                    chatResult=result["chat_res"],
                    prompts=prompts,
                )
            else:
                chat_result = ChatResult(
                    chatResult=result["chat_res"],
                )
            return ResultResponse(
                logId=serving_utils.generate_log_id(),
                errorCode=0,
                errorMsg="Success",
                result=chat_result,
            )

        except Exception as e:
            logging.exception(e)
            raise HTTPException(status_code=500, detail="Internal server error")

    return app
