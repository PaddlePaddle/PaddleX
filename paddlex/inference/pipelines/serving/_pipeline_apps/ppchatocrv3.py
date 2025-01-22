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

from typing import List, Literal, Optional, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypeAlias, assert_never

from .....utils import logging
from .... import results
from ...ppchatocrv3 import PPChatOCRPipeline
from .. import utils as serving_utils
from ..app import AppConfig, create_app
from ..models import NoResultResponse, ResultResponse, DataInfo
from ._common import ocr as ocr_common


class AnalyzeImagesRequest(ocr_common.InferRequest):
    useImgOrientationCls: bool = True
    useImgUnwarping: bool = True
    useSealTextDet: bool = True


Point: TypeAlias = Annotated[List[int], Field(min_length=2, max_length=2)]
Polygon: TypeAlias = Annotated[List[Point], Field(min_length=3)]
BoundingBox: TypeAlias = Annotated[List[float], Field(min_length=4, max_length=4)]


class Text(BaseModel):
    poly: Polygon
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


class AnalyzeImagesResult(BaseModel):
    visionResults: List[VisionResult]
    visionInfo: dict
    dataInfo: DataInfo


class QianfanParams(BaseModel):
    apiKey: str
    secretKey: str
    apiType: Literal["qianfan"] = "qianfan"


class AIStudioParams(BaseModel):
    accessToken: str
    apiType: Literal["aistudio"] = "aistudio"


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
LLMParams: TypeAlias = Union[QianfanParams, AIStudioParams]


class BuildVectorStoreRequest(BaseModel):
    visionInfo: dict
    minChars: Optional[int] = None
    llmRequestInterval: Optional[float] = None
    llmName: Optional[LLMName] = None
    llmParams: Optional[Annotated[LLMParams, Field(discriminator="apiType")]] = None


class BuildVectorStoreResult(BaseModel):
    vectorStore: str


class RetrieveKnowledgeRequest(BaseModel):
    keys: List[str]
    vectorStore: str
    llmName: Optional[LLMName] = None
    llmParams: Optional[Annotated[LLMParams, Field(discriminator="apiType")]] = None


class RetrieveKnowledgeResult(BaseModel):
    retrievalResult: str


class ChatRequest(BaseModel):
    keys: List[str]
    visionInfo: dict
    vectorStore: Optional[str] = None
    retrievalResult: Optional[str] = None
    taskDescription: Optional[str] = None
    rules: Optional[str] = None
    fewShot: Optional[str] = None
    llmName: Optional[LLMName] = None
    llmParams: Optional[Annotated[LLMParams, Field(discriminator="apiType")]] = None
    returnPrompts: bool = False


class Prompts(BaseModel):
    ocr: List[str]
    table: Optional[List[str]] = None
    html: Optional[List[str]] = None


class ChatResult(BaseModel):
    chatResult: dict
    prompts: Optional[Prompts] = None


def _llm_params_to_dict(llm_params: LLMParams) -> dict:
    if llm_params.apiType == "qianfan":
        return {
            "api_type": "qianfan",
            "ak": llm_params.apiKey,
            "sk": llm_params.secretKey,
        }
    if llm_params.apiType == "aistudio":
        return {"api_type": "aistudio", "access_token": llm_params.accessToken}
    else:
        assert_never(llm_params.apiType)


def create_pipeline_app(pipeline: PPChatOCRPipeline, app_config: AppConfig) -> FastAPI:
    app, ctx = create_app(
        pipeline=pipeline, app_config=app_config, app_aiohttp_session=True
    )

    ocr_common.update_app_context(ctx)

    @app.post(
        "/chatocr-vision",
        operation_id="analyzeImages",
        responses={422: {"model": NoResultResponse}},
        response_model_exclude_none=True,
    )
    async def _analyze_images(
        request: AnalyzeImagesRequest,
    ) -> ResultResponse[AnalyzeImagesResult]:
        pipeline = ctx.pipeline

        log_id = serving_utils.generate_log_id()

        if request.inferenceParams:
            max_long_side = request.inferenceParams.maxLongSide
            if max_long_side:
                raise HTTPException(
                    status_code=422,
                    detail="`max_long_side` is currently not supported.",
                )

        images, data_info = await ocr_common.get_images(request, ctx)

        try:
            result = await pipeline.call(
                pipeline.pipeline.visual_predict,
                images,
                use_doc_image_ori_cls_model=request.useImgOrientationCls,
                use_doc_image_unwarp_model=request.useImgUnwarping,
                use_seal_text_det_model=request.useSealTextDet,
            )

            vision_results: List[VisionResult] = []
            for i, (img, item) in enumerate(zip(images, result[0])):
                texts: List[Text] = []
                for poly, text, score in zip(
                    item["ocr_result"]["dt_polys"],
                    item["ocr_result"]["rec_text"],
                    item["ocr_result"]["rec_score"],
                ):
                    texts.append(Text(poly=poly, text=text, score=score))
                tables = [
                    Table(bbox=r["layout_bbox"], html=r["html"])
                    for r in item["table_result"]
                ]
                input_img, layout_img, ocr_img = await ocr_common.postprocess_images(
                    log_id=log_id,
                    index=i,
                    app_context=ctx,
                    input_image=img,
                    layout_image=item["layout_result"].img,
                    ocr_image=item["ocr_result"].img,
                )
                vision_result = VisionResult(
                    texts=texts,
                    tables=tables,
                    inputImage=input_img,
                    ocrImage=ocr_img,
                    layoutImage=layout_img,
                )
                vision_results.append(vision_result)

            return ResultResponse[AnalyzeImagesResult](
                logId=log_id,
                result=AnalyzeImagesResult(
                    visionResults=vision_results,
                    visionInfo=result[1],
                    dataInfo=data_info,
                ),
            )

        except Exception:
            logging.exception("Unexpected exception")
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post(
        "/chatocr-vector",
        operation_id="buildVectorStore",
        responses={422: {"model": NoResultResponse}},
        response_model_exclude_none=True,
    )
    async def _build_vector_store(
        request: BuildVectorStoreRequest,
    ) -> ResultResponse[BuildVectorStoreResult]:
        pipeline = ctx.pipeline

        try:
            kwargs = {"visual_info": results.VisualInfoResult(request.visionInfo)}
            if request.minChars is not None:
                kwargs["min_characters"] = request.minChars
            if request.llmRequestInterval is not None:
                kwargs["llm_request_interval"] = request.llmRequestInterval
            if request.llmName is not None:
                kwargs["llm_name"] = request.llmName
            if request.llmParams is not None:
                kwargs["llm_params"] = _llm_params_to_dict(request.llmParams)

            result = await serving_utils.call_async(
                pipeline.pipeline.build_vector, **kwargs
            )

            return ResultResponse[BuildVectorStoreResult](
                logId=serving_utils.generate_log_id(),
                result=BuildVectorStoreResult(vectorStore=result["vector"]),
            )

        except Exception:
            logging.exception("Unexpected exception")
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post(
        "/chatocr-retrieval",
        operation_id="retrieveKnowledge",
        responses={422: {"model": NoResultResponse}},
        response_model_exclude_none=True,
    )
    async def _retrieve_knowledge(
        request: RetrieveKnowledgeRequest,
    ) -> ResultResponse[RetrieveKnowledgeResult]:
        pipeline = ctx.pipeline

        try:
            kwargs = {
                "key_list": request.keys,
                "vector": results.VectorResult({"vector": request.vectorStore}),
            }
            if request.llmName is not None:
                kwargs["llm_name"] = request.llmName
            if request.llmParams is not None:
                kwargs["llm_params"] = _llm_params_to_dict(request.llmParams)

            result = await serving_utils.call_async(
                pipeline.pipeline.retrieval, **kwargs
            )

            return ResultResponse[RetrieveKnowledgeResult](
                logId=serving_utils.generate_log_id(),
                result=RetrieveKnowledgeResult(retrievalResult=result["retrieval"]),
            )

        except Exception:
            logging.exception("Unexpected exception")
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post(
        "/chatocr-chat",
        operation_id="chat",
        responses={422: {"model": NoResultResponse}},
        response_model_exclude_none=True,
    )
    async def _chat(
        request: ChatRequest,
    ) -> ResultResponse[ChatResult]:
        pipeline = ctx.pipeline

        try:
            kwargs = {
                "key_list": request.keys,
                "visual_info": results.VisualInfoResult(request.visionInfo),
            }
            if request.vectorStore is not None:
                kwargs["vector"] = results.VectorResult({"vector": request.vectorStore})
            if request.retrievalResult is not None:
                kwargs["retrieval_result"] = results.RetrievalResult(
                    {"retrieval": request.retrievalResult}
                )
            if request.taskDescription is not None:
                kwargs["user_task_description"] = request.taskDescription
            if request.rules is not None:
                kwargs["rules"] = request.rules
            if request.fewShot is not None:
                kwargs["few_shot"] = request.fewShot
            if request.llmName is not None:
                kwargs["llm_name"] = request.llmName
            if request.llmParams is not None:
                kwargs["llm_params"] = _llm_params_to_dict(request.llmParams)
            kwargs["save_prompt"] = request.returnPrompts

            result = await serving_utils.call_async(pipeline.pipeline.chat, **kwargs)

            if result["prompt"]:
                prompts = Prompts(
                    ocr=result["prompt"]["ocr_prompt"],
                    table=result["prompt"]["table_prompt"] or None,
                    html=result["prompt"]["html_prompt"] or None,
                )
            else:
                prompts = None
            chat_result = ChatResult(
                chatResult=result["chat_res"],
                prompts=prompts,
            )

            return ResultResponse[ChatResult](
                logId=serving_utils.generate_log_id(),
                result=chat_result,
            )

        except Exception:
            logging.exception("Unexpected exception")
            raise HTTPException(status_code=500, detail="Internal server error")

    return app
