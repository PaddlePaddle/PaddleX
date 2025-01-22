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

from typing import List, Type

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypeAlias

from ._common import ocr as ocr_common
from .....utils import logging
from ...seal_recognition import SealOCRPipeline
from .. import utils as serving_utils
from ..app import AppConfig, create_app
from ..models import NoResultResponse, ResultResponse, DataInfo

InferRequest: Type[ocr_common.InferRequest] = ocr_common.InferRequest

Point: TypeAlias = Annotated[List[int], Field(min_length=2, max_length=2)]
Polygon: TypeAlias = Annotated[List[Point], Field(min_length=3)]


class Text(BaseModel):
    poly: Polygon
    text: str
    score: float


class SealRecResult(BaseModel):
    texts: List[Text]
    inputImage: str
    layoutImage: str
    ocrImage: str


class InferResult(BaseModel):
    sealRecResults: List[SealRecResult]
    dataInfo: DataInfo


def create_pipeline_app(pipeline: SealOCRPipeline, app_config: AppConfig) -> FastAPI:
    app, ctx = create_app(
        pipeline=pipeline, app_config=app_config, app_aiohttp_session=True
    )

    ocr_common.update_app_context(ctx)

    @app.post(
        "/seal-recognition",
        operation_id="infer",
        responses={422: {"model": NoResultResponse}},
        response_model_exclude_none=True,
    )
    async def _infer(request: InferRequest) -> ResultResponse[InferResult]:
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
            result = await pipeline.infer(images)

            seal_rec_results: List[SealRecResult] = []
            for i, (img, item) in enumerate(zip(images, result)):
                texts: List[Text] = []
                for poly, text, score in zip(
                    item["ocr_result"]["dt_polys"],
                    item["ocr_result"]["rec_text"],
                    item["ocr_result"]["rec_score"],
                ):
                    texts.append(Text(poly=poly, text=text, score=score))
                input_img, layout_img, ocr_img = await ocr_common.postprocess_images(
                    log_id=log_id,
                    index=i,
                    app_context=ctx,
                    input_image=img,
                    layout_image=item["layout_result"].img,
                    ocr_image=item["ocr_result"].img,
                )
                seal_rec_results.append(
                    SealRecResult(
                        texts=texts,
                        inputImage=input_img,
                        layoutImage=layout_img,
                        ocrImage=ocr_img,
                    )
                )

            return ResultResponse[InferResult](
                logId=log_id,
                result=InferResult(
                    sealRecResults=seal_rec_results,
                    dataInfo=data_info,
                ),
            )

        except Exception:
            logging.exception("Unexpected exception")
            raise HTTPException(status_code=500, detail="Internal server error")

    return app
