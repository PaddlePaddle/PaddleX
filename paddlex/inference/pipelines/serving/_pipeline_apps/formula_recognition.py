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

from typing import List, Optional, Type

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypeAlias

from ._common import ocr as ocr_common
from .....utils import logging
from ...formula_recognition import FormulaRecognitionPipeline
from .. import utils as serving_utils
from ..app import AppConfig, create_app
from ..models import NoResultResponse, ResultResponse, DataInfo

InferRequest: Type[ocr_common.InferRequest] = ocr_common.InferRequest


Point: TypeAlias = Annotated[List[float], Field(min_length=2, max_length=2)]
Polygon: TypeAlias = Annotated[List[Point], Field(min_length=3)]


class Formula(BaseModel):
    poly: Polygon
    latex: str


class FormulaRecResult(BaseModel):
    formulas: List[Formula]
    inputImage: str
    layoutImage: str
    ocrImage: Optional[str] = None


class InferResult(BaseModel):
    formulaRecResults: List[FormulaRecResult]
    dataInfo: DataInfo


def create_pipeline_app(
    pipeline: FormulaRecognitionPipeline, app_config: AppConfig
) -> FastAPI:
    app, ctx = create_app(
        pipeline=pipeline, app_config=app_config, app_aiohttp_session=True
    )

    ocr_common.update_app_context(ctx)
    ctx.extra["return_ocr_imgs"] = False
    if ctx.config.extra:
        if "return_ocr_imgs" in ctx.config.extra:
            ctx.extra["return_ocr_imgs"] = ctx.config.extra["return_ocr_imgs"]

    @app.post(
        "/formula-recognition",
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

            formula_rec_results: List[FormulaRecResult] = []
            for i, (img, item) in enumerate(zip(images, result)):
                formulas: List[Formula] = []
                for poly, latex in zip(item["dt_polys"], item["rec_formula"]):
                    formulas.append(
                        Formula(
                            poly=poly,
                            latex=latex,
                        )
                    )
                layout_img = item["layout_result"].img
                if ctx.extra["return_ocr_imgs"]:
                    ocr_img = item["formula_result"].img
                    if ocr_img is None:
                        raise RuntimeError("Failed to get the OCR image")
                else:
                    ocr_img = None
                output_imgs = await ocr_common.postprocess_images(
                    log_id=log_id,
                    index=i,
                    app_context=ctx,
                    input_image=img,
                    layout_image=layout_img,
                    ocr_image=ocr_img,
                )
                if ocr_img is not None:
                    input_img, layout_img, ocr_img = output_imgs
                else:
                    input_img, layout_img = output_imgs
                formula_rec_results.append(
                    FormulaRecResult(
                        formulas=formulas,
                        inputImage=input_img,
                        layoutImage=layout_img,
                        ocrImage=ocr_img,
                    )
                )

            return ResultResponse[InferResult](
                logId=log_id,
                result=InferResult(
                    formulaRecResults=formula_rec_results,
                    dataInfo=data_info,
                ),
            )

        except Exception:
            logging.exception("Unexpected exception")
            raise HTTPException(status_code=500, detail="Internal server error")

    return app
