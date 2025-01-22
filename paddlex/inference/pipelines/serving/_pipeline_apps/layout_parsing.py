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

from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypeAlias

from ._common import cv as cv_common, ocr as ocr_common
from .....utils import logging
from ...layout_parsing import LayoutParsingPipeline
from .. import utils as serving_utils
from ..app import AppConfig, create_app
from ..models import NoResultResponse, ResultResponse, DataInfo


class InferRequest(ocr_common.InferRequest):
    useImgOrientationCls: bool = True
    useImgUnwarping: bool = True
    useSealTextDet: bool = True


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
    dataInfo: DataInfo


def create_pipeline_app(
    pipeline: LayoutParsingPipeline, app_config: AppConfig
) -> FastAPI:
    app, ctx = create_app(
        pipeline=pipeline, app_config=app_config, app_aiohttp_session=True
    )

    ocr_common.update_app_context(ctx)

    @app.post(
        "/layout-parsing",
        operation_id="infer",
        responses={422: {"model": NoResultResponse}},
        response_model_exclude_none=True,
    )
    async def _infer(
        request: InferRequest,
    ) -> ResultResponse[InferResult]:
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
            result = await pipeline.infer(
                images,
                use_doc_image_ori_cls_model=request.useImgOrientationCls,
                use_doc_image_unwarp_model=request.useImgUnwarping,
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
                            cv_common.postprocess_image,
                            subitem[label]["img"],
                            log_id=log_id,
                            filename=f"image_{i}_{j}.jpg",
                            file_storage=ctx.extra["file_storage"],
                            return_url=ctx.extra["return_img_urls"],
                            max_img_size=ctx.extra["max_output_img_size"],
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

            return ResultResponse[InferResult](
                logId=log_id,
                result=InferResult(
                    layoutParsingResults=layout_parsing_results,
                    dataInfo=data_info,
                ),
            )

        except Exception:
            logging.exception("Unexpected exception")
            raise HTTPException(status_code=500, detail="Internal server error")

    return app
