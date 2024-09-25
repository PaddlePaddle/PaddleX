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

import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypeAlias

from .. import utils as serving_utils
from ..app import AppConfig, create_app
from ..models import Response, ResultResponse
from ...pipelines import DetPipeline


_logger = logging.getLogger(__name__)


class InferRequest(BaseModel):
    image: str


BoundingBox: TypeAlias = Annotated[List[float], Field(min_length=4, max_length=4)]


class DetectedObject(BaseModel):
    bbox: BoundingBox
    categoryId: int
    score: float


class InferResult(BaseModel):
    detectedObjects: List[DetectedObject]
    image: str


def create_pipeline_app(app_config: AppConfig) -> FastAPI:
    app, ctx = create_app(
        pipeline_cls=DetPipeline, app_config=app_config, app_aiohttp_session=True
    )

    @app.post(
        "/object-detection", operation_id="infer", responses={422: {"model": Response}}
    )
    async def _infer(request: InferRequest) -> ResultResponse[InferResult]:
        pipeline = ctx.pipeline
        aiohttp_session = ctx.aiohttp_session

        try:
            file_bytes = await serving_utils.get_raw_bytes(
                request.image, aiohttp_session
            )
            image = serving_utils.image_bytes_to_array(file_bytes)
            top_k: Optional[int] = None
            if request.inferenceParams is not None:
                if request.inferenceParams.topK is not None:
                    top_k = request.inferenceParams.topK

            result = await pipeline.infer(image)

            categories: List[Category] = []
            for id_, score in islice(
                zip(result["class_ids"], result["scores"]), None, top_k
            ):
                if "label_names" in result:
                    name = result["label_names"][id_]
                else:
                    name = str(id_)
                categories.append(cat=Category(id=id_, name=name, score=score))
            output_image_base64 = result.to_base64()

            return ResultResponse(
                logId=serving_utils.generate_log_id(),
                errorCode=0,
                errorMsg="Success",
                result=InferResult(categories=categories, image=output_image_base64),
            )

        except Exception as e:
            _logger.exception(e)
            raise HTTPException(status_code=500, detail="Internal server error")

    return app
