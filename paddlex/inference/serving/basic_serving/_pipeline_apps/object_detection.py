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

from typing import Any, Dict, List

from .....utils.deps import function_requires_deps, is_dep_available
from ...infra import utils as serving_utils
from ...infra.config import AppConfig
from ...infra.models import AIStudioResultResponse
from ...schemas.object_detection import INFER_ENDPOINT, InferRequest, InferResult
from .._app import create_app, primary_operation

if is_dep_available("fastapi"):
    from fastapi import FastAPI


@function_requires_deps("fastapi")
def create_pipeline_app(pipeline: Any, app_config: AppConfig) -> "FastAPI":
    app, ctx = create_app(
        pipeline=pipeline, app_config=app_config, app_aiohttp_session=True
    )

    @primary_operation(
        app,
        INFER_ENDPOINT,
        "infer",
    )
    async def _infer(request: InferRequest) -> AIStudioResultResponse[InferResult]:
        pipeline = ctx.pipeline
        aiohttp_session = ctx.aiohttp_session

        file_bytes = await serving_utils.get_raw_bytes_async(
            request.image, aiohttp_session
        )
        image = serving_utils.image_bytes_to_array(file_bytes)

        result = (
            await pipeline.infer(
                image,
                threshold=request.threshold,
            )
        )[0]

        objects: List[Dict[str, Any]] = []
        for obj in result["boxes"]:
            objects.append(
                dict(
                    bbox=obj["coordinate"],
                    categoryId=obj["cls_id"],
                    categoryName=obj["label"],
                    score=obj["score"],
                )
            )
        if ctx.config.visualize:
            output_image_base64 = serving_utils.base64_encode(
                serving_utils.image_to_bytes(result.img["res"])
            )
        else:
            output_image_base64 = None

        return AIStudioResultResponse[InferResult](
            logId=serving_utils.generate_log_id(),
            result=InferResult(detectedObjects=objects, image=output_image_base64),
        )

    return app
