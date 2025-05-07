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
from ...schemas.ocr import INFER_ENDPOINT, InferRequest, InferResult
from .._app import create_app, primary_operation
from ._common import common
from ._common import ocr as ocr_common

if is_dep_available("fastapi"):
    from fastapi import FastAPI


@function_requires_deps("fastapi")
def create_pipeline_app(pipeline: Any, app_config: AppConfig) -> "FastAPI":
    app, ctx = create_app(
        pipeline=pipeline, app_config=app_config, app_aiohttp_session=True
    )

    ocr_common.update_app_context(ctx)

    @primary_operation(
        app,
        INFER_ENDPOINT,
        "infer",
    )
    async def _infer(request: InferRequest) -> AIStudioResultResponse[InferResult]:
        pipeline = ctx.pipeline

        log_id = serving_utils.generate_log_id()

        images, data_info = await ocr_common.get_images(request, ctx)

        result = await pipeline.infer(
            images,
            use_doc_orientation_classify=request.useDocOrientationClassify,
            use_doc_unwarping=request.useDocUnwarping,
            use_textline_orientation=request.useTextlineOrientation,
            text_det_limit_side_len=request.textDetLimitSideLen,
            text_det_limit_type=request.textDetLimitType,
            text_det_thresh=request.textDetThresh,
            text_det_box_thresh=request.textDetBoxThresh,
            text_det_unclip_ratio=request.textDetUnclipRatio,
            text_rec_score_thresh=request.textRecScoreThresh,
        )

        ocr_results: List[Dict[str, Any]] = []
        for i, (img, item) in enumerate(zip(images, result)):
            pruned_res = common.prune_result(item.json["res"])
            if ctx.config.visualize:
                output_imgs = item.img
                imgs = {
                    "input_img": img,
                    "ocr_img": output_imgs["ocr_res_img"],
                }
                if "preprocessed_img" in output_imgs:
                    imgs["doc_preprocessing_img"] = output_imgs["preprocessed_img"]
                imgs = await serving_utils.call_async(
                    common.postprocess_images,
                    imgs,
                    log_id,
                    filename_template=f"{{key}}_{i}.jpg",
                    file_storage=ctx.extra["file_storage"],
                    return_urls=ctx.extra["return_img_urls"],
                    max_img_size=ctx.extra["max_output_img_size"],
                )
            else:
                imgs = {}
            ocr_results.append(
                dict(
                    prunedResult=pruned_res,
                    ocrImage=imgs.get("ocr_img"),
                    docPreprocessingImage=imgs.get("doc_preprocessing_img"),
                    inputImage=imgs.get("input_img"),
                )
            )

        return AIStudioResultResponse[InferResult](
            logId=log_id,
            result=InferResult(
                ocrResults=ocr_results,
                dataInfo=data_info,
            ),
        )

    return app
