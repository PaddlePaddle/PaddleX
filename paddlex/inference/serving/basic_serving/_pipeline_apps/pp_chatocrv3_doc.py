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
from ...schemas import pp_chatocrv3_doc as schema
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
        schema.ANALYZE_IMAGES_ENDPOINT,
        "analyzeImages",
    )
    async def _analyze_images(
        request: schema.AnalyzeImagesRequest,
    ) -> AIStudioResultResponse[schema.AnalyzeImagesResult]:
        pipeline = ctx.pipeline

        log_id = serving_utils.generate_log_id()

        images, data_info = await ocr_common.get_images(request, ctx)

        result = await pipeline.call(
            pipeline.pipeline.visual_predict,
            images,
            use_doc_orientation_classify=request.useDocOrientationClassify,
            use_doc_unwarping=request.useDocUnwarping,
            use_seal_recognition=request.useSealRecognition,
            use_table_recognition=request.useTableRecognition,
            layout_threshold=request.layoutThreshold,
            layout_nms=request.layoutNms,
            layout_unclip_ratio=request.layoutUnclipRatio,
            layout_merge_bboxes_mode=request.layoutMergeBboxesMode,
            text_det_limit_side_len=request.textDetLimitSideLen,
            text_det_limit_type=request.textDetLimitType,
            text_det_thresh=request.textDetThresh,
            text_det_box_thresh=request.textDetBoxThresh,
            text_det_unclip_ratio=request.textDetUnclipRatio,
            text_rec_score_thresh=request.textRecScoreThresh,
            seal_det_limit_side_len=request.sealDetLimitSideLen,
            seal_det_limit_type=request.sealDetLimitType,
            seal_det_thresh=request.sealDetThresh,
            seal_det_box_thresh=request.sealDetBoxThresh,
            seal_det_unclip_ratio=request.sealDetUnclipRatio,
            seal_rec_score_thresh=request.sealRecScoreThresh,
        )

        layout_parsing_results: List[Dict[str, Any]] = []
        visual_info: List[dict] = []
        for i, (img, item) in enumerate(zip(images, result)):
            pruned_res = common.prune_result(item["layout_parsing_result"].json["res"])
            if ctx.config.visualize:
                imgs = {
                    "input_img": img,
                    **item["layout_parsing_result"].img,
                }
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
            layout_parsing_results.append(
                dict(
                    prunedResult=pruned_res,
                    outputImages=(
                        {k: v for k, v in imgs.items() if k != "input_img"}
                        if imgs
                        else None
                    ),
                    inputImage=imgs.get("input_img"),
                )
            )
            visual_info.append(item["visual_info"])

        return AIStudioResultResponse[schema.AnalyzeImagesResult](
            logId=log_id,
            result=schema.AnalyzeImagesResult(
                layoutParsingResults=layout_parsing_results,
                visualInfo=visual_info,
                dataInfo=data_info,
            ),
        )

    @primary_operation(
        app,
        schema.BUILD_VECTOR_STORE_ENDPOINT,
        "buildVectorStore",
    )
    async def _build_vector_store(
        request: schema.BuildVectorStoreRequest,
    ) -> AIStudioResultResponse[schema.BuildVectorStoreResult]:
        pipeline = ctx.pipeline

        kwargs: Dict[str, Any] = {
            "flag_save_bytes_vector": True,
            "retriever_config": request.retrieverConfig,
        }
        if request.minCharacters is not None:
            kwargs["min_characters"] = request.minCharacters
        if request.blockSize is not None:
            kwargs["block_size"] = request.blockSize

        vector_info = await serving_utils.call_async(
            pipeline.pipeline.build_vector,
            request.visualInfo,
            **kwargs,
        )

        return AIStudioResultResponse[schema.BuildVectorStoreResult](
            logId=serving_utils.generate_log_id(),
            result=schema.BuildVectorStoreResult(vectorInfo=vector_info),
        )

    @primary_operation(
        app,
        schema.CHAT_ENDPOINT,
        "chat",
    )
    async def _chat(
        request: schema.ChatRequest,
    ) -> AIStudioResultResponse[schema.ChatResult]:
        pipeline = ctx.pipeline

        kwargs: Dict[str, Any] = dict(
            vector_info=request.vectorInfo,
            text_task_description=request.textTaskDescription,
            text_output_format=request.textOutputFormat,
            text_rules_str=request.textRulesStr,
            text_few_shot_demo_text_content=request.textFewShotDemoTextContent,
            text_few_shot_demo_key_value_list=request.textFewShotDemoKeyValueList,
            table_task_description=request.tableTaskDescription,
            table_output_format=request.tableOutputFormat,
            table_rules_str=request.tableRulesStr,
            table_few_shot_demo_text_content=request.tableFewShotDemoTextContent,
            table_few_shot_demo_key_value_list=request.tableFewShotDemoKeyValueList,
            chat_bot_config=request.chatBotConfig,
            retriever_config=request.retrieverConfig,
        )
        if request.useVectorRetrieval is not None:
            kwargs["use_vector_retrieval"] = request.useVectorRetrieval
        if request.minCharacters is not None:
            kwargs["min_characters"] = request.minCharacters

        result = await serving_utils.call_async(
            pipeline.pipeline.chat,
            request.keyList,
            request.visualInfo,
            **kwargs,
        )

        return AIStudioResultResponse[schema.ChatResult](
            logId=serving_utils.generate_log_id(),
            result=schema.ChatResult(
                chatResult=result["chat_res"],
            ),
        )

    return app
