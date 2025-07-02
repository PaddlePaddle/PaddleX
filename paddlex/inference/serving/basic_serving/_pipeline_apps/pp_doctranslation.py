# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from ...schemas import pp_doctranslation as schema
from .._app import create_app, primary_operation
from ._common import common
from ._common import ocr as ocr_common

if is_dep_available("fastapi"):
    from fastapi import FastAPI


@function_requires_deps("fastapi")
def create_pipeline_app(pipeline: Any, app_config: AppConfig) -> "FastAPI":
    pipeline.inintial_visual_predictor(pipeline.config)

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
        visualize_enabled = (
            request.visualize if request.visualize is not None else ctx.config.visualize
        )
        images, data_info = await ocr_common.get_images(request, ctx)

        result = await pipeline.call(
            pipeline.pipeline.visual_predict,
            images,
            use_doc_orientation_classify=request.useDocOrientationClassify,
            use_doc_unwarping=request.useDocUnwarping,
            use_textline_orientation=request.useTextlineOrientation,
            use_seal_recognition=request.useSealRecognition,
            use_table_recognition=request.useTableRecognition,
            use_formula_recognition=request.useFormulaRecognition,
            use_chart_recognition=request.useChartRecognition,
            use_region_detection=request.useRegionDetection,
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
            use_wired_table_cells_trans_to_html=request.useWiredTableCellsTransToHtml,
            use_wireless_table_cells_trans_to_html=request.useWirelessTableCellsTransToHtml,
            use_table_orientation_classify=request.useTableOrientationClassify,
            use_ocr_results_with_table_cells=request.useOcrResultsWithTableCells,
            use_e2e_wired_table_rec_model=request.useE2eWiredTableRecModel,
            use_e2e_wireless_table_rec_model=request.useE2eWirelessTableRecModel,
        )

        layout_parsing_results: List[Dict[str, Any]] = []
        for i, (img, item) in enumerate(zip(images, result)):
            pruned_res = common.prune_result(item["layout_parsing_result"].json["res"])
            md_data = item["layout_parsing_result"].markdown
            md_text = md_data["markdown_texts"]
            md_imgs = await serving_utils.call_async(
                common.postprocess_images,
                md_data["markdown_images"],
                log_id,
                filename_template=f"markdown_{i}/{{key}}",
                file_storage=ctx.extra["file_storage"],
                return_urls=ctx.extra["return_img_urls"],
                max_img_size=ctx.extra["max_output_img_size"],
            )
            md_flags = md_data["page_continuation_flags"]
            if visualize_enabled:
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
                    markdown=dict(
                        text=md_text,
                        images=md_imgs,
                        isStart=md_flags[0],
                        isEnd=md_flags[1],
                    ),
                    outputImages=(
                        {k: v for k, v in imgs.items() if k != "input_img"}
                        if imgs
                        else None
                    ),
                    inputImage=imgs.get("input_img"),
                )
            )

        return AIStudioResultResponse[schema.AnalyzeImagesResult](
            logId=log_id,
            result=schema.AnalyzeImagesResult(
                layoutParsingResults=layout_parsing_results,
                dataInfo=data_info,
            ),
        )

    @primary_operation(
        app,
        schema.TRANSLATE_ENDPOINT,
        "translate",
    )
    async def _translate(
        request: schema.TranslateRequest,
    ) -> AIStudioResultResponse[schema.TranslateResult]:
        pipeline = ctx.pipeline

        ori_md_info_list: List[Dict[str, Any]] = []
        for i, item in enumerate(request.markdownList):
            ori_md_info_list.append(
                {
                    "input_path": None,
                    "page_index": i,
                    "markdown_texts": item.text,
                    "page_continuation_flags": (item.isStart, item.isEnd),
                }
            )

        result = await serving_utils.call_async(
            pipeline.pipeline.translate,
            ori_md_info_list,
            target_language=request.targetLanguage,
            chunk_size=request.chunkSize,
            task_description=request.taskDescription,
            output_format=request.outputFormat,
            rules_str=request.rulesStr,
            few_shot_demo_text_content=request.fewShotDemoTextContent,
            few_shot_demo_key_value_list=request.fewShotDemoKeyValueList,
            glossary=request.glossary,
            llm_request_interval=request.llmRequestInterval,
            chat_bot_config=request.chatBotConfig,
        )

        translation_results: List[Dict[str, Any]] = []
        for item in result:
            translation_results.append(
                dict(
                    language=item["language"],
                    markdown=dict(
                        text=item["markdown_texts"],
                        isStart=item["page_continuation_flags"][0],
                        isEnd=item["page_continuation_flags"][1],
                    ),
                )
            )

        return AIStudioResultResponse[schema.TranslateResult](
            logId=serving_utils.generate_log_id(),
            result=schema.TranslateResult(
                translationResults=translation_results,
            ),
        )

    return app
