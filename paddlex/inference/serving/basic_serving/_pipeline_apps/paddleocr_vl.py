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
from ...schemas.paddleocr_vl import (
    INFER_ENDPOINT,
    RESTRUCTURE_PAGES_ENDPOINT,
    InferRequest,
    InferResult,
    RestructurePagesRequest,
    RestructurePagesResult,
)
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
    async def _infer(
        request: InferRequest,
    ) -> AIStudioResultResponse[InferResult]:
        pipeline = ctx.pipeline

        log_id = request.logId if request.logId else serving_utils.generate_log_id()
        visualize_enabled = (
            request.visualize if request.visualize is not None else ctx.config.visualize
        )
        images, data_info = await ocr_common.get_images(request, ctx)

        result = await pipeline.infer(
            images,
            use_doc_orientation_classify=request.useDocOrientationClassify,
            use_doc_unwarping=request.useDocUnwarping,
            use_layout_detection=request.useLayoutDetection,
            use_chart_recognition=request.useChartRecognition,
            use_seal_recognition=request.useSealRecognition,
            use_ocr_for_image_block=request.useOcrForImageBlock,
            layout_threshold=request.layoutThreshold,
            layout_nms=request.layoutNms,
            layout_unclip_ratio=request.layoutUnclipRatio,
            layout_merge_bboxes_mode=request.layoutMergeBboxesMode,
            layout_shape_mode=request.layoutShapeMode,
            prompt_label=request.promptLabel,
            format_block_content=request.formatBlockContent,
            repetition_penalty=request.repetitionPenalty,
            temperature=request.temperature,
            top_p=request.topP,
            min_pixels=request.minPixels,
            max_pixels=request.maxPixels,
            max_new_tokens=request.maxNewTokens,
            merge_layout_blocks=request.mergeLayoutBlocks,
            markdown_ignore_labels=request.markdownIgnoreLabels,
            vlm_extra_args=request.vlmExtraArgs,
        )

        orig_result = result
        if request.restructurePages:
            result = await serving_utils.call_async(
                pipeline.pipeline.restructure_pages,
                result,
                merge_tables=request.mergeTables,
                relevel_titles=request.relevelTitles,
                concatenate_pages=False,
            )
            result = list(result)

        layout_parsing_results: List[Dict[str, Any]] = []
        for i, (img, item) in enumerate(zip(images, result)):
            pruned_res = common.prune_result(item.json["res"])
            # XXX
            md_data = item._to_markdown(
                pretty=request.prettifyMarkdown,
                show_formula_number=request.showFormulaNumber,
            )
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
            if visualize_enabled:
                imgs = {
                    "input_img": img,
                    **item.img,
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
                    ),
                    outputImages=(
                        {k: v for k, v in imgs.items() if k != "input_img"}
                        if imgs
                        else None
                    ),
                    inputImage=imgs.get("input_img"),
                )
            )

        return AIStudioResultResponse[InferResult](
            logId=log_id,
            result=InferResult(
                layoutParsingResults=layout_parsing_results,
                dataInfo=data_info,
            ),
        )

    @primary_operation(
        app,
        RESTRUCTURE_PAGES_ENDPOINT,
        "restructurePages",
    )
    async def _restructure_pages(
        request: RestructurePagesRequest,
    ) -> AIStudioResultResponse[RestructurePagesResult]:
        def _to_original_result(pruned_res, page_index):
            res = {**pruned_res, "input_path": "", "page_index": page_index}
            orig_res = {"res": res}
            return orig_res

        pipeline = ctx.pipeline

        log_id = request.logId if request.logId else serving_utils.generate_log_id()

        original_results = []
        markdown_images = {}
        for i, page in enumerate(request.pages):
            orig_res = _to_original_result(page.prunedResult, i)
            original_results.append(orig_res)
            if request.concatenatePages:
                markdown_images.update(page.markdownImages)

        restructured_results = await serving_utils.call_async(
            pipeline.pipeline.restructure_pages,
            original_results,
            merge_tables=request.mergeTables,
            relevel_titles=request.relevelTitles,
            concatenate_pages=request.concatenatePages,
        )
        restructured_results = list(restructured_results)

        layout_parsing_results = []
        if request.concatenatePages:
            layout_parsing_result = {}
            layout_parsing_result["prunedResult"] = common.prune_result(
                restructured_results[0].json["res"]
            )
            # XXX
            md_data = restructured_results[0]._to_markdown(
                pretty=request.prettifyMarkdown,
                show_formula_number=request.showFormulaNumber,
            )
            layout_parsing_result["markdown"] = dict(
                text=md_data["markdown_texts"],
                images=markdown_images,
            )
            layout_parsing_results.append(layout_parsing_result)
        else:
            for new_res, old_page in zip(restructured_results, request.pages):
                layout_parsing_result = {}
                layout_parsing_result["prunedResult"] = common.prune_result(
                    new_res.json["res"]
                )
                # XXX
                md_data = new_res._to_markdown(
                    pretty=request.prettifyMarkdown,
                    show_formula_number=request.showFormulaNumber,
                )
                layout_parsing_result["markdown"] = dict(
                    text=md_data["markdown_texts"],
                    images=old_page.markdownImages,
                )
                layout_parsing_results.append(layout_parsing_result)

        return AIStudioResultResponse[RestructurePagesResult](
            logId=log_id,
            result=RestructurePagesResult(
                layoutParsingResults=layout_parsing_results,
            ),
        )

    return app
