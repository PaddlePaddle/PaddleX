# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import warnings

from paddlex_hps_server import (
    BaseTritonPythonModel,
    app_common,
    schemas,
)
from paddlex_hps_server.storage import SupportsGetURL, create_storage


class TritonPythonModel(BaseTritonPythonModel):
    @property
    def pipeline_creation_kwargs(self):
        return {"initial_predictor": False}

    def initialize(self, args):
        super().initialize(args)
        self.context = {}
        self.context["file_storage"] = None
        self.context["return_urls"] = self.app_config.return_urls
        self.context["url_expires_in"] = -1
        if self.app_config.extra:
            if "file_storage" in self.app_config.extra:
                self.context["file_storage"] = create_storage(
                    self.app_config.extra["file_storage"]
                )
            if "return_img_urls" in self.app_config.extra:
                warnings.warn(
                    "`Serving.extra.return_img_urls` is deprecated; use the "
                    "top-level `Serving.return_urls` field instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                if self.context["return_urls"] is None:
                    self.context["return_urls"] = bool(
                        self.app_config.extra["return_img_urls"]
                    )
            if "url_expires_in" in self.app_config.extra:
                self.context["url_expires_in"] = self.app_config.extra["url_expires_in"]
        if self.context["return_urls"] is None:
            self.context["return_urls"] = False
        if self.context["return_urls"]:
            file_storage = self.context["file_storage"]
            if not file_storage:
                raise ValueError(
                    "The file storage must be properly configured when URLs need to be returned."
                )
            if not isinstance(file_storage, SupportsGetURL):
                raise TypeError(f"{type(file_storage)} does not support getting URLs.")

    def get_input_model_type(self):
        return schemas.paddleocr_vl.RestructurePagesRequest

    def get_result_model_type(self):
        return schemas.paddleocr_vl.RestructurePagesResult

    def run(self, input, log_id):
        def _to_original_result(pruned_res, page_index):
            res = {**pruned_res, "input_path": "", "page_index": page_index}
            orig_res = {"res": res}
            return orig_res

        original_results = []
        markdown_images = {}
        for i, page in enumerate(input.pages):
            orig_res = _to_original_result(page.prunedResult, i)
            original_results.append(orig_res)
            if input.concatenatePages and page.markdownImages:
                markdown_images.update(page.markdownImages)

        restructured_results = self.pipeline.restructure_pages(
            original_results,
            merge_tables=input.mergeTables,
            relevel_titles=input.relevelTitles,
            concatenate_pages=input.concatenatePages,
        )
        restructured_results = list(restructured_results)

        layout_parsing_results = []
        if input.concatenatePages:
            layout_parsing_result = {}
            layout_parsing_result["prunedResult"] = app_common.prune_result(
                restructured_results[0].json["res"]
            )
            # XXX
            md_data = restructured_results[0]._to_markdown(
                pretty=input.prettifyMarkdown,
                show_formula_number=input.showFormulaNumber,
            )
            layout_parsing_result["markdown"] = dict(
                text=md_data["markdown_texts"],
                images=markdown_images if input.returnMarkdownImages else None,
            )
            if app_common.normalize_output_formats(input.outputFormats):
                res_obj = restructured_results[0]
                app_common.refill_paddleocr_vl_images_from_markdown(
                    res_obj, markdown_images
                )
                layout_parsing_result["exports"] = app_common.build_pipeline_exports(
                    input.outputFormats,
                    res_obj,
                    log_id=log_id,
                    file_storage=self.context["file_storage"],
                    return_urls=self.context["return_urls"],
                    url_expires_in=self.context["url_expires_in"],
                )
            layout_parsing_results.append(layout_parsing_result)
        else:
            for new_res, old_page in zip(restructured_results, input.pages):
                layout_parsing_result = {}
                layout_parsing_result["prunedResult"] = app_common.prune_result(
                    new_res.json["res"]
                )
                # XXX
                md_data = new_res._to_markdown(
                    pretty=input.prettifyMarkdown,
                    show_formula_number=input.showFormulaNumber,
                )
                layout_parsing_result["markdown"] = dict(
                    text=md_data["markdown_texts"],
                    images=(
                        old_page.markdownImages
                        if input.returnMarkdownImages
                        else None
                    ),
                )
                if app_common.normalize_output_formats(input.outputFormats):
                    app_common.refill_paddleocr_vl_images_from_markdown(
                        new_res, old_page.markdownImages
                    )
                    layout_parsing_result["exports"] = (
                        app_common.build_pipeline_exports(
                            input.outputFormats,
                            new_res,
                            log_id=log_id,
                            file_storage=self.context["file_storage"],
                            return_urls=self.context["return_urls"],
                            url_expires_in=self.context["url_expires_in"],
                        )
                    )
                layout_parsing_results.append(layout_parsing_result)

        return schemas.paddleocr_vl.RestructurePagesResult(
            layoutParsingResults=layout_parsing_results,
        )
