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

from paddlex_hps_server import (
    BaseTritonPythonModel,
    app_common,
    schemas,
)


class TritonPythonModel(BaseTritonPythonModel):
    @property
    def pipeline_creation_kwargs(self):
        return {"initial_predictor": False}

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
            if input.concatenatePages:
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
                images=markdown_images,
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
                    images=old_page.markdownImages,
                )
                layout_parsing_results.append(layout_parsing_result)

        return schemas.paddleocr_vl.RestructurePagesResult(
            layoutParsingResults=layout_parsing_results,
        )
