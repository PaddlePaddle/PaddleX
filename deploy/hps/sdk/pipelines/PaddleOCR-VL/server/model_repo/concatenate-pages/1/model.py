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
        return schemas.paddleocr_vl.ConcatenatePagesRequest

    def get_result_model_type(self):
        return schemas.paddleocr_vl.ConcatenatePagesResult

    def run(self, input, log_id):
        def _to_original_result(pruned_res):
            orig_res = {"res": pruned_res}
            return orig_res

        original_results = []
        markdown_images = {}
        for i, page in enumerate(input.pages):
            orig_res = _to_original_result(page.prunedResult)
            original_results.append(orig_res)
            markdown_images.update(page.markdownImages)

        concatenated_result = self.pipeline.concatenate_pages(
            original_results,
            merge_table=input.mergeTable,
            title_level=input.titleLevel,
        )

        layout_parsing_result = {}
        layout_parsing_result["prunedResult"] = app_common.prune_result(
            concatenated_result.json["res"]
        )
        # XXX
        md_data = concatenated_result._to_markdown(
            pretty=input.prettifyMarkdown,
            show_formula_number=input.showFormulaNumber,
        )
        layout_parsing_result["markdown"] = dict(
            text=md_data["markdown_texts"],
            images=markdown_images,
        )

        return schemas.paddleocr_vl.ConcatenatePagesResult(
            layoutParsingResult=layout_parsing_result,
        )
