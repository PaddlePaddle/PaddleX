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

from paddlex.inference.pipelines.paddleocr_vl.result import PaddleOCRVLResult
from paddlex_hps_server import (
    BaseTritonPythonModel,
    app_common,
    logging,
    protocol,
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
        pages = []
        for i, page in enumerate(input.pages):
            try:
                page = PaddleOCRVLResult(page)
            except Exception as e:
                logging.error("Failed to parse page %d: %s", i, e)
                return protocol.create_aistudio_output_without_result(
                    422,
                    "Unsupported file type",
                    log_id=log_id,
                )
            pages.append(page)

        concatenated_result = self.pipeline.concatenate_pages(
            pages,
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
        md_text = md_data["markdown_texts"]
        # TODO: Reuse images from `infer`
        md_imgs = app_common.postprocess_images(
            md_data["markdown_images"],
            log_id,
            filename_template=f"markdown_{i}/{{key}}",
            file_storage=self.context["file_storage"],
            return_urls=self.context["return_img_urls"],
            max_img_size=self.context["max_output_img_size"],
        )
        layout_parsing_result["markdown"] = dict(
            text=md_text,
            images=md_imgs,
        )

        return schemas.paddleocr_vl.ConcatenatePagesResult(
            layoutParsingResult=layout_parsing_result,
        )
