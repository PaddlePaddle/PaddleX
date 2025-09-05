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

from typing import Any, Dict, Final, List, Tuple

from paddlex_hps_server import (
    BaseTritonPythonModel,
    app_common,
    protocol,
    schemas,
    utils,
)
from paddlex_hps_server.storage import SupportsGetURL, create_storage

_DEFAULT_MAX_NUM_INPUT_IMGS: Final[int] = 10
_DEFAULT_MAX_OUTPUT_IMG_SIZE: Final[Tuple[int, int]] = (2000, 2000)


class TritonPythonModel(BaseTritonPythonModel):
    def initialize(self, args):
        super().initialize(args)
        self.context = {}
        self.context["file_storage"] = None
        self.context["return_img_urls"] = False
        self.context["max_num_input_imgs"] = _DEFAULT_MAX_NUM_INPUT_IMGS
        self.context["max_output_img_size"] = _DEFAULT_MAX_OUTPUT_IMG_SIZE
        if self.app_config.extra:
            if "file_storage" in self.app_config.extra:
                self.context["file_storage"] = create_storage(
                    self.app_config.extra["file_storage"]
                )
            if "return_img_urls" in self.app_config.extra:
                self.context["return_img_urls"] = self.app_config.extra[
                    "return_img_urls"
                ]
            if "max_num_input_imgs" in self.app_config.extra:
                self.context["max_num_input_imgs"] = self.app_config.extra[
                    "max_num_input_imgs"
                ]
            if "max_output_img_size" in self.app_config.extra:
                self.context["max_output_img_size"] = self.app_config.extra[
                    "max_output_img_size"
                ]
        if self.context["return_img_urls"]:
            file_storage = self.context["file_storage"]
            if not file_storage:
                raise ValueError(
                    "The file storage must be properly configured when URLs need to be returned."
                )
            if not isinstance(file_storage, SupportsGetURL):
                raise TypeError(f"{type(file_storage)} does not support getting URLs.")

    def get_input_model_type(self):
        return schemas.doc_preprocessor.InferRequest

    def get_result_model_type(self):
        return schemas.doc_preprocessor.InferResult

    def run(self, input, log_id):
        if input.fileType is None:
            if utils.is_url(input.file):
                maybe_file_type = utils.infer_file_type(input.file)
                if maybe_file_type is None or not (
                    maybe_file_type == "PDF" or maybe_file_type == "IMAGE"
                ):
                    return protocol.create_aistudio_output_without_result(
                        422,
                        "Unsupported file type",
                        log_id=log_id,
                    )
                file_type = maybe_file_type
            else:
                return protocol.create_aistudio_output_without_result(
                    422,
                    "File type cannot be determined",
                    log_id=log_id,
                )
        else:
            file_type = "PDF" if input.fileType == 0 else "IMAGE"

        file_bytes = utils.get_raw_bytes(input.file)
        images, data_info = utils.file_to_images(
            file_bytes,
            file_type,
            max_num_imgs=self.context["max_num_input_imgs"],
        )

        result = list(
            self.pipeline(
                images,
                use_doc_orientation_classify=input.useDocOrientationClassify,
                use_doc_unwarping=input.useDocUnwarping,
            )
        )
        visualize_enabled = (
            input.visualize
            if input.visualize is not None
            else self.app_config.visualize
        )

        doc_pp_results: List[Dict[str, Any]] = []
        for i, (img, item) in enumerate(zip(images, result)):
            pruned_res = app_common.prune_result(item.json["res"])
            output_img = app_common.postprocess_image(
                item["output_img"],
                log_id,
                "output_img.png",
                file_storage=self.context["file_storage"],
                return_url=self.context["return_img_urls"],
                max_img_size=self.context["max_output_img_size"],
            )
            if visualize_enabled:
                vis_imgs = {
                    "input_img": img,
                    "doc_preprocessing_img": item.img["preprocessed_img"],
                }
                vis_imgs = app_common.postprocess_images(
                    vis_imgs,
                    log_id,
                    filename_template=f"{{key}}_{i}.jpg",
                    file_storage=self.context["file_storage"],
                    return_urls=self.context["return_img_urls"],
                    max_img_size=self.context["max_output_img_size"],
                )
            else:
                vis_imgs = {}
            doc_pp_results.append(
                dict(
                    outputImage=output_img,
                    prunedResult=pruned_res,
                    docPreprocessingImage=vis_imgs.get("doc_preprocessing_img"),
                    inputImage=vis_imgs.get("input_img"),
                )
            )

        return schemas.doc_preprocessor.InferResult(
            docPreprocessingResults=doc_pp_results,
            dataInfo=data_info,
        )
