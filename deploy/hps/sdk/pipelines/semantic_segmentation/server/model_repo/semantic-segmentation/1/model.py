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

from paddlex_hps_server import BaseTritonPythonModel, schemas, utils


class TritonPythonModel(BaseTritonPythonModel):
    def get_input_model_type(self):
        return schemas.semantic_segmentation.InferRequest

    def get_result_model_type(self):
        return schemas.semantic_segmentation.InferResult

    def run(self, input, log_id):
        file_bytes = utils.get_raw_bytes(input.image)
        image = utils.image_bytes_to_array(file_bytes)
        visualize_enabled = (
            input.visualize
            if input.visualize is not None
            else self.app_config.visualize
        )

        result = list(self.pipeline.predict(image, target_size=input.targetSize))[0]

        pred = result["pred"][0].tolist()
        size = [len(pred), len(pred[0])]
        label_map = [item for sublist in pred for item in sublist]
        if visualize_enabled:
            output_image_base64 = utils.base64_encode(
                utils.image_to_bytes(result.img["res"].convert("RGB"))
            )
        else:
            output_image_base64 = None

        return schemas.semantic_segmentation.InferResult(
            labelMap=label_map, size=size, image=output_image_base64
        )
