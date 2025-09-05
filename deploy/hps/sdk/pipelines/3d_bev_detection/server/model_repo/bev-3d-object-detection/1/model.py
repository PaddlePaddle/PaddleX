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

import os
from typing import Any, Dict, List

from paddlex_hps_server import BaseTritonPythonModel, schemas, utils


class TritonPythonModel(BaseTritonPythonModel):
    def get_input_model_type(self):
        return schemas.m_3d_bev_detection.InferRequest

    def get_result_model_type(self):
        return schemas.m_3d_bev_detection.InferResult

    def run(self, input, log_id):
        file_bytes = utils.get_raw_bytes(input.tar)
        tar_path = utils.write_to_temp_file(
            file_bytes,
            suffix=".tar",
        )

        try:
            result = list(
                self.pipeline(
                    tar_path,
                )
            )[0]
        finally:
            os.unlink(tar_path)

        objects: List[Dict[str, Any]] = []
        for box, label, score in zip(
            result["boxes_3d"], result["labels_3d"], result["scores_3d"]
        ):
            objects.append(
                dict(
                    bbox=box,
                    categoryId=label,
                    score=score,
                )
            )

        return schemas.m_3d_bev_detection.InferResult(detectedObjects=objects)
