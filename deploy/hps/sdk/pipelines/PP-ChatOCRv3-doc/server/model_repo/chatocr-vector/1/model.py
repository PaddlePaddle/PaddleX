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

from paddlex_hps_server import BaseTritonPythonModel, schemas


class TritonPythonModel(BaseTritonPythonModel):
    @property
    def pipeline_creation_kwargs(self):
        return {"initial_predictor": False}

    def get_input_model_type(self):
        return schemas.pp_chatocrv3_doc.BuildVectorStoreRequest

    def get_result_model_type(self):
        return schemas.pp_chatocrv3_doc.BuildVectorStoreResult

    def run(self, input, log_id):
        vector_info = self.pipeline.build_vector(
            input.visualInfo,
            min_characters=input.minCharacters,
            block_size=input.blockSize,
            flag_save_bytes_vector=True,
            retriever_config=input.retrieverConfig,
        )

        return schemas.pp_chatocrv3_doc.BuildVectorStoreResult(vectorInfo=vector_info)
