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

from operator import attrgetter

from paddlex.inference.pipelines.components import IndexData
from paddlex_hps_server import schemas, utils

from common.base_model import BaseFaceRecognitionModel


class TritonPythonModel(BaseFaceRecognitionModel):
    def get_input_model_type(self):
        return schemas.face_recognition.AddImagesToIndexRequest

    def get_result_model_type(self):
        return schemas.face_recognition.AddImagesToIndexResult

    def run(self, input, log_id):
        file_bytes_list = [
            utils.get_raw_bytes(img)
            for img in map(attrgetter("image"), input.imageLabelPairs)
        ]
        images = [utils.image_bytes_to_array(item) for item in file_bytes_list]
        labels = [pair.label for pair in input.imageLabelPairs]

        index_storage = self.context["index_storage"]
        index_data_bytes = index_storage.get(input.indexKey)
        index_data = IndexData.from_bytes(index_data_bytes)

        index_data = self.pipeline.append_index(images, labels, index_data)

        index_data_bytes = index_data.to_bytes()
        index_storage.set(input.indexKey, index_data_bytes)

        return schemas.face_recognition.AddImagesToIndexResult(
            imageCount=len(index_data.id_map)
        )
