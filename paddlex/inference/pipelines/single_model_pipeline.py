# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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

from .base import BasePipeline


class SingleModelPipeline(BasePipeline):

    entities = [
        "image_classification",
        "object_detection",
        "instance_segmentation",
        "semantic_segmentation",
        "ts_fc",
        "ts_ad",
        "ts_cls",
        "multi_label_image_classification",
        "small_object_detection" "anomaly_detection",
    ]

    def __init__(self, model, predictor_kwargs=None):
        super().__init__(predictor_kwargs)
        self.model = self._create_model(model)

    def predict(self, input, **kwargs):
        yield from self.model(input, **kwargs)
