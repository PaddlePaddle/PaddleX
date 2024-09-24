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
    ]

    def __init__(self, model, batch_size=1, device="gpu", predictor_kwargs=None):
        super().__init__(predictor_kwargs)
        self._predict = self._create_predictor(
            model, batch_size=batch_size, device=device
        )

    def predict(self, x):
        yield from self._predict(x)
