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


class _SingleModelPipeline(BasePipeline):

    def __init__(self, model, batch_size=1, predictor_kwargs=None):
        super().__init__(predictor_kwargs)
        self._build_predictor(model)
        self.set_predictor(batch_size)

    def _build_predictor(self, model):
        self.model = self._create_model(model)

    def set_predictor(self, batch_size):
        self.model.set_predictor(batch_size=batch_size)

    def predict(self, input, **kwargs):
        yield from self.model(input, **kwargs)


class ImageClassification(_SingleModelPipeline):
    entities = "image_classification"


class ObjectDetection(_SingleModelPipeline):
    entities = "object_detection"


class InstanceSegmentation(_SingleModelPipeline):
    entities = "instance_segmentation"


class SemanticSegmentation(_SingleModelPipeline):
    entities = "semantic_segmentation"


class TSFc(_SingleModelPipeline):
    entities = "ts_fc"


class TSAd(_SingleModelPipeline):
    entities = "ts_ad"


class TSCls(_SingleModelPipeline):
    entities = "ts_cls"


class MultiLableImageClas(_SingleModelPipeline):
    entities = "multi_label_image_classification"


class SmallObjDet(_SingleModelPipeline):
    entities = "small_object_detection"


class AnomalyDetection(_SingleModelPipeline):
    entities = "anomaly_detection"
