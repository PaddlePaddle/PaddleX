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

from typing import Any, Dict, List

import ultra_infer as ui
import numpy as np
from paddlex.inference.common.batch_sampler import ImageBatchSampler
from paddlex.inference.models_new.anomaly_detection.result import UadResult
from paddlex.modules.anomaly_detection.model_list import MODELS

from paddlex_hpi.models.base import CVPredictor


class UadPredictor(CVPredictor):
    entities = MODELS

    def _build_ui_model(
        self, option: ui.RuntimeOption
    ) -> ui.vision.segmentation.PyOnlyAnomalyDetectionModel:
        model = ui.vision.segmentation.PyOnlyAnomalyDetectionModel(
            str(self.model_path),
            str(self.params_path),
            str(self.config_path),
            runtime_option=option,
        )
        return model

    def _build_batch_sampler(self) -> ImageBatchSampler:
        return ImageBatchSampler()

    def _get_result_class(self) -> type:
        return UadResult

    def process(self, batch_data: List[Any]) -> Dict[str, List[Any]]:
        batch_raw_imgs = self._data_reader(imgs=batch_data)
        imgs = [np.ascontiguousarray(img) for img in batch_raw_imgs]
        ui_results = self._ui_model.batch_predict(imgs)

        preds_list = []
        for ui_result in ui_results:
            pred = np.array(ui_result.label_map, dtype=np.int32).reshape(
                ui_result.shape
            )
            pred = pred[np.newaxis]
            preds_list.append(pred)

        return {
            "input_path": batch_data,
            "input_img": batch_raw_imgs,
            "pred": preds_list,
        }
