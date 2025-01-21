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

from typing import Any, Dict, List, Union

import ultra_infer as ui
import pandas as pd
from paddlex.inference.common.batch_sampler import TSBatchSampler
from paddlex.inference.models_new.ts_classification.result import TSClsResult
from paddlex.modules.ts_classification.model_list import MODELS

from paddlex_hpi.models.base import TSPredictor


class TSClsPredictor(TSPredictor):
    entities = MODELS

    def _build_batch_sampler(self) -> TSBatchSampler:
        return TSBatchSampler()

    def _get_result_class(self) -> type:
        return TSClsResult

    def _build_ui_model(
        self, option: ui.RuntimeOption
    ) -> ui.ts.classification.PyOnlyClassificationModel:
        model = ui.ts.classification.PyOnlyClassificationModel(
            str(self.model_path),
            str(self.params_path),
            str(self.config_path),
            runtime_option=option,
        )
        return model

    def process(self, batch_data: List[Union[str, pd.DataFrame]]) -> Dict[str, Any]:
        batch_raw_ts = self._data_reader(ts_list=batch_data)
        ui_results = self._ui_model.batch_predict(batch_raw_ts)

        classification_list = []
        for ui_result in ui_results:
            classification = pd.DataFrame.from_dict(
                {"classid": [ui_result.class_id], "score": [ui_result.score]}
            )
            classification.index.name = "sample"
            classification_list.append(classification)

        return {
            "input_path": batch_data,
            "input_ts": batch_raw_ts,
            "classification": classification_list,
        }
