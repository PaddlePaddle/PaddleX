# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from ..object_detection import DetEvaluator
from .dataset_spec import get_dataset_spec
from .model_list import MODELS


class LayoutAnalysisEvaluator(DetEvaluator):
    """Layout Analysis Model Evaluator"""

    entities = MODELS

    def update_config(self):
        """update evaluation config"""
        if self.eval_config.log_interval:
            self.pdx_config.update_log_interval(self.eval_config.log_interval)
        # Keep the metric declared in the model config. `DetConfig.update_dataset`
        # defaults `metric` to "COCO" and writes it unconditionally, so omitting
        # it here would silently downgrade e.g. DocLayoutV3Metric to plain COCO
        # and drop reading-order evaluation.
        metric = self.pdx_config.metric if "metric" in self.pdx_config else "COCO"
        dataset_type, data_fields = get_dataset_spec(self.global_config.model)
        self.pdx_config.update_dataset(
            self.global_config.dataset_dir,
            dataset_type,
            data_fields=data_fields,
            metric=metric,
        )
        self.pdx_config.update_weights(self.eval_config.weight_path)
        # Workers run from the external repository; relative metric outputs
        # would otherwise write bbox.json into that checkout.
        output_dir = os.path.abspath(self.global_config.output)
        os.makedirs(output_dir, exist_ok=True)
        self.pdx_config.update_save_dir(output_dir)
        self.pdx_config.update({"output_eval": output_dir})
