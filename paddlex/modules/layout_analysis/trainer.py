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


from ..object_detection import DetTrainer
from .dataset_spec import get_dataset_spec
from .model_list import MODELS


class LayoutAnalysisTrainer(DetTrainer):
    """Layout Analysis Model Trainer"""

    entities = MODELS

    def update_config(self):
        super().update_config()
        # Training-time validation must also keep bbox.json out of the
        # external PaddleDetection working directory.
        self.pdx_config.update({"output_eval": self.pdx_config.save_dir})
        if (
            self.global_config.model == "PP-DocLayoutV4"
            and self.pdx_config.dict.get("deform_only_train", False)
            and not self.pdx_config.dict.get("pretrain_weights")
            and not self.train_config.get("resume_path")
        ):
            raise ValueError(
                "Geometry-only training requires a trained V4 detector. Set "
                "Train.pretrain_weight_path or Train.resume_path before training."
            )

    def _update_dataset(self):
        """update dataset settings"""
        metric = self.pdx_config.metric if "metric" in self.pdx_config else "COCO"
        dataset_type, data_fields = get_dataset_spec(self.global_config.model)
        self.pdx_config.update_dataset(
            self.global_config.dataset_dir,
            dataset_type,
            data_fields=data_fields,
            metric=metric,
        )
