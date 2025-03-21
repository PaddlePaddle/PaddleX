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
from .model_list import MODELS


class KeypointTrainer(DetTrainer):
    """Human Pose Estimation Model Trainer"""

    entities = MODELS

    def _update_dataset(self):
        """update dataset settings"""
        metric = self.pdx_config.metric
        data_fields = (
            self.pdx_config.TrainDataset["data_fields"]
            if "data_fields" in self.pdx_config.TrainDataset
            else None
        )

        self.pdx_config.update_dataset(
            self.global_config.dataset_dir,
            "KeypointTopDownCocoDataset",
            data_fields=data_fields,
            metric=metric,
        )
