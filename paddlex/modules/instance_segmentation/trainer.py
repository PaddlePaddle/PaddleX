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


class InstanceSegTrainer(DetTrainer):
    """Instance Segmentation Model Trainer"""

    entities = MODELS

    def _update_dataset(self):
        """update dataset settings"""
        self.pdx_config.update_dataset(
            self.global_config.dataset_dir,
            "COCOInstSegDataset",
            data_fields=["image", "gt_bbox", "gt_class", "gt_poly", "is_crowd"],
        )
