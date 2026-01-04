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

from ..base import BaseEvaluator
from .model_list import MODELS


class VideoClsEvaluator(BaseEvaluator):
    """Image Classification Model Evaluator"""

    entities = MODELS

    def update_config(self):
        """update evaluation config"""
        if self.eval_config.log_interval:
            self.pdx_config.update_log_interval(self.eval_config.log_interval)
        self.pdx_config.update_dataset(
            self.global_config.dataset_dir, "VideoClsDataset"
        )
        if self.eval_config.batch_size is not None:
            self.pdx_config.update_batch_size(self.eval_config.batch_size, mode="eval")
        self.pdx_config.update_pretrained_weights(self.eval_config.weight_path)

    def get_eval_kwargs(self) -> dict:
        """get key-value arguments of model evaluation function

        Returns:
            dict: the arguments of evaluation function.
        """
        return {
            "weight_path": self.eval_config.weight_path,
            "device": self.get_device(using_device_number=1),
        }
