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

from ...utils.misc import abspath
from ..base import BaseEvaluator
from .model_list import MODELS


class JDEEvaluator(BaseEvaluator):
    """Object Detection Model Evaluator"""

    entities = MODELS

    def _update_dataset(self):
        """update dataset settings"""
        data_path = abspath(self.global_config.dataset_dir)
        metric = "MOT"
        ds_cfg = {
            "TrainDataset": {
                "dataset_dir": data_path,
                "image_lists": "mot.train",
            },
            "EvalMOTDataset": {
                "dataset_dir": data_path,
                "data_root": "val/images",
            },
        }
        self.pdx_config.update(ds_cfg)
        self.pdx_config.set_val("metric", metric)

    def update_config(self):
        """update evalution config"""
        if self.eval_config.log_interval:
            self.pdx_config.update_log_interval(self.eval_config.log_interval)
        self._update_dataset()
        self.pdx_config.update_weights(self.eval_config.weight_path)

    def get_eval_kwargs(self) -> dict:
        """get key-value arguments of model evalution function

        Returns:
            dict: the arguments of evaluation function.
        """
        return {
            "weight_path": self.eval_config.weight_path,
            "device": self.get_device(using_device_number=1),
        }
