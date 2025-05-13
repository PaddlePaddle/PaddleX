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
from pathlib import Path

from ..base import BaseEvaluator
from .model_list import MODELS


class UadEvaluator(BaseEvaluator):
    """Semantic Segmentation Model Evaluator"""

    entities = MODELS

    def update_config(self):
        """update evaluation config"""
        self.pdx_config.update_dataset(self.global_config.dataset_dir, "SegDataset")
        self.pdx_config.update_pretrained_weights(None, is_backbone=True)

    def get_config_path(self, weight_path):
        """
        get config path

        Args:
            weight_path (str): The path to the weight

        Returns:
            config_path (str): The path to the config

        """

        config_path = Path(weight_path).parent.parent / "config.yaml"

        return config_path

    def get_eval_kwargs(self) -> dict:
        """get key-value arguments of model evaluation function

        Returns:
            dict: the arguments of evaluation function.
        """
        device = self.get_device()
        # XXX:
        os.environ.pop("FLAGS_npu_jit_compile", None)
        return {"weight_path": self.eval_config.weight_path, "device": device}
