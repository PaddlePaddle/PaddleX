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


from pathlib import Path

from ..base import BaseEvaluator
from .model_list import MODELS


class TextRecEvaluator(BaseEvaluator):
    """Text Recognition Model Evaluator"""

    entities = MODELS

    def update_config(self):
        """update evaluation config"""
        if self.eval_config.log_interval:
            self.pdx_config.update_log_interval(self.eval_config.log_interval)
        if self.global_config["model"] == "LaTeX_OCR_rec":
            self.pdx_config.update_dataset(
                self.global_config.dataset_dir, "LaTeXOCRDataSet"
            )
        elif "PP-OCRv3" in self.global_config["model"]:
            self.pdx_config.update_dataset(
                self.global_config.dataset_dir, "SimpleDataSet"
            )
        else:
            self.pdx_config.update_dataset(
                self.global_config.dataset_dir, "MSTextRecDataset"
            )
        label_dict_path = None
        if self.eval_config.get("label_dict_path"):
            label_dict_path = self.eval_config.label_dict_path
        else:
            label_dict_path = (
                Path(self.eval_config.weight_path).parent / "label_dict.txt"
            )
            if not label_dict_path.exists():
                label_dict_path = None
        if label_dict_path is not None:
            self.pdx_config.update_label_dict_path(label_dict_path)

    def get_eval_kwargs(self) -> dict:
        """get key-value arguments of model evaluation function

        Returns:
            dict: the arguments of evaluation function.
        """
        return {
            "weight_path": self.eval_config.weight_path,
            "device": self.get_device(),
        }
