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


import tarfile
from pathlib import Path

from ..base import BaseEvaluator
from .model_list import MODELS


class TSCLSEvaluator(BaseEvaluator):
    """TS Classification Model Evaluator"""

    entities = MODELS

    def update_config(self):
        """update evalution config"""
        self.pdx_config.update_dataset(self.global_config.dataset_dir, "TSCLSDataset")

    def get_eval_kwargs(self) -> dict:
        """get key-value arguments of model evalution function

        Returns:
            dict: the arguments of evaluation function.
        """
        return {
            "weight_path": self.eval_config.weight_path,
            "device": self.get_device(using_device_number=1),
        }

    def uncompress_tar_file(self):
        """unpackage the tar file containing training outputs and update weight path"""
        if tarfile.is_tarfile(self.eval_config.weight_path):
            dest_path = Path(self.eval_config.weight_path).parent
            with tarfile.open(self.eval_config.weight_path, "r") as tar:
                tar.extractall(path=dest_path)
            self.eval_config.weight_path = dest_path.joinpath(
                "best_accuracy.pdparams/best_model/model.pdparams"
            )

    def evaluate(self):
        """firstly, update evaluation config, then evaluate model, finally return the evaluation result"""
        self.uncompress_tar_file()
        return super().evaluate()
