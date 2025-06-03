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


from ..base import BaseTrainer
from .model_list import MODELS


class BEVFusionTrainer(BaseTrainer):
    """3D BEV Detection Model Trainer"""

    entities = MODELS

    def _update_dataset(self):
        """update dataset settings"""
        self.pdx_config.update_dataset(
            self.global_config.dataset_dir,
            self.global_config.get("datart_prefix", True),
            "NuscenesMMDataset",
            version=self.global_config.get("version", "mini"),
        )

    def _update_pretrained_model(self):
        self.pdx_config.update_pretrained_model(
            self.global_config.load_cam_from, self.global_config.load_lidar_from
        )

    def update_config(self):
        """update training config"""
        self._update_dataset()
        self._update_pretrained_model()

        if self.train_config.batch_size is not None:
            self.pdx_config.update_batch_size(self.train_config.batch_size)
        if self.train_config.learning_rate is not None:
            self.pdx_config.update_learning_rate(self.train_config.learning_rate)
        if self.train_config.epochs_iters is not None:
            self.pdx_config.update_epochs(self.train_config.epochs_iters)
            self.train_config.epochs_iters
        else:
            self.pdx_config.get_epochs_iters()
        if self.global_config.output is not None:
            self.pdx_config.update_save_dir(self.global_config.output)

    def get_train_kwargs(self) -> dict:
        """get key-value arguments of model training function

        Returns:
            dict: the arguments of training function.
        """
        train_args = {"device": self.get_device()}
        train_args["dy2st"] = self.train_config.get("dy2st", False)
        # amp support 'O1', 'O2', 'OFF'
        train_args["amp"] = self.train_config.get("amp", "OFF")
        if self.global_config.output is not None:
            train_args["save_dir"] = self.global_config.output
        return train_args
