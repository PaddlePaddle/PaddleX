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

from ..base import BaseTrainer
from .model_list import MODELS


class SegTrainer(BaseTrainer):
    """Semantic Segmentation Model Trainer"""

    entities = MODELS

    def update_config(self):
        """update training config"""
        self.pdx_config.update_dataset(self.global_config.dataset_dir, "SegDataset")
        if self.train_config.num_classes is not None:
            self.pdx_config.update_num_classes(self.train_config.num_classes)
        if (
            self.train_config.pretrain_weight_path
            and self.train_config.pretrain_weight_path != ""
        ):
            self.pdx_config.update_pretrained_weights(
                self.train_config.pretrain_weight_path, is_backbone=True
            )

    def get_train_kwargs(self) -> dict:
        """get key-value arguments of model training function

        Returns:
            dict: the arguments of training function.
        """
        train_args = {"device": self.get_device()}
        # XXX:
        os.environ.pop("FLAGS_npu_jit_compile", None)
        if self.train_config.batch_size is not None:
            train_args["batch_size"] = self.train_config.batch_size
        if self.train_config.learning_rate is not None:
            train_args["learning_rate"] = self.train_config.learning_rate
        if self.train_config.epochs_iters is not None:
            train_args["epochs_iters"] = self.train_config.epochs_iters
        if (
            self.train_config.resume_path is not None
            and self.train_config.resume_path != ""
        ):
            train_args["resume_path"] = self.train_config.resume_path
        if self.global_config.output is not None:
            train_args["save_dir"] = self.global_config.output
        if self.train_config.log_interval:
            train_args["log_iters"] = self.train_config.log_interval
        if self.train_config.eval_interval:
            train_args["do_eval"] = True
            train_args["save_interval"] = self.train_config.eval_interval
        train_args["dy2st"] = self.train_config.get("dy2st", False)
        if self.train_config.get("input_shape") is not None:
            train_args["input_shape"] = self.train_config.input_shape
        # amp support 'O1', 'O2', 'OFF'
        train_args["amp"] = self.train_config.get("amp", "OFF")
        return train_args
