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


from ...utils import logging
from ..base import BaseTrainer
from .model_list import MODELS


class DetTrainer(BaseTrainer):
    """Object Detection Model Trainer"""

    entities = MODELS

    def _update_dataset(self):
        """update dataset settings"""
        metric = self.pdx_config.metric if "metric" in self.pdx_config else "COCO"
        data_fields = (
            self.pdx_config.TrainDataset["data_fields"]
            if "data_fields" in self.pdx_config.TrainDataset
            else None
        )

        self.pdx_config.update_dataset(
            self.global_config.dataset_dir,
            "COCODetDataset",
            data_fields=data_fields,
            metric=metric,
        )

    def update_config(self):
        """update training config"""
        if self.train_config.log_interval:
            self.pdx_config.update_log_interval(self.train_config.log_interval)
        if self.train_config.eval_interval:
            self.pdx_config.update_eval_interval(self.train_config.eval_interval)

        self._update_dataset()

        if self.train_config.num_classes is not None:
            self.pdx_config.update_num_class(self.train_config.num_classes)
        if (
            self.train_config.pretrain_weight_path
            and self.train_config.pretrain_weight_path != ""
        ):
            self.pdx_config.update_pretrained_weights(
                self.train_config.pretrain_weight_path
            )
        if self.train_config.batch_size is not None:
            self.pdx_config.update_batch_size(self.train_config.batch_size)
        if self.train_config.learning_rate is not None:
            self.pdx_config.update_learning_rate(self.train_config.learning_rate)
        if self.train_config.epochs_iters is not None:
            self.pdx_config.update_epochs(self.train_config.epochs_iters)
            epochs_iters = self.train_config.epochs_iters
        else:
            epochs_iters = self.pdx_config.get_epochs_iters()
        if self.train_config.warmup_steps is not None:
            self.pdx_config.update_warmup_steps(self.train_config.warmup_steps)
        if self.global_config.output is not None:
            self.pdx_config.update_save_dir(self.global_config.output)

        if "PicoDet" in self.global_config.model:
            assigner_epochs = max(int(epochs_iters / 10), 1)
            try:
                self.pdx_config.update_static_assigner_epochs(assigner_epochs)
            except Exception:
                logging.info(
                    f"The model({self.global_config.model}) don't support to update_static_assigner_epochs!"
                )

    def get_train_kwargs(self) -> dict:
        """get key-value arguments of model training function

        Returns:
            dict: the arguments of training function.
        """
        train_args = {"device": self.get_device()}
        if (
            self.train_config.resume_path is not None
            and self.train_config.resume_path != ""
        ):
            train_args["resume_path"] = self.train_config.resume_path
        train_args["dy2st"] = self.train_config.get("dy2st", False)
        # amp support 'O1', 'O2', 'OFF'
        train_args["amp"] = self.train_config.get("amp", "OFF")
        return train_args
