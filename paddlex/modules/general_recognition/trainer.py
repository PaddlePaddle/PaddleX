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

from ..image_classification import ClsTrainer
from .model_list import MODELS


class ShiTuRecTrainer(ClsTrainer):
    """ShiTu Recognition Model Trainer"""

    entities = MODELS

    def update_config(self):
        """update training config"""
        if self.train_config.log_interval:
            self.pdx_config.update_log_interval(self.train_config.log_interval)
        if self.train_config.eval_interval:
            self.pdx_config.update_eval_interval(self.train_config.eval_interval)
        if self.train_config.save_interval:
            self.pdx_config.update_save_interval(self.train_config.save_interval)

        self.pdx_config.update_dataset(
            self.global_config.dataset_dir, "ShiTuRecDataset"
        )
        if self.train_config.num_classes is not None:
            self.pdx_config.update_num_classes(self.train_config.num_classes)
        if self.train_config.pretrain_weight_path != "":
            self.pdx_config.update_pretrained_weights(
                self.train_config.pretrain_weight_path
            )

        if self.train_config.batch_size is not None:
            self.pdx_config.update_batch_size(self.train_config.batch_size)
        if self.train_config.learning_rate is not None:
            self.pdx_config.update_learning_rate(self.train_config.learning_rate)
        if self.train_config.epochs_iters is not None:
            self.pdx_config._update_epochs(self.train_config.epochs_iters)
        if self.train_config.warmup_steps is not None:
            self.pdx_config.update_warmup_epochs(self.train_config.warmup_steps)
        if self.global_config.output is not None:
            self.pdx_config._update_output_dir(self.global_config.output)
