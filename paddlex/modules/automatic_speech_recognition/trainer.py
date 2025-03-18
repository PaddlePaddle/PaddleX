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

from pathlib import Path
from ..base import BaseTrainer
from ...utils.config import AttrDict
from ...utils import logging
from .model_list import MODELS

class ChunkConformerTrainer(BaseTrainer):
    """Automatic Speech Recognition Model Trainer"""

    entities = MODELS

    def _update_dataset(self):
        """Update dataset settings for speech recognition"""
        self.pdx_config.update_dataset(
            self.global_config.dataset_dir,
            self.global_config.get("datart_prefix", True),
            "ASRDataset",
            sample_rate=self.global_config.get("sample_rate", 16000),
            audio_format=self.global_config.get("audio_format", "wav")
        )

    def _update_pretrained_model(self):
        self.pdx_config.update_pretrained_model(
            self.global_config.pretrained_model_path
        )

    def update_config(self):
        """Update training configuration"""
        self._update_dataset()
        self._update_pretrained_model()

        if self.train_config.batch_size is not None:
            self.pdx_config.update_batch_size(self.train_config.batch_size)
        if self.train_config.learning_rate is not None:
            self.pdx_config.update_learning_rate(self.train_config.learning_rate)
        if self.train_config.epochs_iters is not None:
            self.pdx_config.update_epochs(self.train_config.epochs_iters)
        if self.global_config.output is not None:
            self.pdx_config.update_save_dir(self.global_config.output)

    def get_train_kwargs(self) -> dict:
        """Get training arguments"""
        train_args = {
            "device": self.get_device(),
            "sample_rate": self.global_config.get("sample_rate", 16000)
        }
        if self.global_config.output is not None:
            train_args["save_dir"] = self.global_config.output
        return train_args
