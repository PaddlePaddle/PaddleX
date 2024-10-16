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

import os
import json
import time
import tarfile
from pathlib import Path
import lazy_paddle as paddle

from ..base import BaseTrainer
from ...utils.config import AttrDict
from .model_list import MODELS


class TSCLSTrainer(BaseTrainer):
    """TS Classification Model Trainer"""

    entities = MODELS

    def train(self):
        """firstly, update and dump train config, then train model"""
        # XXX: using super().train() instead when the train_hook() is supported.
        os.makedirs(self.global_config.output, exist_ok=True)
        self.update_config()
        self.dump_config()
        train_result = self.pdx_model.train(**self.get_train_kwargs())
        assert (
            train_result.returncode == 0
        ), f"Encountered an unexpected error({train_result.returncode}) in \
training!"

        self.make_tar_file()

    def make_tar_file(self):
        """make tar file to package the training outputs"""
        tar_path = Path(self.global_config.output) / "best_accuracy.pdparams.tar"
        with tarfile.open(tar_path, "w") as tar:
            tar.add(self.global_config.output, arcname="best_accuracy.pdparams")

    def update_config(self):
        """update training config"""
        self.pdx_config.update_dataset(self.global_config.dataset_dir, "TSCLSDataset")
        if self.train_config.time_col is not None:
            self.pdx_config.update_basic_info({"time_col": self.train_config.time_col})
        if self.train_config.target_cols is not None:
            self.pdx_config.update_basic_info(
                {"target_cols": self.train_config.target_cols.split(",")}
            )
        if self.train_config.group_id is not None:
            self.pdx_config.update_basic_info({"group_id": self.train_config.group_id})
        if self.train_config.static_cov_cols is not None:
            self.pdx_config.update_basic_info(
                {"static_cov_cols": self.train_config.static_cov_cols}
            )
        if self.train_config.freq is not None:
            try:
                self.train_config.freq = int(self.train_config.freq)
            except ValueError:
                pass
            self.pdx_config.update_basic_info({"freq": self.train_config.freq})
        if self.train_config.batch_size is not None:
            self.pdx_config.update_batch_size(self.train_config.batch_size)
        if self.train_config.learning_rate is not None:
            self.pdx_config.update_learning_rate(self.train_config.learning_rate)
        if self.train_config.epochs_iters is not None:
            self.pdx_config.update_epochs(self.train_config.epochs_iters)
        if self.train_config.log_interval is not None:
            self.pdx_config.update_log_interval(self.train_config.log_interval)
        if self.global_config.output is not None:
            self.pdx_config.update_save_dir(self.global_config.output)

    def get_train_kwargs(self) -> dict:
        """get key-value arguments of model training function

        Returns:
            dict: the arguments of training function.
        """
        train_args = {"device": self.get_device()}
        if self.global_config.output is not None:
            train_args["save_dir"] = self.global_config.output
        return train_args
