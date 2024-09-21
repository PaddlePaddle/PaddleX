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
import shutil
from pathlib import Path
import lazy_paddle as paddle

from ..base import BaseTrainer, BaseTrainDeamon
from ...utils.config import AttrDict
from .model_list import MODELS


class TextRecTrainer(BaseTrainer):
    """Text Recognition Model Trainer"""

    entities = MODELS

    def dump_label_dict(self, src_label_dict_path: str):
        """dump label dict config

        Args:
            src_label_dict_path (str): path to label dict file to be saved.
        """
        dst_label_dict_path = Path(self.global_config.output).joinpath("label_dict.txt")
        shutil.copyfile(src_label_dict_path, dst_label_dict_path)

    def build_deamon(self, config: AttrDict) -> "TextRecTrainDeamon":
        """build deamon thread for saving training outputs timely

        Args:
            config (AttrDict): PaddleX pipeline config, which is loaded from pipeline yaml file.

        Returns:
            TextRecTrainDeamon: the training deamon thread object for saving training outputs timely.
        """
        return TextRecTrainDeamon(config)

    def update_config(self):
        """update training config"""
        if self.train_config.log_interval:
            self.pdx_config.update_log_interval(self.train_config.log_interval)
        if self.train_config.eval_interval:
            self.pdx_config._update_eval_interval_by_epoch(
                self.train_config.eval_interval
            )
        if self.train_config.save_interval:
            self.pdx_config.update_save_interval(self.train_config.save_interval)

        if self.global_config["model"] == "LaTeX_OCR_rec":
            self.pdx_config.update_dataset(
                self.global_config.dataset_dir, "LaTeXOCRDataSet"
            )
        else:
            self.pdx_config.update_dataset(
                self.global_config.dataset_dir, "MSTextRecDataset"
            )

        label_dict_path = Path(self.global_config.dataset_dir).joinpath("dict.txt")
        if label_dict_path.exists():
            self.pdx_config.update_label_dict_path(label_dict_path)
            self.dump_label_dict(label_dict_path)

        if self.train_config.pretrain_weight_path:
            self.pdx_config.update_pretrained_weights(
                self.train_config.pretrain_weight_path
            )

        if self.global_config["model"] == "LaTeX_OCR_rec":
            if (
                self.train_config.batch_size_train is not None
                and self.train_config.batch_size_val
            ):
                self.pdx_config.update_batch_size_pair(
                    self.train_config.batch_size_train, self.train_config.batch_size_val
                )
        else:
            if self.train_config.batch_size is not None:
                self.pdx_config.update_batch_size(self.train_config.batch_size)

        if self.train_config.learning_rate is not None:
            self.pdx_config.update_learning_rate(self.train_config.learning_rate)
        if self.train_config.epochs_iters is not None:
            self.pdx_config._update_epochs(self.train_config.epochs_iters)
        if (
            self.train_config.resume_path is not None
            and self.train_config.resume_path != ""
        ):
            self.pdx_config._update_checkpoints(self.train_config.resume_path)
        if self.global_config.output is not None:
            self.pdx_config._update_output_dir(self.global_config.output)

    def get_train_kwargs(self) -> dict:
        """get key-value arguments of model training function

        Returns:
            dict: the arguments of training function.
        """
        return {
            "device": self.get_device(),
            "dy2st": self.train_config.get("dy2st", False),
        }


class TextRecTrainDeamon(BaseTrainDeamon):
    """TableRecTrainDeamon"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_the_pdparams_suffix(self):
        """get the suffix of pdparams file"""
        return "pdparams"

    def get_the_pdema_suffix(self):
        """get the suffix of pdema file"""
        return "pdema"

    def get_the_pdopt_suffix(self):
        """get the suffix of pdopt file"""
        return "pdopt"

    def get_the_pdstates_suffix(self):
        """get the suffix of pdstates file"""
        return "states"

    def get_ith_ckp_prefix(self, epoch_id):
        """get the prefix of the epoch_id checkpoint file"""
        return f"iter_epoch_{epoch_id}"

    def get_best_ckp_prefix(self):
        """get the prefix of the best checkpoint file"""
        return "best_accuracy"

    def get_score(self, pdstates_path):
        """get the score by pdstates file"""
        if not Path(pdstates_path).exists():
            return 0
        if self.global_config["model"] == "LaTeX_OCR_rec":
            return paddle.load(pdstates_path)["best_model_dict"]["exp_rate"]
        else:
            return paddle.load(pdstates_path)["best_model_dict"]["acc"]

    def get_epoch_id_by_pdparams_prefix(self, pdparams_prefix):
        """get the epoch_id by pdparams file"""
        return int(pdparams_prefix.split("_")[-1])
