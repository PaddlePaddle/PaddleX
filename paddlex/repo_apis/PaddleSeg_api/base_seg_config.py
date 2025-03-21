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

from urllib.parse import urlparse

import yaml

from ...utils.misc import abspath
from ..base import BaseConfig


class BaseSegConfig(BaseConfig):
    """BaseSegConfig"""

    def update(self, dict_like_obj):
        """update"""
        from paddleseg.cvlibs.config import merge_config_dicts

        dict_ = merge_config_dicts(dict_like_obj, self.dict)
        self.reset_from_dict(dict_)

    def load(self, config_path):
        """load"""
        from paddleseg.cvlibs.config import parse_from_yaml

        dict_ = parse_from_yaml(config_path)
        if not isinstance(dict_, dict):
            raise TypeError
        self.reset_from_dict(dict_)

    def dump(self, config_path):
        """dump"""
        from paddleseg.utils import NoAliasDumper

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.dict, f, Dumper=NoAliasDumper)

    def update_learning_rate(self, learning_rate):
        """update_learning_rate"""
        if "lr_scheduler" not in self:
            raise RuntimeError(
                "Not able to update learning rate, because no LR scheduler config was found."
            )
        self.lr_scheduler["learning_rate"] = learning_rate

    def update_batch_size(self, batch_size, mode="train"):
        """update_batch_size"""
        if mode == "train":
            self.set_val("batch_size", batch_size)
        else:
            raise ValueError(
                f"Setting `batch_size` in {repr(mode)} mode is not supported."
            )

    def update_log_ranks(self, device):
        """update log ranks

        Args:
            device (str): the running device to set
        """
        log_ranks = device.split(":")[1]
        self.set_val("log_ranks", log_ranks)

    def update_print_mem_info(self, print_mem_info: bool):
        """setting print memory info"""
        assert isinstance(print_mem_info, bool), "print_mem_info should be a bool"
        self.set_val("print_mem_info", print_mem_info)

    def update_shuffle(self, shuffle: bool):
        """setting print memory info"""
        assert isinstance(shuffle, bool), "shuffle should be a bool"
        self.set_val("shuffle", shuffle)

    def update_pretrained_weights(self, weight_path, is_backbone=False):
        """update_pretrained_weights"""
        if "model" not in self:
            raise RuntimeError(
                "Not able to update pretrained weight path, because no model config was found."
            )
        if isinstance(weight_path, str):
            if urlparse(weight_path).scheme == "":
                # If `weight_path` is a string but not URL (with scheme present),
                # it will be recognized as a local file path.
                weight_path = abspath(weight_path)
        else:
            if weight_path is not None:
                raise TypeError("`weight_path` must be string or None.")
        if is_backbone:
            if "backbone" not in self.model:
                raise RuntimeError(
                    "Not able to update pretrained weight path of backbone, because no backbone config was found."
                )
            self.model["backbone"]["pretrained"] = weight_path
        else:
            self.model["pretrained"] = weight_path

    def update_dy2st(self, dy2st):
        """update_dy2st"""
        self.set_val("to_static_training", dy2st)

    def update_dataset(self, dataset_dir, dataset_type=None):
        """update_dataset"""
        raise NotImplementedError

    def get_epochs_iters(self):
        """get_epochs_iters"""
        raise NotImplementedError

    def get_learning_rate(self):
        """get_learning_rate"""
        raise NotImplementedError

    def get_batch_size(self, mode="train"):
        """get_batch_size"""
        raise NotImplementedError

    def get_qat_epochs_iters(self):
        """get_qat_epochs_iters"""
        return self.get_epochs_iters() // 2

    def get_qat_learning_rate(self):
        """get_qat_learning_rate"""
        return self.get_learning_rate() / 2
