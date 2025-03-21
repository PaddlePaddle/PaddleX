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

import codecs

import yaml

from ...utils.misc import abspath
from ..base import BaseConfig


class PP3DConfig(BaseConfig):
    # Refer to https://github.com/PaddlePaddle/Paddle3D/blob/release/1.0/paddle3d/apis/config.py
    def update(self, dict_like_obj):
        def _merge_config_dicts(dict_from, dict_to):
            # According to
            # https://github.com/PaddlePaddle/Paddle3D/blob/3cf884ecbc94330be0e2db780434bb60b9b4fe8c/paddle3d/apis/config.py#L90
            for key, val in dict_from.items():
                if isinstance(val, dict) and key in dict_to:
                    dict_to[key] = _merge_config_dicts(val, dict_to[key])
                else:
                    dict_to[key] = val
            return dict_to

        dict_ = _merge_config_dicts(dict_like_obj, self.dict)
        self.reset_from_dict(dict_)

    def load(self, config_path):
        with codecs.open(config_path, "r", "utf-8") as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)
        dict_ = dic
        self.reset_from_dict(dict_)

    def dump(self, config_path):
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.dict, f)

    def update_learning_rate(self, learning_rate):
        if "lr_scheduler" not in self:
            raise RuntimeError(
                "Not able to update learning rate, because no LR scheduler config was found."
            )

        # Some lr_scheduler in Paddle3D has not learning_rate parameter.
        if self.lr_scheduler["type"] == "OneCycle":
            self.lr_scheduler["lr_max"] = learning_rate
        elif self.lr_scheduler["type"] == "OneCycleWarmupDecayLr":
            self.lr_scheduler["base_learning_rate"] = learning_rate
        else:
            self.lr_scheduler["learning_rate"] = learning_rate

    def update_batch_size(self, batch_size, mode="train"):
        if mode == "train":
            self.set_val("batch_size", batch_size)
        else:
            raise ValueError(
                f"Setting `batch_size` in {repr(mode)} mode is not supported."
            )

    def update_epochs(self, epochs, mode="train"):
        if mode == "train":
            self.set_val("epochs", epochs)
        else:
            raise ValueError(f"Setting `epochs` in {repr(mode)} mode is not supported.")

    def update_pretrained_weights(self, weight_path, is_backbone=False):
        raise NotImplementedError

    def get_epochs_iters(self):
        if "iters" in self:
            return self.iters
        else:
            assert "epochs" in self
            return self.epochs

    def get_learning_rate(self):
        if "lr_scheduler" not in self or "learning_rate" not in self.lr_scheduler:
            # Default lr
            return 0.0001
        else:
            lr = self.lr_scheduler["learning_rate"]
            while isinstance(lr, dict):
                lr = lr["learning_rate"]
            return lr

    def get_batch_size(self, mode="train"):
        if "batch_size" in self:
            return self.batch_size
        else:
            # Default batch size
            return 1

    def get_qat_epochs_iters(self):
        assert (
            "finetune_config" in self
        ), "QAT training yaml should contain finetune_config key"
        if "iters" in self.finetune_config:
            return self.finetune_config["iters"]
        else:
            assert "epochs" in self.finetune_config
            return self.finetune_config["epochs"]

    def get_qat_learning_rate(self):
        assert (
            "finetune_config" in self
        ), "QAT training yaml should contain finetune_config key"
        cfg = self.finetune_config
        if "lr_scheduler" in cfg or "learning_rate" not in cfg.lr_scheduler:
            # Default lr
            return 1.25e-4
        else:
            lr = cfg.lr_scheduler["learning_rate"]
            while isinstance(lr, dict):
                lr = lr["learning_rate"]
            return lr

    def update_warmup_steps(self, steps):
        self.lr_scheduler["warmup_steps"] = steps

    def update_end_lr(self, learning_rate):
        self.lr_scheduler["end_lr"] = learning_rate

    def update_iters(self, iters):
        self.set_val("iters", iters)
        if "epochs" in self:
            self.set_val("epochs", None)

    def update_finetune_iters(self, iters):
        self.finetune_config["iters"] = iters
        if "epochs" in self.finetune_config:
            self.finetune_config["epochs"] = None

    def update_save_dir(self, save_dir: str):
        self["save_dir"] = abspath(save_dir)
