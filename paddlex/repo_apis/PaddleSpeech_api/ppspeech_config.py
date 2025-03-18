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

import codecs
import yaml
from ...utils.misc import abspath
from ..base import BaseConfig

class PPSpeechConfig(BaseConfig):
    """Speech recognition configuration handler"""

    def update(self, dict_like_obj):
        def _merge_config_dicts(dict_from, dict_to):
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
        self.reset_from_dict(dic)

    def dump(self, config_path):
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.dict, f)

    def update_learning_rate(self, learning_rate):
        if "lr_scheduler" not in self:
            raise RuntimeError("No LR scheduler config found")

        if self.lr_scheduler["type"] in ["TransformerLRScheduler", "WarmupDecay"]:
            self.lr_scheduler["learning_rate"] = learning_rate
        else:
            self.lr_scheduler["base_lr"] = learning_rate

    def update_batch_size(self, batch_size, mode="train"):
        if mode == "train":
            self.set_val("batch_size", batch_size)
        else:
            raise ValueError(f"Setting batch_size in {mode} mode not supported")

    def update_audio_params(self, sample_rate: int, audio_format: str):
        self.set_val("sample_rate", sample_rate)
        self.set_val("audio_format", audio_format)

    def update_vocab(self, vocab_path: str):
        self.set_val("vocab_path", abspath(vocab_path))

    def update_epochs(self, epochs, mode="train"):
        if mode == "train":
            self.set_val("epochs", epochs)
        else:
            raise ValueError(f"Setting epochs in {mode} mode not supported")

    def update_pretrained_weights(self, weight_path):
        self.set_val("pretrained_weights", abspath(weight_path))

    def get_learning_rate(self):
        if "lr_scheduler" not in self:
            return 0.00025  # Default ASR learning rate
        return self.lr_scheduler.get("learning_rate", 0.00025)

    def update_save_dir(self, save_dir: str):
        self["save_dir"] = abspath(save_dir)

    def update_dataset(self, dataset_dir: str, dataset_type: str, **kwargs):
        self.set_val("dataset_dir", abspath(dataset_dir))
        self.set_val("dataset_type", dataset_type)
        for key, value in kwargs.items():
            self.set_val(key, value)
