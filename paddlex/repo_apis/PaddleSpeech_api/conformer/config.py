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


from ....utils.misc import abspath
from ..ppspeech_config import PPSpeechConfig


class ChunkConformerConfig(PPSpeechConfig):
    """Configuration class for Chunk Conformer speech recognition models"""

    def update_dataset(
        self, dataset_dir, datart_prefix=True, dataset_type=None, *, version=None
    ):
        """Update dataset configuration for speech recognition"""
        dataset_dir = abspath(dataset_dir)
        if dataset_type is None:
            dataset_type = "SpeechDataset"

        ds_cfg = {
            "train_dataset": {
                "type": dataset_type,
                "data_dir": dataset_dir,
                "datart_prefix": datart_prefix,
            },
            "val_dataset": {
                "type": dataset_type,
                "data_dir": dataset_dir,
                "datart_prefix": datart_prefix,
            },
        }

        # Prune old config
        keys_to_keep = ("transforms", "mode", "class_names", "modality")
        for key in list(k for k in self.train_dataset if k not in keys_to_keep):
            self.train_dataset.pop(key)
        for key in list(k for k in self.val_dataset if k not in keys_to_keep):
            self.val_dataset.pop(key)

        self.update(ds_cfg)

    def update_audio_config(self, sample_rate=16000, frame_length=25, frame_shift=10):
        """Update audio feature extraction settings"""
        self.update(
            {
                "sample_rate": sample_rate,
                "frame_length": frame_length,
                "frame_shift": frame_shift,
            }
        )

    def _update_amp(self, amp):
        amp_cfg = {
            "amp_cfg": {
                "use_amp": False,
                "enable": False,
                "level": amp,
                "scaler": {"init_loss_scaling": 512.0},
                "custom_black_list": ["matmul_v2", "elementwise_mul"],
            }
        }
        self.update(amp_cfg)

    def update_class_names(self, class_names):
        if "train_dataset" in self and "transforms" in getattr(self, "train_dataset"):
            self.train_dataset["class_names"] = class_names
            tf_cfg_list = self.train_dataset["transforms"]
            for tf_cfg in tf_cfg_list:
                if tf_cfg["type"] == "SampleNameFilter":
                    tf_cfg["classes"] = class_names
                    break
        if "val_dataset" in self:
            self.val_dataset["class_names"] = class_names

    def update_pretrained_model(self, load_from: str):
        """Update model pretrained weight path"""
        self.model["load_from"] = load_from

    def update_weights(self, weight_path: str):
        """Update model weight path"""
        self["weights"] = weight_path

    def update_model_params(self, num_blocks=12, d_model=512, num_heads=8):
        """Update core model architecture parameters"""
        self.model.update(
            {"num_blocks": num_blocks, "d_model": d_model, "num_heads": num_heads}
        )
