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


from ....utils.misc import abspath
from ..pp3d_config import PP3DConfig


class BEVFusionConfig(PP3DConfig):
    def update_dataset(
        self, dataset_dir, datart_prefix=True, dataset_type=None, *, version=None
    ):
        dataset_dir = abspath(dataset_dir)
        if dataset_type is None:
            dataset_type = "NuscenesMMDataset"
        if dataset_type == "NuscenesMMDataset":
            ds_cfg = self._make_nuscenes_mm_dataset_config(
                dataset_dir, datart_prefix, version=version
            )
        else:
            raise ValueError(f"{dataset_type} is not supported.")
        # Prune old config
        keys_to_keep = ("transforms", "mode", "class_names", "modality")
        if "train_dataset" in self:
            for key in list(k for k in self.train_dataset if k not in keys_to_keep):
                self.train_dataset.pop(key)
        if "val_dataset" in self:
            for key in list(k for k in self.val_dataset if k not in keys_to_keep):
                self.val_dataset.pop(key)
        self.update(ds_cfg)

    def _make_nuscenes_mm_dataset_config(
        self, dataset_root_path, datart_prefix, version
    ):
        if version is None:
            # Default version
            version = "trainval"
        if version == "trainval":
            train_mode = "train"
            val_mode = "val"
        elif version == "mini":
            train_mode = "mini_train"
            val_mode = "mini_val"
        else:
            raise ValueError("Unsupported version.")
        return {
            "train_dataset": {
                "type": "NuscenesMMDataset",
                "data_root": dataset_root_path,
                "ann_file": f"{dataset_root_path}/nuscenes_infos_train.pkl",
                "mode": train_mode,
                "datart_prefix": datart_prefix,
            },
            "val_dataset": {
                "type": "NuscenesMMDataset",
                "data_root": dataset_root_path,
                "ann_file": f"{dataset_root_path}/nuscenes_infos_val.pkl",
                "mode": val_mode,
                "datart_prefix": datart_prefix,
            },
        }

    def _update_amp(self, amp):
        # XXX: Currently, we hard-code the AMP settings according to
        # https://github.com/PaddlePaddle/Paddle3D/blob/3cf884ecbc94330be0e2db780434bb60b9b4fe8c/configs/smoke/smoke_dla34_no_dcn_kitti_amp.yml#L6
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
            # TODO: Provide another method to customize `SampleNameFilter` classes names
            # TODO: Give an explicit warning for the implicit behavior
            tf_cfg_list = self.train_dataset["transforms"]
            for tf_cfg in tf_cfg_list:
                if tf_cfg["type"] == "SampleNameFilter":
                    tf_cfg["classes"] = class_names
                    # We assume that there is at most one `SampleNameFilter` in `tf_cfg_list`
                    break
        if "val_dataset" in self:
            self.val_dataset["class_names"] = class_names

    def update_pretrained_model(self, load_cam_from: str, load_lidar_from: str):
        """update model pretrained weight

        Args:
            load_cam_from (str): the path to cam weight file of model.
            load_lidar_from (str): the path to lidar weight file of model.
        """
        self.model["load_cam_from"] = load_cam_from
        self.model["load_lidar_from"] = load_lidar_from

    def update_weights(self, weight_path: str):
        """update model weight

        Args:
            weight_path (str): the path to weight file of model.
        """
        self["weights"] = weight_path
