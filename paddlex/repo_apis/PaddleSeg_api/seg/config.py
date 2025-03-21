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


import os
from typing import Union

from ....utils import logging
from ....utils.misc import abspath
from ..base_seg_config import BaseSegConfig


class SegConfig(BaseSegConfig):
    """Semantic Segmentation Config"""

    def update_dataset(self, dataset_path: str, dataset_type: str = None):
        """update dataset settings

        Args:
            dataset_path (str): the root path of dataset.
            dataset_type (str, optional): dataset type. Defaults to None.

        Raises:
            ValueError: the dataset_type error.
        """
        dataset_dir = abspath(dataset_path)
        if dataset_type is None:
            dataset_type = "SegDataset"
        if dataset_type == "SegDataset":
            # TODO: Prune extra keys
            ds_cfg = self._make_custom_dataset_config(dataset_dir)
            self.update(ds_cfg)
        elif dataset_type == "_dummy":
            # XXX: A special dataset type to tease PaddleSeg val dataset checkers
            self.update(
                {
                    "val_dataset": {
                        "type": "SegDataset",
                        "dataset_root": dataset_dir,
                        "val_path": os.path.join(dataset_dir, "val.txt"),
                        "mode": "val",
                    },
                }
            )
        else:
            raise ValueError(f"{repr(dataset_type)} is not supported.")

    def update_num_classes(self, num_classes: int):
        """update classes number

        Args:
            num_classes (int): the classes number value to set.
        """
        if "train_dataset" in self:
            self.train_dataset["num_classes"] = num_classes
        if "val_dataset" in self:
            self.val_dataset["num_classes"] = num_classes
        if "model" in self:
            self.model["num_classes"] = num_classes

        if self.model_name in ["MaskFormer_tiny", "MaskFormer_small"]:
            for tf_cfg in self.train_dataset["transforms"]:
                if tf_cfg["type"] == "GenerateInstanceTargets":
                    tf_cfg["num_classes"] = num_classes

            losses = self.loss["types"]
            for loss_cfg in losses:
                loss_cfg["num_classes"] = num_classes

    def update_train_crop_size(self, crop_size: Union[int, list]):
        """update the image cropping size of training preprocessing

        Args:
            crop_size (int | list): the size of image to be cropped.

        Raises:
            ValueError: the `crop_size` error.
        """
        # XXX: This method is highly coupled to PaddleSeg's internal functions
        if isinstance(crop_size, int):
            crop_size = [crop_size, crop_size]
        else:
            crop_size = list(crop_size)
            if len(crop_size) != 2:
                raise ValueError
            crop_size = [int(crop_size[0]), int(crop_size[1])]

        tf_cfg_list = self.train_dataset["transforms"]
        modified = False
        for tf_cfg in tf_cfg_list:
            if tf_cfg["type"] == "RandomPaddingCrop":
                tf_cfg["crop_size"] = crop_size
                modified = True
        if not modified:
            logging.warning(
                "Could not find configuration item of image cropping transformation operator. "
                "Therefore, the crop size was not updated."
            )

    def get_epochs_iters(self) -> int:
        """get epochs

        Returns:
            int: the epochs value, i.e., `Global.epochs` in config.
        """
        if "iters" in self:
            return self.iters
        else:
            # Default iters
            return 1000

    def get_learning_rate(self) -> float:
        """get learning rate

        Returns:
            float: the learning rate value, i.e., `Optimizer.lr.learning_rate` in config.
        """
        if "lr_scheduler" not in self or "learning_rate" not in self.lr_scheduler:
            # Default lr
            return 0.0001
        else:
            return self.lr_scheduler["learning_rate"]

    def get_batch_size(self, mode="train") -> int:
        """get batch size

        Args:
            mode (str, optional): the mode that to be get batch size value, must be one of 'train', 'eval', 'test'.
                Defaults to 'train'.

        Raises:
            ValueError: the `mode` error. `train` is supported only.

        Returns:
            int: the batch size value of `mode`, i.e., `DataLoader.{mode}.sampler.batch_size` in config.
        """
        if mode == "train":
            if "batch_size" in self:
                return self.batch_size
            else:
                # Default batch size
                return 4
        else:
            raise ValueError(
                f"Getting `batch_size` in {repr(mode)} mode is not supported."
            )

    def _make_custom_dataset_config(self, dataset_root_path: str) -> dict:
        """construct the dataset config that meets the format requirements

        Args:
            dataset_root_path (str): the root directory of dataset.

        Returns:
            dict: the dataset config.
        """
        ds_cfg = {
            "train_dataset": {
                "type": "SegDataset",
                "dataset_root": dataset_root_path,
                "train_path": os.path.join(dataset_root_path, "train.txt"),
                "mode": "train",
            },
            "val_dataset": {
                "type": "SegDataset",
                "dataset_root": dataset_root_path,
                "val_path": os.path.join(dataset_root_path, "val.txt"),
                "mode": "val",
            },
        }

        return ds_cfg
