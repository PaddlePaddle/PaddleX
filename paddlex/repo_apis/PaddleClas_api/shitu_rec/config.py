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
from ..cls import ClsConfig


class ShiTuRecConfig(ClsConfig):
    """ShiTu Recognition Config"""

    def update_dataset(
        self,
        dataset_path: str,
        dataset_type: str = None,
        *,
        train_list_path: str = None,
    ):
        """update dataset settings

        Args:
            dataset_path (str): the root path of dataset.
            dataset_type (str, optional): dataset type. Defaults to None.
            train_list_path (str, optional): the path of train dataset annotation file . Defaults to None.

        Raises:
            ValueError: the dataset_type error.
        """
        dataset_path = abspath(dataset_path)

        dataset_type = "ShiTuRecDataset"
        if train_list_path:
            train_list_path = f"{train_list_path}"
        else:
            train_list_path = f"{dataset_path}/train.txt"

        ds_cfg = [
            f"DataLoader.Train.dataset.name={dataset_type}",
            f"DataLoader.Train.dataset.image_root={dataset_path}",
            f"DataLoader.Train.dataset.cls_label_path={train_list_path}",
            f"DataLoader.Eval.Query.dataset.name={dataset_type}",
            f"DataLoader.Eval.Query.dataset.image_root={dataset_path}",
            f"DataLoader.Eval.Query.dataset.cls_label_path={dataset_path}/query.txt",
            f"DataLoader.Eval.Gallery.dataset.name={dataset_type}",
            f"DataLoader.Eval.Gallery.dataset.image_root={dataset_path}",
            f"DataLoader.Eval.Gallery.dataset.cls_label_path={dataset_path}/gallery.txt",
        ]

        self.update(ds_cfg)

    def update_batch_size(self, batch_size: int, mode: str = "train"):
        """update batch size setting

        Args:
            batch_size (int): the batch size number to set.
            mode (str, optional): the mode that to be set batch size, must be one of 'train', 'eval'.
                Defaults to 'train'.

        Raises:
            ValueError: `mode` error.
        """
        if mode == "train":
            if self.DataLoader["Train"]["sampler"].get("batch_size", False):
                _cfg = [f"DataLoader.Train.sampler.batch_size={batch_size}"]
            else:
                _cfg = [f"DataLoader.Train.sampler.first_bs={batch_size}"]
                _cfg = [f"DataLoader.Train.dataset.name=MultiScaleDataset"]
        elif mode == "eval":
            _cfg = [f"DataLoader.Eval.Query.sampler.batch_size={batch_size}"]
            _cfg = [f"DataLoader.Eval.Gallery.sampler.batch_size={batch_size}"]
        else:
            raise ValueError("The input `mode` should be train or eval")
        self.update(_cfg)

    def update_num_classes(self, num_classes: int):
        """update classes number

        Args:
            num_classes (int): the classes number value to set.
        """
        update_str_list = [f"Arch.Head.class_num={num_classes}"]
        self.update(update_str_list)

    def update_num_workers(self, num_workers: int):
        """update workers number of train and eval dataloader

        Args:
            num_workers (int): the value of train and eval dataloader workers number to set.
        """
        _cfg = [
            f"DataLoader.Train.loader.num_workers={num_workers}",
            f"DataLoader.Eval.Query.loader.num_workers={num_workers}",
            f"DataLoader.Eval.Gallery.loader.num_workers={num_workers}",
        ]
        self.update(_cfg)

    def update_shared_memory(self, shared_memeory: bool):
        """update shared memory setting of train and eval dataloader

        Args:
            shared_memeory (bool): whether or not to use shared memory
        """
        assert isinstance(shared_memeory, bool), "shared_memeory should be a bool"
        _cfg = [
            f"DataLoader.Train.loader.use_shared_memory={shared_memeory}",
            f"DataLoader.Eval.Query.loader.use_shared_memory={shared_memeory}",
            f"DataLoader.Eval.Gallery.loader.use_shared_memory={shared_memeory}",
        ]
        self.update(_cfg)

    def update_shuffle(self, shuffle: bool):
        """update shuffle setting of train and eval dataloader

        Args:
            shuffle (bool): whether or not to shuffle the data
        """
        assert isinstance(shuffle, bool), "shuffle should be a bool"
        _cfg = [
            f"DataLoader.Train.loader.shuffle={shuffle}",
            f"DataLoader.Eval.Query.loader.shuffle={shuffle}",
            f"DataLoader.Eval.Gallery.loader.shuffle={shuffle}",
        ]
        self.update(_cfg)

    def _get_backbone_name(self) -> str:
        """get backbone name of rec model

        Returns:
            str: the model backbone name, i.e., `Arch.Backbone.name` in config.
        """
        return self.dict["Arch"]["Backbone"]["name"]
