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

from typing import Any, Dict, List, Optional, Union
import os
import ast
from pathlib import Path
import numpy as np
import pickle
import tarfile


from ....utils import logging
from ....utils.download import download
from ....utils.cache import CACHE_DIR
from .base_batch_sampler import BaseBatchSampler


class Det3DBatchSampler(BaseBatchSampler):

    # XXX: auto download for url
    def _download_from_url(self, in_path: str) -> str:
        file_name = Path(in_path).name
        save_path = Path(CACHE_DIR) / "predict_input" / file_name
        download(in_path, save_path, overwrite=True)
        return save_path.as_posix()

    @property
    def batch_size(self) -> int:
        """Gets the batch size."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int) -> None:
        """Sets the batch size.

        Args:
            batch_size (int): The batch size to set.
        """
        if batch_size != 1:
            logging.warning(
                "inference for 3D models only support batch_size equal to 1"
            )
        self._batch_size = batch_size

    def load_annotations(self, ann_file: str, data_root_dir: str) -> List[Dict]:
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.
            data_root_dir: (str): Path of the data root directory.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = pickle.load(open(ann_file, "rb"))
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        # append root_dir to image and lidar filepaths
        for item in data_infos:
            # lidar data
            lidar_path = item["lidar_path"]
            new_lidar_path = os.path.join(data_root_dir, lidar_path)
            item["lidar_path"] = new_lidar_path
            # camera data
            cam_data = item["cams"]
            for cam_data_item_key in cam_data:
                cam_data_item = cam_data[cam_data_item_key]
                cam_data_item_path = cam_data_item["data_path"]
                new_cam_data_item_path = os.path.join(data_root_dir, cam_data_item_path)
                cam_data_item["data_path"] = new_cam_data_item_path
            # sweep data
            sweeps = item["sweeps"]
            for sweep_item in sweeps:
                sweep_item_path = sweep_item["data_path"]
                new_sweep_item_path = os.path.join(data_root_dir, sweep_item_path)
                sweep_item["data_path"] = new_sweep_item_path
        return data_infos

    def sample(self, inputs: Union[List[str], str]):
        if not isinstance(inputs, list):
            inputs = [inputs]

        sample_set = []
        for input in inputs:
            if isinstance(input, str):
                ann_path = (
                    self._download_from_url(input)
                    if input.startswith("http")
                    else input
                )
            else:
                logging.warning(
                    f"Not supported input data type! Only `str` is supported! So has been ignored: {input}."
                )
            # extract tar file
            tar_root_dir = os.path.dirname(ann_path)
            self.extract_tar(ann_path, tar_root_dir)
            data_root_dir, _ = os.path.splitext(ann_path)
            ann_pkl_path = os.path.join(data_root_dir, "nuscenes_infos_val.pkl")
            self.data_infos = self.load_annotations(ann_pkl_path, data_root_dir)
            sample_set.extend(self.data_infos)

        batch = []
        for sample in sample_set:
            batch.append(sample)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if len(batch) > 0:
            yield batch

    def _rand_batch(self, data_size: int) -> List[Any]:
        raise NotImplementedError(
            "rand batch is not supported for 3D detection annotation data"
        )

    def extract_tar(self, tar_path, extract_path="."):
        try:
            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(path=extract_path)
                print(f"file extract to {extract_path}")
        except Exception as e:
            print(f"error occurred while extracting tar file: {e}")
