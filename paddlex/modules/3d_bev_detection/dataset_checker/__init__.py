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

import pickle
from pathlib import Path

from ...base import BaseDatasetChecker
from ..model_list import MODELS
from .dataset_src import check, deep_analyse


class BEVFusionDatasetChecker(BaseDatasetChecker):
    entities = MODELS

    def check_dataset(self, dataset_dir: str) -> dict:
        """check if the dataset meets the specifications and get dataset summary

        Args:
            dataset_dir (str): the root directory of dataset.
            sample_num (int): the number to be sampled.
        Returns:
            dict: dataset summary.
        """
        return check(dataset_dir)

    def analyse(self, dataset_dir: str) -> dict:
        """deep analyse dataset

        Args:
            dataset_dir (str): the root directory of dataset.

        Returns:
            dict: the deep analysis results.
        """
        return deep_analyse(dataset_dir, self.output)

    def get_data(self, ann_file, max_sample_num):
        infos = self.data_infos(ann_file, max_sample_num)
        meta = []
        for info in infos:
            image_paths = []
            cam_orders = [
                "CAM_FRONT_LEFT",
                "CAM_FRONT",
                "CAM_FRONT_RIGHT",
                "CAM_BACK_RIGHT",
                "CAM_BACK",
                "CAM_BACK_LEFT",
            ]
            for cam_type in cam_orders:
                cam_info = info["cams"][cam_type]
                cam_data_path = cam_info["data_path"]
                image_paths.append(cam_data_path)

            meta.append(
                {
                    "sample_idx": info["token"],
                    "lidar_path": info["lidar_path"],
                    "image_paths": image_paths,
                }
            )
        return meta

    def data_infos(self, ann_file, max_sample_num):
        data = pickle.load(open(ann_file, "rb"))
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:max_sample_num]
        return data_infos

    def get_show_type(self) -> str:
        """get the show type of dataset

        Returns:
            str: show type
        """
        return "txt"

    def get_dataset_type(self) -> str:
        """return the dataset type

        Returns:
            str: dataset type
        """
        return "NuscenesMMDataset"
