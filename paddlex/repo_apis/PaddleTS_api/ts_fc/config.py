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

from ....utils.misc import abspath
from ..ts_base.config import BaseTSConfig


class LongForecastConfig(BaseTSConfig):
    """Long Forecast Config"""

    def update_input_len(self, seq_len: int):
        """
        update the input sequence length

        Args:
            seq_len (int): input length

        Raises:
            TypeError: if seq_len is not dict, raising TypeError
        """
        if "seq_len" not in self:
            raise RuntimeError(
                "Not able to update seq_len, because no seq_len config was found."
            )
        self.set_val("seq_len", seq_len)

    def update_predict_len(self, predict_len: int):
        """
        updaet the predict sequence length

        Args:
            predict_len (int): predict length

        Raises:
            RuntimeError: if predict_len is not set, raising RuntimeError
        """
        if "predict_len" not in self:
            raise RuntimeError(
                "Not able to update predict_len, because no predict_len config was found."
            )
        self.set_val("predict_len", predict_len)

    def update_sampling_stride(self, sampling_stride: int):
        """
        updaet the sampling stride of sequence to reduce the training time

        Args:
            sampling_stride (int): sampling rate

        Raises:
            RuntimeError: if sampling stride is not set, raising RuntimeError
        """
        if "sampling_stride" not in self:
            raise RuntimeError(
                "Not able to update sampling_stride, because no sampling_stride config was found."
            )
        self.set_val("sampling_stride", sampling_stride)

    def update_dataset(self, dataset_dir: str, dataset_type: str = None):
        """
        update the dataset

        Args:
            dataset_dir (str): dataset root path
            dataset_type (str, optional): type to set for dataset. Default='TSDataset'
        """
        if dataset_type is None:
            dataset_type = "TSDataset"
        dataset_dir = abspath(dataset_dir)
        ds_cfg = self._make_custom_dataset_config(dataset_dir)
        self.update(ds_cfg)

    def update_basic_info(self, info_params: dict):
        """
        update basic info including time_col, freq, target_cols.

        Args:
            info_params (dict): update basic info

        Raises:
            TypeError: if info_params is not dict, raising TypeError
        """
        if isinstance(info_params, dict):
            self.update({"info_params": info_params})
        else:
            raise TypeError("`info_params` must be dict.")

    def update_patience(self, patience: int):
        """
        update patience.

        Args:
            patience (int): update patience

        Raises:
            RuntimeError: if patience is not found, raising RuntimeError
        """
        if "patience" not in self.model["model_cfg"]:
            raise RuntimeError(
                "Not able to update patience, because no patience config was found."
            )
        self.model["model_cfg"]["patience"] = patience

    def _make_custom_dataset_config(self, dataset_root_path: str):
        """construct the dataset config that meets the format requirements

        Args:
            dataset_root_path (str): the root directory of dataset.

        Returns:
            dict: the dataset config.
        """
        ds_cfg = {
            "dataset": {
                "name": "TSDataset",
                "dataset_root": dataset_root_path,
                "train_path": os.path.join(dataset_root_path, "train.csv"),
                "val_path": os.path.join(dataset_root_path, "val.csv"),
            },
        }

        return ds_cfg
