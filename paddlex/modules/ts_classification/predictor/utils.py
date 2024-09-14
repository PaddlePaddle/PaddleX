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
import os

from ....utils import logging
from ...base.predictor.transforms import ts_common


class InnerConfig(object):
    """Inner Config"""

    def __init__(self, config_path, model_dir=None):
        self.inner_cfg = self.load(config_path)
        self.model_dir = model_dir

    def load(self, config_path):
        """load config"""
        with codecs.open(config_path, "r", "utf-8") as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)
        return dic

    @property
    def pre_transforms(self):
        """read preprocess transforms from  config file"""

        tfs = []
        if self.inner_cfg.get("info_params", False):

            if self.inner_cfg.get("scale", False):
                scaler_file_path = os.path.join(self.model_dir, "scaler.pkl")
                if not os.path.exists(scaler_file_path):
                    raise FileNotFoundError(
                        f"Cannot find scaler file: {scaler_file_path}"
                    )
                tf = ts_common.TSNormalize(
                    scaler_file_path, self.inner_cfg["info_params"]
                )
                tfs.append(tf)

            tf = ts_common.BuildTSDataset(self.inner_cfg["info_params"])
            tfs.append(tf)

            tf = ts_common.BuildPadMask(self.inner_cfg["input_data"])
            tfs.append(tf)

            tf = ts_common.TStoArray(self.inner_cfg["input_data"])
            tfs.append(tf)
        else:
            raise ValueError("info_params is not found in config file")

        return tfs
