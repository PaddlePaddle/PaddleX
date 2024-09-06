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

from ....utils import logging
from ...base.predictor.transforms import image_common
from .transforms import SaveDetResults, PadStride, DetResize, Pad


class InnerConfig(object):
    """Inner Config"""

    def __init__(self, config_path):
        self.inner_cfg = self.load(config_path)

    def load(self, config_path):
        """load infer config"""
        with codecs.open(config_path, "r", "utf-8") as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)
        return dic

    @property
    def pre_transforms(self):
        """read preprocess transforms from  config file"""
        tfs_cfg = self.inner_cfg["Preprocess"]
        tfs = []
        for cfg in tfs_cfg:
            if cfg["type"] == "NormalizeImage":
                mean = cfg.get("mean", 0.5)
                std = cfg.get("std", 0.5)
                scale = 1.0 / 255.0 if cfg.get("is_scale", True) else 1

                norm_type = cfg.get("norm_type", "mean_std")
                if norm_type != "mean_std":
                    mean = 0
                    std = 1

                tf = image_common.Normalize(mean=mean, std=std, scale=scale)
            elif cfg["type"] == "Resize":
                interp = cfg.get("interp", "LINEAR")
                if isinstance(interp, int):
                    interp = {
                        0: "NEAREST",
                        1: "LINEAR",
                        2: "CUBIC",
                        3: "AREA",
                        4: "LANCZOS4",
                    }[interp]
                tf = DetResize(
                    target_hw=cfg["target_size"],
                    keep_ratio=cfg.get("keep_ratio", True),
                    interp=interp,
                )
            elif cfg["type"] == "Permute":
                tf = image_common.ToCHWImage()
            elif cfg["type"] == "PadStride":
                stride = cfg.get("stride", 32)
                tf = PadStride(stride=stride)
            elif cfg["type"] == "Pad":
                fill_value = cfg.get("fill_value", [114.0, 114.0, 114.0])
                size = cfg.get("size", [640, 640])
                tf = Pad(size=size, fill_value=fill_value)
            else:
                raise RuntimeError(f"Unsupported type: {cfg['type']}")
            tfs.append(tf)
        return tfs

    @property
    def labels(self):
        """the labels in inner config"""
        return self.inner_cfg["label_list"]
