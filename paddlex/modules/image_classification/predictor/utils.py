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

from ...base.predictor.transforms import image_common
from . import transforms as T


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
        """read preprocess transforms from config file"""
        if "RecPreProcess" in list(self.inner_cfg.keys()):
            tfs_cfg = self.inner_cfg["RecPreProcess"]["transform_ops"]
        else:
            tfs_cfg = self.inner_cfg["PreProcess"]["transform_ops"]
        tfs = []
        for cfg in tfs_cfg:
            tf_key = list(cfg.keys())[0]
            if tf_key == "NormalizeImage":
                tf = image_common.Normalize(
                    mean=cfg["NormalizeImage"].get("mean", [0.485, 0.456, 0.406]),
                    std=cfg["NormalizeImage"].get("std", [0.229, 0.224, 0.225]),
                )
            elif tf_key == "ResizeImage":
                if "resize_short" in list(cfg[tf_key].keys()):
                    tf = image_common.ResizeByShort(
                        target_short_edge=cfg["ResizeImage"].get("resize_short", 224),
                        size_divisor=None,
                        interp="LINEAR",
                    )
                else:
                    tf = image_common.Resize(
                        target_size=cfg["ResizeImage"].get("size", (224, 224))
                    )
            elif tf_key == "CropImage":
                tf = image_common.Crop(crop_size=cfg["CropImage"].get("size", 224))
            elif tf_key == "ToCHWImage":
                tf = image_common.ToCHWImage()
            else:
                raise RuntimeError(f"Unsupported type: {tf_key}")
            tfs.append(tf)
        return tfs

    @property
    def post_transforms(self):
        """read postprocess transforms from config file"""
        IGNORE_OPS = ["main_indicator", "SavePreLabel"]
        tfs_cfg = self.inner_cfg["PostProcess"]
        tfs = []
        for tf_key in tfs_cfg:
            if tf_key == "Topk":
                tf = T.Topk(
                    topk=tfs_cfg["Topk"]["topk"],
                    class_ids=tfs_cfg["Topk"].get("label_list", None),
                )
            elif tf_key == "MultiLabelThreshOutput":
                tf = T.MultiLabelThreshOutput(
                    threshold=0.5,
                    class_ids=tfs_cfg["MultiLabelThreshOutput"].get("label_list", None),
                )
            elif tf_key in IGNORE_OPS:
                continue
            else:
                raise RuntimeError(f"Unsupported type: {tf_key}")
            tfs.append(tf)
        return tfs

    @property
    def labels(self):
        """the labels in inner config"""
        postprocess_name = self.inner_cfg["PostProcess"].keys()
        if "Topk" in postprocess_name:
            return self.inner_cfg["PostProcess"]["Topk"].get("label_list", None)
        elif "MultiLabelThreshOutput" in postprocess_name:
            return self.inner_cfg["PostProcess"]["MultiLabelThreshOutput"].get(
                "label_list", None
            )
        else:
            return None
