# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from ...common.transformers.transformers import PretrainedConfig

DEFAULT_CONFIG = {
    "model_name": "PP-OCRv5_mobile_det",
    "algorithm": "DB",
    "backbone": {
        "name": "PPLCNetV3",
        "scale": 1.0,
        "det": True,
        "conv_kxk_num": 4,
        "reduction": 4,
        "lr_mult_list": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "lab_lr": 0.1,
        "out_channels": 512,
        "act": "hswish",
        "net_config": {
            "blocks2": [[3, 16, 24, 1, False]],
            "blocks3": [[3, 24, 48, 2, False], [3, 48, 48, 1, False]],
            "blocks4": [[3, 48, 96, 2, False], [3, 96, 96, 1, False]],
            "blocks5": [
                [3, 96, 192, 2, False],
                [5, 192, 192, 1, False],
                [5, 192, 192, 1, False],
                [5, 192, 192, 1, False],
                [5, 192, 192, 1, False],
            ],
            "blocks6": [
                [5, 192, 384, 2, True],
                [5, 384, 384, 1, True],
                [5, 384, 384, 1, False],
                [5, 384, 384, 1, False],
            ],
            "layer_list_out_channels": [12, 18, 42, 360],
        },
    },
    "neck": {"name": "RSEFPN", "out_channels": 96, "shortcut": True},
    "head": {"name": "DBHead", "k": 50, "kernel_list": [3, 2, 2], "fix_nan": False},
}


class PPOCRV5MobileDetConfig(PretrainedConfig):
    model_type = "det"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model_name = kwargs.get("model_name", DEFAULT_CONFIG["model_name"])
        self.algorithm = kwargs.get("algorithm", DEFAULT_CONFIG["algorithm"])

        backbone_cfg = kwargs.get("backbone", DEFAULT_CONFIG["backbone"])
        self.backbone_name = backbone_cfg.get(
            "name", DEFAULT_CONFIG["backbone"]["name"]
        )
        self.backbone_scale = backbone_cfg.get(
            "scale", DEFAULT_CONFIG["backbone"]["scale"]
        )
        self.backbone_det = backbone_cfg.get("det", DEFAULT_CONFIG["backbone"]["det"])
        self.backbone_conv_kxk_num = backbone_cfg.get(
            "conv_kxk_num", DEFAULT_CONFIG["backbone"]["conv_kxk_num"]
        )
        self.backbone_reduction = backbone_cfg.get(
            "reduction", DEFAULT_CONFIG["backbone"]["reduction"]
        )
        self.backbone_lr_mult_list = backbone_cfg.get(
            "lr_mult_list", DEFAULT_CONFIG["backbone"]["lr_mult_list"]
        )
        self.backbone_lab_lr = backbone_cfg.get(
            "lab_lr", DEFAULT_CONFIG["backbone"]["lab_lr"]
        )
        self.backbone_net_config = backbone_cfg.get(
            "net_config", DEFAULT_CONFIG["backbone"]["net_config"]
        )
        self.backbone_out_channels = backbone_cfg.get(
            "out_channels", DEFAULT_CONFIG["backbone"]["out_channels"]
        )
        self.backbone_act = backbone_cfg.get("act", DEFAULT_CONFIG["backbone"]["act"])

        neck_cfg = kwargs.get("neck", DEFAULT_CONFIG["neck"])
        self.neck_name = neck_cfg.get("name", DEFAULT_CONFIG["neck"]["name"])
        self.neck_out_channels = neck_cfg.get(
            "out_channels", DEFAULT_CONFIG["neck"]["out_channels"]
        )
        self.neck_shortcut = neck_cfg.get(
            "shortcut", DEFAULT_CONFIG["neck"]["shortcut"]
        )

        head_cfg = kwargs.get("head", DEFAULT_CONFIG["head"])
        self.head_name = head_cfg.get("name", DEFAULT_CONFIG["head"]["name"])
        self.head_k = head_cfg.get("k", DEFAULT_CONFIG["head"]["k"])
        self.head_kernel_list = head_cfg.get(
            "kernel_list", DEFAULT_CONFIG["head"]["kernel_list"]
        )
        self.head_fix_nan = head_cfg.get("fix_nan", DEFAULT_CONFIG["head"]["fix_nan"])
