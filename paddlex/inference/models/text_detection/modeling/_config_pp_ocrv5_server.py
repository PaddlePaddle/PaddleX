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
    "model_type": "det",
    "model_name": "PP-OCRv5_server_det",
    "algorithm": "DB",
    "upsample_mode": "nearest",
    "upsample_align_mode": 1,
    "backbone": {
        "name": "PPHGNetV2",
        "stem_channels": [3, 32, 48],
        "stage_config": {
            "stage1": [48, 48, 128, 1, False, False, 3, 6, 2],
            "stage2": [128, 96, 512, 1, True, False, 3, 6, 2],
            "stage3": [512, 192, 1024, 3, True, True, 5, 6, 2],
            "stage4": [1024, 384, 2048, 1, True, True, 5, 6, 2],
        },
        "use_lab": False,
        "use_last_conv": True,
        "class_expand": 2048,
        "dropout_prob": 0.0,
        "class_num": 1000,
        "lr_mult_list": [1.0, 1.0, 1.0, 1.0, 1.0],
        "det": True,
        "out_indices": [0, 1, 2, 3],
    },
    "neck": {
        "name": "LKPAN",
        "out_channels": 256,
        "mode": "large",
        "reduce_factor": 2,
        "intraclblock_config": {
            "reduce_channel": [1, 1, 0],
            "return_channel": [1, 1, 0],
            "v_layer_7x1": [[7, 1], [1, 1], [3, 0]],
            "v_layer_5x1": [[5, 1], [1, 1], [2, 0]],
            "v_layer_3x1": [[3, 1], [1, 1], [1, 0]],
            "q_layer_1x7": [[1, 7], [1, 1], [0, 3]],
            "q_layer_1x5": [[1, 5], [1, 1], [0, 2]],
            "q_layer_1x3": [[1, 3], [1, 1], [0, 1]],
            "c_layer_7x7": [[7, 7], [1, 1], [3, 3]],
            "c_layer_5x5": [[5, 5], [1, 1], [2, 2]],
            "c_layer_3x3": [[3, 3], [1, 1], [1, 1]],
        },
    },
    "head": {
        "name": "PFHeadLocal",
        "in_channels": 1024,
        "k": 50,
        "mode": "large",
        "scale_factor": 2,
        "act": "relu",
        "kernel_list": [3, 2, 2],
        "fix_nan": False,
    },
}


class PPOCRV5ServerDetConfig(PretrainedConfig):
    model_type = "det"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model_name = kwargs.get("model_name", DEFAULT_CONFIG["model_name"])
        self.algorithm = kwargs.get("algorithm", DEFAULT_CONFIG["algorithm"])
        self.upsample_mode = kwargs.get(
            "upsample_mode", DEFAULT_CONFIG["upsample_mode"]
        )
        self.upsample_align_mode = kwargs.get(
            "upsample_align_mode", DEFAULT_CONFIG["upsample_align_mode"]
        )

        backbone_cfg = kwargs.get("backbone", DEFAULT_CONFIG["backbone"])
        self.backbone_name = backbone_cfg.get(
            "name", DEFAULT_CONFIG["backbone"]["name"]
        )
        self.backbone_stem_channels = backbone_cfg.get(
            "stem_channels", DEFAULT_CONFIG["backbone"]["stem_channels"]
        )
        self.backbone_stage_config = backbone_cfg.get(
            "stage_config", DEFAULT_CONFIG["backbone"]["stage_config"]
        )
        self.backbone_use_lab = backbone_cfg.get(
            "use_lab", DEFAULT_CONFIG["backbone"]["use_lab"]
        )
        self.backbone_use_last_conv = backbone_cfg.get(
            "use_last_conv", DEFAULT_CONFIG["backbone"]["use_last_conv"]
        )
        self.backbone_class_expand = backbone_cfg.get(
            "class_expand", DEFAULT_CONFIG["backbone"]["class_expand"]
        )
        self.backbone_class_num = backbone_cfg.get(
            "class_num", DEFAULT_CONFIG["backbone"]["class_num"]
        )
        self.backbone_lr_mult_list = backbone_cfg.get(
            "lr_mult_list", DEFAULT_CONFIG["backbone"]["lr_mult_list"]
        )
        self.backbone_det = backbone_cfg.get("det", DEFAULT_CONFIG["backbone"]["det"])
        self.backbone_out_indices = backbone_cfg.get(
            "out_indices", DEFAULT_CONFIG["backbone"]["out_indices"]
        )

        neck_cfg = kwargs.get("neck", DEFAULT_CONFIG["neck"])
        self.neck_name = neck_cfg.get("name", DEFAULT_CONFIG["neck"]["name"])
        self.neck_out_channels = neck_cfg.get(
            "out_channels", DEFAULT_CONFIG["neck"]["out_channels"]
        )
        self.neck_mode = neck_cfg.get("mode", DEFAULT_CONFIG["neck"]["mode"])
        self.neck_reduce_factor = neck_cfg.get(
            "reduce_factor", DEFAULT_CONFIG["neck"]["reduce_factor"]
        )
        self.neck_intraclblock_config = neck_cfg.get(
            "intraclblock_config", DEFAULT_CONFIG["neck"]["intraclblock_config"]
        )

        head_cfg = kwargs.get("head", DEFAULT_CONFIG["head"])
        self.head_name = head_cfg.get("name", DEFAULT_CONFIG["head"]["name"])
        self.head_in_channels = head_cfg.get(
            "in_channels", DEFAULT_CONFIG["head"]["in_channels"]
        )
        self.head_k = head_cfg.get("k", DEFAULT_CONFIG["head"]["k"])
        self.head_mode = head_cfg.get("mode", DEFAULT_CONFIG["head"]["mode"])
        self.head_scale_factor = head_cfg.get(
            "scale_factor", DEFAULT_CONFIG["head"]["scale_factor"]
        )
        self.head_act = head_cfg.get("act", DEFAULT_CONFIG["head"]["act"])
        self.head_kernel_list = head_cfg.get(
            "kernel_list", DEFAULT_CONFIG["head"]["kernel_list"]
        )
        self.head_fix_nan = head_cfg.get("fix_nan", DEFAULT_CONFIG["head"]["fix_nan"])
