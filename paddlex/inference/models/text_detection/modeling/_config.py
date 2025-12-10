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


class PPOCRV5MobileDetConfig(PretrainedConfig):
    model_type = "det"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model_name = kwargs["model_name"]
        self.algorithm = kwargs["algorithm"]

        backbone_cfg = kwargs["backbone"]
        self.backbone_name = backbone_cfg["name"]
        self.backbone_scale = backbone_cfg["scale"]
        self.backbone_det = backbone_cfg["det"]
        self.backbone_conv_kxk_num = backbone_cfg["conv_kxk_num"]
        self.backbone_lr_mult_list = backbone_cfg["lr_mult_list"]
        self.backbone_lab_lr = backbone_cfg["lab_lr"]
        self.backbone_net_config = backbone_cfg["net_config"]
        self.backbone_out_channels = backbone_cfg["out_channels"]

        neck_cfg = kwargs["neck"]
        self.neck_name = neck_cfg["name"]
        self.neck_out_channels = neck_cfg["out_channels"]
        self.neck_shortcut = neck_cfg["shortcut"]

        head_cfg = kwargs["head"]
        self.head_name = head_cfg["name"]
        self.head_k = head_cfg["k"]
        self.head_fix_nan = head_cfg["fix_nan"]


class PPOCRV5ServerDetConfig(PretrainedConfig):
    model_type = "det"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model_name = kwargs["model_name"]
        self.algorithm = kwargs["algorithm"]

        # Backbone
        backbone_cfg = kwargs["backbone"]
        self.backbone_name = backbone_cfg["name"]
        self.backbone_stem_channels = backbone_cfg["stem_channels"]
        self.backbone_stage_config = backbone_cfg["stage_config"]
        self.backbone_use_lab = backbone_cfg["use_lab"]
        self.backbone_use_last_conv = backbone_cfg["use_last_conv"]
        self.backbone_class_expand = backbone_cfg["class_expand"]
        self.backbone_dropout_prob = backbone_cfg["dropout_prob"]
        self.backbone_class_num = backbone_cfg["class_num"]
        self.backbone_lr_mult_list = backbone_cfg["lr_mult_list"]
        self.backbone_det = backbone_cfg["det"]
        self.backbone_out_indices = backbone_cfg["out_indices"]

        # Neck
        neck_cfg = kwargs["neck"]
        self.neck_name = neck_cfg["name"]
        self.neck_out_channels = neck_cfg["out_channels"]
        self.neck_mode = neck_cfg["mode"]

        # Head
        head_cfg = kwargs["head"]
        self.head_name = head_cfg["name"]
        self.head_in_channels = head_cfg["in_channels"]
        self.head_k = head_cfg["k"]
        self.head_mode = head_cfg["mode"]
