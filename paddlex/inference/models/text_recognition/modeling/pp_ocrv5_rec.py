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


import paddle

from ....utils.benchmark import add_inference_operations, benchmark
from ...common.transformers.transformers import PretrainedConfig, PretrainedModel
from .pp_ocrv5_rec_modules.rec_lcnetv3 import PPLCNetV3
from .pp_ocrv5_rec_modules.rec_multi_head import MultiHead
from .pp_ocrv5_rec_modules.rec_pphgnetv2 import PPHGNetV2

__all__ = ["PPOCRV5Rec"]


class PPOCRV5RecConfig(PretrainedConfig):
    def __init__(
        self,
        backbone,
        MultiHead,
    ):
        self.backbone_name = backbone["name"]
        if self.backbone_name == "PPLCNetV3":
            self.net_config = backbone["net_config"]
            self.scale = backbone["scale"]
            self.conv_kxk_num = backbone["conv_kxk_num"]
            self.lr_mult_list = backbone["lr_mult_list"]
            self.lab_lr = backbone["lab_lr"]
        elif self.backbone_name == "PPHGNetV2":
            self.text_rec = backbone["text_rec"]
            self.stem_channels = backbone["stem_channels"]
            self.stage_config = backbone["stage_config"]
            self.det = backbone["det"]
            self.use_lab = backbone["use_lab"]
            self.use_last_conv = backbone["use_last_conv"]
            self.class_expand = backbone["class_expand"]
            self.dropout_prob = backbone["dropout_prob"]
            self.class_num = backbone["class_num"]
            self.lr_mult_list = backbone["lr_mult_list"]
            self.out_indices = backbone["out_indices"]
        else:
            raise RuntimeError(
                f"There is no dynamic graph implementation for backbone {backbone['name']}."
            )
        self.head_list = MultiHead["head_list"]
        self.decode_list = MultiHead["decode_list"]
        self.tensor_parallel_degree = 1


class PPOCRV5Rec(PretrainedModel):

    config_class = PPOCRV5RecConfig

    def __init__(self, config: PPOCRV5RecConfig):
        super().__init__(config)
        if self.config.backbone_name == "PPLCNetV3":
            self.backbone = PPLCNetV3(
                scale=self.config.scale,
                net_config=self.config.net_config,
                conv_kxk_num=self.config.conv_kxk_num,
                lr_mult_list=self.config.lr_mult_list,
                lab_lr=self.config.lab_lr,
            )
        elif self.config.backbone_name == "PPHGNetV2":
            self.backbone = PPHGNetV2(
                stage_config=self.config.stage_config,
                stem_channels=self.config.stem_channels,
                text_rec=self.config.text_rec,
                det=self.config.det,
                use_lab=self.config.use_lab,
                use_last_conv=self.config.use_last_conv,
                class_expand=self.config.class_expand,
                dropout_prob=self.config.dropout_prob,
                class_num=self.config.class_num,
                lr_mult_list=self.config.lr_mult_list,
                out_indices=self.config.out_indices,
            )
        self.head = MultiHead(
            in_channels=self.backbone.out_channels,
            out_channels_list=self.config.decode_list,
            head_list=self.config.head_list,
        )

    add_inference_operations("pp_ocrv5_rec_forward")

    @benchmark.timeit_with_options(name="pp_ocrv5_rec_forward")
    def forward(self, x):
        x = paddle.to_tensor(x[0])
        x = self.backbone(x)
        x = self.head(x)
        return [x.cpu().numpy()]

    def get_transpose_weight_keys(self):
        transpose_keys = ["fc", "out_proj", "attn.qkv"]
        need_to_transpose = []
        all_weight_keys = []
        for name, param in self.head.named_parameters():
            all_weight_keys.append("head." + name)
        for i in range(len(all_weight_keys)):
            for j in range(len(transpose_keys)):
                if (transpose_keys[j] in all_weight_keys[i]) and (
                    "bias" not in all_weight_keys[i]
                ):
                    need_to_transpose.append(all_weight_keys[i])
        if self.config.backbone_name == "PPHGNetV2":
            need_to_transpose.append("backbone.fc.weight")
        return need_to_transpose

    def get_hf_state_dict(self, *args, **kwargs):

        model_state_dict = self.state_dict(*args, **kwargs)

        hf_state_dict = {}
        for old_key, value in model_state_dict.items():
            if "_mean" in old_key:
                new_key = old_key.replace("_mean", "running_mean")
            elif "_variance" in old_key:
                new_key = old_key.replace("_variance", "running_var")
            else:
                new_key = old_key
            hf_state_dict[new_key] = value

        return hf_state_dict

    def set_hf_state_dict(self, state_dict, *args, **kwargs):

        key_mapping = {}
        for old_key in list(state_dict.keys()):
            if "running_mean" in old_key:
                key_mapping[old_key] = old_key.replace("running_mean", "_mean")
            elif "running_var" in old_key:
                key_mapping[old_key] = old_key.replace("running_var", "_variance")

        for old_key, new_key in key_mapping.items():
            state_dict[new_key] = state_dict.pop(old_key)

        return self.set_state_dict(state_dict, *args, **kwargs)
