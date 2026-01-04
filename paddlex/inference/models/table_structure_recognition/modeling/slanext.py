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
from .slanext_modules.rec_vary_vit import Vary_VIT_B
from .slanext_modules.table_att_head import SLAHead

__all__ = ["SLANeXt"]


class SLANeXtConfig(PretrainedConfig):
    def __init__(
        self,
        backbone,
        SLAHead,
    ):
        if backbone["name"] == "Vary_VIT_B":
            self.image_size = backbone["image_size"]
            self.encoder_embed_dim = backbone["encoder_embed_dim"]
            self.encoder_depth = backbone["encoder_depth"]
            self.encoder_num_heads = backbone["encoder_num_heads"]
            self.encoder_global_attn_indexes = backbone["encoder_global_attn_indexes"]
        else:
            raise RuntimeError(
                f"There is no dynamic graph implementation for backbone {backbone['name']}."
            )
        self.out_channels = SLAHead["out_channels"]
        self.hidden_size = SLAHead["hidden_size"]
        self.max_text_length = SLAHead["max_text_length"]
        self.loc_reg_num = SLAHead["loc_reg_num"]
        self.tensor_parallel_degree = 1


class SLANeXt(PretrainedModel):

    config_class = SLANeXtConfig

    def __init__(self, config: SLANeXtConfig):
        super().__init__(config)
        self.backbone = Vary_VIT_B(
            image_size=self.config.image_size,
            encoder_embed_dim=self.config.encoder_embed_dim,
            encoder_depth=self.config.encoder_depth,
            encoder_num_heads=self.config.encoder_num_heads,
            encoder_global_attn_indexes=self.config.encoder_global_attn_indexes,
        )
        self.head = SLAHead(
            in_channels=self.backbone.out_channels,
            out_channels=self.config.out_channels,
            hidden_size=self.config.hidden_size,
            max_text_length=self.config.max_text_length,
            loc_reg_num=self.config.loc_reg_num,
        )

    add_inference_operations("slanext_forward")

    @benchmark.timeit_with_options(name="slanext_forward")
    def forward(self, x):
        x = paddle.to_tensor(x[0])
        x = self.backbone(x)
        x = self.head(x)
        return [x["loc_preds"], x["structure_probs"]]

    def get_transpose_weight_keys(self):
        transpose_keys = ["mlp.lin2", "attn.qkv", "mlp.lin1"]
        need_to_transpose = []
        all_weight_keys = []
        for name, param in self.backbone.named_parameters():
            all_weight_keys.append("backbone." + name)
        for i in range(len(all_weight_keys)):
            for j in range(len(transpose_keys)):
                if (transpose_keys[j] in all_weight_keys[i]) and (
                    "bias" not in all_weight_keys[i]
                ):
                    need_to_transpose.append(all_weight_keys[i])
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
