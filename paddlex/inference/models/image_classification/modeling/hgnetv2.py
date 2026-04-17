# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from typing import Any, List

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Constant

from ....utils.benchmark import add_inference_operations, benchmark
from ...common.transformers.activations import ACT2FN
from ...common.transformers.transformers import (
    BatchNormHFStateDictMixin,
    PretrainedModel,
)
from ._config_hgnetv2 import HGNetV2Config


class HGNetV2LearnableAffineBlock(nn.Layer):
    def __init__(self, scale_value=1.0, bias_value=0.0):
        super().__init__()
        self.scale = self.create_parameter(
            shape=[1],
            default_initializer=Constant(value=scale_value),
        )
        self.bias = self.create_parameter(
            shape=[1],
            default_initializer=Constant(value=bias_value),
        )

    def forward(self, hidden_state):
        return self.scale * hidden_state + self.bias


class HGNetV2ConvLayer(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        groups=1,
        activation="relu",
        use_learnable_affine_block=False,
    ):
        super().__init__()
        self.convolution = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=(kernel_size - 1) // 2,
            bias_attr=False,
        )
        self.normalization = nn.BatchNorm2D(out_channels)
        self.activation = (
            ACT2FN[activation] if activation is not None else nn.Identity()
        )
        if activation and use_learnable_affine_block:
            self.lab = HGNetV2LearnableAffineBlock()
        else:
            self.lab = nn.Identity()

    def forward(self, input):
        hidden_state = self.convolution(input)
        hidden_state = self.normalization(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.lab(hidden_state)
        return hidden_state


class HGNetV2ConvLayerLight(nn.Layer):
    def __init__(
        self, in_channels, out_channels, kernel_size, use_learnable_affine_block=False
    ):
        super().__init__()
        self.conv1 = HGNetV2ConvLayer(
            in_channels,
            out_channels,
            kernel_size=1,
            activation=None,
            use_learnable_affine_block=use_learnable_affine_block,
        )
        self.conv2 = HGNetV2ConvLayer(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            groups=out_channels,
            use_learnable_affine_block=use_learnable_affine_block,
        )

    def forward(self, hidden_state):
        hidden_state = self.conv1(hidden_state)
        hidden_state = self.conv2(hidden_state)
        return hidden_state


class HGNetV2Embeddings(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.stem1 = HGNetV2ConvLayer(
            config.stem_channels[0],
            config.stem_channels[1],
            kernel_size=3,
            stride=config.stem_strides[0],
            activation=config.hidden_act,
            use_learnable_affine_block=config.use_learnable_affine_block,
        )
        self.stem2a = HGNetV2ConvLayer(
            config.stem_channels[1],
            config.stem_channels[1] // 2,
            kernel_size=2,
            stride=config.stem_strides[1],
            activation=config.hidden_act,
            use_learnable_affine_block=config.use_learnable_affine_block,
        )
        self.stem2b = HGNetV2ConvLayer(
            config.stem_channels[1] // 2,
            config.stem_channels[1],
            kernel_size=2,
            stride=config.stem_strides[2],
            activation=config.hidden_act,
            use_learnable_affine_block=config.use_learnable_affine_block,
        )
        self.stem3 = HGNetV2ConvLayer(
            config.stem_channels[1] * 2,
            config.stem_channels[1],
            kernel_size=3,
            stride=config.stem_strides[3],
            activation=config.hidden_act,
            use_learnable_affine_block=config.use_learnable_affine_block,
        )
        self.stem4 = HGNetV2ConvLayer(
            config.stem_channels[1],
            config.stem_channels[2],
            kernel_size=1,
            stride=config.stem_strides[4],
            activation=config.hidden_act,
            use_learnable_affine_block=config.use_learnable_affine_block,
        )
        self.pool = nn.MaxPool2D(kernel_size=2, stride=1, ceil_mode=True)
        self.num_channels = config.num_channels

    def forward(self, pixel_values):
        embedding = self.stem1(pixel_values)
        embedding = F.pad(embedding, [0, 1, 0, 1])
        emb_stem_2a = self.stem2a(embedding)
        emb_stem_2a = F.pad(emb_stem_2a, [0, 1, 0, 1])
        emb_stem_2a = self.stem2b(emb_stem_2a)
        pooled_emb = self.pool(embedding)
        embedding = paddle.concat([pooled_emb, emb_stem_2a], axis=1)
        embedding = self.stem3(embedding)
        embedding = self.stem4(embedding)
        return embedding


class HGNetV2BasicLayer(nn.Layer):
    def __init__(
        self,
        in_channels,
        middle_channels,
        out_channels,
        layer_num,
        kernel_size=3,
        residual=False,
        light_block=False,
        drop_path=0.0,
        use_learnable_affine_block=False,
    ):
        super().__init__()
        self.residual = residual

        self.layers = nn.LayerList()
        for i in range(layer_num):
            temp_in_channels = in_channels if i == 0 else middle_channels
            if light_block:
                block = HGNetV2ConvLayerLight(
                    in_channels=temp_in_channels,
                    out_channels=middle_channels,
                    kernel_size=kernel_size,
                    use_learnable_affine_block=use_learnable_affine_block,
                )
            else:
                block = HGNetV2ConvLayer(
                    in_channels=temp_in_channels,
                    out_channels=middle_channels,
                    kernel_size=kernel_size,
                    use_learnable_affine_block=use_learnable_affine_block,
                    stride=1,
                )
            self.layers.append(block)

        total_channels = in_channels + layer_num * middle_channels
        aggregation_squeeze_conv = HGNetV2ConvLayer(
            total_channels,
            out_channels // 2,
            kernel_size=1,
            stride=1,
            use_learnable_affine_block=use_learnable_affine_block,
        )
        aggregation_excitation_conv = HGNetV2ConvLayer(
            out_channels // 2,
            out_channels,
            kernel_size=1,
            stride=1,
            use_learnable_affine_block=use_learnable_affine_block,
        )
        self.aggregation = nn.Sequential(
            aggregation_squeeze_conv,
            aggregation_excitation_conv,
        )
        self.drop_path = nn.Dropout(p=drop_path) if drop_path else nn.Identity()

    def forward(self, hidden_state):
        identity = hidden_state
        output = [hidden_state]
        for layer in self.layers:
            hidden_state = layer(hidden_state)
            output.append(hidden_state)
        hidden_state = paddle.concat(output, axis=1)
        hidden_state = self.aggregation(hidden_state)
        if self.residual:
            hidden_state = self.drop_path(hidden_state) + identity
        return hidden_state


class HGNetV2Stage(nn.Layer):
    def __init__(self, config, stage_index, drop_path=0.0):
        super().__init__()
        in_channels = config.stage_in_channels[stage_index]
        mid_channels = config.stage_mid_channels[stage_index]
        out_channels = config.stage_out_channels[stage_index]
        num_blocks = config.stage_num_blocks[stage_index]
        num_layers = config.stage_numb_of_layers[stage_index]
        downsample = config.stage_downsample[stage_index]
        light_block = config.stage_light_block[stage_index]
        kernel_size = config.stage_kernel_size[stage_index]
        use_learnable_affine_block = config.use_learnable_affine_block
        stride = config.stage_downsample_strides[stage_index]

        if downsample:
            self.downsample = HGNetV2ConvLayer(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                groups=in_channels,
                activation=None,
            )
        else:
            self.downsample = nn.Identity()

        blocks_list = []
        for i in range(num_blocks):
            blocks_list.append(
                HGNetV2BasicLayer(
                    in_channels if i == 0 else out_channels,
                    mid_channels,
                    out_channels,
                    num_layers,
                    residual=(i != 0),
                    kernel_size=kernel_size,
                    light_block=light_block,
                    drop_path=drop_path,
                    use_learnable_affine_block=use_learnable_affine_block,
                )
            )
        self.blocks = nn.LayerList(blocks_list)

    def forward(self, hidden_state):
        hidden_state = self.downsample(hidden_state)
        for block in self.blocks:
            hidden_state = block(hidden_state)
        return hidden_state


class HGNetV2Encoder(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.stages = nn.LayerList()
        for stage_index in range(len(config.stage_in_channels)):
            self.stages.append(HGNetV2Stage(config, stage_index))

    def forward(self, hidden_state):
        hidden_states = [hidden_state]
        for stage in self.stages:
            hidden_state = stage(hidden_state)
            hidden_states.append(hidden_state)
        return hidden_state, hidden_states


class HGNetV2Backbone(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.embedder = HGNetV2Embeddings(config)
        self.encoder = HGNetV2Encoder(config)
        self.out_channels = list(config.stage_out_channels)

    def forward(self, pixel_values):
        embedding_output = self.embedder(pixel_values)
        _, hidden_states = self.encoder(embedding_output)
        return hidden_states[1:]


class HGNetV2ForImageClassification(BatchNormHFStateDictMixin, PretrainedModel):
    config_class = HGNetV2Config

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.embedder = HGNetV2Embeddings(config)
        self.encoder = HGNetV2Encoder(config)
        self.avg_pool = nn.AdaptiveAvgPool2D((1, 1))
        self.flatten = nn.Flatten()
        self.fc = (
            nn.Linear(config.hidden_sizes[-1], config.num_labels)
            if config.num_labels > 0
            else nn.Identity()
        )
        self.classifier = nn.LayerList([self.avg_pool, self.flatten])
        self.out_act = nn.Softmax(axis=-1)

    add_inference_operations("hgnetv2_forward")

    @benchmark.timeit_with_options(name="hgnetv2_forward")
    def forward(self, x: List) -> List:
        x = paddle.to_tensor(x[0])
        embedding_output = self.embedder(x)
        last_hidden_state, _ = self.encoder(embedding_output)
        for layer in self.classifier:
            last_hidden_state = layer(last_hidden_state)
        logits = self.fc(last_hidden_state)
        result = self.out_act(logits)
        return [result.cpu().numpy()]

    def get_transpose_weight_keys(self):
        t_layers = ["fc"]
        keys = []
        for key, _ in self.get_hf_state_dict().items():
            for t_layer in t_layers:
                if t_layer in key and key.endswith("weight"):
                    keys.append(key)
        return keys
