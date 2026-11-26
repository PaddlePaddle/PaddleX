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

"""Dynamic-graph PP-DocLayoutV4, for loading the safetensors package.

PP-DocLayoutV4 is PP-DocLayoutV3's backbone/encoder/decoder stack with the mask
branch removed and the following additions:

* ``num_coords = 10`` -- the bbox heads regress a 4-point quad as
  ``[cx, cy, dx1, dy1, ..., dx4, dy4]`` in sigmoid space instead of a
  ``cxcywh`` box. Deformable attention still samples on the enclosing rect,
  which :func:`quad10d_to_rect` derives.
* per-decoder-layer bbox/class heads. V3 ties them to the encoder heads; V4
  keeps six independent copies, so the checkpoint carries
  ``decoder.bbox_embed.{0..5}`` / ``decoder.class_embed.{0..5}``.
* reading order via S2R: a ROOR (direct-successor) head alongside V3's
  relative-order head, fused as ``rel = a * closure(roor) + b * rel_dir``.
  Unlike V3, the reading order is NOT decoded in the graph -- the raw
  ``rel``/``roor`` logit matrices and the query indices are returned so that
  ``LayoutAnalysisProcess`` can run the DAG decode on the filtered boxes.

* optional page deformation prediction from stride-16 encoder features,
  returning rectifying (roll, confidence, shear, kx, ky) parameters.

Everything that is bit-identical to V3 (backbone, AIFI, deformable attention,
decoder layer, CSPRep/RepVgg blocks, ...) is imported from
:mod:`.pp_doclayout_v3` rather than copied, so the two stay in lockstep.
"""

from __future__ import absolute_import, division, print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ...common.transformers.transformers import (
    BatchNormHFStateDictMixin,
    PretrainedModel,
)
from ._config_pp_doclayout_v4 import PPDocLayoutV4Config
from .pp_doclayout_v2 import (
    _apply_rt_detr_key_conversion,
    _reverse_rt_detr_key_conversion,
)
from .pp_doclayout_v3 import (
    PPDocLayoutV3AIFILayer,
    PPDocLayoutV3ConvEncoder,
    PPDocLayoutV3ConvNormLayer,
    PPDocLayoutV3CSPRepLayer,
    PPDocLayoutV3DecoderLayer,
    PPDocLayoutV3MLPPredictionHead,
    inverse_sigmoid,
)

__all__ = ["PPDocLayoutV4"]


def quad10d_to_rect(quad):
    """Convert the 10-D quad parameterization to ``(cx, cy, w, h)``.

    ``quad`` is ``[..., 10]`` in sigmoid space with the corner offsets shifted
    by +0.5. The returned centre is the *predicted* ``cx, cy`` (not the centroid
    of the corners) and ``w, h`` are the full offset extents.
    """
    cx = quad[..., 0:1]
    cy = quad[..., 1:2]
    dx = quad[..., 2::2] - 0.5
    dy = quad[..., 3::2] - 0.5
    w = dx.max(-1, keepdim=True) - dx.min(-1, keepdim=True)
    h = dy.max(-1, keepdim=True) - dy.min(-1, keepdim=True)
    return paddle.concat([cx, cy, w, h], axis=-1)


class PPDocLayoutV4GlobalPointer(nn.Layer):
    """Pairwise scorer for the reading-order heads.

    ``antisymmetric=True`` (relative order) enforces ``logits[i, j] ==
    -logits[j, i]``; ``antisymmetric=False`` (ROOR / direct successor) only
    suppresses self-loops on the diagonal. Note there is no rotary embedding
    here, and the ``1 / sqrt(head_size)`` scaling is applied *before* the
    antisymmetrization.
    """

    def __init__(self, config, antisymmetric):
        super().__init__()
        self.head_size = config.global_pointer_head_size
        self.antisymmetric = antisymmetric
        self.dense = nn.Linear(config.d_model, self.head_size * 2)
        self.dropout = nn.Dropout(config.gp_dropout_value)

    def forward(self, inputs):
        batch_size, sequence_length, _ = inputs.shape
        projection = self.dense(inputs).reshape(
            [batch_size, sequence_length, 2, self.head_size]
        )
        projection = self.dropout(projection)
        queries, keys = projection[:, :, 0, :], projection[:, :, 1, :]

        raw_logits = paddle.matmul(queries, keys.transpose([0, 2, 1])) / (
            self.head_size**0.5
        )
        if self.antisymmetric:
            return raw_logits - raw_logits.transpose([0, 2, 1])
        eye_mask = paddle.eye(sequence_length, dtype=raw_logits.dtype).unsqueeze(0)
        return raw_logits - eye_mask * 1e4


class PPDocLayoutV4S2RFusion(nn.Layer):
    """Fuse the ROOR successor matrix's transitive closure into the relative order.

    ``rel = a * antisym(closure(roor)) + b * rel_dir``. With ``a_init = 0`` the
    module starts numerically identical to using ``rel_dir`` alone, and with
    ``learnable_b = False`` ``b`` stays a plain Python float -- which is why the
    checkpoint only carries ``a`` and ``b`` has to come from the config.
    """

    def __init__(self, config):
        super().__init__()
        self.steps = config.s2r_steps
        self.damping = config.s2r_damping
        self.a = self.create_parameter(
            shape=[1],
            default_initializer=nn.initializer.Constant(config.s2r_a_init),
        )
        self.learnable_b = config.s2r_learnable_b
        if self.learnable_b:
            self.b = self.create_parameter(
                shape=[1],
                default_initializer=nn.initializer.Constant(config.s2r_b_init),
            )
        else:
            self.b = float(config.s2r_b_init)

    def forward(self, rel_dir, roor_logits):
        num_queries = roor_logits.shape[-1]
        eye = paddle.eye(num_queries, dtype=roor_logits.dtype).unsqueeze(0)

        # Soft directed adjacency: A[i, j] ~= P(i directly precedes j).
        adjacency = F.sigmoid(roor_logits) * (1.0 - eye)
        # Clip row sums from below at 1 so dense rows are damped without
        # amplifying weak edges or terminal nodes.
        adjacency = adjacency / paddle.clip(adjacency.sum(-1, keepdim=True), min=1.0)

        # Soft transitive closure S = sum_k damping^(k-1) * A^k.
        closure = adjacency
        power = adjacency
        for _ in range(self.steps - 1):
            power = self.damping * paddle.bmm(adjacency, power)
            closure = closure + power

        prop = closure - closure.transpose([0, 2, 1])
        return self.a * prop + self.b * rel_dir


class PPDocLayoutV4HybridEncoder(nn.Layer):
    """AIFI + top-down FPN + bottom-up PAN. V3's encoder without the mask head."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_channels = config.encoder_in_channels
        self.encoder_hidden_dim = config.encoder_hidden_dim
        self.encode_proj_layers = config.encode_proj_layers
        self.num_fpn_stages = len(self.in_channels) - 1
        self.num_pan_stages = len(self.in_channels) - 1

        self.aifi = nn.LayerList(
            [
                PPDocLayoutV3AIFILayer(config)
                for _ in range(len(self.encode_proj_layers))
            ]
        )

        self.lateral_convs = nn.LayerList()
        self.fpn_blocks = nn.LayerList()
        for _ in range(self.num_fpn_stages):
            self.lateral_convs.append(
                PPDocLayoutV3ConvNormLayer(
                    config,
                    in_channels=self.encoder_hidden_dim,
                    out_channels=self.encoder_hidden_dim,
                    kernel_size=1,
                    stride=1,
                    activation=config.activation_function,
                )
            )
            self.fpn_blocks.append(PPDocLayoutV3CSPRepLayer(config))

        self.downsample_convs = nn.LayerList()
        self.pan_blocks = nn.LayerList()
        for _ in range(self.num_pan_stages):
            self.downsample_convs.append(
                PPDocLayoutV3ConvNormLayer(
                    config,
                    in_channels=self.encoder_hidden_dim,
                    out_channels=self.encoder_hidden_dim,
                    kernel_size=3,
                    stride=2,
                    activation=config.activation_function,
                )
            )
            self.pan_blocks.append(PPDocLayoutV3CSPRepLayer(config))

    def forward(self, feature_maps):
        if self.config.encoder_layers > 0:
            for i, enc_ind in enumerate(self.encode_proj_layers):
                feature_maps[enc_ind] = self.aifi[i](feature_maps[enc_ind])

        # top-down FPN. Note the lateral output is written back into
        # `fpn_feature_maps[-1]`, so the PAN stage consumes the post-lateral
        # feature, not the raw encoder output.
        fpn_feature_maps = [feature_maps[-1]]
        for idx, (lateral_conv, fpn_block) in enumerate(
            zip(self.lateral_convs, self.fpn_blocks)
        ):
            backbone_feature_map = feature_maps[self.num_fpn_stages - idx - 1]
            top_fpn_feature_map = lateral_conv(fpn_feature_maps[-1])
            fpn_feature_maps[-1] = top_fpn_feature_map
            top_fpn_feature_map = F.interpolate(
                top_fpn_feature_map, scale_factor=2.0, mode="nearest"
            )
            fpn_feature_maps.append(
                fpn_block(
                    paddle.concat([top_fpn_feature_map, backbone_feature_map], axis=1)
                )
            )

        fpn_feature_maps.reverse()

        # bottom-up PAN
        pan_feature_maps = [fpn_feature_maps[0]]
        for idx, (downsample_conv, pan_block) in enumerate(
            zip(self.downsample_convs, self.pan_blocks)
        ):
            downsampled_feature_map = downsample_conv(pan_feature_maps[-1])
            pan_feature_maps.append(
                pan_block(
                    paddle.concat(
                        [downsampled_feature_map, fpn_feature_maps[idx + 1]], axis=1
                    )
                )
            )

        return pan_feature_maps


class PPDocLayoutV4Decoder(nn.Layer):
    """Deformable decoder with per-layer bbox/class heads and quad references."""

    def __init__(self, config):
        super().__init__()
        self.num_queries = config.num_queries
        self.num_coords = config.num_coords
        self.layers = nn.LayerList(
            [PPDocLayoutV3DecoderLayer(config) for _ in range(config.decoder_layers)]
        )
        self.query_pos_head = PPDocLayoutV3MLPPredictionHead(
            config.num_coords, 2 * config.d_model, config.d_model, num_layers=2
        )
        # Untied, one per decoder layer (V3 ties a single head to the encoder).
        self.bbox_embed = nn.LayerList(
            [
                PPDocLayoutV3MLPPredictionHead(
                    config.d_model, config.d_model, config.num_coords, num_layers=3
                )
                for _ in range(config.decoder_layers)
            ]
        )
        self.class_embed = nn.LayerList(
            [
                nn.Linear(config.d_model, config.num_labels)
                for _ in range(config.decoder_layers)
            ]
        )

    def forward(
        self,
        inputs_embeds,
        encoder_hidden_states,
        reference_points,
        spatial_shapes,
        spatial_shapes_list,
        level_start_index,
        order_head,
        global_pointer,
        roor_order_head,
        roor_global_pointer,
        s2r_fusion,
    ):
        hidden_states = inputs_embeds
        reference_points = F.sigmoid(reference_points)

        last_index = len(self.layers) - 1
        for idx, decoder_layer in enumerate(self.layers):
            # Deformable attention samples on the enclosing rect of the quad.
            if self.num_coords > 4:
                reference_points_input = quad10d_to_rect(reference_points).unsqueeze(2)
            else:
                reference_points_input = reference_points.unsqueeze(2)

            hidden_states = decoder_layer(
                hidden_states,
                object_queries_position_embeddings=self.query_pos_head(
                    reference_points
                ),
                encoder_hidden_states=encoder_hidden_states,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
                spatial_shapes_list=spatial_shapes_list,
                level_start_index=level_start_index,
                encoder_attention_mask=None,
            )

            reference_points = F.sigmoid(
                self.bbox_embed[idx](hidden_states) + inverse_sigmoid(reference_points)
            )

            if idx != last_index:
                continue

            # Only the last layer's class/order predictions are used at
            # inference, matching the reference implementation's `eval_idx`.
            logits = self.class_embed[idx](hidden_states)
            valid_query = (
                hidden_states[:, -self.num_queries :]
                if self.num_queries is not None
                else hidden_states
            )
            roor_logits = roor_global_pointer(roor_order_head[idx](valid_query))
            rel_dir = global_pointer(order_head[idx](valid_query))
            rel_logits = (
                s2r_fusion(rel_dir, roor_logits) if s2r_fusion is not None else rel_dir
            )

        return {
            "pred_boxes": reference_points,
            "logits": logits,
            "rel_order_logits": rel_logits,
            "roor_order_logits": roor_logits,
        }


class PPDocLayoutV4DeformConvHead(nn.Layer):
    """Dilated conv stack over one encoder level + 2 normalised coordinate
    channels, followed by spatial-softmax attention pooling.

    Each location emits one attention logit and a full ``out_dim`` prediction;
    the prediction is the attention-weighted sum.  Local geometric evidence
    (page edges, text-line slopes) is read where it occurs, and the coordinate
    channels tell every location where it sits relative to the canvas centre,
    which the canvas-referenced (roll, shear, kx, ky) need.  The receptive field
    is 15 cells (~240px at stride 16).

    Parameters live under ``deform_head.body.*``, so the head can be trained
    alone by freezing every other prefix.
    """

    def __init__(self, in_ch, out_dim, width=128):
        super(PPDocLayoutV4DeformConvHead, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2D(in_ch + 2, width, 3, padding=1),
            nn.ReLU(),
            nn.Conv2D(width, width, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2D(width, width, 3, padding=4, dilation=4),
            nn.ReLU(),
            nn.Conv2D(width, out_dim + 1, 1),
        )

    def forward(self, feat):
        shp = paddle.shape(feat)
        ys = paddle.linspace(-1.0, 1.0, shp[2], dtype=feat.dtype)
        xs = paddle.linspace(-1.0, 1.0, shp[3], dtype=feat.dtype)
        gy, gx = paddle.meshgrid(ys, xs)
        coord = paddle.stack([gx, gy], 0).unsqueeze(0) + paddle.zeros_like(feat[:, :2])
        o = self.body(paddle.concat([feat, coord], 1)).flatten(2)
        att = F.softmax(o[:, :1], axis=-1)
        return (att * o[:, 1:]).sum(-1)


class PPDocLayoutV4Model(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.backbone = PPDocLayoutV3ConvEncoder(config)
        intermediate_channel_sizes = self.backbone.intermediate_channel_sizes

        # The backbone always returns 4 stages; V4 uses stages 2-4 only (V3 also
        # consumed the stride-4 feature, but that fed the mask branch).
        encoder_input_proj_list = []
        for in_channels in intermediate_channel_sizes:
            encoder_input_proj_list.append(
                nn.Sequential(
                    nn.Conv2D(
                        in_channels,
                        config.encoder_hidden_dim,
                        kernel_size=1,
                        bias_attr=False,
                    ),
                    nn.BatchNorm2D(config.encoder_hidden_dim),
                )
            )
        self.encoder_input_proj = nn.LayerList(encoder_input_proj_list[1:])

        self.encoder = PPDocLayoutV4HybridEncoder(config)
        if config.deform_pred:
            self.deform_head = PPDocLayoutV4DeformConvHead(
                config.encoder_hidden_dim,
                6 + 2 * config.deform_kbins,
                config.deform_conv_width,
            )
            self.register_buffer(
                "deform_bin_centers",
                paddle.linspace(
                    -config.deform_krange, config.deform_krange, config.deform_kbins
                ),
            )

        self.denoising_class_embed = nn.Embedding(config.num_labels, config.d_model)
        if config.learn_initial_query:
            self.weight_embedding = nn.Embedding(config.num_queries, config.d_model)

        self.enc_output = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model, epsilon=config.layer_norm_eps),
        )
        self.enc_score_head = nn.Linear(config.d_model, config.num_labels)
        self.enc_bbox_head = PPDocLayoutV3MLPPredictionHead(
            config.d_model, config.d_model, config.num_coords, num_layers=3
        )

        # A second 1x1 projection on top of the encoder outputs. This is a real
        # trained layer, not an identity -- easy to miss.
        decoder_input_proj_list = []
        for in_channels in config.decoder_in_channels:
            decoder_input_proj_list.append(
                nn.Sequential(
                    nn.Conv2D(
                        in_channels, config.d_model, kernel_size=1, bias_attr=False
                    ),
                    nn.BatchNorm2D(config.d_model, epsilon=config.batch_norm_eps),
                )
            )
        self.decoder_input_proj = nn.LayerList(decoder_input_proj_list)

        self.decoder = PPDocLayoutV4Decoder(config)

        # Reading order: relative-order head (antisymmetric) + ROOR head.
        self.decoder_order_head = nn.LayerList(
            [
                nn.Linear(config.d_model, config.d_model)
                for _ in range(config.decoder_layers)
            ]
        )
        self.decoder_global_pointer = PPDocLayoutV4GlobalPointer(
            config, antisymmetric=True
        )
        self.decoder_roor_order_head = nn.LayerList(
            [
                nn.Linear(config.d_model, config.d_model)
                for _ in range(config.decoder_layers)
            ]
        )
        self.decoder_roor_global_pointer = PPDocLayoutV4GlobalPointer(
            config, antisymmetric=False
        )
        self.s2r_fusion = PPDocLayoutV4S2RFusion(config) if config.use_s2r else None

    def generate_anchors(self, spatial_shapes, grid_size=0.05, dtype="float32"):
        """Build 10-D quad anchors in inverse-sigmoid space.

        Layout is ``[cx, cy, x1+0.5, y1+0.5, ..., x4+0.5, y4+0.5]`` with corners
        in TL, TR, BR, BL order. Only ``cx, cy`` vary spatially -- the four
        offsets are a per-level constant, so validity is decided by the centres.
        """
        anchors = []
        for level, (height, width) in enumerate(spatial_shapes):
            grid_y, grid_x = paddle.meshgrid(
                paddle.arange(end=height).astype(dtype),
                paddle.arange(end=width).astype(dtype),
            )
            grid_xy = paddle.stack([grid_x, grid_y], axis=-1)
            valid_wh = paddle.to_tensor([width, height]).astype(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_wh

            half = grid_size * (2.0**level) / 2.0
            offsets = paddle.to_tensor(
                [
                    [
                        0.5 - half,
                        0.5 - half,
                        0.5 + half,
                        0.5 - half,
                        0.5 + half,
                        0.5 + half,
                        0.5 - half,
                        0.5 + half,
                    ]
                ],
                dtype=dtype,
            ).expand([1, height * width, 8])
            anchors.append(
                paddle.concat(
                    [grid_xy.reshape([1, height * width, 2]), offsets], axis=-1
                )
            )

        eps = 1e-2
        anchors = paddle.concat(anchors, axis=1)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(axis=-1, keepdim=True)
        anchors = paddle.log(anchors / (1 - anchors))
        anchors = paddle.where(
            valid_mask, anchors, paddle.full_like(anchors, float("inf"))
        )
        return anchors, valid_mask

    def forward(self, pixel_values):
        features = self.backbone(pixel_values)
        # features[0] is the stride-4 map, unused without the mask branch.
        proj_feats = [
            self.encoder_input_proj[level](source)
            for level, source in enumerate(features[1:])
        ]

        encoder_outputs = self.encoder(proj_feats)

        sources = [
            self.decoder_input_proj[level](source)
            for level, source in enumerate(encoder_outputs)
        ]

        source_flatten = []
        spatial_shapes_list = []
        spatial_shapes = paddle.empty([len(sources), 2], dtype="int64")
        for level, source in enumerate(sources):
            height, width = source.shape[-2:]
            spatial_shapes[level, 0] = height
            spatial_shapes[level, 1] = width
            spatial_shapes_list.append((height, width))
            source_flatten.append(source.flatten(2).transpose([0, 2, 1]))
        source_flatten = paddle.concat(source_flatten, axis=1)
        level_start_index = paddle.concat(
            [paddle.zeros([1], dtype="int64"), spatial_shapes.prod(1).cumsum(0)[:-1]]
        )

        anchors, valid_mask = self.generate_anchors(
            tuple(spatial_shapes_list), dtype=source_flatten.dtype
        )

        memory = valid_mask.astype(source_flatten.dtype) * source_flatten
        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_logits = self.enc_bbox_head(output_memory) + anchors

        # Class-agnostic top-k over positions.
        _, topk_ind = paddle.topk(
            enc_outputs_class.max(-1), self.config.num_queries, axis=1
        )
        reference_points_unact = paddle.take_along_axis(
            enc_outputs_coord_logits,
            topk_ind.unsqueeze(-1).expand([-1, -1, enc_outputs_coord_logits.shape[-1]]),
            axis=1,
        )

        if self.config.learn_initial_query:
            target = self.weight_embedding.weight.unsqueeze(0).expand(
                [source_flatten.shape[0], -1, -1]
            )
        else:
            target = paddle.take_along_axis(
                output_memory,
                topk_ind.unsqueeze(-1).expand([-1, -1, output_memory.shape[-1]]),
                axis=1,
            )

        outputs = self.decoder(
            inputs_embeds=target,
            encoder_hidden_states=source_flatten,
            reference_points=reference_points_unact,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            order_head=self.decoder_order_head,
            global_pointer=self.decoder_global_pointer,
            roor_order_head=self.decoder_roor_order_head,
            roor_global_pointer=self.decoder_roor_global_pointer,
            s2r_fusion=self.s2r_fusion,
        )
        if self.config.deform_pred and self.config.deform_out:
            raw = self.deform_head(encoder_outputs[self.config.deform_level].detach())
            prob = F.softmax(raw[:, :4], axis=-1)
            quadrant = paddle.argmax(prob, axis=-1).astype(raw.dtype)
            angle = 90.0 * quadrant + 45.0 * raw[:, 4]
            angle = (
                paddle.mod(angle + 180.0, paddle.full([1], 360.0, dtype=raw.dtype))
                - 180.0
            )
            bins = self.config.deform_kbins
            centers = self.deform_bin_centers.astype(raw.dtype)
            kx = (F.softmax(raw[:, 6 : 6 + bins], axis=-1) * centers).sum(-1)
            ky = (F.softmax(raw[:, 6 + bins : 6 + 2 * bins], axis=-1) * centers).sum(-1)
            outputs["warp_params"] = paddle.stack(
                [angle, prob.max(-1), 45.0 * raw[:, 5], kx, ky], axis=-1
            )
        return outputs


class PPDocLayoutV4PostProcess:
    """Decode quads to absolute corners and select the top-k query/class pairs.

    Deliberately does NOT decode the reading order: the raw ``rel``/``roor``
    logits and the query indices are handed to ``LayoutAnalysisProcess``, which
    runs the DAG decode on the score-filtered boxes.
    """

    def __init__(self, num_classes=25, num_top_queries=300, num_coords=10):
        self.num_classes = num_classes
        self.num_top_queries = num_top_queries
        self.num_coords = num_coords

    def __call__(self, head_out, im_shape, scale_factor):
        bboxes, logits, rel_order_logits, roor_order_logits = head_out
        batch_size, num_queries = bboxes.shape[:2]

        # `im_shape` is the resized H, W; dividing by scale_factor recovers the
        # original H, W with round-half-up.
        origin_shape = paddle.floor(im_shape / scale_factor + 0.5)
        width = origin_shape[:, 1:2].unsqueeze(1)
        height = origin_shape[:, 0:1].unsqueeze(1)

        if self.num_coords == 10:
            center_x = bboxes[..., 0:1] * width
            center_y = bboxes[..., 1:2] * height
            offset_x = (bboxes[..., 2::2] - 0.5) * width
            offset_y = (bboxes[..., 3::2] - 0.5) * height
            bbox_pred = paddle.stack(
                [center_x + offset_x, center_y + offset_y], axis=-1
            ).reshape([batch_size, num_queries, 8])
        else:
            raise ValueError(f"Unsupported num_coords: {self.num_coords}.")

        scores = F.sigmoid(logits)
        # Flattened top-k over query x class, so one query may appear several
        # times under different labels -- `qi` carries the original query index.
        scores, flat_ind = paddle.topk(scores.flatten(1), self.num_top_queries, axis=-1)
        labels = flat_ind % self.num_classes
        qi = flat_ind // self.num_classes
        batch_ind = (
            paddle.arange(end=batch_size).unsqueeze(-1).tile([1, self.num_top_queries])
        )
        bbox_pred = paddle.gather_nd(bbox_pred, paddle.stack([batch_ind, qi], axis=-1))

        bbox_pred = paddle.concat(
            [
                labels.unsqueeze(-1).astype("float32"),
                scores.unsqueeze(-1),
                bbox_pred,
            ],
            axis=-1,
        ).reshape([-1, 2 + 8])
        # Every query is emitted; the score threshold is applied downstream.
        bbox_num = paddle.to_tensor(self.num_top_queries, dtype="int32").tile(
            [batch_size]
        )
        return bbox_pred, bbox_num, qi, rel_order_logits, roor_order_logits


class PPDocLayoutV4(BatchNormHFStateDictMixin, PretrainedModel):

    config_class = PPDocLayoutV4Config

    def __init__(self, config):
        super(PPDocLayoutV4, self).__init__(config)
        self.config = config
        self.model = PPDocLayoutV4Model(config)
        self.post_process = PPDocLayoutV4PostProcess(
            num_classes=config.num_labels,
            num_top_queries=config.num_queries,
            num_coords=config.num_coords,
        )

    def forward(self, inputs):
        """Return detection/order tensors and optional trailing [B, 5] warp_params.

        Inputs are (img_size, img, scale_factors), matching the static runner.
        """
        im_shape = paddle.to_tensor(inputs[0])
        pixel_values = paddle.to_tensor(inputs[1])
        scale_factor = paddle.to_tensor(inputs[2])

        decoder_outputs = self.model(pixel_values)

        bbox, bbox_num, qi, rel_logits, roor_logits = self.post_process(
            (
                decoder_outputs["pred_boxes"],
                decoder_outputs["logits"],
                decoder_outputs["rel_order_logits"],
                decoder_outputs["roor_order_logits"],
            ),
            im_shape,
            scale_factor,
        )
        outputs = [bbox, bbox_num, qi, rel_logits, roor_logits]
        if "warp_params" in decoder_outputs:
            outputs.append(decoder_outputs["warp_params"])
        return outputs

    def get_transpose_weight_keys(self):
        t_layers = [
            "fc",
            "o_proj",
            "out_proj",
            "output_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "enc_bbox_head",
            "enc_output",
            "query_pos_head",
            "enc_score_head",
            "encoder_attn",
            "decoder.bbox_embed",
            "decoder.class_embed",
            "decoder_order_head",
            "decoder_global_pointer",
            "decoder_roor_order_head",
            "decoder_roor_global_pointer",
            "sampling_offsets",
            "attention_weights",
            "value_proj",
        ]
        keys = []
        for key, _ in self.get_hf_state_dict().items():
            for t_layer in t_layers:
                if (
                    t_layer in key
                    and key.endswith("weight")
                    and "norm" not in key
                    and "LayerNorm" not in key
                    and "layer_norm" not in key
                    and "enc_output.1" not in key
                ):
                    keys.append(key)
        keys.extend(
            f"model.{head}.dense.weight"
            for head in ("decoder_global_pointer", "decoder_roor_global_pointer")
        )
        return keys

    @staticmethod
    def _normalize_hf_geometry_keys(state_dict):
        state_dict = dict(state_dict)
        old_key = "model.deform_bin_centers"
        new_key = "model.deform_head.bin_centers"
        if old_key in state_dict:
            if new_key in state_dict:
                raise ValueError("Checkpoint contains both geometry bin-center names")
            state_dict[new_key] = state_dict.pop(old_key)
        return state_dict

    @classmethod
    def _load_pretrained_model(cls, model, state_dict, loaded_keys, **kwargs):
        if kwargs.get("convert_from_hf"):
            # HF tensors have already been transposed to Paddle's [in, out]
            # layout by load_state_dict. Accept the former fused format too.
            if state_dict is not None:
                state_dict = cls._normalize_hf_geometry_keys(state_dict)
                for head in ("decoder_global_pointer", "decoder_roor_global_pointer"):
                    prefix = f"model.{head}."
                    for suffix in ("weight", "bias"):
                        fused = prefix + "dense." + suffix
                        if fused in state_dict:
                            if any(
                                prefix + part + "." + suffix in state_dict
                                for part in ("q_proj", "k_proj")
                            ):
                                raise ValueError(
                                    f"Checkpoint contains fused and split projections for {prefix}"
                                )
                            q, k = paddle.split(state_dict.pop(fused), 2, axis=-1)
                            state_dict[prefix + "q_proj." + suffix] = q
                            state_dict[prefix + "k_proj." + suffix] = k
                # PyTorch BatchNorm counters have no Paddle equivalent.
                state_dict = {
                    k: v
                    for k, v in state_dict.items()
                    if not k.endswith(".num_batches_tracked")
                }
                loaded_keys = list(state_dict)
            expected = model.get_hf_state_dict()
            critical = [
                k for k in expected if "global_pointer." in k or ".deform_head." in k
            ]
            missing = sorted(set(critical) - set(loaded_keys))
            if missing:
                raise ValueError(
                    f"Incomplete PP-DocLayoutV4 checkpoint; missing critical weights: {missing}"
                )
        return super()._load_pretrained_model(model, state_dict, loaded_keys, **kwargs)

    def set_hf_state_dict(self, state_dict, *args, **kwargs):
        state_dict = self._normalize_hf_geometry_keys(state_dict)
        for head in ("decoder_global_pointer", "decoder_roor_global_pointer"):
            prefix = f"model.{head}."
            for suffix in ("weight", "bias"):
                q, k = (prefix + part + "." + suffix for part in ("q_proj", "k_proj"))
                if q in state_dict or k in state_dict:
                    if q not in state_dict or k not in state_dict:
                        raise ValueError(
                            f"Incomplete query/key projection pair for {prefix}{suffix}"
                        )
                    state_dict[prefix + "dense." + suffix] = paddle.concat(
                        [state_dict.pop(q), state_dict.pop(k)], axis=-1
                    )
        key = "model.deform_head.bin_centers"
        if key in state_dict:
            state_dict["model.deform_bin_centers"] = state_dict.pop(key)
        return super().set_hf_state_dict(
            _apply_rt_detr_key_conversion(state_dict), *args, **kwargs
        )

    def get_hf_state_dict(self, *args, **kwargs):
        state_dict = _reverse_rt_detr_key_conversion(
            super().get_hf_state_dict(*args, **kwargs)
        )
        for head in ("decoder_global_pointer", "decoder_roor_global_pointer"):
            prefix = f"model.{head}."
            for suffix in ("weight", "bias"):
                q, k = paddle.split(
                    state_dict.pop(prefix + "dense." + suffix), 2, axis=-1
                )
                state_dict[prefix + "q_proj." + suffix] = q
                state_dict[prefix + "k_proj." + suffix] = k
        return self._normalize_hf_geometry_keys(state_dict)
