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

from __future__ import absolute_import, division, print_function

import paddle
import paddle.nn.functional as F

from .pp_doclayout_v2_modules.pp_doclayout_v2_transformer import PPDocLayoutTransformer
from .rt_detr import RTDETR, DETRPostProcess, RTDETRConfig
from .rtdetrl_modules.modules.utils import bbox_cxcywh_to_xyxy

__all__ = ["PPDocLayoutV2"]


def get_order(order_logits):
    order_scores = paddle.nn.functional.sigmoid(order_logits)
    B, N, _ = order_scores.shape
    one = paddle.ones([N, N], dtype=order_scores.dtype)
    upper = paddle.triu(one, 1)
    lower = paddle.tril(one, -1)
    Q = order_scores * upper + (1.0 - paddle.transpose(order_scores, [0, 2, 1])) * lower
    order_votes = paddle.sum(Q, axis=1)
    order_pointers = paddle.argsort(order_votes, axis=1)
    order_seq = paddle.full(order_pointers.shape, -1, dtype=order_pointers.dtype)
    batch_indices = paddle.arange(B).reshape([-1, 1]).expand([B, N])
    order_seq[batch_indices, order_pointers] = paddle.arange(N).expand([B, N])

    return order_seq, order_votes


class PPDocLayoutPostProcess(DETRPostProcess):
    def __init__(self, **kwargs):
        kwargs.setdefault("num_classes", 25)
        super(PPDocLayoutPostProcess, self).__init__(**kwargs)

    def __call__(self, head_out, order_logits, im_shape, scale_factor, pad_shape):
        """
        Decode the bbox and mask.

        Args:
            head_out (tuple): bbox_pred, cls_logit and masks of bbox_head output.
            order_logits (Tensor): The result from ReadingOrder.
            im_shape (Tensor): The shape of the input image without padding.
            scale_factor (Tensor): The scale factor of the input image.
            pad_shape (Tensor): The shape of the input image with padding.
        Returns:
            bbox_pred (Tensor): The output prediction with shape [N, 6], including
                labels, scores and bboxes. The size of bboxes are corresponding
                to the input image, the bboxes may be used in other branch.
            bbox_num (Tensor): The number of prediction boxes of each batch with
                shape [bs], and is N.
        """
        bboxes, logits, masks = head_out
        if self.dual_queries:
            num_queries = logits.shape[1]
            logits, bboxes = (
                logits[:, : int(num_queries // (self.dual_groups + 1)), :],
                bboxes[:, : int(num_queries // (self.dual_groups + 1)), :],
            )

        bbox_pred = bbox_cxcywh_to_xyxy(bboxes)

        # calculate the original shape of the image
        origin_shape = paddle.floor(im_shape / scale_factor + 0.5)
        img_h, img_w = paddle.split(origin_shape, 2, axis=-1)
        if self.bbox_decode_type == "pad":
            # calculate the shape of the image with padding
            out_shape = pad_shape / im_shape * origin_shape
            out_shape = out_shape.flip(1).tile([1, 2]).unsqueeze(1)
        elif self.bbox_decode_type == "origin":
            out_shape = origin_shape.flip(1).tile([1, 2]).unsqueeze(1)
        else:
            raise Exception(f"Wrong `bbox_decode_type`: {self.bbox_decode_type}.")
        bbox_pred *= out_shape

        scores = (
            F.sigmoid(logits) if self.use_focal_loss else F.softmax(logits)[:, :, :-1]
        )

        pad_order_seq, pad_order_votes = get_order(order_logits)

        if not self.use_focal_loss:
            scores, labels = scores.max(-1), scores.argmax(-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = paddle.topk(scores, self.num_top_queries, axis=-1)
                batch_ind = (
                    paddle.arange(end=scores.shape[0])
                    .unsqueeze(-1)
                    .tile([1, self.num_top_queries])
                )
                index = paddle.stack([batch_ind, index], axis=-1)
                labels = paddle.gather_nd(labels, index)
                bbox_pred = paddle.gather_nd(bbox_pred, index)
                pad_order_seq = paddle.gather_nd(pad_order_seq, index)
                pad_order_votes = paddle.gather_nd(pad_order_votes, index)
        else:
            scores, index = paddle.topk(
                scores.flatten(1), self.num_top_queries, axis=-1
            )
            labels = index % self.num_classes
            index = index // self.num_classes
            batch_ind = (
                paddle.arange(end=scores.shape[0])
                .unsqueeze(-1)
                .tile([1, self.num_top_queries])
            )
            index = paddle.stack([batch_ind, index], axis=-1)
            bbox_pred = paddle.gather_nd(bbox_pred, index)
            pad_order_seq = paddle.gather_nd(pad_order_seq, index)
            pad_order_votes = paddle.gather_nd(pad_order_votes, index)

        mask_pred = None
        if self.with_mask:
            assert masks is not None
            assert masks.shape[0] == 1
            masks = paddle.gather_nd(masks, index)
            if self.bbox_decode_type == "pad":
                masks = F.interpolate(
                    masks,
                    scale_factor=self.mask_stride,
                    mode="bilinear",
                    align_corners=False,
                )
                # TODO: Support prediction with bs>1.
                # remove padding for input image
                h, w = im_shape.astype("int32")[0]
                masks = masks[..., :h, :w]
            # get pred_mask in the original resolution.
            img_h = img_h[0].astype("int32")
            img_w = img_w[0].astype("int32")
            masks = F.interpolate(
                masks, size=[img_h, img_w], mode="bilinear", align_corners=False
            )
            mask_pred, scores = self._mask_postprocess(masks, scores)

        bbox_pred = paddle.concat(
            [
                labels.unsqueeze(-1).astype("float32"),
                scores.unsqueeze(-1),
                bbox_pred,
                pad_order_seq.unsqueeze(-1).astype("float32"),
                pad_order_votes.unsqueeze(-1).astype("float32"),
            ],
            axis=-1,
        )
        bbox_num = paddle.to_tensor(self.num_top_queries, dtype="int32").tile(
            [bbox_pred.shape[0]]
        )
        bbox_pred = bbox_pred.reshape([-1, 8])
        return bbox_pred, bbox_num, mask_pred


class PPDocLayoutV2Config(RTDETRConfig):
    pass


class PPDocLayoutV2(RTDETR):

    config_class = PPDocLayoutV2Config

    def __init__(self, config: PPDocLayoutV2Config):
        super(PPDocLayoutV2, self).__init__(config)

        self.transformer = PPDocLayoutTransformer(
            num_queries=self.config.tf_num_queries,
            position_embed_type=self.config.tf_position_embed_type,
            feat_strides=self.config.tf_feat_strides,
            backbone_feat_channels=self.config.tf_backbone_feat_channels,
            num_levels=self.config.tf_num_levels,
            nhead=self.config.tf_nhead,
            num_decoder_layers=self.config.tf_num_decoder_layers,
            dim_feedforward=self.config.tf_dim_feedforward,
            dropout=self.config.tf_dropout,
            activation=self.config.tf_activation,
            num_denoising=self.config.tf_num_denoising,
            label_noise_ratio=self.config.tf_label_noise_ratio,
            box_noise_scale=self.config.tf_box_noise_scale,
            learnt_init_query=self.config.tf_learnt_init_query,
        )

        self.post_process = PPDocLayoutPostProcess(
            num_top_queries=self.config.num_top_queries,
            use_focal_loss=self.config.use_focal_loss,
        )

    def forward(self, inputs):
        x = paddle.to_tensor(inputs[1])
        x = self.backbone(x)
        x_neck = self.neck(x)
        x = self.transformer(x_neck)
        order_logits = x[-1]
        preds = self.head(x[:-1], x_neck)
        bbox, bbox_num, mask = self.post_process(
            preds,
            order_logits,
            paddle.to_tensor(inputs[0]),
            paddle.to_tensor(inputs[2]),
            inputs[1][2:].shape,
        )

        output = [bbox, bbox_num]
        return output

    def get_transpose_weight_keys(self):
        t_layers = [
            "fc",
            "out_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "linear_1",
            "linear_2",
            "enc_bbox_head",
            "spatial_proj",
            "query",
            "key",
            "value",
            "intermediate",
            "attention",
            "output",
            "relative_head",
            "query_pos_head",
            "enc_score_head",
            "in_proj_weight",
            "linear1",
            "linear2",
            "label_features_projection",
            "reading_order_predictor.encoder.layer",
            "encoder_attn",
            "decoder.bbox_embed",
            "decoder.class_embed",
        ]
        keys = []
        for key, _ in self.get_hf_state_dict().items():
            for t_layer in t_layers:
                if (
                    t_layer in key
                    and key.endswith("weight")
                    and "LayerNorm" not in key
                    and "layer_norm" not in key
                    and "enc_output.1" not in key
                ):
                    keys.append(key)

        return keys

    def set_hf_state_dict(self, state_dict, *args, **kwargs):
        import re

        mapping = {
            # --- Backbone ---
            r"model.backbone.model.embedder.stem(\d+)a.normalization": r"backbone.stem.stem\1a.bn",
            r"model.backbone.model.embedder.stem(\d+)b.normalization": r"backbone.stem.stem\1b.bn",
            r"model.backbone.model.embedder.stem(\d+)a.convolution": r"backbone.stem.stem\1a.conv",
            r"model.backbone.model.embedder.stem(\d+)b.convolution": r"backbone.stem.stem\1b.conv",
            r"model.backbone.model.embedder.stem(\d+).normalization": r"backbone.stem.stem\1.bn",
            r"model.backbone.model.embedder.stem(\d+).convolution": r"backbone.stem.stem\1.conv",
            r"model.backbone.model.encoder.stages.(\d+).blocks.(\d+).layers.(\d+).conv(\d+).normalization": r"backbone.stages.\1.blocks.\2.layers.\3.conv\4.bn",
            r"model.backbone.model.encoder.stages.(\d+).blocks.(\d+).layers.(\d+).conv(\d+).convolution": r"backbone.stages.\1.blocks.\2.layers.\3.conv\4.conv",
            r"model.backbone.model.encoder.stages.(\d+).blocks.(\d+).layers.(\d+).normalization": r"backbone.stages.\1.blocks.\2.layers.\3.bn",
            r"model.backbone.model.encoder.stages.(\d+).blocks.(\d+).layers.(\d+).convolution": r"backbone.stages.\1.blocks.\2.layers.\3.conv",
            r"model.backbone.model.encoder.stages.(\d+).blocks.(\d+).aggregation.0.normalization": r"backbone.stages.\1.blocks.\2.aggregation_squeeze_conv.bn",
            r"model.backbone.model.encoder.stages.(\d+).blocks.(\d+).aggregation.0.convolution": r"backbone.stages.\1.blocks.\2.aggregation_squeeze_conv.conv",
            r"model.backbone.model.encoder.stages.(\d+).blocks.(\d+).aggregation.1.convolution": r"backbone.stages.\1.blocks.\2.aggregation_excitation_conv.conv",
            r"model.backbone.model.encoder.stages.(\d+).blocks.(\d+).aggregation.1.normalization": r"backbone.stages.\1.blocks.\2.aggregation_excitation_conv.bn",
            r"model.backbone.model.encoder.stages.(\d+).downsample.normalization": r"backbone.stages.\1.downsample.bn",
            r"model.backbone.model.encoder.stages.(\d+).downsample.convolution": r"backbone.stages.\1.downsample.conv",
            # --- Decoder ---
            r"model.decoder_input_proj.(\d+).0": r"transformer.input_proj.\1.conv",
            r"model.decoder_input_proj.(\d+).1": r"transformer.input_proj.\1.norm",
            r"model.decoder.layers.(\d+).self_attn_layer_norm": r"transformer.decoder.layers.\1.norm1",
            r"model.decoder.layers.(\d+).encoder_attn_layer_norm": r"transformer.decoder.layers.\1.norm2",
            r"model.decoder.layers.(\d+).final_layer_norm": r"transformer.decoder.layers.\1.norm3",
            r"model.decoder.layers.(\d+).encoder_attn": r"transformer.decoder.layers.\1.cross_attn",
            r"model.decoder.layers.(\d+).fc(\d+)": r"transformer.decoder.layers.\1.linear\2",
            # --- Encoder ---
            r"model.encoder.encoder.(\d+).layers.(\d+).self_attn_layer_norm": r"neck.encoder.\1.layers.\2.norm1",
            r"model.encoder.encoder.(\d+).layers.(\d+).final_layer_norm": r"neck.encoder.\1.layers.\2.norm2",
            r"model.encoder.encoder.(\d+).layers.(\d+).fc(\d+)": r"neck.encoder.\1.layers.\2.linear\3",
            r"model.encoder.encoder.(\d+).layers.(\d+).fc(\d+).bias": r"neck.encoder.\1.layers.\2.norm\3.bias",
            r"model.encoder.fpn_blocks.(\d+).bottlenecks.(\d+).conv(\d+).norm": r"neck.fpn_blocks.\1.bottlenecks.\2.conv\3.bn",
            r"model.encoder.pan_blocks.(\d+).bottlenecks.(\d+).conv(\d+).norm": r"neck.pan_blocks.\1.bottlenecks.\2.conv\3.bn",
            r"model.encoder.fpn_blocks.(\d+).bottlenecks.(\d+).conv(\d+).conv": r"neck.fpn_blocks.\1.bottlenecks.\2.conv\3.conv",
            r"model.encoder.pan_blocks.(\d+).bottlenecks.(\d+).conv(\d+).conv": r"neck.pan_blocks.\1.bottlenecks.\2.conv\3.conv",
            r"model.encoder.fpn_blocks.(\d+).conv(\d+).norm": r"neck.fpn_blocks.\1.conv\2.bn",
            r"model.encoder.pan_blocks.(\d+).conv(\d+).norm": r"neck.pan_blocks.\1.conv\2.bn",
            r"model.encoder.lateral_convs.(\d+).norm": r"neck.lateral_convs.\1.bn",
            r"model.encoder.downsample_convs.(\d+).norm": r"neck.downsample_convs.\1.bn",
            # --- General ---
            "model.backbone.model.encoder.stages": "backbone.stages",
            "model.decoder.layers": "transformer.decoder.layers",
            "model.decoder.bbox_embed": "transformer.dec_bbox_head",
            "model.decoder.class_embed": "transformer.dec_score_head",
            "model.decoder.query_pos_head": "transformer.query_pos_head",
            "reading_order": "transformer.reading_order_predictor",
            "model.encoder_input_proj": "neck.input_proj",
            "model.encoder": "neck",
            "model": "transformer",
        }

        def _convert_key(key):
            for pattern, replacement in mapping.items():
                new_key, n = re.subn(pattern, replacement, key)
                if n > 0:
                    return new_key
            return key

        def _convert_state_dict(state_dict):
            keys = state_dict.keys()
            new_tensors = {}
            for key in keys:
                tensor = state_dict[key]
                new_key = _convert_key(key)

                if "q_proj.weight" in new_key or "q_proj.bias" in new_key:
                    k_proj = state_dict.get(key.replace("q_proj", "k_proj"), None)
                    v_proj = state_dict.get(key.replace("q_proj", "v_proj"), None)
                    if k_proj is not None and v_proj is not None:
                        merged_tensor = paddle.cat([tensor, k_proj, v_proj], dim=-1)
                        merged_key = new_key.replace("q_proj.", "in_proj_")
                        new_tensors[merged_key] = merged_tensor
                else:
                    new_tensors[new_key] = tensor

            return new_tensors

        state_dict = _convert_state_dict(state_dict)
        key_mapping = {}
        rules = self._get_reverse_key_rules()
        for old_key in list(state_dict.keys()):
            for match_key, old_sub, new_sub in rules:
                if match_key in old_key:
                    key_mapping[old_key] = old_key.replace(old_sub, new_sub)
                    break
        for old_key, new_key in key_mapping.items():
            state_dict[new_key] = state_dict.pop(old_key)

        return self.set_state_dict(state_dict, *args, **kwargs)

    def get_hf_state_dict(self, *args, **kwargs):
        import re

        mapping = {
            # --- Backbone ---
            r"backbone\.stem\.stem(\d+)a\.bn": r"model.backbone.model.embedder.stem\1a.normalization",
            r"backbone\.stem\.stem(\d+)b\.bn": r"model.backbone.model.embedder.stem\1b.normalization",
            r"backbone\.stem\.stem(\d+)a\.conv": r"model.backbone.model.embedder.stem\1a.convolution",
            r"backbone\.stem\.stem(\d+)b\.conv": r"model.backbone.model.embedder.stem\1b.convolution",
            r"backbone\.stem\.stem(\d+)\.bn": r"model.backbone.model.embedder.stem\1.normalization",
            r"backbone\.stem\.stem(\d+)\.conv": r"model.backbone.model.embedder.stem\1.convolution",
            r"backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.conv(\d+)\.bn": r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.conv\4.normalization",
            r"backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.conv(\d+)\.conv": r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.conv\4.convolution",
            r"backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.bn": r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.normalization",
            r"backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.conv\b": r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.convolution",
            r"backbone\.stages\.(\d+)\.blocks\.(\d+)\.aggregation_squeeze_conv\.bn": r"model.backbone.model.encoder.stages.\1.blocks.\2.aggregation.0.normalization",
            r"backbone\.stages\.(\d+)\.blocks\.(\d+)\.aggregation_squeeze_conv\.conv": r"model.backbone.model.encoder.stages.\1.blocks.\2.aggregation.0.convolution",
            r"backbone\.stages\.(\d+)\.blocks\.(\d+)\.aggregation_excitation_conv\.conv": r"model.backbone.model.encoder.stages.\1.blocks.\2.aggregation.1.convolution",
            r"backbone\.stages\.(\d+)\.blocks\.(\d+)\.aggregation_excitation_conv\.bn": r"model.backbone.model.encoder.stages.\1.blocks.\2.aggregation.1.normalization",
            r"backbone\.stages\.(\d+)\.downsample\.bn": r"model.backbone.model.encoder.stages.\1.downsample.normalization",
            r"backbone\.stages\.(\d+)\.downsample\.conv": r"model.backbone.model.encoder.stages.\1.downsample.convolution",
            # --- Decoder ---
            r"transformer\.input_proj\.(\d+)\.conv": r"model.decoder_input_proj.\1.0",
            r"transformer\.input_proj\.(\d+)\.norm": r"model.decoder_input_proj.\1.1",
            r"transformer\.decoder\.layers\.(\d+)\.norm1": r"model.decoder.layers.\1.self_attn_layer_norm",
            r"transformer\.decoder\.layers\.(\d+)\.norm2": r"model.decoder.layers.\1.encoder_attn_layer_norm",
            r"transformer\.decoder\.layers\.(\d+)\.norm3": r"model.decoder.layers.\1.final_layer_norm",
            r"transformer\.decoder\.layers\.(\d+)\.cross_attn": r"model.decoder.layers.\1.encoder_attn",
            r"transformer\.decoder\.layers\.(\d+)\.linear(\d+)": r"model.decoder.layers.\1.fc\2",
            # --- Encoder ---
            r"neck\.encoder\.(\d+)\.layers\.(\d+)\.norm1": r"model.encoder.encoder.\1.layers.\2.self_attn_layer_norm",
            r"neck\.encoder\.(\d+)\.layers\.(\d+)\.norm2": r"model.encoder.encoder.\1.layers.\2.final_layer_norm",
            r"neck\.encoder\.(\d+)\.layers\.(\d+)\.linear(\d+)": r"model.encoder.encoder.\1.layers.\2.fc\3",
            r"neck\.encoder\.(\d+)\.layers\.(\d+)\.norm(\d+)\.bias": r"model.encoder.encoder.\1.layers.\2.fc\3.bias",
            r"neck\.fpn_blocks\.(\d+)\.bottlenecks\.(\d+)\.conv(\d+)\.bn": r"model.encoder.fpn_blocks.\1.bottlenecks.\2.conv\3.norm",
            r"neck\.pan_blocks\.(\d+)\.bottlenecks\.(\d+)\.conv(\d+)\.bn": r"model.encoder.pan_blocks.\1.bottlenecks.\2.conv\3.norm",
            r"neck\.fpn_blocks\.(\d+)\.bottlenecks\.(\d+)\.conv(\d+)\.conv": r"model.encoder.fpn_blocks.\1.bottlenecks.\2.conv\3.conv",
            r"neck\.pan_blocks\.(\d+)\.bottlenecks\.(\d+)\.conv(\d+)\.conv": r"model.encoder.pan_blocks.\1.bottlenecks.\2.conv\3.conv",
            r"neck\.fpn_blocks\.(\d+)\.conv(\d+)\.bn": r"model.encoder.fpn_blocks.\1.conv\2.norm",
            r"neck\.pan_blocks\.(\d+)\.conv(\d+)\.bn": r"model.encoder.pan_blocks.\1.conv\2.norm",
            r"neck\.lateral_convs\.(\d+)\.bn": r"model.encoder.lateral_convs.\1.norm",
            r"neck\.downsample_convs\.(\d+)\.bn": r"model.encoder.downsample_convs.\1.norm",
            # --- General ---
            "backbone.stages": "model.backbone.model.encoder.stages",
            "transformer.decoder.layers": "model.decoder.layers",
            "transformer.dec_bbox_head": "model.decoder.bbox_embed",
            "transformer.dec_score_head": "model.decoder.class_embed",
            "transformer.query_pos_head": "model.decoder.query_pos_head",
            "transformer.reading_order_predictor": "reading_order",
            "transformer": "model",
            "neck.input_proj": "model.encoder_input_proj",
            "neck": "model.encoder",
        }

        def _convert_key(key):
            for pattern, replacement in mapping.items():
                new_key, n = re.subn(pattern, replacement, key)
                if n > 0:
                    return new_key
            return key

        def _split_linear(tensor, key):
            encoder_hidden_dim = 256
            if "in_proj_weight" in key:
                q = key.replace("in_proj_weight", "q_proj.weight")
                q_tensor = tensor[:encoder_hidden_dim, :].clone()
                k = key.replace("in_proj_weight", "k_proj.weight")
                k_tensor = tensor[
                    encoder_hidden_dim : 2 * encoder_hidden_dim, :
                ].clone()
                v = key.replace("in_proj_weight", "v_proj.weight")
                v_tensor = tensor[-encoder_hidden_dim:, :].clone()
            elif "in_proj_bias" in key:
                q = key.replace("in_proj_bias", "q_proj.bias")
                q_tensor = tensor[:encoder_hidden_dim].clone()
                k = key.replace("in_proj_bias", "k_proj.bias")
                k_tensor = tensor[encoder_hidden_dim : 2 * encoder_hidden_dim].clone()
                v = key.replace("in_proj_bias", "v_proj.bias")
                v_tensor = tensor[-encoder_hidden_dim:].clone()

            return q, k, v, q_tensor, k_tensor, v_tensor

        def _convert_state_dict(current_state_dict):
            keys = current_state_dict.keys()
            new_tensors = {}
            for key in keys:
                tensor = current_state_dict[key]
                new_key = _convert_key(key)

                if "in_proj_weight" in new_key or "in_proj_bias" in new_key:
                    q, k, v, q_tensor, k_tensor, v_tensor = _split_linear(
                        tensor, new_key
                    )
                    new_tensors[q] = q_tensor
                    new_tensors[k] = k_tensor
                    new_tensors[v] = v_tensor
                else:
                    new_tensors[new_key] = tensor
            return new_tensors

        model_state_dict = self.state_dict(*args, **kwargs)
        hf_state_dict = {}
        rules = self._get_forward_key_rules()
        for old_key, value in model_state_dict.items():
            new_key = old_key
            for match_key, old_sub, new_sub in rules:
                if match_key in old_key:
                    new_key = old_key.replace(old_sub, new_sub)
                    break
            hf_state_dict[new_key] = value

        hf_state_dict = _convert_state_dict(hf_state_dict)
        return hf_state_dict
