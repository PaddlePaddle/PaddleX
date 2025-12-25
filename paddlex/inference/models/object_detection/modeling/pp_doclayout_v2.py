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
            "channelwise",
            "mapper_crp",
            "mapper_sca",
            ".mapper.",
            "txt_mapper",
            "txt_pooled_mapper",
            "clip_img_mapper",
            "kv_mapper",
            "clip_mapper",
            "out_proj",
            # "patch_embedding",
            "q_proj",
            "k_proj",
            "v_proj",
            "lm_head",
            "gate_proj",
            "up_proj",
            "down_proj",
            "o_proj",
            "lm_head",
            "linear_1",
            "linear_2",
            # doclayout
            "dec_bbox_head",
            "dec_score_head",
            "enc_bbox_head",
            "spatial_proj",
            "query",
            "key",
            "value",
            "intermediate",
            "attention",
            "output",
            "global_agg",
            "visual_features_projection",
            "relative_head",
            "global_visual_proj",
            "query_pos_head",
            "enc_score_head",
            "cross_attn",
            "out_proj",
            "in_proj_weight",
            "linear1",
            "linear2",
            "label_features_projection",
            "reading_order_predictor.encoder.layer",
        ]
        keys = []
        for key, _ in self.get_hf_state_dict().items():
            for t_layer in t_layers:
                if (
                    t_layer in key
                    and key.endswith("weight")
                    and "LayerNorm" not in key
                    and "enc_output.1" not in key
                ):
                    # if "enc_output" in key:
                    # breakpoint()
                    keys.append(key)

        return keys

    # def get_hf_state_dict(self):
