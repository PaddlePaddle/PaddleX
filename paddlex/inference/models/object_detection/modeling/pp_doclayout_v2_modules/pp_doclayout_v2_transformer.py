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
#
# Modified from Deformable-DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Modified from detrex (https://github.com/IDEA-Research/detrex)
# Copyright 2022 The IDEA Authors. All rights reserved.

from __future__ import absolute_import, division, print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..rtdetrl_modules.modules.detr_ops import _get_clones, inverse_sigmoid
from ..rtdetrl_modules.modules.utils import (
    bbox_cxcywh_to_xyxy,
    get_contrastive_denoising_training_group,
)
from ..rtdetrl_modules.rtdetr_transformer import (
    RTDETRTransformer,
    TransformerDecoderLayer,
)
from .reading_order_predictor import ReadingOrderPredictor

__all__ = ["PPDocLayoutTransformer"]


DET_TO_ORDER_MAP = {
    "paragraph_title": ("paragraph_title", 0),
    "image": ("image", 1),
    "table": ("image", 1),
    "chart": ("image", 1),
    "text": ("text", 2),
    "reference": ("text", 2),
    "algorithm": ("text", 2),
    "reference_content": ("text", 2),
    "inline_formula": ("text", 2),
    "number": ("number", 3),
    "abstract": ("abstract", 4),
    "content": ("content", 5),
    "figure_title": ("figure_title", 6),
    "vision_footnote": ("figure_title", 6),
    "formula": ("display_formula", 7),
    "display_formula": ("display_formula", 7),
    "doc_title": ("doc_title", 8),
    "footnote": ("footnote", 9),
    "header": ("header", 10),
    "header_image": ("header", 10),
    "footer": ("footer", 11),
    "footer_image": ("footer", 11),
    "seal": ("seal", 12),
    "formula_number": ("formula_number", 13),
    "aside_text": ("aside_text", 14),
    "vertical_text": ("vertical_text", 15),
}


def get_label_map():
    categories = [
        "abstract",
        "algorithm",
        "aside_text",
        "chart",
        "content",
        "display_formula",
        "doc_title",
        "figure_title",
        "footer",
        "footer_image",
        "footnote",
        "formula_number",
        "header",
        "header_image",
        "image",
        "inline_formula",
        "number",
        "paragraph_title",
        "reference",
        "reference_content",
        "seal",
        "table",
        "text",
        "vertical_text",
        "vision_footnote",
    ]

    label_map = []
    for det_id, det_name in enumerate(categories):
        order_name, order_id = DET_TO_ORDER_MAP[det_name]
        label_map.append((det_id, order_id))

    sorted_label_map = sorted(label_map, key=lambda x: x[0])
    return [i[1] for i in sorted_label_map]


def _get_global_visual_feature(memory, spatial_shapes, level_start_index):
    """
    从 encoder 的 memory 提取全局视觉向量。
    这里用第0层特征图的全局平均池化，形状 [bs, hidden_dim=256]
    """
    bs, _, hidden_dim = memory.shape
    h0, w0 = spatial_shapes[0]
    memory_lvl0 = (
        memory[:, : level_start_index[1], :]
        .reshape([bs, h0, w0, hidden_dim])
        .transpose([0, 3, 1, 2])
    )
    # [bs, C, H, W] -> [bs, C]
    g = F.adaptive_avg_pool2d(memory_lvl0, output_size=1).flatten(1)
    return g  # [bs, 256]


class TransformerDecoder(nn.Layer):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

        # 放在 TransformerDecoder.__init__ 里
        threshold_dict = {
            0: 0.50,  # abstract
            1: 0.50,  # algorithm
            2: 0.50,  # aside_text
            3: 0.50,  # chart
            4: 0.50,  # content
            5: 0.40,  # formula
            6: 0.40,  # doc_title
            7: 0.50,  # figure_title
            8: 0.50,  # footer
            9: 0.50,  # footer
            10: 0.50,  # footnote
            11: 0.50,  # formula_number
            12: 0.50,  # header
            13: 0.50,  # header
            14: 0.50,  # image
            15: 0.40,  # formula
            16: 0.50,  # number
            17: 0.40,  # paragraph_title
            18: 0.50,  # reference
            19: 0.50,  # reference_content
            20: 0.45,  # seal
            21: 0.50,  # table
            22: 0.40,  # text
            23: 0.40,  # text
            24: 0.50,  # vision_footnote
        }

        # 转成 tensor，shape=[25]
        self.class_thresholds = paddle.to_tensor(
            [threshold_dict[i] for i in range(25)], dtype="float32"
        )

        self.class_map = get_label_map()

        self.ro_mask_aug_cfg = dict(
            enable=True,
            prob=0.1,  # 60% 的样本做 mask 增强
            mode_weights={"bernoulli": 0.5, "span": 0.4, "headtail": 0.1},
            bernoulli_p=0.12,
            span_max_ratio=0.35,
            headtail_max_ratio=0.25,
            min_keep=2,
        )

    def _ro_semantic_map_anyshape(self, labels: paddle.Tensor) -> paddle.Tensor:
        idx = paddle.cast(labels, "int64")
        flat = paddle.reshape(idx, [-1])
        _RO_LABEL_MAP = paddle.to_tensor(self.class_map, dtype="int64")
        map_t = paddle.to_tensor(_RO_LABEL_MAP, dtype="int64", stop_gradient=True)
        out = paddle.gather(map_t, flat)
        out = paddle.reshape(out, paddle.shape(idx))

        return out

    def forward(
        self,
        tgt,
        ref_points_unact,
        memory,
        memory_spatial_shapes,
        memory_level_start_index,
        bbox_head,
        score_head,
        reading_order_predictor,
        query_pos_head,
        attn_mask=None,
        memory_mask=None,
        query_pos_head_inv_sig=False,
        gt_meta=None,
    ):

        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []

        ref_points_detach = F.sigmoid(ref_points_unact)
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            if not query_pos_head_inv_sig:
                query_pos_embed = query_pos_head(ref_points_detach)
            else:
                query_pos_embed = query_pos_head(inverse_sigmoid(ref_points_detach))

            output = layer(
                output,
                ref_points_input,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask,
                memory_mask,
                query_pos_embed,
            )

            inter_ref_bbox = F.sigmoid(
                bbox_head[i](output) + inverse_sigmoid(ref_points_detach)
            )
            score_logits = score_head[i](output)

            if self.training:
                dec_out_logits.append(score_logits)
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(
                        F.sigmoid(
                            bbox_head[i](output) + inverse_sigmoid(ref_points_detach)
                        )
                    )
            elif i == self.eval_idx:
                dec_out_logits.append(score_logits)
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = (
                inter_ref_bbox.detach() if self.training else inter_ref_bbox
            )

        bs = output.shape[0]

        if self.training and gt_meta is not None and "gt_bbox" in gt_meta:
            shuffled_gt_bboxes_list = []
            shuffled_gt_labels_list = []
            final_gt_read_order_list = []

            for i in range(bs):
                gt_bboxes = gt_meta["gt_bbox"][i]

                num_gt = gt_bboxes.shape[0]

                gt_read_order = gt_meta["gt_read_order"][i][:num_gt]
                valid_gt_labels = gt_meta["gt_class"][i][:num_gt]

                num_gt = gt_bboxes.shape[0]

                if num_gt > 0:
                    shuffl_indices = paddle.randperm(num_gt)
                    shuffled_gt_bboxes_list.append(
                        paddle.gather(gt_bboxes, shuffl_indices, axis=0)
                    )
                    final_gt_read_order_list.append(
                        paddle.gather(gt_read_order, shuffl_indices, axis=0)
                    )
                    shuffled_gt_labels_list.append(
                        paddle.gather(valid_gt_labels, shuffl_indices, axis=0)
                    )
                else:
                    shuffled_gt_bboxes_list.append(gt_bboxes)
                    final_gt_read_order_list.append(gt_read_order)
                    shuffled_gt_labels_list.append(valid_gt_labels)

            gt_bboxes_list = shuffled_gt_bboxes_list

            batch_boxes_list = []
            for boxes_cxcywh in gt_bboxes_list:
                if boxes_cxcywh.shape[0] > 0:
                    boxes_xyxy = (
                        bbox_cxcywh_to_xyxy(boxes_cxcywh) * 1000
                    )  # 存疑，gt框没有归一化
                    boxes_xyxy = boxes_xyxy.clip(min=0, max=1000)
                    batch_boxes_list.append(boxes_xyxy.astype("int64").numpy().tolist())
                else:
                    batch_boxes_list.append([])

            global_visual = _get_global_visual_feature(
                memory, memory_spatial_shapes, memory_level_start_index
            )
            padded_ro_logits = reading_order_predictor(
                boxes_list=batch_boxes_list,
                labels_list=shuffled_gt_labels_list,
                global_visual=global_visual,  # 仍保留
                global_memory=memory,  # NEW
                global_spatial_shapes=memory_spatial_shapes,  # NEW
                global_level_start_index=memory_level_start_index,  # NEW
            )

            max_gt_len = max(len(b) for b in batch_boxes_list)
            num_order_classes = padded_ro_logits.shape[-1]

            final_fg_indices = paddle.zeros([bs, max_gt_len], dtype="int64")
            final_fg_masks = paddle.zeros([bs, max_gt_len], dtype="bool")

            for i in range(bs):
                num_gt = len(batch_boxes_list[i])
                if num_gt > 0:
                    final_fg_masks[i, :num_gt] = paddle.arange(num_gt)
                    final_fg_masks[i, :num_gt] = True

            out_read_orders = (
                padded_ro_logits,
                final_fg_indices,
                final_fg_masks,
                final_gt_read_order_list,
            )

        else:
            raw_bboxes = paddle.stack(dec_out_bboxes)[0]  # (batch_size, 300, 4)
            bboxes = bbox_cxcywh_to_xyxy(raw_bboxes).astype("float32") * 1000
            bboxes = paddle.clip(bboxes, min=0.0, max=1000.0).astype("int64")
            logits = paddle.stack(dec_out_logits)[0]  # (batch_size, 300, 1)

            # 1. 得到每个框最大logit和对应的类别ID
            probs = F.sigmoid(logits)
            max_probs = paddle.max(probs, axis=-1)  # (batch_size, 300)
            class_ids = paddle.argmax(probs, axis=-1)  # (batch_size, 300)

            # 2. 有效框mask
            inline_formula_id = 15
            thresholds = paddle.index_select(
                self.class_thresholds, class_ids.reshape([-1]), 0
            )
            thresholds = thresholds.reshape(class_ids.shape)
            mask = (
                max_probs >= thresholds
            )  # & (class_ids != inline_formula_id)           # (batch_size, 300)
            mask = mask.astype("int64")

            # 3. 排序，把有效框排前面（无效的自动补0）
            sorted_mask = mask.sort(axis=1, descending=True)
            indices = mask.argsort(axis=1, descending=True)  # (batch_size, 300)

            # 4. 重排类别和boxes
            sorted_class_ids = paddle.take_along_axis(
                class_ids, indices, axis=1
            )  # (batch_size, 300)
            sorted_boxes = paddle.take_along_axis(
                bboxes, indices.unsqueeze(-1).expand(shape=[-1, -1, 4]), axis=1
            )  # (batch_size, 300, 4)
            sorted_raw_boxes = paddle.take_along_axis(
                raw_bboxes, indices.unsqueeze(-1).expand(shape=[-1, -1, 4]), axis=1
            )  # (batch_size, 300, 4)
            sorted_logits = paddle.take_along_axis(
                logits, indices.unsqueeze(-1), axis=1
            )  # (batch_size, 300, 1)

            # 5. 补0
            mask_expand = sorted_mask.unsqueeze(-1).expand(
                shape=[-1, -1, 4]
            )  # (batch_size, 300, 4)

            pad_boxes = sorted_boxes * mask_expand  # 无效框box置零
            pad_class_ids = sorted_class_ids * sorted_mask  # 无效框类别置零

            pad_class_ids = self._ro_semantic_map_anyshape(pad_class_ids)

            order_logits = reading_order_predictor(  # [B, Nq, C_order] [B, 300, 510]
                boxes=pad_boxes,
                labels=pad_class_ids,
                mask=mask,
            )
            order_logits = order_logits[:, :, :300]
        return (
            sorted_raw_boxes.unsqueeze(axis=0),
            sorted_logits.unsqueeze(axis=0),
            order_logits,
        )


class PPDocLayoutTransformer(RTDETRTransformer):
    def __init__(
        self,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        num_decoder_points=4,
        eval_idx=-1,
        reading_order_config=None,
        **kwargs
    ):
        kwargs.setdefault("num_classes", 25)
        super(PPDocLayoutTransformer, self).__init__(**kwargs)

        decoder_layer = TransformerDecoderLayer(
            self.hidden_dim,
            self.nhead,
            dim_feedforward,
            dropout,
            activation,
            self.num_levels,
            num_decoder_points,
        )
        self.decoder = TransformerDecoder(
            self.hidden_dim, decoder_layer, self.num_decoder_layers, eval_idx
        )
        self.reading_order_predictor = ReadingOrderPredictor(reading_order_config)

    def forward(self, feats, pad_mask=None, gt_meta=None, is_teacher=False):
        # input projection and embedding
        (memory, spatial_shapes, level_start_index) = self._get_encoder_input(feats)

        # prepare denoising training
        if self.training:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = (
                get_contrastive_denoising_training_group(
                    gt_meta,
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed.weight,
                    self.num_denoising,
                    self.label_noise_ratio,
                    self.box_noise_scale,
                )
            )
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = (
                None,
                None,
                None,
                None,
            )

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = (
            self._get_decoder_input(
                memory,
                spatial_shapes,
                denoising_class,
                denoising_bbox_unact,
                is_teacher,
            )
        )

        # decoder
        out_bboxes, out_logits, out_read_orders = self.decoder(
            target,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.reading_order_predictor,
            self.query_pos_head,
            attn_mask=attn_mask,
            memory_mask=None,
            query_pos_head_inv_sig=self.query_pos_head_inv_sig,
            gt_meta=gt_meta,
        )
        return (
            out_bboxes,
            out_logits,
            enc_topk_bboxes,
            enc_topk_logits,
            dn_meta,
            out_read_orders,
        )
