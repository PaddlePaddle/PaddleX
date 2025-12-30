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

from ...common.transformers.transformers import (
    BatchNormHFStateDictMixin,
    PretrainedConfig,
    PretrainedModel,
)
from .rtdetrl_modules.detr_head import DINOHead
from .rtdetrl_modules.hgnet_v2 import PPHGNetV2
from .rtdetrl_modules.hybrid_encoder import HybridEncoder, TransformerLayer
from .rtdetrl_modules.modules.detr_loss import DINOLoss
from .rtdetrl_modules.modules.matchers import HungarianMatcher
from .rtdetrl_modules.modules.utils import bbox_cxcywh_to_xyxy
from .rtdetrl_modules.rtdetr_transformer import RTDETRTransformer

__all__ = ["RTDETR"]


class DETRPostProcess(object):
    __shared__ = ["num_classes", "use_focal_loss", "with_mask"]
    __inject__ = []

    def __init__(
        self,
        num_classes=80,
        num_top_queries=100,
        dual_queries=False,
        dual_groups=0,
        use_focal_loss=False,
        with_mask=False,
        mask_stride=4,
        mask_threshold=0.5,
        use_avg_mask_score=False,
        bbox_decode_type="origin",
    ):
        super(DETRPostProcess, self).__init__()
        assert bbox_decode_type in ["origin", "pad"]

        self.num_classes = num_classes
        self.num_top_queries = num_top_queries
        self.dual_queries = dual_queries
        self.dual_groups = dual_groups
        self.use_focal_loss = use_focal_loss
        self.with_mask = with_mask
        self.mask_stride = mask_stride
        self.mask_threshold = mask_threshold
        self.use_avg_mask_score = use_avg_mask_score
        self.bbox_decode_type = bbox_decode_type

    def _mask_postprocess(self, mask_pred, score_pred):
        mask_score = F.sigmoid(mask_pred)
        mask_pred = (mask_score > self.mask_threshold).astype(mask_score.dtype)
        if self.use_avg_mask_score:
            avg_mask_score = (mask_pred * mask_score).sum([-2, -1]) / (
                mask_pred.sum([-2, -1]) + 1e-6
            )
            score_pred *= avg_mask_score

        return mask_pred.flatten(0, 1).astype("int32"), score_pred

    def __call__(self, head_out, im_shape, scale_factor, pad_shape):
        """
        Decode the bbox and mask.

        Args:
            head_out (tuple): bbox_pred, cls_logit and masks of bbox_head output.
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
            [labels.unsqueeze(-1).astype("float32"), scores.unsqueeze(-1), bbox_pred],
            axis=-1,
        )
        bbox_num = paddle.to_tensor(self.num_top_queries, dtype="int32").tile(
            [bbox_pred.shape[0]]
        )
        bbox_pred = bbox_pred.reshape([-1, 6])
        return bbox_pred, bbox_num, mask_pred


class RTDETRConfig(PretrainedConfig):
    def __init__(
        self,
        backbone,
        HybridEncoder,
        RTDETRTransformer,
        DINOHead,
        DETRPostProcess,
    ):
        if backbone["name"] == "PPHGNetV2":
            self.arch = backbone["arch"]
            self.return_idx = backbone["return_idx"]
            self.freeze_stem_only = backbone["freeze_stem_only"]
            self.freeze_at = backbone["freeze_at"]
            self.freeze_norm = backbone["freeze_norm"]
            self.lr_mult_list = backbone["lr_mult_list"]
        else:
            raise RuntimeError(
                f"There is no dynamic graph implementation for backbone {backbone['name']}."
            )
        self.hidden_dim = HybridEncoder["hidden_dim"]
        self.use_encoder_idx = HybridEncoder["use_encoder_idx"]
        self.num_encoder_layers = HybridEncoder["num_encoder_layers"]
        self.el_d_model = HybridEncoder["encoder_layer"]["d_model"]
        self.el_nhead = HybridEncoder["encoder_layer"]["nhead"]
        self.el_dim_feedforward = HybridEncoder["encoder_layer"]["dim_feedforward"]
        self.el_dropout = HybridEncoder["encoder_layer"]["dropout"]
        self.el_activation = HybridEncoder["encoder_layer"]["activation"]
        self.expansion = HybridEncoder["expansion"]
        self.tf_num_queries = RTDETRTransformer["num_queries"]
        self.tf_position_embed_type = RTDETRTransformer["position_embed_type"]
        self.tf_feat_strides = RTDETRTransformer["feat_strides"]
        self.tf_num_levels = RTDETRTransformer["num_levels"]
        self.tf_nhead = RTDETRTransformer["nhead"]
        self.tf_num_decoder_layers = RTDETRTransformer["num_decoder_layers"]
        self.tf_backbone_feat_channels = RTDETRTransformer["backbone_feat_channels"]
        self.tf_dim_feedforward = RTDETRTransformer["dim_feedforward"]
        self.tf_dropout = RTDETRTransformer["dropout"]
        self.tf_activation = RTDETRTransformer["activation"]
        self.tf_num_denoising = RTDETRTransformer["num_denoising"]
        self.tf_label_noise_ratio = RTDETRTransformer["label_noise_ratio"]
        self.tf_box_noise_scale = RTDETRTransformer["box_noise_scale"]
        self.tf_learnt_init_query = RTDETRTransformer["learnt_init_query"]
        self.loss_coeff = DINOHead["loss"]["loss_coeff"]
        self.aux_loss = DINOHead["loss"]["aux_loss"]
        self.use_vfl = DINOHead["loss"]["use_vfl"]
        self.matcher_coeff = DINOHead["loss"]["matcher"]["matcher_coeff"]
        self.num_top_queries = DETRPostProcess["num_top_queries"]
        self.use_focal_loss = DETRPostProcess["use_focal_loss"]
        self.tensor_parallel_degree = 1


class RTDETR(BatchNormHFStateDictMixin, PretrainedModel):

    config_class = RTDETRConfig

    def __init__(self, config: RTDETRConfig):
        super().__init__(config)

        self.backbone = PPHGNetV2(
            arch=self.config.arch,
            lr_mult_list=self.config.lr_mult_list,
            return_idx=self.config.return_idx,
            freeze_stem_only=self.config.freeze_stem_only,
            freeze_at=self.config.freeze_at,
            freeze_norm=self.config.freeze_norm,
        )
        self.neck = HybridEncoder(
            hidden_dim=self.config.hidden_dim,
            use_encoder_idx=self.config.use_encoder_idx,
            num_encoder_layers=self.config.num_encoder_layers,
            encoder_layer=TransformerLayer(
                d_model=self.config.el_d_model,
                nhead=self.config.el_nhead,
                dim_feedforward=self.config.el_dim_feedforward,
                dropout=self.config.el_dropout,
                activation=self.config.el_activation,
            ),
            expansion=self.config.expansion,
        )
        self.transformer = RTDETRTransformer(
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
        self.head = DINOHead(
            loss=DINOLoss(
                loss_coeff=self.config.loss_coeff,
                aux_loss=self.config.aux_loss,
                use_vfl=self.config.use_vfl,
                matcher=HungarianMatcher(
                    matcher_coeff=self.config.matcher_coeff,
                ),
            )
        )
        self.post_process = DETRPostProcess(
            num_top_queries=self.config.num_top_queries,
            use_focal_loss=self.config.use_focal_loss,
        )

    def forward(self, inputs):
        x = paddle.to_tensor(inputs[1])
        x = self.backbone(x)
        x_neck = self.neck(x)
        x = self.transformer(x_neck)
        preds = self.head(x, x_neck)
        bbox, bbox_num, mask = self.post_process(
            preds,
            paddle.to_tensor(inputs[0]),
            paddle.to_tensor(inputs[2]),
            inputs[1][2:].shape,
        )
        output = [bbox, bbox_num]
        return output

    def get_transpose_weight_keys(self):
        need_to_transpose = []
        all_weight_keys = []
        for name, param in self.neck.named_parameters():
            all_weight_keys.append("neck." + name)
        for name, param in self.transformer.named_parameters():
            all_weight_keys.append("transformer." + name)
        for i in range(len(all_weight_keys)):
            if ("out_proj" in all_weight_keys[i]) and (
                "bias" not in all_weight_keys[i]
            ):
                need_to_transpose.append(all_weight_keys[i])
        return need_to_transpose
