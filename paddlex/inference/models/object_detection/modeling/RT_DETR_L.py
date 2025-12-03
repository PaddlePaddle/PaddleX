# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import inspect

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn import Conv2D, BatchNorm2D, ReLU, AdaptiveAvgPool2D, MaxPool2D
from paddle.nn.initializer import KaimingNormal, Constant
from paddle.regularizer import L2Decay

from ...common.transformers.transformers import PretrainedConfig, PretrainedModel
from .detr_head import DINOHead
from .hgnet_v2 import PPHGNetV2
from .hybrid_encoder import HybridEncoder, TransformerLayer
from .modules.csp_darknet import BaseConv
from .modules.cspresnet import RepVggBlock
from .modules.detr_loss import DINOLoss
from .modules.detr_ops import ShapeSpec
from .modules.initializer import linear_init_, conv_init_, xavier_uniform_, normal_
from .modules.layers import MultiHeadAttention, _convert_attention_mask
from .modules.matchers import HungarianMatcher
from .modules.ops import get_act_fn
from .modules.position_encoding import PositionEmbedding
from .modules.detr_ops import _get_clones
from .modules.utils import bbox_cxcywh_to_xyxy
from .rtdetr_transformer import RTDETRTransformer


__all__ = ['RTDETRL']


class DETRPostProcess(object):
    __shared__ = ['num_classes', 'use_focal_loss', 'with_mask']
    __inject__ = []

    def __init__(self,
                 num_classes=80,
                 num_top_queries=100,
                 dual_queries=False,
                 dual_groups=0,
                 use_focal_loss=False,
                 with_mask=False,
                 mask_stride=4,
                 mask_threshold=0.5,
                 use_avg_mask_score=False,
                 bbox_decode_type='origin'):
        super(DETRPostProcess, self).__init__()
        assert bbox_decode_type in ['origin', 'pad']

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
                mask_pred.sum([-2, -1]) + 1e-6)
            score_pred *= avg_mask_score

        return mask_pred.flatten(0, 1).astype('int32'), score_pred

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
            logits, bboxes = logits[:, :int(num_queries // (self.dual_groups + 1)), :], \
                             bboxes[:, :int(num_queries // (self.dual_groups + 1)), :]

        bbox_pred = bbox_cxcywh_to_xyxy(bboxes)
        # calculate the original shape of the image
        origin_shape = paddle.floor(im_shape / scale_factor + 0.5)
        img_h, img_w = paddle.split(origin_shape, 2, axis=-1)
        if self.bbox_decode_type == 'pad':
            # calculate the shape of the image with padding
            out_shape = pad_shape / im_shape * origin_shape
            out_shape = out_shape.flip(1).tile([1, 2]).unsqueeze(1)
        elif self.bbox_decode_type == 'origin':
            out_shape = origin_shape.flip(1).tile([1, 2]).unsqueeze(1)
        else:
            raise Exception(
                f'Wrong `bbox_decode_type`: {self.bbox_decode_type}.')
        bbox_pred *= out_shape

        scores = F.sigmoid(logits) if self.use_focal_loss else F.softmax(
            logits)[:, :, :-1]

        if not self.use_focal_loss:
            scores, labels = scores.max(-1), scores.argmax(-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = paddle.topk(
                    scores, self.num_top_queries, axis=-1)
                batch_ind = paddle.arange(
                    end=scores.shape[0]).unsqueeze(-1).tile(
                        [1, self.num_top_queries])
                index = paddle.stack([batch_ind, index], axis=-1)
                labels = paddle.gather_nd(labels, index)
                bbox_pred = paddle.gather_nd(bbox_pred, index)
        else:
            scores, index = paddle.topk(
                scores.flatten(1), self.num_top_queries, axis=-1)
            labels = index % self.num_classes
            index = index // self.num_classes
            batch_ind = paddle.arange(end=scores.shape[0]).unsqueeze(-1).tile(
                [1, self.num_top_queries])
            index = paddle.stack([batch_ind, index], axis=-1)
            bbox_pred = paddle.gather_nd(bbox_pred, index)

        mask_pred = None
        if self.with_mask:
            assert masks is not None
            assert masks.shape[0] == 1
            masks = paddle.gather_nd(masks, index)
            if self.bbox_decode_type == 'pad':
                masks = F.interpolate(
                    masks,
                    scale_factor=self.mask_stride,
                    mode="bilinear",
                    align_corners=False)
                # TODO: Support prediction with bs>1.
                # remove padding for input image
                h, w = im_shape.astype('int32')[0]
                masks = masks[..., :h, :w]
            # get pred_mask in the original resolution.
            img_h = img_h[0].astype('int32')
            img_w = img_w[0].astype('int32')
            masks = F.interpolate(
                masks,
                size=[img_h, img_w],
                mode="bilinear",
                align_corners=False)
            mask_pred, scores = self._mask_postprocess(masks, scores)

        bbox_pred = paddle.concat(
            [
                labels.unsqueeze(-1).astype('float32'), scores.unsqueeze(-1),
                bbox_pred
            ],
            axis=-1)
        bbox_num = paddle.to_tensor(
            self.num_top_queries, dtype='int32').tile([bbox_pred.shape[0]])
        bbox_pred = bbox_pred.reshape([-1, 6])
        return bbox_pred, bbox_num, mask_pred


class RTDETRLConfig(PretrainedConfig):
    def __init__(self, config: PretrainedConfig):
        self.config = config
    
    def init_block_from_config(self, config, target_class, other_param={}):
        sig = inspect.signature(target_class.__init__)
        kwargs = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            if param_name in config:
                value = config[param_name]
                if isinstance(value, str):
                    if (type(value) is dict) and ("name" in value):
                        continue
                    elif value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
                    elif value.lower() == "name":
                        continue
                kwargs[param_name] = value
        for key, value in other_param.items():
            kwargs[key] = value
        return target_class(**kwargs)
    
    def init_PPHGNetV2(self):
        return self.init_block_from_config(self.config.PPHGNetV2, PPHGNetV2)
    
    def init_HybridEncoder(self):
        self.TransformerLayer = self.init_block_from_config(
            self.config.HybridEncoder["encoder_layer"], 
            TransformerLayer
        )
        return self.init_block_from_config(
            self.config.HybridEncoder,
            HybridEncoder,
            other_param={
                "encoder_layer": self.TransformerLayer
            }
        )

    def init_RTDETRTransformer(self):
        return self.init_block_from_config(
            self.config.RTDETRTransformer, 
            RTDETRTransformer
        )
    
    def init_DINOHead(self):
        self.matcher = self.init_block_from_config(
            self.config.DINOHead["loss"]["matcher"], 
            HungarianMatcher
        )
        self.loss = self.init_block_from_config(
            self.config.DINOHead["loss"],
            DINOLoss,
            other_param={
                "matcher": self.matcher
            }
        )
        return self.init_block_from_config(
            self.config.DINOHead,
            DINOHead,
            other_param={
                "loss": self.loss
            }
        )
    
    def init_DETRPostProcess(self):
        return self.init_block_from_config(
            self.config.DETRPostProcess,
            DETRPostProcess
        )


class RTDETRL(PretrainedModel):

    config_class = PretrainedConfig

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.rtdetrl_config = RTDETRLConfig(config)
        self.backbone = self.rtdetrl_config.init_PPHGNetV2()
        self.neck = self.rtdetrl_config.init_HybridEncoder()
        self.transformer = self.rtdetrl_config.init_RTDETRTransformer()
        self.head = self.rtdetrl_config.init_DINOHead()
        self.post_process = self.rtdetrl_config.init_DETRPostProcess()

    def forward(self, inputs):
        x = paddle.to_tensor(inputs[1])
        x = self.backbone(x)
        x_neck = self.neck(x)
        x = self.transformer(x_neck)
        preds = self.head(x, x_neck)
        bbox, bbox_num, mask = self.post_process(
                    preds, paddle.to_tensor(inputs[0]), paddle.to_tensor(inputs[2]),
                    inputs[1][2:].shape)
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
                if (("out_proj" in all_weight_keys[i]) and 
                    ("bias" not in all_weight_keys[i])):
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