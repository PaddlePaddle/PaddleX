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

import copy
import math
from collections import namedtuple

import paddle
import paddle.nn as nn


class ShapeSpec(namedtuple("_ShapeSpec", ["channels", "height", "width", "stride"])):
    def __new__(cls, channels=None, height=None, width=None, stride=None):
        return super(ShapeSpec, cls).__new__(cls, channels, height, width, stride)


def delta2bbox(deltas, boxes, weights=[1.0, 1.0, 1.0, 1.0], max_shape=None):
    """Decode deltas to boxes. Used in RCNNBox,CascadeHead,RCNNHead,RetinaHead.
    Note: return tensor shape [n,1,4]
        If you want to add a reshape, please add after the calling code instead of here.
    """
    clip_scale = math.log(1000.0 / 16)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = deltas[:, 0::4] / wx
    dy = deltas[:, 1::4] / wy
    dw = deltas[:, 2::4] / ww
    dh = deltas[:, 3::4] / wh
    # Prevent sending too large values into paddle.exp()
    dw = paddle.clip(dw, max=clip_scale)
    dh = paddle.clip(dh, max=clip_scale)

    pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
    pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
    pred_w = paddle.exp(dw) * widths.unsqueeze(1)
    pred_h = paddle.exp(dh) * heights.unsqueeze(1)

    pred_boxes = []
    pred_boxes.append(pred_ctr_x - 0.5 * pred_w)
    pred_boxes.append(pred_ctr_y - 0.5 * pred_h)
    pred_boxes.append(pred_ctr_x + 0.5 * pred_w)
    pred_boxes.append(pred_ctr_y + 0.5 * pred_h)
    pred_boxes = paddle.stack(pred_boxes, axis=-1)

    if max_shape is not None:
        pred_boxes[..., 0::2] = pred_boxes[..., 0::2].clip(min=0, max=max_shape[1])
        pred_boxes[..., 1::2] = pred_boxes[..., 1::2].clip(min=0, max=max_shape[0])
    return pred_boxes


def _get_clones(module, N):
    return nn.LayerList([copy.deepcopy(module) for _ in range(N)])


def inverse_sigmoid(x, eps=1e-5):
    x = x.clip(min=0.0, max=1.0)
    return paddle.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def get_valid_ratio(mask):
    _, H, W = mask.shape
    valid_ratio_h = paddle.sum(mask[:, :, 0], 1) / H
    valid_ratio_w = paddle.sum(mask[:, 0, :], 1) / W
    # [b, 2]
    return paddle.stack([valid_ratio_w, valid_ratio_h], -1)


def get_sine_pos_embed(
    pos_tensor, num_pos_feats=128, temperature=10000, exchange_xy=True
):
    """generate sine position embedding from a position tensor

    Args:
        pos_tensor (Tensor): Shape as `(None, n)`.
        num_pos_feats (int): projected shape for each float in the tensor. Default: 128
        temperature (int): The temperature used for scaling
            the position embedding. Default: 10000.
        exchange_xy (bool, optional): exchange pos x and pos y. \
            For example, input tensor is `[x, y]`, the results will  # noqa
            be `[pos(y), pos(x)]`. Defaults: True.

    Returns:
        Tensor: Returned position embedding  # noqa
        with shape `(None, n * num_pos_feats)`.
    """
    scale = 2.0 * math.pi
    dim_t = 2.0 * paddle.floor_divide(paddle.arange(num_pos_feats), paddle.to_tensor(2))
    dim_t = scale / temperature ** (dim_t / num_pos_feats)

    def sine_func(x):
        x *= dim_t
        return paddle.stack((x[:, :, 0::2].sin(), x[:, :, 1::2].cos()), axis=3).flatten(
            2
        )

    pos_res = [sine_func(x) for x in pos_tensor.split(pos_tensor.shape[-1], -1)]
    if exchange_xy:
        pos_res[0], pos_res[1] = pos_res[1], pos_res[0]
    pos_res = paddle.concat(pos_res, axis=2)
    return pos_res


def bbox_iou(box1, box2, giou=False, diou=False, ciou=False, eps=1e-9):
    """calculate the iou of box1 and box2

    Args:
        box1 (list): [x, y, w, h], all have the shape [b, na, h, w, 1]
        box2 (list): [x, y, w, h], all have the shape [b, na, h, w, 1]
        giou (bool): whether use giou or not, default False
        diou (bool): whether use diou or not, default False
        ciou (bool): whether use ciou or not, default False
        eps (float): epsilon to avoid divide by zero

    Return:
        iou (Tensor): iou of box1 and box1, with the shape [b, na, h, w, 1]
    """
    px1, py1, px2, py2 = box1
    gx1, gy1, gx2, gy2 = box2
    x1 = paddle.maximum(px1, gx1)
    y1 = paddle.maximum(py1, gy1)
    x2 = paddle.minimum(px2, gx2)
    y2 = paddle.minimum(py2, gy2)

    overlap = ((x2 - x1).clip(0)) * ((y2 - y1).clip(0))

    area1 = (px2 - px1) * (py2 - py1)
    area1 = area1.clip(0)

    area2 = (gx2 - gx1) * (gy2 - gy1)
    area2 = area2.clip(0)

    union = area1 + area2 - overlap + eps
    iou = overlap / union

    if giou or ciou or diou:
        # convex w, h
        cw = paddle.maximum(px2, gx2) - paddle.minimum(px1, gx1)
        ch = paddle.maximum(py2, gy2) - paddle.minimum(py1, gy1)
        if giou:
            c_area = cw * ch + eps
            return iou - (c_area - union) / c_area
        else:
            # convex diagonal squared
            c2 = cw**2 + ch**2 + eps
            # center distance
            rho2 = ((px1 + px2 - gx1 - gx2) ** 2 + (py1 + py2 - gy1 - gy2) ** 2) / 4
            if diou:
                return iou - rho2 / c2
            else:
                w1, h1 = px2 - px1, py2 - py1 + eps
                w2, h2 = gx2 - gx1, gy2 - gy1 + eps
                delta = paddle.atan(w1 / h1) - paddle.atan(w2 / h2)
                v = (4 / math.pi**2) * paddle.pow(delta, 2)
                alpha = v / (1 + eps - iou + v)
                alpha.stop_gradient = True
                return iou - (rho2 / c2 + v * alpha)
    else:
        return iou
