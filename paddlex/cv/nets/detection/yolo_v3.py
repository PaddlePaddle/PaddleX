# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

import paddle
from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
from collections import OrderedDict
from .ops import MultiClassNMS, MultiClassSoftNMS, MatrixNMS
from .ops import DropBlock
from .loss.yolo_loss import YOLOv3Loss
from .loss.iou_loss import IouLoss
from .loss.iou_aware_loss import IouAwareLoss
from .iou_aware import get_iou_aware_score
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence


class YOLOv3:
    def __init__(
            self,
            backbone,
            mode='train',
            # YOLOv3Head
            num_classes=80,
            anchors=None,
            anchor_masks=None,
            coord_conv=False,
            iou_aware=False,
            iou_aware_factor=0.4,
            scale_x_y=1.,
            spp=False,
            drop_block=False,
            use_matrix_nms=False,
            # YOLOv3Loss
            batch_size=8,
            ignore_threshold=0.7,
            label_smooth=False,
            use_fine_grained_loss=False,
            use_iou_loss=False,
            iou_loss_weight=2.5,
            iou_aware_loss_weight=1.0,
            max_height=608,
            max_width=608,
            # NMS
            nms_score_threshold=0.01,
            nms_topk=1000,
            nms_keep_topk=100,
            nms_iou_threshold=0.45,
            fixed_input_shape=None):
        if anchors is None:
            anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                       [59, 119], [116, 90], [156, 198], [373, 326]]
        if anchor_masks is None:
            anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self._parse_anchors(anchors)
        self.mode = mode
        self.num_classes = num_classes
        self.backbone = backbone
        self.norm_decay = 0.0
        self.prefix_name = ''
        self.use_fine_grained_loss = use_fine_grained_loss
        self.fixed_input_shape = fixed_input_shape
        self.coord_conv = coord_conv
        self.iou_aware = iou_aware
        self.iou_aware_factor = iou_aware_factor
        self.scale_x_y = scale_x_y
        self.use_spp = spp
        self.drop_block = drop_block

        if use_matrix_nms:
            self.nms = MatrixNMS(
                background_label=-1,
                keep_top_k=nms_keep_topk,
                normalized=False,
                score_threshold=nms_score_threshold,
                post_threshold=0.01)
        else:
            self.nms = MultiClassNMS(
                background_label=-1,
                keep_top_k=nms_keep_topk,
                nms_threshold=nms_iou_threshold,
                nms_top_k=nms_topk,
                normalized=False,
                score_threshold=nms_score_threshold)
        self.iou_loss = None
        self.iou_aware_loss = None
        if use_iou_loss:
            self.iou_loss = IouLoss(
                loss_weight=iou_loss_weight,
                max_height=max_height,
                max_width=max_width)
        if iou_aware:
            self.iou_aware_loss = IouAwareLoss(
                loss_weight=iou_aware_loss_weight,
                max_height=max_height,
                max_width=max_width)
        self.yolo_loss = YOLOv3Loss(
            batch_size=batch_size,
            ignore_thresh=ignore_threshold,
            scale_x_y=scale_x_y,
            label_smooth=label_smooth,
            use_fine_grained_loss=self.use_fine_grained_loss,
            iou_loss=self.iou_loss,
            iou_aware_loss=self.iou_aware_loss)
        self.conv_block_num = 2
        self.block_size = 3
        self.keep_prob = 0.9
        self.downsample = [32, 16, 8]
        self.clip_bbox = True

    def _head(self, input, is_train=True):
        outputs = []

        # get last out_layer_num blocks in reverse order
        out_layer_num = len(self.anchor_masks)
        blocks = input[-1:-out_layer_num - 1:-1]

        route = None
        for i, block in enumerate(blocks):
            if i > 0:  # perform concat in first 2 detection_block
                block = fluid.layers.concat(input=[route, block], axis=1)
            route, tip = self._detection_block(
                block,
                channel=64 * (2**out_layer_num) // (2**i),
                is_first=i == 0,
                is_test=(not is_train),
                conv_block_num=self.conv_block_num,
                name=self.prefix_name + "yolo_block.{}".format(i))

            # out channel number = mask_num * (5 + class_num)
            if self.iou_aware:
                num_filters = len(self.anchor_masks[i]) * (
                    self.num_classes + 6)
            else:
                num_filters = len(self.anchor_masks[i]) * (
                    self.num_classes + 5)
            with fluid.name_scope('yolo_output'):
                block_out = fluid.layers.conv2d(
                    input=tip,
                    num_filters=num_filters,
                    filter_size=1,
                    stride=1,
                    padding=0,
                    act=None,
                    param_attr=ParamAttr(
                        name=self.prefix_name +
                        "yolo_output.{}.conv.weights".format(i)),
                    bias_attr=ParamAttr(
                        regularizer=L2Decay(0.),
                        name=self.prefix_name +
                        "yolo_output.{}.conv.bias".format(i)))
                outputs.append(block_out)

            if i < len(blocks) - 1:
                # do not perform upsample in the last detection_block
                route = self._conv_bn(
                    input=route,
                    ch_out=256 // (2**i),
                    filter_size=1,
                    stride=1,
                    padding=0,
                    is_test=(not is_train),
                    name=self.prefix_name + "yolo_transition.{}".format(i))
                # upsample
                route = self._upsample(route)

        return outputs

    def _parse_anchors(self, anchors):
        self.anchors = []
        self.mask_anchors = []

        assert len(anchors) > 0, "ANCHORS not set."
        assert len(self.anchor_masks) > 0, "ANCHOR_MASKS not set."

        for anchor in anchors:
            assert len(anchor) == 2, "anchor {} len should be 2".format(anchor)
            self.anchors.extend(anchor)

        anchor_num = len(anchors)
        for masks in self.anchor_masks:
            self.mask_anchors.append([])
            for mask in masks:
                assert mask < anchor_num, "anchor mask index overflow"
                self.mask_anchors[-1].extend(anchors[mask])

    def _create_tensor_from_numpy(self, numpy_array):
        paddle_array = fluid.layers.create_global_var(
            shape=numpy_array.shape, value=0., dtype=numpy_array.dtype)
        fluid.layers.assign(numpy_array, paddle_array)
        return paddle_array

    def _add_coord(self, input, is_test=True):
        if not self.coord_conv:
            return input

        # NOTE: here is used for exporting model for TensorRT inference,
        #       only support batch_size=1 for input shape should be fixed,
        #       and we create tensor with fixed shape from numpy array
        if is_test and input.shape[2] > 0 and input.shape[3] > 0:
            batch_size = 1
            grid_x = int(input.shape[3])
            grid_y = int(input.shape[2])
            idx_i = np.array(
                [[i / (grid_x - 1) * 2.0 - 1 for i in range(grid_x)]],
                dtype='float32')
            gi_np = np.repeat(idx_i, grid_y, axis=0)
            gi_np = np.reshape(gi_np, newshape=[1, 1, grid_y, grid_x])
            gi_np = np.tile(gi_np, reps=[batch_size, 1, 1, 1])

            x_range = self._create_tensor_from_numpy(gi_np.astype(np.float32))
            x_range.stop_gradient = True
            y_range = self._create_tensor_from_numpy(
                gi_np.transpose([0, 1, 3, 2]).astype(np.float32))
            y_range.stop_gradient = True

        # NOTE: in training mode, H and W is variable for random shape,
        #       implement add_coord with shape as Variable
        else:
            input_shape = fluid.layers.shape(input)
            b = input_shape[0]
            h = input_shape[2]
            w = input_shape[3]

            x_range = fluid.layers.range(0, w, 1, 'float32') / ((w - 1.) / 2.)
            x_range = x_range - 1.
            x_range = fluid.layers.unsqueeze(x_range, [0, 1, 2])
            x_range = fluid.layers.expand(x_range, [b, 1, h, 1])
            x_range.stop_gradient = True
            y_range = fluid.layers.transpose(x_range, [0, 1, 3, 2])
            y_range.stop_gradient = True

        return fluid.layers.concat([input, x_range, y_range], axis=1)

    def _conv_bn(self,
                 input,
                 ch_out,
                 filter_size,
                 stride,
                 padding,
                 act='leaky',
                 is_test=False,
                 name=None):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            act=None,
            param_attr=ParamAttr(name=name + '.conv.weights'),
            bias_attr=False)
        bn_name = name + '.bn'
        bn_param_attr = ParamAttr(
            regularizer=L2Decay(self.norm_decay), name=bn_name + '.scale')
        bn_bias_attr = ParamAttr(
            regularizer=L2Decay(self.norm_decay), name=bn_name + '.offset')
        out = fluid.layers.batch_norm(
            input=conv,
            act=None,
            is_test=is_test,
            param_attr=bn_param_attr,
            bias_attr=bn_bias_attr,
            moving_mean_name=bn_name + '.mean',
            moving_variance_name=bn_name + '.var')
        if act == 'leaky':
            out = fluid.layers.leaky_relu(x=out, alpha=0.1)
        return out

    def _spp_module(self, input, is_test=True, name=""):
        output1 = input
        output2 = fluid.layers.pool2d(
            input=output1,
            pool_size=5,
            pool_stride=1,
            pool_padding=2,
            ceil_mode=False,
            pool_type='max')
        output3 = fluid.layers.pool2d(
            input=output1,
            pool_size=9,
            pool_stride=1,
            pool_padding=4,
            ceil_mode=False,
            pool_type='max')
        output4 = fluid.layers.pool2d(
            input=output1,
            pool_size=13,
            pool_stride=1,
            pool_padding=6,
            ceil_mode=False,
            pool_type='max')
        output = fluid.layers.concat(
            input=[output1, output2, output3, output4], axis=1)
        return output

    def _upsample(self, input, scale=2, name=None):
        out = fluid.layers.resize_nearest(
            input=input, scale=float(scale), name=name, align_corners=False)
        return out

    def _detection_block(self,
                         input,
                         channel,
                         conv_block_num=2,
                         is_first=False,
                         is_test=True,
                         name=None):
        assert channel % 2 == 0, \
            "channel {} cannot be divided by 2 in detection block {}" \
            .format(channel, name)

        conv = input
        for j in range(conv_block_num):
            conv = self._add_coord(conv, is_test=is_test)
            conv = self._conv_bn(
                conv,
                channel,
                filter_size=1,
                stride=1,
                padding=0,
                is_test=is_test,
                name='{}.{}.0'.format(name, j))
            if self.use_spp and is_first and j == 1:
                conv = self._spp_module(conv, is_test=is_test, name="spp")
                conv = self._conv_bn(
                    conv,
                    512,
                    filter_size=1,
                    stride=1,
                    padding=0,
                    is_test=is_test,
                    name='{}.{}.spp.conv'.format(name, j))
            conv = self._conv_bn(
                conv,
                channel * 2,
                filter_size=3,
                stride=1,
                padding=1,
                is_test=is_test,
                name='{}.{}.1'.format(name, j))
            if self.drop_block and j == 0 and not is_first:
                conv = DropBlock(
                    conv,
                    block_size=self.block_size,
                    keep_prob=self.keep_prob,
                    is_test=is_test)

        if self.drop_block and is_first:
            conv = DropBlock(
                conv,
                block_size=self.block_size,
                keep_prob=self.keep_prob,
                is_test=is_test)
        conv = self._add_coord(conv, is_test=is_test)
        route = self._conv_bn(
            conv,
            channel,
            filter_size=1,
            stride=1,
            padding=0,
            is_test=is_test,
            name='{}.2'.format(name))
        new_route = self._add_coord(route, is_test=is_test)
        tip = self._conv_bn(
            new_route,
            channel * 2,
            filter_size=3,
            stride=1,
            padding=1,
            is_test=is_test,
            name='{}.tip'.format(name))
        return route, tip

    def _get_loss(self, inputs, gt_box, gt_label, gt_score, targets):
        loss = self.yolo_loss(inputs, gt_box, gt_label, gt_score, targets,
                              self.anchors, self.anchor_masks,
                              self.mask_anchors, self.num_classes,
                              self.prefix_name)
        total_loss = fluid.layers.sum(list(loss.values()))
        return total_loss

    def _get_prediction(self, inputs, im_size):
        boxes = []
        scores = []
        for i, input in enumerate(inputs):
            if self.iou_aware:
                input = get_iou_aware_score(input,
                                            len(self.anchor_masks[i]),
                                            self.num_classes,
                                            self.iou_aware_factor)
            scale_x_y = self.scale_x_y if not isinstance(
                self.scale_x_y, Sequence) else self.scale_x_y[i]

            if paddle.__version__ < '1.8.4' and paddle.__version__ != '0.0.0':
                box, score = fluid.layers.yolo_box(
                    x=input,
                    img_size=im_size,
                    anchors=self.mask_anchors[i],
                    class_num=self.num_classes,
                    conf_thresh=self.nms.score_threshold,
                    downsample_ratio=self.downsample[i],
                    name=self.prefix_name + 'yolo_box' + str(i),
                    clip_bbox=self.clip_bbox)
            else:
                box, score = fluid.layers.yolo_box(
                    x=input,
                    img_size=im_size,
                    anchors=self.mask_anchors[i],
                    class_num=self.num_classes,
                    conf_thresh=self.nms.score_threshold,
                    downsample_ratio=self.downsample[i],
                    name=self.prefix_name + 'yolo_box' + str(i),
                    clip_bbox=self.clip_bbox,
                    scale_x_y=self.scale_x_y)

            boxes.append(box)
            scores.append(fluid.layers.transpose(score, perm=[0, 2, 1]))

        yolo_boxes = fluid.layers.concat(boxes, axis=1)
        yolo_scores = fluid.layers.concat(scores, axis=2)
        if type(self.nms) is MultiClassSoftNMS:
            yolo_scores = fluid.layers.transpose(yolo_scores, perm=[0, 2, 1])
        pred = self.nms(bboxes=yolo_boxes, scores=yolo_scores)
        return pred

    def generate_inputs(self):
        inputs = OrderedDict()
        if self.fixed_input_shape is not None:
            input_shape = [
                None, 3, self.fixed_input_shape[1], self.fixed_input_shape[0]
            ]
            inputs['image'] = fluid.data(
                dtype='float32', shape=input_shape, name='image')
        else:
            inputs['image'] = fluid.data(
                dtype='float32', shape=[None, 3, None, None], name='image')
        if self.mode == 'train':
            inputs['gt_box'] = fluid.data(
                dtype='float32', shape=[None, None, 4], name='gt_box')
            inputs['gt_label'] = fluid.data(
                dtype='int32', shape=[None, None], name='gt_label')
            inputs['gt_score'] = fluid.data(
                dtype='float32', shape=[None, None], name='gt_score')
            inputs['im_size'] = fluid.data(
                dtype='int32', shape=[None, 2], name='im_size')
            if self.use_fine_grained_loss:
                downsample = 32
                for i, mask in enumerate(self.anchor_masks):
                    if self.fixed_input_shape is not None:
                        target_shape = [
                            self.fixed_input_shape[1] // downsample,
                            self.fixed_input_shape[0] // downsample
                        ]
                    else:
                        target_shape = [None, None]
                    inputs['target{}'.format(i)] = fluid.data(
                        dtype='float32',
                        lod_level=0,
                        shape=[
                            None, len(mask), 6 + self.num_classes,
                            target_shape[0], target_shape[1]
                        ],
                        name='target{}'.format(i))
                    downsample //= 2
        elif self.mode == 'eval':
            inputs['im_size'] = fluid.data(
                dtype='int32', shape=[None, 2], name='im_size')
            inputs['im_id'] = fluid.data(
                dtype='int32', shape=[None, 1], name='im_id')
            inputs['gt_box'] = fluid.data(
                dtype='float32', shape=[None, None, 4], name='gt_box')
            inputs['gt_label'] = fluid.data(
                dtype='int32', shape=[None, None], name='gt_label')
            inputs['is_difficult'] = fluid.data(
                dtype='int32', shape=[None, None], name='is_difficult')
        elif self.mode == 'test':
            inputs['im_size'] = fluid.data(
                dtype='int32', shape=[None, 2], name='im_size')
        return inputs

    def build_net(self, inputs):
        image = inputs['image']
        feats = self.backbone(image)
        if isinstance(feats, OrderedDict):
            feat_names = list(feats.keys())
            feats = [feats[name] for name in feat_names]

        head_outputs = self._head(feats, self.mode == 'train')
        if self.mode == 'train':
            gt_box = inputs['gt_box']
            gt_label = inputs['gt_label']
            gt_score = inputs['gt_score']
            im_size = inputs['im_size']
            num_boxes = fluid.layers.shape(gt_box)[1]
            im_size_wh = fluid.layers.reverse(im_size, axis=1)
            whwh = fluid.layers.concat([im_size_wh, im_size_wh], axis=1)
            whwh = fluid.layers.unsqueeze(whwh, axes=[1])
            whwh = fluid.layers.expand(whwh, expand_times=[1, num_boxes, 1])
            whwh = fluid.layers.cast(whwh, dtype='float32')
            whwh.stop_gradient = True
            normalized_box = fluid.layers.elementwise_div(gt_box, whwh)

            targets = []
            if self.use_fine_grained_loss:
                for i, mask in enumerate(self.anchor_masks):
                    k = 'target{}'.format(i)
                    if k in inputs:
                        targets.append(inputs[k])
            return self._get_loss(head_outputs, normalized_box, gt_label,
                                  gt_score, targets)
        else:
            im_size = inputs['im_size']
            return self._get_prediction(head_outputs, im_size)
