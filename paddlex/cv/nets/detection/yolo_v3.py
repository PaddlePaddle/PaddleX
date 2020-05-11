# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
from collections import OrderedDict


class YOLOv3:
    def __init__(self,
                 backbone,
                 num_classes,
                 mode='train',
                 anchors=None,
                 anchor_masks=None,
                 ignore_threshold=0.7,
                 label_smooth=False,
                 nms_score_threshold=0.01,
                 nms_topk=1000,
                 nms_keep_topk=100,
                 nms_iou_threshold=0.45,
                 train_random_shapes=[
                     320, 352, 384, 416, 448, 480, 512, 544, 576, 608
                 ],
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
        self.ignore_thresh = ignore_threshold
        self.label_smooth = label_smooth
        self.nms_score_threshold = nms_score_threshold
        self.nms_topk = nms_topk
        self.nms_keep_topk = nms_keep_topk
        self.nms_iou_threshold = nms_iou_threshold
        self.norm_decay = 0.0
        self.prefix_name = ''
        self.train_random_shapes = train_random_shapes
        self.fixed_input_shape = fixed_input_shape

    def _head(self, feats):
        outputs = []
        out_layer_num = len(self.anchor_masks)
        blocks = feats[-1:-out_layer_num - 1:-1]
        route = None

        for i, block in enumerate(blocks):
            if i > 0:
                block = fluid.layers.concat(input=[route, block], axis=1)
            route, tip = self._detection_block(
                block,
                channel=512 // (2**i),
                name=self.prefix_name + 'yolo_block.{}'.format(i))

            num_filters = len(self.anchor_masks[i]) * (self.num_classes + 5)
            block_out = fluid.layers.conv2d(
                input=tip,
                num_filters=num_filters,
                filter_size=1,
                stride=1,
                padding=0,
                act=None,
                param_attr=ParamAttr(name=self.prefix_name +
                                     'yolo_output.{}.conv.weights'.format(i)),
                bias_attr=ParamAttr(
                    regularizer=L2Decay(0.0),
                    name=self.prefix_name +
                    'yolo_output.{}.conv.bias'.format(i)))
            outputs.append(block_out)

            if i < len(blocks) - 1:
                route = self._conv_bn(
                    input=route,
                    ch_out=256 // (2**i),
                    filter_size=1,
                    stride=1,
                    padding=0,
                    name=self.prefix_name + 'yolo_transition.{}'.format(i))
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

    def _upsample(self, input, scale=2, name=None):
        out = fluid.layers.resize_nearest(
            input=input, scale=float(scale), name=name)
        return out

    def _detection_block(self, input, channel, name=None):
        assert channel % 2 == 0, "channel({}) cannot be divided by 2 in detection block({})".format(
            channel, name)

        is_test = False if self.mode == 'train' else True
        conv = input
        for i in range(2):
            conv = self._conv_bn(
                conv,
                channel,
                filter_size=1,
                stride=1,
                padding=0,
                is_test=is_test,
                name='{}.{}.0'.format(name, i))
            conv = self._conv_bn(
                conv,
                channel * 2,
                filter_size=3,
                stride=1,
                padding=1,
                is_test=is_test,
                name='{}.{}.1'.format(name, i))
        route = self._conv_bn(
            conv,
            channel,
            filter_size=1,
            stride=1,
            padding=0,
            is_test=is_test,
            name='{}.2'.format(name))
        tip = self._conv_bn(
            route,
            channel * 2,
            filter_size=3,
            stride=1,
            padding=1,
            is_test=is_test,
            name='{}.tip'.format(name))
        return route, tip

    def _get_loss(self, inputs, gt_box, gt_label, gt_score):
        losses = []
        downsample = 32
        for i, input in enumerate(inputs):
            loss = fluid.layers.yolov3_loss(
                x=input,
                gt_box=gt_box,
                gt_label=gt_label,
                gt_score=gt_score,
                anchors=self.anchors,
                anchor_mask=self.anchor_masks[i],
                class_num=self.num_classes,
                ignore_thresh=self.ignore_thresh,
                downsample_ratio=downsample,
                use_label_smooth=self.label_smooth,
                name=self.prefix_name + 'yolo_loss' + str(i))
            losses.append(fluid.layers.reduce_mean(loss))
            downsample //= 2
        return sum(losses)

    def _get_prediction(self, inputs, im_size):
        boxes = []
        scores = []
        downsample = 32
        for i, input in enumerate(inputs):
            box, score = fluid.layers.yolo_box(
                x=input,
                img_size=im_size,
                anchors=self.mask_anchors[i],
                class_num=self.num_classes,
                conf_thresh=self.nms_score_threshold,
                downsample_ratio=downsample,
                name=self.prefix_name + 'yolo_box' + str(i))
            boxes.append(box)
            scores.append(fluid.layers.transpose(score, perm=[0, 2, 1]))
            downsample //= 2
        yolo_boxes = fluid.layers.concat(boxes, axis=1)
        yolo_scores = fluid.layers.concat(scores, axis=2)
        pred = fluid.layers.multiclass_nms(
            bboxes=yolo_boxes,
            scores=yolo_scores,
            score_threshold=self.nms_score_threshold,
            nms_top_k=self.nms_topk,
            keep_top_k=self.nms_keep_topk,
            nms_threshold=self.nms_iou_threshold,
            normalized=False,
            nms_eta=1.0,
            background_label=-1)
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
        if self.mode == 'train':
            if isinstance(self.train_random_shapes,
                          (list, tuple)) and len(self.train_random_shapes) > 0:
                import numpy as np
                shapes = np.array(self.train_random_shapes)
                shapes = np.stack([shapes, shapes], axis=1).astype('float32')
                shapes_tensor = fluid.layers.assign(shapes)
                index = fluid.layers.uniform_random(
                    shape=[1], dtype='float32', min=0.0, max=1)
                index = fluid.layers.cast(
                    index * len(self.train_random_shapes), dtype='int32')
                shape = fluid.layers.gather(shapes_tensor, index)
                shape = fluid.layers.reshape(shape, [-1])
                shape = fluid.layers.cast(shape, dtype='int32')
                image = fluid.layers.resize_nearest(
                    image, out_shape=shape, align_corners=False)
        feats = self.backbone(image)
        if isinstance(feats, OrderedDict):
            feat_names = list(feats.keys())
            feats = [feats[name] for name in feat_names]

        head_outputs = self._head(feats)
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
            return self._get_loss(head_outputs, normalized_box, gt_label,
                                  gt_score)
        else:
            im_size = inputs['im_size']
            return self._get_prediction(head_outputs, im_size)
