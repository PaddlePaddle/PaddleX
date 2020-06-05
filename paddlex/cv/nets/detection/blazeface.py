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

class BlazeFace:
    def __init__(self,
                 backbone,
                 min_sizes=[[16., 24.], [32., 48., 64., 80., 96., 128.]],
                 max_sizes=None,
                 steps=[8., 16.],
                 num_classes=2,
                 use_density_prior_box=False,
                 densities=[[2, 2], [2, 1, 1, 1, 1, 1]],
                 nms_threshold=0.3,
                 nms_topk=5000,
                 nms_keep_topk=750,
                 score_threshold=0.01,
                 nms_eta=1.0,
                 fixed_input_shape=None):
        self.backbone = backbone
        self.num_classes = num_classes
        self.output_decoder = output_decoder
        self.min_sizes = min_sizes
        self.max_sizes = max_sizes
        self.steps = steps
        self.use_density_prior_box = use_density_prior_box
        self.densities = densities
        self.fixed_input_shape = fixed_input_shape
        self.nms_threshold = nms_threshold
        self.nms_topk = nms_topk
        self.nms_keep_topk = nms_keep_topk
        self.score_threshold  = score_threshold
        self.nms_eta = nms_eta
        self.background_label = 0

    
    def _multi_box_head(self,
                        inputs,
                        image,
                        num_classes=2,
                        use_density_prior_box=False):
        def permute_and_reshape(input, last_dim):
            trans = fluid.layers.transpose(input, perm=[0, 2, 3, 1])
            compile_shape = [0, -1, last_dim]
            return fluid.layers.reshape(trans, shape=compile_shape)

        def _is_list_or_tuple_(data):
            return (isinstance(data, list) or isinstance(data, tuple))

        locs, confs = [], []
        boxes, vars = [], []
        b_attr = ParamAttr(learning_rate=2., regularizer=L2Decay(0.))

        for i, input in enumerate(inputs):
            min_size = self.min_sizes[i]

            if use_density_prior_box:
                densities = self.densities[i]
                box, var = fluid.layers.density_prior_box(
                    input,
                    image,
                    densities=densities,
                    fixed_sizes=min_size,
                    fixed_ratios=[1.],
                    clip=False,
                    offset=0.5,
                    steps=[self.steps[i]] * 2)
            else:
                box, var = fluid.layers.prior_box(
                    input,
                    image,
                    min_sizes=min_size,
                    max_sizes=None,
                    steps=[self.steps[i]] * 2,
                    aspect_ratios=[1.],
                    clip=False,
                    flip=False,
                    offset=0.5)

            num_boxes = box.shape[2]

            box = fluid.layers.reshape(box, shape=[-1, 4])
            var = fluid.layers.reshape(var, shape=[-1, 4])
            num_loc_output = num_boxes * 4
            num_conf_output = num_boxes * num_classes
            # get loc
            mbox_loc = fluid.layers.conv2d(
                input, num_loc_output, 3, 1, 1, bias_attr=b_attr)
            loc = permute_and_reshape(mbox_loc, 4)
            # get conf
            mbox_conf = fluid.layers.conv2d(
                input, num_conf_output, 3, 1, 1, bias_attr=b_attr)
            conf = permute_and_reshape(mbox_conf, 2)

            locs.append(loc)
            confs.append(conf)
            boxes.append(box)
            vars.append(var)

        face_mbox_loc = fluid.layers.concat(locs, axis=1)
        face_mbox_conf = fluid.layers.concat(confs, axis=1)
        prior_boxes = fluid.layers.concat(boxes)
        box_vars = fluid.layers.concat(vars)
        return face_mbox_loc, face_mbox_conf, prior_boxes, box_vars
    
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
                dtype='float32', shape=[None, None, 4], lod_level=1, name='gt_box')
            inputs['gt_label'] = fluid.data(
                dtype='int32', shape=[None, None], lod_level=1, name='gt_label')
            inputs['im_size'] = fluid.data(
                dtype='int32', shape=[None, 2], name='im_size')
        elif self.mode == 'eval':
            inputs['gt_box'] = fluid.data(
                dtype='float32', shape=[None, None, 4], lod_level=1, name='gt_box')
            inputs['gt_label'] = fluid.data(
                dtype='int32', shape=[None, None], lod_level=1, name='gt_label')
            inputs['is_difficult'] = fluid.data(
                dtype='int32', shape=[None, 1], lod_level=1, name='is_difficult')
            inputs['im_id'] = fluid.data(
                dtype='int32', shape=[None, 1], name='im_id')
        elif self.mode == 'test':
            inputs['im_size'] = fluid.data(
                dtype='int32', shape=[None, 2], name='im_size')
        return inputs
    
    
    def build_net(self, inputs):
        image = inputs['image']
        if self.mode == 'train':
            gt_bbox = inputs['gt_bbox']
            gt_label = inputs['gt_label']
            im_size = inputs['im_size']
            num_boxes = fluid.layers.shape(gt_box)[1]
            im_size_wh = fluid.layers.reverse(im_size, axis=1)
            whwh = fluid.layers.concat([im_size_wh, im_size_wh], axis=1)
            whwh = fluid.layers.unsqueeze(whwh, axes=[1])
            whwh = fluid.layers.expand(whwh, expand_times=[1, num_boxes, 1])
            whwh = fluid.layers.cast(whwh, dtype='float32')
            whwh.stop_gradient = True
            normalized_box = fluid.layers.elementwise_div(gt_box, whwh)
        body_feats = self.backbone(image)
        locs, confs, box, box_var = self._multi_box_head(
            inputs=body_feats,
            image=image,
            num_classes=self.num_classes,
            use_density_prior_box=self.use_density_prior_box)
        if mode == 'train':
            loss = fluid.layers.ssd_loss(
                locs,
                confs,
                gt_bbox,
                gt_label,
                box,
                box_var,
                overlap_threshold=0.35,
                neg_overlap=0.35)
            loss = fluid.layers.reduce_sum(loss)
            loss.persistable = True
            return loss
        else:
            pred = fluid.layers.detection_output(
                locs, 
                confs, 
                box, 
                box_var,
                background_label=self.background_label,
                nms_threshold=self.nms_threshold,
                nms_top_k=self.nms_keep_topk,
                keep_top_k=self.nms_keep_topk,
                score_threshold=self.score_threshold,
                nms_eta=self.nms_eta)
            return pred