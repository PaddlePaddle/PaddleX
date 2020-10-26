# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import os
import os.path as osp
import time
import cv2
import numpy as np
import yaml
from six import text_type as _text_type
from openvino.inference_engine import IECore


class Predictor:
    def __init__(self, model_xml, model_yaml, device="CPU"):
        self.device = device
        if not osp.exists(model_xml):
            print("model xml file is not exists in {}".format(model_xml))
        self.model_xml = model_xml
        self.model_bin = osp.splitext(model_xml)[0] + ".bin"
        if not osp.exists(model_yaml):
            print("model yaml file is not exists in {}".format(model_yaml))
        with open(model_yaml) as f:
            self.info = yaml.load(f.read(), Loader=yaml.Loader)
        self.model_type = self.info['_Attributes']['model_type']
        self.model_name = self.info['Model']
        self.num_classes = self.info['_Attributes']['num_classes']
        self.labels = self.info['_Attributes']['labels']
        transforms_mode = self.info.get('TransformsMode', 'RGB')
        if transforms_mode == 'RGB':
            to_rgb = True
        else:
            to_rgb = False
        self.transforms = self.build_transforms(self.info['Transforms'],
                                                to_rgb)
        self.predictor, self.net = self.create_predictor()
        self.total_time = 0
        self.count_num = 0

    def create_predictor(self):

        #initialization for specified device
        print("Creating Inference Engine")
        ie = IECore()
        print("Loading network files:\n\t{}\n\t{}".format(self.model_xml,
                                                          self.model_bin))
        net = ie.read_network(model=self.model_xml, weights=self.model_bin)
        net.batch_size = 1
        network_config = {}
        if self.device == "MYRIAD":
            network_config = {'VPU_HW_STAGES_OPTIMIZATION': 'NO'}
        exec_net = ie.load_network(
            network=net, device_name=self.device, config=network_config)
        return exec_net, net

    def build_transforms(self, transforms_info, to_rgb=True):
        if self.model_type == "classifier":
            import transforms.cls_transforms as transforms
        elif self.model_type == "detector":
            import transforms.det_transforms as transforms
        elif self.model_type == "segmenter":
            import transforms.seg_transforms as transforms
        op_list = list()
        for op_info in transforms_info:
            op_name = list(op_info.keys())[0]
            op_attr = op_info[op_name]
            if not hasattr(transforms, op_name):
                raise Exception(
                    "There's no operator named '{}' in transforms of {}".
                    format(op_name, self.model_type))
            op_list.append(getattr(transforms, op_name)(**op_attr))
        eval_transforms = transforms.Compose(op_list)
        if hasattr(eval_transforms, 'to_rgb'):
            eval_transforms.to_rgb = to_rgb
        self.arrange_transforms(eval_transforms)
        return eval_transforms

    def arrange_transforms(self, eval_transforms):
        if self.model_type == 'classifier':
            import transforms.cls_transforms as transforms
            arrange_transform = transforms.ArrangeClassifier
        elif self.model_type == 'segmenter':
            import transforms.seg_transforms as transforms
            arrange_transform = transforms.ArrangeSegmenter
        elif self.model_type == 'detector':
            import transforms.det_transforms as transforms
            arrange_name = 'Arrange{}'.format(self.model_name)
            arrange_transform = getattr(transforms, arrange_name)
        else:
            raise Exception("Unrecognized model type: {}".format(
                self.model_type))
        if type(eval_transforms.transforms[-1]).__name__.startswith('Arrange'):
            eval_transforms.transforms[-1] = arrange_transform(mode='test')
        else:
            eval_transforms.transforms.append(arrange_transform(mode='test'))

    def raw_predict(self, preprocessed_input):
        self.count_num += 1
        feed_dict = {}
        if self.model_name == "YOLOv3":
            inputs = self.net.inputs
            for name in inputs:
                if (len(inputs[name].shape) == 2):
                    feed_dict[name] = preprocessed_input['im_size']
                elif (len(inputs[name].shape) == 4):
                    feed_dict[name] = preprocessed_input['image']
                else:
                    pass
        else:
            input_blob = next(iter(self.net.inputs))
            feed_dict[input_blob] = preprocessed_input['image']
        #Start sync inference
        print("Starting inference in synchronous mode")
        res = self.predictor.infer(inputs=feed_dict)

        #Processing output blob
        print("Processing output blob")
        return res

    def preprocess(self, image):
        res = dict()
        if self.model_type == "classifier":
            im = self.transforms(image)
            im = np.expand_dims(im, axis=0).copy()
            res['image'] = im
        elif self.model_type == "detector":
            if self.model_name == "YOLOv3":
                im, im_shape = self.transforms(image)
                im = np.expand_dims(im, axis=0).copy()
                im_shape = np.expand_dims(im_shape, axis=0).copy()
                res['image'] = im
                res['im_size'] = im_shape
        elif self.model_type == "segmenter":
            im, im_info = self.transforms(image)
            im = np.expand_dims(im, axis=0).copy()
            res['image'] = im
            res['im_info'] = im_info
        return res

    def classifier_postprocess(self, preds, topk=1):
        """ 对分类模型的预测结果做后处理
        """
        true_topk = min(self.num_classes, topk)
        output_name = next(iter(self.net.outputs))
        pred_label = np.argsort(-preds[output_name][0])[:true_topk]
        result = [{
            'category_id': l,
            'category': self.labels[l],
            'score': preds[output_name][0][l],
        } for l in pred_label]
        print(result)
        return result

    def segmenter_postprocess(self, preds, preprocessed_inputs):
        """ 对语义分割结果做后处理
        """
        it = iter(self.net.outputs)
        next(it)
        label_name = next(it)
        label_map = np.squeeze(preds[label_name]).astype('uint8')
        score_name = next(it)
        score_map = np.squeeze(preds[score_name])
        score_map = np.transpose(score_map, (1, 2, 0))
        
        im_info = preprocessed_inputs['im_info']
        
        for info in im_info[0][::-1]:
            if info[0] == 'resize':
                w, h = info[1][1], info[1][0]
                label_map = cv2.resize(label_map, (w, h), cv2.INTER_NEAREST)
                score_map = cv2.resize(score_map, (w, h), cv2.INTER_LINEAR)
            elif info[0] == 'padding':
                w, h = info[1][1], info[1][0]
                label_map = label_map[0:h, 0:w]
                score_map = score_map[0:h, 0:w, :]
        return {'label_map': label_map, 'score_map': score_map}

    def detector_postprocess(self, preds, preprocessed_inputs):
        """对图像检测结果做后处理
        """
        outputs = self.net.outputs
        for name in outputs:
            if (len(outputs[name].shape) == 3):
                output = preds[name][0]
        result = []
        for out in output:
            if (out[0] >= 0):
                result.append(out.tolist())
            else:
                pass
        return result

    def predict(self, image, topk=1, threshold=0.5):
        preprocessed_input = self.preprocess(image)
        model_pred = self.raw_predict(preprocessed_input)
        if self.model_type == "classifier":
            results = self.classifier_postprocess(model_pred, topk)
        elif self.model_type == "detector":
            results = self.detector_postprocess(model_pred, preprocessed_input)
        elif self.model_type == "segmenter":
            results = self.segmenter_postprocess(model_pred,
                                                 preprocessed_input)
