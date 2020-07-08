# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
import cv2
import numpy as np
import yaml
from six import text_type as _text_type
from openvino.inference_engine import IECore
from utils import logging 




class Predictor:
    def __init__(self,
                 model_xml,
                 model_yaml,
                 device="CPU"):
        self.device = device
        if not osp.exists(model_xml):
            logging.error("model xml file is not exists in {}".format(model_xml))
        self.model_xml = model_xml
        self.model_bin = osp.splitext(model_xml)[0] + ".bin"
        if not osp.exists(model_yaml):
            logging,error("model yaml file is not exists in {}".format(model_yaml))
        with open(model_yaml) as f:
            self.info = yaml.load(f.read(), Loader=yaml.Loader)
        self.model_type = self.info['_Attributes']['model_type']
        self.model_name = self.info['Model']
        self.num_classes = self.info['_Attributes']['num_classes']
        self.labels = self.info['_Attributes']['labels']
        if self.info['Model'] == 'MaskRCNN':
            if self.info['_init_params']['with_fpn']:
                self.mask_head_resolution = 28
            else:
                self.mask_head_resolution = 14
        transforms_mode = self.info.get('TransformsMode', 'RGB')
        if transforms_mode == 'RGB':
            to_rgb = True
        else:
            to_rgb = False
        self.transforms = self.build_transforms(self.info['Transforms'], to_rgb)
        self.predictor, self.net = self.create_predictor()




    def create_predictor(self):

        #initialization for specified device
        logging.info("Creating Inference Engine")
        ie = IECore()
        logging.info("Loading network files:\n\t{}\n\t{}".format(self.model_xml, self.model_bin))
        net = ie.read_network(model=self.model_xml,weights=self.model_bin)
        net.batch_size = 1
        exec_net = ie.load_network(network=net,device_name=self.device)
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
            import transforms.det_transforms as transforms
            arrange_transform = transforms.ArrangeSegmenter
        elif self.model_type == 'detector':
            import transforms.seg_transforms as transforms
            arrange_name = 'Arrange{}'.format(self.model_name)
            arrange_transform = getattr(transforms, arrange_name)
        else:
            raise Exception("Unrecognized model type: {}".format(
                self.model_type))
        if type(eval_transforms.transforms[-1]).__name__.startswith('Arrange'):
            eval_transforms.transforms[-1] = arrange_transform(mode='test')
        else:
            eval_transforms.transforms.append(arrange_transform(mode='test'))


    def raw_predict(self, images):
        input_blob = next(iter(self.net.inputs))
        out_blob = next(iter(self.net.outputs)) 
        #Start sync inference
        logging.info("Starting inference in synchronous mode")
        res = self.predictor.infer(inputs={input_blob:images})
    
        #Processing output blob
        logging.info("Processing output blob")
        res = res[out_blob]
        print("res: ",res)

    def preprocess(self, image):
        
        if self.model_type == "classifier":
            im, = self.transforms(image)
            im = np.expand_dims(im, axis=0).copy()
            #res['image'] = im
        '''elif self.model_type == "detector":
            if self.model_name == "YOLOv3":
                im, im_shape = self.transforms(image)
                im = np.expand_dims(im, axis=0).copy()
                im_shape = np.expand_dims(im_shape, axis=0).copy()
                res['image'] = im
                res['im_size'] = im_shape
            if self.model_name.count('RCNN') > 0:
                im, im_resize_info, im_shape = self.transforms(image)
                im = np.expand_dims(im, axis=0).copy()
                im_resize_info = np.expand_dims(im_resize_info, axis=0).copy()
                im_shape = np.expand_dims(im_shape, axis=0).copy()
                res['image'] = im
                res['im_info'] = im_resize_info
                res['im_shape'] = im_shape
        elif self.model_type == "segmenter":
            im, im_info = self.transforms(image)
            im = np.expand_dims(im, axis=0).copy()
            res['image'] = im
            res['im_info'] = im_info'''
        return im


    def predict(self, image, topk=1, threshold=0.5):
        preprocessed_input = self.preprocess(image)
        model_pred = self.raw_predict(preprocessed_input)

