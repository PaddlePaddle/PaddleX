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
import time
import cv2
import numpy as np
import yaml
from six import text_type as _text_type
from paddlelite.lite import *


class Predictor:
    def __init__(self, model_nb, model_yaml, thread_num):
        if not osp.exists(model_nb):
            print("model nb file is not exists in {}".format(model_xml))
        self.model_nb = model_nb
        config = MobileConfig()
        config.set_model_from_file(model_nb)
        config.set_threads(thread_num)
        if not osp.exists(model_yaml):
            print("model yaml file is not exists in {}".format(model_yaml))
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
        self.transforms = self.build_transforms(self.info['Transforms'],
                                                to_rgb)
        self.predictor = create_paddle_predictor(config)
        self.total_time = 0
        self.count_num = 0

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
        input_tensor = self.predictor.get_input(0)
        im_shape = preprocessed_input['image'].shape
        input_tensor.resize(self.shape)
        input_tensor.set_float_data(preprocessed_input['image'])
        if self.model_name == "YOLOv3":
            input_size_tensor = self.predictor.get_input(1)
            input_size_tensor.resize([1, 2])
            input_size_tensor.set_float_data(preprocessed_input['im_size'])
        #Start  inference
        start_time = time.time()
        self.predictor.run()
        time_use = time.time() - start_time
        if (self.count_num >= 20):
            self.total_time += time_use
        if (self.count_num >= 120):
            print("avgtime:", self.total_time * 10)

        #Processing output blob
        print("Processing output blob")
        return

    def preprocess(self, image):
        res = dict()
        if self.model_type == "classifier":
            im, = self.transforms(image)
            self.shape = [1] + list(im.shape)
            im = np.expand_dims(im, axis=0).copy()
            im = im.flatten()
            res['image'] = im
        elif self.model_type == "detector":
            if self.model_name == "YOLOv3":
                im, im_shape = self.transforms(image)
                self.shape = [1] + list(im.shape)
                im = np.expand_dims(im, axis=0).copy()
                im_shape = np.expand_dims(im_shape, axis=0).copy()
                im = im.flatten()
                im_shape = im_shape.flatten()
                res['image'] = im
                res['im_size'] = im_shape
            if self.model_name.count('RCNN') > 0:
                im, im_resize_info, im_shape = self.transforms(image)
                self.shape = [1] + list(im.shape)
                im = np.expand_dims(im, axis=0).copy()
                im_resize_info = np.expand_dims(im_resize_info, axis=0).copy()
                im_shape = np.expand_dims(im_shape, axis=0).copy()
                res['image'] = im
                res['im_info'] = im_resize_info
                res['im_shape'] = im_shape
        elif self.model_type == "segmenter":
            im, im_info = self.transforms(image)
            self.shape = [1] + list(im.shape)
            im = np.expand_dims(im, axis=0).copy()
            #np.savetxt('./input_data.txt',im.flatten())
            res['image'] = im
            res['im_info'] = im_info
        return res

    def classifier_postprocess(self, topk=1):
        output_tensor = self.predictor.get_output(0)
        output_data = output_tensor.float_data()
        true_topk = min(self.num_classes, topk)
        pred_label = np.argsort(-np.array(output_data))[:true_topk]
        result = [{
            'category_id': l,
            'category': self.labels[l],
            'score': output_data[l],
        } for l in pred_label]
        print(result)
        return result

    def segmenter_postprocess(self, preprocessed_inputs):
        out_label_tensor = self.predictor.get_output(0)
        out_label = out_label_tensor.float_data()
        label_shape = tuple(out_label_tensor.shape())
        label_map = np.array(out_label).astype('uint8')
        label_map = label_map.reshape(label_shape)
        label_map = np.squeeze(label_map)

        out_score_tensor = self.predictor.get_output(1)
        out_score = out_score_tensor.float_data()
        score_shape = tuple(out_score_tensor.shape())
        score_map = np.array(out_score)
        score_map = score_map.reshap(score_shape)
        score_map = np.transpose(score_map, (1, 2, 0))

        im_info = preprocessed_inputs['im_info']
        for info in im_info[::-1]:
            if info[0] == 'resize':
                w, h = info[1][1], info[1][0]
                label_map = cv2.resize(label_map, (w, h), cv2.INTER_NEAREST)
                score_map = cv2.resize(score_map, (w, h), cv2.INTER_LINEAR)
            elif info[0] == 'padding':
                w, h = info[1][1], info[1][0]
                label_map = label_map[0:h, 0:w]
                score_map = score_map[0:h, 0:w, :]
            else:
                raise Exception("Unexpected info '{}' in im_info".format(info[
                    0]))
        return {'label_map': label_map, 'score_map': score_map}

    def detector_postprocess(self, preprocessed_inputs):
        out_tensor = self.predictor.get_output(0)
        out_data = out_tensor.float_data()
        print("@@@@@@@@@@@")
        print(out_data)
        out_shape = tuple(out_tensor.shape())
        out_data = np.array(out_data)
        outputs = out_data.reshape(out_shape)

        result = []
        for out in outputs:
            result.append(out.tolist())
        #print(result)
        return result

    def predict(self, image, topk=1, threshold=0.5):
        preprocessed_input = self.preprocess(image)
        self.raw_predict(preprocessed_input)
        if self.model_type == "classifier":
            results = self.classifier_postprocess(topk)
        elif self.model_type == "detector":
            results = self.detector_postprocess(preprocessed_input)
        elif self.model_type == "segmenter":
            pass
            results = self.segmenter_postprocess(preprocessed_input)
