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
import os
import os.path as osp
import cv2
import numpy as np
import yaml
import paddlex
import paddle.fluid as fluid


class Predictor:
    def __init__(self,
                 model_dir,
                 use_gpu=True,
                 gpu_id=0,
                 use_mkl=False,
                 use_trt=False,
                 use_glog=False,
                 memory_optimize=True):
        """ 创建Paddle Predictor

            Args:
                model_dir: 模型路径（必须是导出的部署或量化模型）
                use_gpu: 是否使用gpu，默认True
                gpu_id: 使用gpu的id，默认0
                use_mkl: 是否使用mkldnn计算库，CPU情况下使用，默认False
                use_trt: 是否使用TensorRT，默认False
                use_glog: 是否启用glog日志, 默认False
                memory_optimize: 是否启动内存优化，默认True
        """
        if not osp.isdir(model_dir):
            raise Exception("[ERROR] Path {} not exist.".format(model_dir))
        if not osp.exists(osp.join(model_dir, "model.yml")):
            raise Exception("There's not model.yml in {}".format(model_dir))
        with open(osp.join(model_dir, "model.yml")) as f:
            self.info = yaml.load(f.read(), Loader=yaml.Loader)

        self.status = self.info['status']

        if self.status != "Quant" and self.status != "Infer":
            raise Exception("[ERROR] Only quantized model or exported "
                            "inference model is supported.")

        self.model_dir = model_dir
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
        self.predictor = self.create_predictor(
            use_gpu, gpu_id, use_mkl, use_trt, use_glog, memory_optimize)

    def create_predictor(self,
                         use_gpu=True,
                         gpu_id=0,
                         use_mkl=False,
                         use_trt=False,
                         use_glog=False,
                         memory_optimize=True):
        config = fluid.core.AnalysisConfig(
            os.path.join(self.model_dir, '__model__'),
            os.path.join(self.model_dir, '__params__'))

        if use_gpu:
            # 设置GPU初始显存(单位M)和Device ID
            config.enable_use_gpu(100, gpu_id)
        else:
            config.disable_gpu()
        if use_mkl:
            config.enable_mkldnn()
        if use_glog:
            config.enable_glog_info()
        else:
            config.disable_glog_info()
        if memory_optimize:
            config.enable_memory_optim()

        # 开启计算图分析优化，包括OP融合等
        config.switch_ir_optim(True)
        # 关闭feed和fetch OP使用，使用ZeroCopy接口必须设置此项
        config.switch_use_feed_fetch_ops(False)
        predictor = fluid.core.create_paddle_predictor(config)
        return predictor

    def build_transforms(self, transforms_info, to_rgb=True):
        if self.model_type == "classifier":
            from paddlex.cls import transforms
        elif self.model_type == "detector":
            from paddlex.det import transforms
        elif self.model_type == "segmenter":
            from paddlex.seg import transforms
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

    def arrange_transforms(self, transforms):
        if self.model_type == 'classifier':
            arrange_transform = paddlex.cls.transforms.ArrangeClassifier
        elif self.model_type == 'segmenter':
            arrange_transform = paddlex.seg.transforms.ArrangeSegmenter
        elif self.model_type == 'detector':
            arrange_name = 'Arrange{}'.format(self.model_name)
            arrange_transform = getattr(paddlex.det.transforms, arrange_name)
        else:
            raise Exception("Unrecognized model type: {}".format(
                self.model_type))
        if type(transforms.transforms[-1]).__name__.startswith('Arrange'):
            transforms.transforms[-1] = arrange_transform(mode='test')
        else:
            transforms.transforms.append(arrange_transform(mode='test'))

    def preprocess(self, image):
        """ 对图像做预处理

            Args:
                image(str|np.ndarray): 图片路径或np.ndarray，如为后者，要求是BGR格式
        """
        res = dict()
        if self.model_type == "classifier":
            im, = self.transforms(image)
            im = np.expand_dims(im, axis=0).copy()
            res['image'] = im
        elif self.model_type == "detector":
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
            res['im_info'] = im_info
        return res

    def raw_predict(self, inputs):
        """ 接受预处理过后的数据进行预测

            Args:
                inputs(tuple): 预处理过后的数据
        """
        for k, v in inputs.items():
            try:
                tensor = self.predictor.get_input_tensor(k)
            except:
                continue
            tensor.copy_from_cpu(v)
        self.predictor.zero_copy_run()
        output_names = self.predictor.get_output_names()
        output_results = list()
        for name in output_names:
            output_tensor = self.predictor.get_output_tensor(name)
            output_results.append(output_tensor.copy_to_cpu())
        return output_results

    def classifier_postprocess(self, preds, topk=1):
        """ 对分类模型的预测结果做后处理
        """
        true_topk = min(self.num_classes, topk)
        pred_label = np.argsort(preds[0][0])[::-1][:true_topk]
        result = [{
            'category_id': l,
            'category': self.labels[l],
            'score': preds[0][0, l],
        } for l in pred_label]
        return result

    def segmenter_postprocess(self, preds, preprocessed_inputs):
        """ 对语义分割结果做后处理
        """
        label_map = np.squeeze(preds[0]).astype('uint8')
        score_map = np.squeeze(preds[1])
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

    def detector_postprocess(self, preds, preprocessed_inputs):
        """ 对目标检测和实例分割结果做后处理
        """
        bboxes = {'bbox': (np.array(preds[0]), [[len(preds[0])]])}
        bboxes['im_id'] = (np.array([[0]]).astype('int32'), [])
        clsid2catid = dict({i: i for i in range(self.num_classes)})
        xywh_results = paddlex.cv.models.utils.detection_eval.bbox2out(
            [bboxes], clsid2catid)
        results = list()
        for xywh_res in xywh_results:
            del xywh_res['image_id']
            xywh_res['category'] = self.labels[xywh_res['category_id']]
            results.append(xywh_res)
        if len(preds) > 1:
            im_shape = preprocessed_inputs['im_shape']
            bboxes['im_shape'] = (im_shape, [])
            bboxes['mask'] = (np.array(preds[1]), [[len(preds[1])]])
            segm_results = paddlex.cv.models.utils.detection_eval.mask2out(
                [bboxes], clsid2catid, self.mask_head_resolution)
            import pycocotools.mask as mask_util
            for i in range(len(results)):
                results[i]['mask'] = mask_util.decode(segm_results[i][
                    'segmentation'])
        return results

    def predict(self, image, topk=1, threshold=0.5):
        """ 图片预测

            Args:
                image(str|np.ndarray): 图片路径或np.ndarray格式，如果后者，要求为BGR输入格式
                topk(int): 分类预测时使用，表示预测前topk的结果
        """
        preprocessed_input = self.preprocess(image)
        model_pred = self.raw_predict(preprocessed_input)

        if self.model_type == "classifier":
            results = self.classifier_postprocess(model_pred, topk)
        elif self.model_type == "detector":
            results = self.detector_postprocess(model_pred, preprocessed_input)
        elif self.model_type == "segmenter":
            results = self.segmenter_postprocess(model_pred,
                                                 preprocessed_input)
        return results
