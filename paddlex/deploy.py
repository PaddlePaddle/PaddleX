# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import multiprocessing as mp
import paddlex
import paddle.fluid as fluid
from paddlex.cv.transforms import build_transforms
from paddlex.cv.models import BaseClassifier
from paddlex.cv.models import PPYOLO, FasterRCNN, MaskRCNN
from paddlex.cv.models import DeepLabv3p
import paddlex.utils.logging as logging


class Predictor:
    def __init__(self,
                 model_dir,
                 use_gpu=True,
                 gpu_id=0,
                 use_mkl=True,
                 mkl_thread_num=4,
                 use_trt=False,
                 use_glog=False,
                 memory_optimize=True):
        """ 创建Paddle Predictor

            Args:
                model_dir: 模型路径（必须是导出的部署或量化模型）
                use_gpu: 是否使用gpu，默认True
                gpu_id: 使用gpu的id，默认0
                use_mkl: 是否使用mkldnn计算库，CPU情况下使用，默认False
                mkl_thread_num: mkldnn计算线程数，默认为4
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
        self.transforms = build_transforms(self.model_type,
                                           self.info['Transforms'], to_rgb)
        self.predictor = self.create_predictor(use_gpu, gpu_id, use_mkl,
                                               mkl_thread_num, use_trt,
                                               use_glog, memory_optimize)
        # 线程池，在模型在预测时用于对输入数据以图片为单位进行并行处理
        # 主要用于batch_predict接口
        thread_num = mp.cpu_count() if mp.cpu_count() < 8 else 8
        self.thread_pool = mp.pool.ThreadPool(thread_num)

    def reset_thread_pool(self, thread_num):
        self.thread_pool.close()
        self.thread_pool.join()
        self.thread_pool = mp.pool.ThreadPool(thread_num)

    def create_predictor(self,
                         use_gpu=True,
                         gpu_id=0,
                         use_mkl=False,
                         mkl_thread_num=4,
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
        if use_mkl and not use_gpu:
            if self.model_name not in ["HRNet", "DeepLabv3p", "PPYOLO"]:
                config.enable_mkldnn()
                config.set_cpu_math_library_num_threads(mkl_thread_num)
            else:
                logging.warning(
                    "HRNet/DeepLabv3p/PPYOLO are not supported for the use of mkldnn\n"
                )
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

    def preprocess(self, image, thread_pool=None):
        """ 对图像做预处理

            Args:
                image(list|tuple): 数组中的元素可以是图像路径，也可以是解码后的排列格式为（H，W，C）
                    且类型为float32且为BGR格式的数组。
        """
        res = dict()
        if self.model_type == "classifier":
            im = BaseClassifier._preprocess(
                image,
                self.transforms,
                self.model_type,
                self.model_name,
                thread_pool=thread_pool)
            res['image'] = im
        elif self.model_type == "detector":
            if self.model_name in ["PPYOLO", "YOLOv3"]:
                im, im_size = PPYOLO._preprocess(
                    image,
                    self.transforms,
                    self.model_type,
                    self.model_name,
                    thread_pool=thread_pool)
                res['image'] = im
                res['im_size'] = im_size
            if self.model_name.count('RCNN') > 0:
                im, im_resize_info, im_shape = FasterRCNN._preprocess(
                    image,
                    self.transforms,
                    self.model_type,
                    self.model_name,
                    thread_pool=thread_pool)
                res['image'] = im
                res['im_info'] = im_resize_info
                res['im_shape'] = im_shape
        elif self.model_type == "segmenter":
            im, im_info = DeepLabv3p._preprocess(
                image,
                self.transforms,
                self.model_type,
                self.model_name,
                thread_pool=thread_pool)
            res['image'] = im
            res['im_info'] = im_info
        return res

    def postprocess(self,
                    results,
                    topk=1,
                    batch_size=1,
                    im_shape=None,
                    im_info=None):
        """ 对预测结果做后处理

            Args:
                results (list): 预测结果
                topk (int): 分类预测时前k个最大值
                batch_size (int): 预测时图像批量大小
                im_shape (list): MaskRCNN的图像输入大小
                im_info (list)：RCNN系列和分割网络的原图大小
        """

        def offset_to_lengths(lod):
            offset = lod[0]
            lengths = [
                offset[i + 1] - offset[i] for i in range(len(offset) - 1)
            ]
            return [lengths]

        if self.model_type == "classifier":
            true_topk = min(self.num_classes, topk)
            preds = BaseClassifier._postprocess([results[0][0]], true_topk,
                                                self.labels)
        elif self.model_type == "detector":
            res = {'bbox': (results[0][0], offset_to_lengths(results[0][1])), }
            res['im_id'] = (np.array(
                [[i] for i in range(batch_size)]).astype('int32'), [[]])
            if self.model_name in ["PPYOLO", "YOLOv3"]:
                preds = PPYOLO._postprocess(res, batch_size, self.num_classes,
                                            self.labels)
            elif self.model_name == "FasterRCNN":
                preds = FasterRCNN._postprocess(res, batch_size,
                                                self.num_classes, self.labels)
            elif self.model_name == "MaskRCNN":
                res['mask'] = (results[1][0], offset_to_lengths(results[1][1]))
                res['im_shape'] = (im_shape, [])
                preds = MaskRCNN._postprocess(
                    res, batch_size, self.num_classes,
                    self.mask_head_resolution, self.labels)
        elif self.model_type == "segmenter":
            res = [results[0][0], results[1][0]]
            preds = DeepLabv3p._postprocess(res, im_info)
        return preds

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
            output_tensor_lod = output_tensor.lod()
            output_results.append(
                [output_tensor.copy_to_cpu(), output_tensor_lod])
        return output_results

    def predict(self, image, topk=1, transforms=None):
        """ 图片预测

            Args:
                image(str|np.ndarray): 图像路径；或者是解码后的排列格式为（H, W, C）且类型为float32且为BGR格式的数组。
                topk(int): 分类预测时使用，表示预测前topk的结果。
                transforms (paddlex.cls.transforms): 数据预处理操作。
        """
        if transforms is not None:
            self.transforms = transforms
        preprocessed_input = self.preprocess([image])
        model_pred = self.raw_predict(preprocessed_input)
        im_shape = None if 'im_shape' not in preprocessed_input else preprocessed_input[
            'im_shape']
        im_info = None if 'im_info' not in preprocessed_input else preprocessed_input[
            'im_info']
        results = self.postprocess(
            model_pred,
            topk=topk,
            batch_size=1,
            im_shape=im_shape,
            im_info=im_info)

        return results[0]

    def batch_predict(self, image_list, topk=1, transforms=None):
        """ 图片预测

            Args:
                image_list(list|tuple): 对列表（或元组）中的图像同时进行预测，列表中的元素可以是图像路径
                    也可以是解码后的排列格式为（H，W，C）且类型为float32且为BGR格式的数组。

                topk(int): 分类预测时使用，表示预测前topk的结果。
                transforms (paddlex.cls.transforms): 数据预处理操作。
        """
        if transforms is not None:
            self.transforms = transforms
        preprocessed_input = self.preprocess(image_list, self.thread_pool)
        model_pred = self.raw_predict(preprocessed_input)
        im_shape = None if 'im_shape' not in preprocessed_input else preprocessed_input[
            'im_shape']
        im_info = None if 'im_info' not in preprocessed_input else preprocessed_input[
            'im_info']
        results = self.postprocess(
            model_pred,
            topk=topk,
            batch_size=len(image_list),
            im_shape=im_shape,
            im_info=im_info)

        return results
