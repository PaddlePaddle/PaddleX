# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os.path as osp
import numpy as np
from paddle.inference import Config
from paddle.inference import create_predictor
from paddle.inference import PrecisionType
from paddlex.cv.models import load_model
from paddlex.utils import logging, Timer


class Predictor(object):
    def __init__(self,
                 model_dir,
                 use_gpu=False,
                 gpu_id=0,
                 cpu_thread_num=1,
                 use_mkl=True,
                 mkl_thread_num=4,
                 use_trt=False,
                 use_glog=False,
                 memory_optimize=True,
                 max_trt_batch_size=1,
                 trt_precision_mode='float32'):
        """ 创建Paddle Predictor
            Args:
                model_dir: 模型路径（必须是导出的部署或量化模型）
                use_gpu: 是否使用gpu，默认False
                gpu_id: 使用gpu的id，默认0
                cpu_thread_num：使用cpu进行预测时的线程数，默认为1
                use_mkl: 是否使用mkldnn计算库，CPU情况下使用，默认False
                mkl_thread_num: mkldnn计算线程数，默认为4
                use_trt: 是否使用TensorRT，默认False
                use_glog: 是否启用glog日志, 默认False
                memory_optimize: 是否启动内存优化，默认True
                max_trt_batch_size: 在使用TensorRT时配置的最大batch size，默认1
                trt_precision_mode：在使用TensorRT时采用的精度，可选值['float32', 'float16']。默认'float32',
        """
        self.model_dir = model_dir
        self._model = load_model(model_dir, with_net=False)

        if trt_precision_mode.lower() == 'float32':
            trt_precision_mode = PrecisionType.Float32
        elif trt_precision_mode.lower() == 'float16':
            trt_precision_mode = PrecisionType.Float16
        else:
            logging.error(
                "TensorRT precision mode {} is invalid. Supported modes are float32 and float16."
                .format(trt_precision_mode),
                exit=True)

        self.predictor = self.create_predictor(
            use_gpu=use_gpu,
            gpu_id=gpu_id,
            cpu_thread_num=cpu_thread_num,
            use_mkl=use_mkl,
            mkl_thread_num=mkl_thread_num,
            use_trt=use_trt,
            use_glog=use_glog,
            memory_optimize=memory_optimize,
            max_trt_batch_size=max_trt_batch_size,
            trt_precision_mode=trt_precision_mode)
        self.timer = Timer()

    def create_predictor(self,
                         use_gpu=True,
                         gpu_id=0,
                         cpu_thread_num=1,
                         use_mkl=True,
                         mkl_thread_num=4,
                         use_trt=False,
                         use_glog=False,
                         memory_optimize=True,
                         max_trt_batch_size=1,
                         trt_precision_mode=PrecisionType.Float32):
        config = Config(
            osp.join(self.model_dir, 'model.pdmodel'),
            osp.join(self.model_dir, 'model.pdiparams'))

        if use_gpu:
            # 设置GPU初始显存(单位M)和Device ID
            config.enable_use_gpu(100, gpu_id)
            config.switch_ir_optim(True)
            if use_trt:
                if self._model.model_type == 'segmenter':
                    logging.warning(
                        "Semantic segmentation models do not support TensorRT acceleration, "
                        "TensorRT is forcibly disabled.")
                elif 'RCNN' in self._model.__class__.__name__:
                    logging.warning(
                        "RCNN models do not support TensorRT acceleration, "
                        "TensorRT is forcibly disabled.")
                else:
                    config.enable_tensorrt_engine(
                        workspace_size=1 << 10,
                        max_batch_size=max_trt_batch_size,
                        min_subgraph_size=3,
                        precision_mode=trt_precision_mode,
                        use_static=False,
                        use_calib_mode=False)
        else:
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(cpu_thread_num)
            if use_mkl:
                try:
                    # cache 10 different shapes for mkldnn to avoid memory leak
                    config.set_mkldnn_cache_capacity(10)
                    config.enable_mkldnn()
                    config.set_cpu_math_library_num_threads(mkl_thread_num)
                except Exception as e:
                    logging.warning(
                        "The current environment does not support `mkldnn`, so disable mkldnn."
                    )
                    pass

        if not use_glog:
            config.disable_glog_info()
        if memory_optimize:
            config.enable_memory_optim()
        config.switch_use_feed_fetch_ops(False)
        predictor = create_predictor(config)
        return predictor

    def preprocess(self, images, transforms):
        preprocessed_samples = self._model._preprocess(
            images, transforms, to_tensor=False)
        if self._model.model_type == 'classifier':
            preprocessed_samples = {'image': preprocessed_samples[0]}
        elif self._model.model_type == 'segmenter':
            preprocessed_samples = {
                'image': preprocessed_samples[0],
                'ori_shape': preprocessed_samples[1]
            }
        elif self._model.model_type == 'detector':
            pass
        else:
            logging.error(
                "Invalid model type {}".format(self._model.model_type),
                exit=True)
        return preprocessed_samples

    def postprocess(self, net_outputs, topk=1, ori_shape=None,
                    transforms=None):
        if self._model.model_type == 'classifier':
            true_topk = min(self._model.num_classes, topk)
            preds = self._model._postprocess(net_outputs[0], true_topk)
            if len(preds) == 1:
                preds = preds[0]
        elif self._model.model_type == 'segmenter':
            label_map, score_map = self._model._postprocess(
                net_outputs,
                batch_origin_shape=ori_shape,
                transforms=transforms.transforms)
            label_map = np.squeeze(label_map)
            score_map = np.squeeze(score_map)
            if score_map.ndim == 3:
                preds = {'label_map': label_map, 'score_map': score_map}
            else:
                preds = [{
                    'label_map': l,
                    'score_map': s
                } for l, s in zip(label_map, score_map)]
        elif self._model.model_type == 'detector':
            net_outputs = {
                k: v
                for k, v in zip(['bbox', 'bbox_num', 'mask'], net_outputs)
            }
            preds = self._model._postprocess(net_outputs)
            if len(preds) == 1:
                preds = preds[0]
        else:
            logging.error(
                "Invalid model type {}.".format(self._model.model_type),
                exit=True)

        return preds

    def raw_predict(self, inputs):
        """ 接受预处理过后的数据进行预测
            Args:
                inputs(dict): 预处理过后的数据
        """
        input_names = self.predictor.get_input_names()
        for name in input_names:
            input_tensor = self.predictor.get_input_handle(name)
            input_tensor.copy_from_cpu(inputs[name])

        self.predictor.run()
        output_names = self.predictor.get_output_names()
        net_outputs = list()
        for name in output_names:
            output_tensor = self.predictor.get_output_handle(name)
            net_outputs.append(output_tensor.copy_to_cpu())

        return net_outputs

    def _run(self, images, topk=1, transforms=None):
        self.timer.preprocess_time_s.start()
        preprocessed_input = self.preprocess(images, transforms)
        self.timer.preprocess_time_s.end(iter_num=len(images))

        self.timer.inference_time_s.start()
        net_outputs = self.raw_predict(preprocessed_input)
        self.timer.inference_time_s.end(iter_num=1)

        self.timer.postprocess_time_s.start()
        results = self.postprocess(
            net_outputs,
            topk,
            ori_shape=preprocessed_input.get('ori_shape', None),
            transforms=transforms)
        self.timer.postprocess_time_s.end(iter_num=len(images))

        return results

    def predict(self,
                img_file,
                topk=1,
                transforms=None,
                warmup_iters=0,
                repeats=1):
        """ 图片预测
            Args:
                img_file(List[np.ndarray or str], str or np.ndarray):
                    图像路径；或者是解码后的排列格式为（H, W, C）且类型为float32且为BGR格式的数组。
                topk(int): 分类预测时使用，表示预测前topk的结果。默认值为1。
                transforms (paddlex.transforms): 数据预处理操作。默认值为None, 即使用`model.yml`中保存的数据预处理操作。
                warmup_iters (int): 预热轮数，用于评估模型推理以及前后处理速度。若大于1，会预先重复预测warmup_iters，而后才开始正式的预测及其速度评估。默认为0。
                repeats (int): 重复次数，用于评估模型推理以及前后处理速度。若大于1，会预测repeats次取时间平均值。默认值为1。
        """
        if repeats < 1:
            logging.error("`repeats` must be greater than 1.", exit=True)
        if transforms is None and not hasattr(self._model, 'test_transforms'):
            raise Exception("Transforms need to be defined, now is None.")
        if transforms is None:
            transforms = self._model.test_transforms
        if isinstance(img_file, (str, np.ndarray)):
            images = [img_file]
        else:
            images = img_file

        for _ in range(warmup_iters):
            self._run(images=images, topk=topk, transforms=transforms)
        self.timer.reset()

        for _ in range(repeats):
            results = self._run(
                images=images, topk=topk, transforms=transforms)

        self.timer.repeats = repeats
        self.timer.info(average=True)

        return results
