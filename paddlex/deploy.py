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
import numpy as np
import yaml
from paddle.inference import Config
from paddle.inference import create_predictor
from paddle.inference import PrecisionType
from paddlex.cv.transforms import build_transforms
from paddlex.utils import logging


class Predictor(object):
    def __init__(self,
                 model_dir,
                 use_gpu=True,
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
                use_gpu: 是否使用gpu，默认True
                gpu_id: 使用gpu的id，默认0
                cpu_thread_num=1：使用cpu进行预测时的线程数，默认为1
                use_mkl: 是否使用mkldnn计算库，CPU情况下使用，默认False
                mkl_thread_num: mkldnn计算线程数，默认为4
                use_trt: 是否使用TensorRT，默认False
                use_glog: 是否启用glog日志, 默认False
                memory_optimize: 是否启动内存优化，默认True
                max_trt_batch_size: 在使用TensorRT时配置的最大batch size，默认1
                trt_precision_mode：在使用TensorRT时采用的精度，默认float32
        """
        if not osp.isdir(model_dir):
            logging.error(
                "{} is not a valid model directory.".format(model_dir),
                exit=True)

        if trt_precision_mode == 'float32':
            trt_precision_mode = PrecisionType.Float32
        elif trt_precision_mode == 'float16':
            trt_precision_mode = PrecisionType.Float16
        else:
            logging.error(
                "TensorRT precision mode {} is invalid. Supported modes are float32 and float16."
                .format(trt_precision_mode),
                exit=True)

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
            prog_file=osp.join(self.model_dir, 'model.pdmodel'),
            params_file=osp.join(self.model_dir, 'model.pdiparams'))

        if use_gpu:
            # 设置GPU初始显存(单位M)和Device ID
            config.enable_use_gpu(100, gpu_id)
            config.switch_ir_optim(True)
            if use_trt:
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

        if use_glog:
            config.enable_glog_info()
        else:
            config.disable_glog_info()
        if memory_optimize:
            config.enable_memory_optim()
        config.switch_use_feed_fetch_ops(False)
        predictor = create_predictor(config)
        return predictor
