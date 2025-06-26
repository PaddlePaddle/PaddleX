# copyright (c) 2025 PaddlePaddle Authors. All Rights Reserve.
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

from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import os
import lazy_paddle as paddle
# import paddle
import numpy as np

def get_predictor(
        model_dir: Optional[os.PathLike]=None,
        model_file: Optional[os.PathLike]=None,
        params_file: Optional[os.PathLike]=None,
        device: str='cpu',
        # for gpu
        use_trt: bool=False,
        device_id: int=0,
        # for trt
        use_dynamic_shape: bool=True,
        min_subgraph_size: int=5,
        # for cpu
        cpu_threads: int=1,
        use_mkldnn: bool=False,
        # for trt or mkldnn
        precision: int="fp32"):
    """
    Args:
        model_dir (os.PathLike): root path of model.pdmodel and model.pdiparams.
        model_file (os.PathLike): name of model_file.
        params_file (os.PathLike): name of params_file.
        device (str): Choose the device you want to run, it can be: cpu/gpu, default is cpu.
        use_trt (bool): whether to use TensorRT or not in GPU.
        device_id (int): Choose your device id, only valid when the device is gpu, default 0.
        use_dynamic_shape (bool): use dynamic shape or not in TensorRT.
        use_mkldnn (bool): whether to use MKLDNN or not in CPU.
        cpu_threads (int): num of thread when use CPU.
        precision (str): mode of running (fp32/fp16/bf16/int8).  
    """
    rerun_flag = False
    if device != "gpu" and use_trt:
        raise ValueError(
            "Predict by TensorRT mode: {}, expect device=='gpu', but device == {}".
            format(precision, device))

    model_name = str(model_file).rsplit('.', 1)[0]
    assert model_name == str(params_file).split('.')[0], "The prefix of model_file and params_file should be same."
    config = paddle.inference.Config(model_dir, model_name)

    if paddle.__version__ <= "2.5.2" and paddle.__version__ != "0.0.0":
        config.enable_memory_optim()
    config.switch_ir_optim(True)
    if device == "gpu":
        config.enable_use_gpu(100, device_id)
    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(cpu_threads)
        if use_mkldnn:
            # fp32
            config.enable_mkldnn()
            if precision == "int8":
                config.enable_mkldnn_int8({
                    "conv2d_transpose", "conv2d", "depthwise_conv2d", "pool2d",
                    "transpose2", "elementwise_mul"
                })
                # config.enable_mkldnn_int8()
            elif precision in {"fp16", "bf16"}:
                config.enable_mkldnn_bfloat16()
            print("MKLDNN with {}".format(precision))
    if use_trt:
        if precision == "bf16":
            print("paddle trt does not support bf16, switching to fp16.")
            precision = "fp16"
        precision_map = {
            "int8": paddle.inference.Config.Precision.Int8,
            "fp32": paddle.inference.Config.Precision.Float32,
            "fp16": paddle.inference.Config.Precision.Half,
        }
        assert precision in precision_map.keys()
        pdtxt_name = model_file.split(".")[0] + "_" + precision + ".txt"
        if use_dynamic_shape:
            dynamic_shape_file = os.path.join(model_dir, pdtxt_name)
            if os.path.exists(dynamic_shape_file):
                config.enable_tuned_tensorrt_dynamic_shape(dynamic_shape_file,
                                                           True)
                # for fastspeech2
                config.exp_disable_tensorrt_ops(["reshape2"])
                print("trt set dynamic shape done!")
            else:
                # In order to avoid memory overflow when collecting dynamic shapes, it is changed to use CPU.
                config.disable_gpu()
                config.set_cpu_math_library_num_threads(10)
                config.collect_shape_range_info(dynamic_shape_file)
                print("Start collect dynamic shape...")
                rerun_flag = True

        if not rerun_flag:
            print("Tensor RT with {}".format(precision))
            config.enable_tensorrt_engine(
                workspace_size=1 << 30,
                max_batch_size=1,
                min_subgraph_size=min_subgraph_size,
                precision_mode=precision_map[precision],
                use_static=True,
                use_calib_mode=False, )

    predictor = paddle.inference.create_predictor(config)
    return predictor

def get_am_output(input: list,
                  am_predictor: paddle.nn.Layer,
                  am: str,
                  lang: str='zh',
                  merge_sentences: bool=True,
                  speaker_dict: Optional[os.PathLike]=None,
                  spk_id: int=0,
                  add_blank: bool=False):
    am_name = am[:am.rindex('_')]
    am_dataset = am[am.rindex('_') + 1:]
    am_input_names = am_predictor.get_input_names()
    get_spk_id = False
    get_tone_ids = False
    if am_name == 'speedyspeech':
        get_tone_ids = True
    if am_dataset in {"aishell3", "vctk", "mix", "canton"} and speaker_dict:
        get_spk_id = True
        spk_id = np.array([spk_id])
    if get_tone_ids:
        tone_ids = frontend_dict['tone_ids']
        tones = tone_ids[0].numpy()
        tones_handle = am_predictor.get_input_handle(am_input_names[1])
        tones_handle.reshape(tones.shape)
        tones_handle.copy_from_cpu(tones)
    if get_spk_id:
        spk_id_handle = am_predictor.get_input_handle(am_input_names[1])
        spk_id_handle.reshape(spk_id.shape)
        spk_id_handle.copy_from_cpu(spk_id)
    phone_ids = input
    phones = np.array(phone_ids[0])
    phones_handle = am_predictor.get_input_handle(am_input_names[0])
    phones_handle.reshape(phones.shape)
    phones_handle.copy_from_cpu(phones)

    am_predictor.run()
    am_output_names = am_predictor.get_output_names()
    am_output_handle = am_predictor.get_output_handle(am_output_names[0])
    am_output_data = am_output_handle.copy_to_cpu()
    return [am_output_data]

