# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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


import os
from functools import wraps, partial
from paddle.inference import Config, create_predictor

from .....utils import logging


def register(register_map, key):
    """register the option setting func
    """

    def decorator(func):
        register_map[key] = func

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


class PaddleInferenceOption(object):
    """Paddle Inference Engine Option
    """
    SUPPORT_RUN_MODE = ('paddle', 'trt_fp32', 'trt_fp16', 'trt_int8', 'mkldnn',
                        'mkldnn_bf16')
    SUPPORT_DEVICE = ('gpu', 'cpu', 'npu', 'xpu', 'mlu')
    _REGISTER_MAP = {}

    register2self = partial(register, _REGISTER_MAP)

    def __init__(self, **kwargs):
        super().__init__()
        self._cfg = {}
        self._init_option(**kwargs)

    def _init_option(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self._REGISTER_MAP:
                raise Exception(
                    f"{k} is not supported to set! The supported option is: \
{list(self._REGISTER_MAP.keys())}")
            self._REGISTER_MAP.get(k)(self, v)
        for k, v in self._get_default_config().items():
            self._cfg.setdefault(k, v)

    def _get_default_config(cls):
        """ get default config """
        return {
            'run_mode': 'paddle',
            'batch_size': 1,
            'device': 'gpu',
            'min_subgraph_size': 3,
            'shape_info_filename': None,
            'trt_calib_mode': False,
            'cpu_threads': 1,
            'trt_use_static': False
        }

    @register2self('run_mode')
    def set_run_mode(self, run_mode: str):
        """set run mode
        """
        if run_mode not in self.SUPPORT_RUN_MODE:
            support_run_mode_str = ", ".join(self.SUPPORT_RUN_MODE)
            raise ValueError(
                f"`run_mode` must be {support_run_mode_str}, but received {repr(run_mode)}."
            )
        self._cfg['run_mode'] = run_mode

    @register2self('batch_size')
    def set_batch_size(self, batch_size: int):
        """set batch size
        """
        if not isinstance(batch_size, int) or batch_size < 1:
            raise Exception()
        self._cfg['batch_size'] = batch_size

    @register2self('device')
    def set_device(self, device: str):
        """set device
        """
        device = device.split(":")[0]
        if device.lower() not in self.SUPPORT_DEVICE:
            support_run_mode_str = ", ".join(self.SUPPORT_DEVICE)
            raise ValueError(
                f"`device` must be {support_run_mode_str}, but received {repr(device)}."
            )
        self._cfg['device'] = device.lower()

    @register2self('min_subgraph_size')
    def set_min_subgraph_size(self, min_subgraph_size: int):
        """set min subgraph size
        """
        if not isinstance(min_subgraph_size, int):
            raise Exception()
        self._cfg['min_subgraph_size'] = min_subgraph_size

    @register2self('shape_info_filename')
    def set_shape_info_filename(self, shape_info_filename: str):
        """set shape info filename
        """
        self._cfg['shape_info_filename'] = shape_info_filename

    @register2self('trt_calib_mode')
    def set_trt_calib_mode(self, trt_calib_mode):
        """set trt calib mode
        """
        self._cfg['trt_calib_mode'] = trt_calib_mode

    @register2self('cpu_threads')
    def set_cpu_threads(self, cpu_threads):
        """set cpu threads
        """
        if not isinstance(cpu_threads, int) or cpu_threads < 1:
            raise Exception()
        self._cfg['cpu_threads'] = cpu_threads

    @register2self('trt_use_static')
    def set_trt_use_static(self, trt_use_static):
        """set trt use static
        """
        self._cfg['trt_use_static'] = trt_use_static

    def get_support_run_mode(self):
        """get supported run mode
        """
        return self.SUPPORT_RUN_MODE

    def get_support_device(self):
        """get supported device
        """
        return self.SUPPORT_DEVICE

    def __str__(self):
        return "\n  " + "\n  ".join([f"{k}: {v}" for k, v in self._cfg.items()])

    def __getattr__(self, key):
        if key not in self._cfg:
            raise Exception()
        return self._cfg.get(key)


class _PaddleInferencePredictor(object):
    """ Predictor based on Paddle Inference """

    def __init__(self, param_path, model_path, option, delete_pass=[]):
        super().__init__()
        self.predictor, self.inference_config, self.input_names, self.input_handlers, self.output_handlers = \
self._create(param_path, model_path, option, delete_pass=delete_pass)

    def _create(self, param_path, model_path, option, delete_pass):
        """ _create """
        if not os.path.exists(model_path) or not os.path.exists(param_path):
            raise FileNotFoundError(
                f"Please ensure {model_path} and {param_path} exist.")

        model_buffer, param_buffer = self._read_model_param(model_path,
                                                            param_path)
        config = Config()
        config.set_model_buffer(model_buffer,
                                len(model_buffer), param_buffer,
                                len(param_buffer))

        if option.device == 'gpu':
            config.enable_use_gpu(200, 0)
        elif option.device == 'npu':
            config.enable_custom_device('npu')
        elif option.device == 'xpu':
            config.enable_custom_device('npu')
        elif option.device == 'mlu':
            config.enable_custom_device('mlu')
        else:
            assert option.device == 'cpu'
            config.disable_gpu()
            if 'mkldnn' in option.run_mode:
                try:
                    config.enable_mkldnn()
                    config.set_cpu_math_library_num_threads(option.cpu_threads)
                    if 'bf16' in option.run_mode:
                        config.enable_mkldnn_bfloat16()
                except Exception as e:
                    logging.warning(
                        "MKL-DNN is not available. We will disable MKL-DNN.")

        precision_map = {
            'trt_int8': Config.Precision.Int8,
            'trt_fp32': Config.Precision.Float32,
            'trt_fp16': Config.Precision.Half
        }
        if option.run_mode in precision_map.keys():
            config.enable_tensorrt_engine(
                workspace_size=(1 << 25) * option.batch_size,
                max_batch_size=option.batch_size,
                min_subgraph_size=option.min_subgraph_size,
                precision_mode=precision_map[option.run_mode],
                trt_use_static=option.trt_use_static,
                use_calib_mode=option.trt_calib_mode)

            if option.shape_info_filename is not None:
                if not os.path.exists(option.shape_info_filename):
                    config.collect_shape_range_info(option.shape_info_filename)
                    logging.info(
                        f"Dynamic shape info is collected into: {option.shape_info_filename}"
                    )
                else:
                    logging.info(
                        f"A dynamic shape info file ( {option.shape_info_filename} ) already exists. \
No need to generate again.")
                config.enable_tuned_tensorrt_dynamic_shape(
                    option.shape_info_filename, True)

        # Disable paddle inference logging
        config.disable_glog_info()
        for del_p in delete_pass:
            config.delete_pass(del_p)
        # Enable shared memory
        config.enable_memory_optim()
        config.switch_ir_optim(True)
        # Disable feed, fetch OP, needed by zero_copy_run
        config.switch_use_feed_fetch_ops(False)
        predictor = create_predictor(config)

        # Get input and output handlers
        input_names = predictor.get_input_names()
        input_handlers = []
        output_handlers = []
        for input_name in input_names:
            input_handler = predictor.get_input_handle(input_name)
            input_handlers.append(input_handler)
        output_names = predictor.get_output_names()
        for output_name in output_names:
            output_handler = predictor.get_output_handle(output_name)
            output_handlers.append(output_handler)
        return predictor, config, input_names, input_handlers, output_handlers

    def _read_model_param(self, model_path, param_path):
        """ read model and param """
        model_file = open(model_path, 'rb')
        param_file = open(param_path, 'rb')
        model_buffer = model_file.read()
        param_buffer = param_file.read()
        return model_buffer, param_buffer

    def get_input_names(self):
        """ get input names """
        return self.input_names

    def predict(self, x):
        """ predict """
        for idx in range(len(x)):
            self.input_handlers[idx].reshape(x[idx].shape)
            self.input_handlers[idx].copy_from_cpu(x[idx])

        self.predictor.run()

        res = []
        for out_tensor in self.output_handlers:
            out_arr = out_tensor.copy_to_cpu()
            res.append(out_arr)
        return res
