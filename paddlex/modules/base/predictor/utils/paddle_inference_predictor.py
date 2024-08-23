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
import paddle
from paddle.inference import Config, create_predictor

from .....utils import logging


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
            config.enable_use_gpu(200, option.device_id)
            if paddle.is_compiled_with_rocm():
                os.environ['FLAGS_conv_workspace_size_limit'] = '2000'
            else:
                config.enable_new_ir(True)
        elif option.device == 'npu':
            config.enable_custom_device('npu')
            os.environ["FLAGS_npu_jit_compile"] = "0"
            os.environ["FLAGS_use_stride_kernel"] = "0"
            os.environ["FLAGS_allocator_strategy"] = "auto_growth"
            os.environ[
                "CUSTOM_DEVICE_BLACK_LIST"] = "pad3d,pad3d_grad,set_value,set_value_with_tensor"
            os.environ["FLAGS_npu_scale_aclnn"] = "True"
            os.environ["FLAGS_npu_split_aclnn"] = "True"
        elif option.device == 'xpu':
            os.environ["BKCL_FORCE_SYNC"] = "1"
            os.environ["BKCL_TIMEOUT"] = "1800"
            os.environ["FLAGS_use_stride_kernel"] = "0"
        elif option.device == 'mlu':
            config.enable_custom_device('mlu')
            os.environ["FLAGS_use_stride_kernel"] = "0"
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
