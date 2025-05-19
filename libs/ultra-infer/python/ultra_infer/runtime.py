# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import absolute_import
import logging
import numpy as np
from . import ModelFormat
from . import c_lib_wrap as C


class Runtime:
    """UltraInfer Runtime object."""

    def __init__(self, runtime_option):
        """Initialize a UltraInfer Runtime object.

        :param runtime_option: (ultra_infer.RuntimeOption)Options for UltraInfer Runtime
        """

        self._runtime = C.Runtime()
        self.runtime_option = runtime_option
        assert self._runtime.init(
            self.runtime_option._option
        ), "Initialize Runtime Failed!"

    def forward(self, *inputs):
        """[Only for Poros backend] Inference with input data for poros

        :param data: (list[str : numpy.ndarray])The input data list
        :return list of numpy.ndarray
        """
        if self.runtime_option._option.model_format != ModelFormat.TORCHSCRIPT:
            raise Exception(
                "The forward function is only used for Poros backend, please call infer function"
            )
        inputs_dict = dict()
        for i in range(len(inputs)):
            inputs_dict["x" + str(i)] = inputs[i]
        return self.infer(inputs_dict)

    def infer(self, data):
        """Inference with input data.

        :param data: (dict[str : numpy.ndarray])The input data dict, key value must keep same with the loaded model
        :return list of numpy.ndarray
        """
        assert isinstance(data, dict) or isinstance(
            data, list
        ), "The input data should be type of dict or list."
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, np.ndarray) and not v.data.contiguous:
                    data[k] = np.ascontiguousarray(data[k])

        return self._runtime.infer(data)

    def bind_input_tensor(self, name, fdtensor):
        """Bind FDTensor by name, no copy and share input memory

        :param name: (str)The name of input data.
        :param fdtensor: (ultra_infer.FDTensor)The input FDTensor.
        """
        self._runtime.bind_input_tensor(name, fdtensor)

    def bind_output_tensor(self, name, fdtensor):
        """Bind FDTensor by name, no copy and share output memory

        :param name: (str)The name of output data.
        :param fdtensor: (ultra_infer.FDTensor)The output FDTensor.
        """
        self._runtime.bind_output_tensor(name, fdtensor)

    def zero_copy_infer(self):
        """No params inference the model.

        the input and output data need to pass through the bind_input_tensor and get_output_tensor interfaces.
        """
        self._runtime.infer()

    def get_output_tensor(self, name):
        """Get output FDTensor by name, no copy and share backend output memory

        :param name: (str)The name of output data.
        :return ultra_infer.FDTensor
        """
        return self._runtime.get_output_tensor(name)

    def compile(self, warm_datas):
        """[Only for Poros backend] compile with prewarm data for poros

        :param data: (list[str : numpy.ndarray])The prewarm data list
        :return TorchScript Model
        """
        if self.runtime_option._option.model_format != ModelFormat.TORCHSCRIPT:
            raise Exception(
                "The compile function is only used for Poros backend, please call infer function"
            )
        assert isinstance(warm_datas, list), "The prewarm data should be type of list."
        for i in range(len(warm_datas)):
            warm_data = warm_datas[i]
            if isinstance(warm_data[0], np.ndarray):
                warm_data = list(data for data in warm_data)
            else:
                warm_data = list(data.numpy() for data in warm_data)
            warm_datas[i] = warm_data
        return self._runtime.compile(warm_datas, self.runtime_option._option)

    def num_inputs(self):
        """Get number of inputs of the loaded model."""
        return self._runtime.num_inputs()

    def num_outputs(self):
        """Get number of outputs of the loaded model."""
        return self._runtime.num_outputs()

    def get_input_info(self, index):
        """Get input information of the loaded model.

        :param index: (int)Index of the input
        :return ultra_infer.TensorInfo
        """
        assert isinstance(
            index, int
        ), "The input parameter index should be type of int."
        assert (
            index < self.num_inputs()
        ), "The input parameter index:{} should less than number of inputs:{}.".format(
            index, self.num_inputs
        )
        return self._runtime.get_input_info(index)

    def get_output_info(self, index):
        """Get output information of the loaded model.

        :param index: (int)Index of the output
        :return ultra_infer.TensorInfo
        """
        assert isinstance(
            index, int
        ), "The input parameter index should be type of int."
        assert (
            index < self.num_outputs()
        ), "The input parameter index:{} should less than number of outputs:{}.".format(
            index, self.num_outputs
        )
        return self._runtime.get_output_info(index)

    def get_profile_time(self):
        """Get profile time of Runtime after the profile process is done."""
        return self._runtime.get_profile_time()


class RuntimeOption:
    """Options for UltraInfer Runtime."""

    __slots__ = ["_option"]

    def __init__(self):
        """Initialize a UltraInfer RuntimeOption object."""

        self._option = C.RuntimeOption()

    def set_model_path(
        self, model_path, params_path="", model_format=ModelFormat.PADDLE
    ):
        """Set path of model file and parameters file

        :param model_path: (str)Path of model file
        :param params_path: (str)Path of parameters file
        :param model_format: (ModelFormat)Format of model, support ModelFormat.PADDLE/ModelFormat.ONNX/ModelFormat.TORCHSCRIPT
        """
        return self._option.set_model_path(model_path, params_path, model_format)

    def set_model_buffer(
        self, model_buffer, params_buffer="", model_format=ModelFormat.PADDLE
    ):
        """Specify the memory buffer of model and parameter. Used when model and params are loaded directly from memory
        :param model_buffer: (bytes)The memory buffer of model
        :param params_buffer: (bytes)The memory buffer of the parameters
        :param model_format: (ModelFormat)Format of model, support ModelFormat.PADDLE/ModelFormat.ONNX/ModelFormat.TORCHSCRIPT
        """
        return self._option.set_model_buffer(model_buffer, params_buffer, model_format)

    def use_gpu(self, device_id=0):
        """Inference with Nvidia GPU

        :param device_id: (int)The index of GPU will be used for inference, default 0
        """
        if not C.is_built_with_gpu():
            logging.warning(
                "The installed ultra_infer-python package is not built with GPU, will force to use CPU. To use GPU, following the commands to install ultra_infer-gpu-python."
            )
            return
        return self._option.use_gpu(device_id)

    def use_kunlunxin(
        self,
        device_id=0,
        l3_workspace_size=16 * 1024 * 1024,
        locked=False,
        autotune=True,
        autotune_file="",
        precision="int16",
        adaptive_seqlen=False,
        enable_multi_stream=False,
        gm_default_size=0,
    ):
        """Inference with KunlunXin XPU

        :param device_id: (int)The index of KunlunXin XPU will be used for inference, default 0
        :param l3_workspace_size: (int)The size of the video memory allocated by the l3 cache, the maximum is 16M, default 16M
        :param locked: (bool)Whether the allocated L3 cache can be locked. If false, it means that the L3 cache is not locked,
                        and the allocated L3 cache can be shared by multiple models, and multiple models
        :param autotune: (bool)Whether to autotune the conv operator in the model.
                        If true, when the conv operator of a certain dimension is executed for the first time,
                        it will automatically search for a better algorithm to improve the performance of subsequent conv operators of the same dimension.
        :param autotune_file: (str)Specify the path of the autotune file. If autotune_file is specified,
                        the algorithm specified in the file will be used and autotune will not be performed again.
        :param precision: (str)Calculation accuracy of multi_encoder
        :param adaptive_seqlen: (bool)adaptive_seqlen Is the input of multi_encoder variable length
        :param enable_multi_stream: (bool)Whether to enable the multi stream of KunlunXin XPU.
        :param gm_default_size The default size of context global memory of KunlunXin XPU.
        """
        return self._option.use_kunlunxin(
            device_id,
            l3_workspace_size,
            locked,
            autotune,
            autotune_file,
            precision,
            adaptive_seqlen,
            enable_multi_stream,
            gm_default_size,
        )

    def use_cpu(self):
        """Inference with CPU"""
        return self._option.use_cpu()

    def use_rknpu2(
        self, rknpu2_name=C.CpuName.RK356X, rknpu2_core=C.CoreMask.RKNN_NPU_CORE_AUTO
    ):
        return self._option.use_rknpu2(rknpu2_name, rknpu2_core)

    def use_sophgo(self):
        """Inference with SOPHGO TPU"""
        return self._option.use_sophgo()

    def use_ascend(self, device_id=0):
        """Inference with Huawei Ascend NPU"""
        return self._option.use_ascend(device_id)

    def disable_valid_backend_check(self):
        """Disable checking validity of backend during inference"""
        return self._option.disable_valid_backend_check()

    def enable_valid_backend_check(self):
        """Enable checking validity of backend during inference"""
        return self._option.enable_valid_backend_check()

    def set_cpu_thread_num(self, thread_num=-1):
        """Set number of threads if inference with CPU

        :param thread_num: (int)Number of threads, if not positive, means the number of threads is decided by the backend, default -1
        """
        return self._option.set_cpu_thread_num(thread_num)

    def set_ort_graph_opt_level(self, level=-1):
        """Set graph optimization level for ONNX Runtime backend

        :param level: (int)Optimization level, -1 means the default setting
        """
        logging.warning(
            "`RuntimeOption.set_ort_graph_opt_level` will be deprecated in v1.2.0, please use `RuntimeOption.graph_optimize_level = 99` instead."
        )
        self._option.ort_option.graph_optimize_level = level

    def use_paddle_backend(self):
        """Use Paddle Inference backend, support inference Paddle model on CPU/Nvidia GPU."""
        return self._option.use_paddle_backend()

    def use_paddle_infer_backend(self):
        """Wrapper function of use_paddle_backend(), use Paddle Inference backend, support inference Paddle model on CPU/Nvidia GPU."""
        return self.use_paddle_backend()

    def use_poros_backend(self):
        """Use Poros backend, support inference TorchScript model on CPU/Nvidia GPU."""
        return self._option.use_poros_backend()

    def use_ort_backend(self):
        """Use ONNX Runtime backend, support inference Paddle/ONNX model on CPU/Nvidia GPU."""
        return self._option.use_ort_backend()

    def use_tvm_backend(self):
        """Use TVM Runtime backend, support inference TVM model on CPU."""
        return self._option.use_tvm_backend()

    def use_trt_backend(self):
        """Use TensorRT backend, support inference Paddle/ONNX model on Nvidia GPU."""
        return self._option.use_trt_backend()

    def use_openvino_backend(self):
        """Use OpenVINO backend, support inference Paddle/ONNX model on CPU."""
        return self._option.use_openvino_backend()

    def use_lite_backend(self):
        """Use Paddle Lite backend, support inference Paddle model on ARM CPU."""
        return self._option.use_lite_backend()

    def use_paddle_lite_backend(self):
        """Wrapper function of use_lite_backend(), use Paddle Lite backend, support inference Paddle model on ARM CPU."""
        return self.use_lite_backend()

    def use_om_backend(self):
        """Use Om backend, support inference Om model on NPU"""
        return self._option.use_om_backend()

    def set_lite_context_properties(self, context_properties):
        """Set nnadapter context properties for Paddle Lite backend."""
        logging.warning(
            "`RuntimeOption.set_lite_context_properties` will be deprecated in v1.2.0, please use `RuntimeOption.paddle_lite_option.nnadapter_context_properties = ...` instead."
        )
        self._option.paddle_lite_option.nnadapter_context_properties = (
            context_properties
        )

    def set_lite_model_cache_dir(self, model_cache_dir):
        """Set nnadapter model cache dir for Paddle Lite backend."""
        logging.warning(
            "`RuntimeOption.set_lite_model_cache_dir` will be deprecated in v1.2.0, please use `RuntimeOption.paddle_lite_option.nnadapter_model_cache_dir = ...` instead."
        )

        self._option.paddle_lite_option.nnadapter_model_cache_dir = model_cache_dir

    def set_lite_dynamic_shape_info(self, dynamic_shape_info):
        """Set nnadapter dynamic shape info for Paddle Lite backend."""
        logging.warning(
            "`RuntimeOption.set_lite_dynamic_shape_info` will be deprecated in v1.2.0, please use `RuntimeOption.paddle_lite_option.nnadapter_dynamic_shape_info = ...` instead."
        )
        self._option.paddle_lite_option.nnadapter_dynamic_shape_info = (
            dynamic_shape_info
        )

    def set_lite_subgraph_partition_path(self, subgraph_partition_path):
        """Set nnadapter subgraph partition path for Paddle Lite backend."""
        logging.warning(
            "`RuntimeOption.set_lite_subgraph_partition_path` will be deprecated in v1.2.0, please use `RuntimeOption.paddle_lite_option.nnadapter_subgraph_partition_config_path = ...` instead."
        )
        self._option.paddle_lite_option.nnadapter_subgraph_partition_config_path = (
            subgraph_partition_path
        )

    def set_lite_subgraph_partition_config_buffer(self, subgraph_partition_buffer):
        """Set nnadapter subgraph partition buffer for Paddle Lite backend."""
        logging.warning(
            "`RuntimeOption.set_lite_subgraph_partition_buffer` will be deprecated in v1.2.0, please use `RuntimeOption.paddle_lite_option.nnadapter_subgraph_partition_config_buffer = ...` instead."
        )
        self._option.paddle_lite_option.nnadapter_subgraph_partition_config_buffer = (
            subgraph_partition_buffer
        )

    def set_lite_mixed_precision_quantization_config_path(
        self, mixed_precision_quantization_config_path
    ):
        """Set nnadapter mixed precision quantization config path for Paddle Lite backend.."""
        logging.warning(
            "`RuntimeOption.set_lite_mixed_precision_quantization_config_path` will be deprecated in v1.2.0, please use `RuntimeOption.paddle_lite_option.nnadapter_mixed_precision_quantization_config_path = ...` instead."
        )
        self._option.paddle_lite_option.nnadapter_mixed_precision_quantization_config_path = (
            mixed_precision_quantization_config_path
        )

    def set_paddle_mkldnn(self, use_mkldnn=True):
        """Enable/Disable MKLDNN while using Paddle Inference backend, mkldnn is enabled by default."""
        logging.warning(
            "`RuntimeOption.set_paddle_mkldnn` will be derepcated in v1.2.0, please use `RuntimeOption.paddle_infer_option.enable_mkldnn = True` instead."
        )
        self._option.paddle_infer_option.enable_mkldnn = True

    def set_openvino_device(self, name="CPU"):
        """Set device name for OpenVINO, default 'CPU', can also be 'AUTO', 'GPU', 'GPU.1'....
        This interface is deprecated, please use `RuntimeOption.openvino_option.set_device` instead.
        """
        logging.warning(
            "`RuntimeOption.set_openvino_device` will be deprecated in v1.2.0, please use `RuntimeOption.openvino_option.set_device` instead."
        )
        self._option.openvino_option.set_device(name)

    def set_openvino_shape_info(self, shape_info):
        """Set shape information of the models' inputs, used for GPU to fix the shape
           This interface is deprecated, please use `RuntimeOption.openvino_option.set_shape_info` instead.

        :param shape_info: (dict{str, list of int})Shape information of model's inputs, e.g {"image": [1, 3, 640, 640], "scale_factor": [1, 2]}
        """
        logging.warning(
            "`RuntimeOption.set_openvino_shape_info` will be deprecated in v1.2.0, please use `RuntimeOption.openvino_option.set_shape_info` instead."
        )
        self._option.openvino_option.set_shape_info(shape_info)

    def set_openvino_cpu_operators(self, operators):
        """While using OpenVINO backend and intel GPU, this interface specifies unsupported operators to run on CPU
           This interface is deprecated, please use `RuntimeOption.openvino_option.set_cpu_operators` instead.

        :param operators: (list of string)list of operators' name, e.g ["MulticlasNms"]
        """
        logging.warning(
            "`RuntimeOption.set_openvino_cpu_operators` will be deprecated in v1.2.0, please use `RuntimeOption.openvino_option.set_cpu_operators` instead."
        )
        self._option.openvino_option.set_cpu_operators(operators)

    def enable_paddle_log_info(self):
        """Enable print out the debug log information while using Paddle Inference backend, the log information is disabled by default."""
        logging.warning(
            "RuntimeOption.enable_paddle_log_info` will be deprecated in v1.2.0, please use `RuntimeOption.paddle_infer_option.enable_log_info = True` instead."
        )
        self._option.paddle_infer_option.enable_log_info = True

    def disable_paddle_log_info(self):
        """Disable print out the debug log information while using Paddle Inference backend, the log information is disabled by default."""
        logging.warning(
            "RuntimeOption.disable_paddle_log_info` will be deprecated in v1.2.0, please use `RuntimeOption.paddle_infer_option.enable_log_info = False` instead."
        )
        self._option.paddle_infer_option.enable_log_info = False

    def set_paddle_mkldnn_cache_size(self, cache_size):
        """Set size of shape cache while using Paddle Inference backend with MKLDNN enabled, default will cache all the dynamic shape."""
        logging.warning(
            "RuntimeOption.set_paddle_mkldnn_cache_size` will be deprecated in v1.2.0, please use `RuntimeOption.paddle_infer_option.mkldnn_cache_size = {}` instead.".format(
                cache_size
            )
        )
        self._option.paddle_infer_option.mkldnn_cache_size = cache_size

    def enable_lite_fp16(self):
        """Enable half precision inference while using Paddle Lite backend on ARM CPU, fp16 is disabled by default."""
        logging.warning(
            "`RuntimeOption.enable_lite_fp16` will be deprecated in v1.2.0, please use `RuntimeOption.paddle_lite_option.enable_fp16 = True` instead."
        )
        self._option.paddle_lite_option.enable_fp16 = True

    def disable_lite_fp16(self):
        """Disable half precision inference while using Paddle Lite backend on ARM CPU, fp16 is disabled by default."""
        logging.warning(
            "`RuntimeOption.disable_lite_fp16` will be deprecated in v1.2.0, please use `RuntimeOption.paddle_lite_option.enable_fp16 = False` instead."
        )
        self._option.paddle_lite_option.enable_fp16 = False

    def set_lite_power_mode(self, mode):
        """Set POWER mode while using Paddle Lite backend on ARM CPU."""
        logging.warning(
            "`RuntimeOption.set_lite_powermode` will be deprecated in v1.2.0, please use `RuntimeOption.paddle_lite_option.power_mode = {}` instead.".format(
                mode
            )
        )
        self._option.paddle_lite_option.power_mode = mode

    def set_trt_input_shape(
        self, tensor_name, min_shape, opt_shape=None, max_shape=None
    ):
        """Set shape range information while using TensorRT backend with loadding a model contains dynamic input shape. While inference with a new input shape out of the set shape range, the tensorrt engine will be rebuilt to expand the shape range information.

        :param tensor_name: (str)Name of input which has dynamic shape
        :param min_shape: (list of int)Minimum shape of the input, e.g [1, 3, 224, 224]
        :param opt_shape: (list of int)Optimize shape of the input, this often set as the most common input shape, if set to None, it will keep same with min_shape
        :param max_shape: (list of int)Maximum shape of the input, e.g [8, 3, 224, 224], if set to None, it will keep same with the min_shape
        """
        logging.warning(
            "`RuntimeOption.set_trt_input_shape` will be deprecated in v1.2.0, please use `RuntimeOption.trt_option.set_shape()` instead."
        )
        if opt_shape is None and max_shape is None:
            opt_shape = min_shape
            max_shape = min_shape
        else:
            assert (
                opt_shape is not None and max_shape is not None
            ), "Set min_shape only, or set min_shape, opt_shape, max_shape both."
        return self._option.trt_option.set_shape(
            tensor_name, min_shape, opt_shape, max_shape
        )

    def set_trt_input_data(
        self, tensor_name, min_input_data, opt_input_data=None, max_input_data=None
    ):
        """Set input data while using TensorRT backend with loadding a model contains dynamic input shape.

        :param tensor_name: (str)Name of input which has dynamic shape
        :param min_input_data: (list of int)Input data for Minimum shape of the input.
        :param opt_input_data: (list of int)Input data for Optimize shape of the input, if set to None, it will keep same with min_input_data
        :param max_input_data: (list of int)Input data for Maximum shape of the input, if set to None, it will keep same with the min_input_data
        """
        logging.warning(
            "`RuntimeOption.set_trt_input_data` will be deprecated in v1.2.0, please use `RuntimeOption.trt_option.set_input_data()` instead."
        )
        if opt_input_data is None and max_input_data is None:
            opt_input_data = min_input_data
            opt_input_data = min_input_data
        else:
            assert (
                opt_input_data is not None and max_input_data is not None
            ), "Set min_input_data only, or set min_input_data, opt_input_data, max_input_data both."
        return self._option.trt_option.set_input_data(
            tensor_name, min_input_data, opt_input_data, max_input_data
        )

    def set_trt_cache_file(self, cache_file_path):
        """Set a cache file path while using TensorRT backend. While loading a Paddle/ONNX model with set_trt_cache_file("./tensorrt_cache/model.trt"), if file `./tensorrt_cache/model.trt` exists, it will skip building tensorrt engine and load the cache file directly; if file `./tensorrt_cache/model.trt` doesn't exist, it will building tensorrt engine and save the engine as binary string to the cache file.

        :param cache_file_path: (str)Path of tensorrt cache file
        """
        logging.warning(
            "`RuntimeOption.set_trt_cache_file` will be deprecated in v1.2.0, please use `RuntimeOption.trt_option.serialize_file = {}` instead.".format(
                cache_file_path
            )
        )
        self._option.trt_option.serialize_file = cache_file_path

    def enable_trt_fp16(self):
        """Enable half precision inference while using TensorRT backend, notice that not all the Nvidia GPU support FP16, in those cases, will fallback to FP32 inference."""
        logging.warning(
            "`RuntimeOption.enable_trt_fp16` will be deprecated in v1.2.0, please use `RuntimeOption.trt_option.enable_fp16 = True` instead."
        )
        self._option.trt_option.enable_fp16 = True

    def disable_trt_fp16(self):
        """Disable half precision inference while suing TensorRT backend."""
        logging.warning(
            "`RuntimeOption.disable_trt_fp16` will be deprecated in v1.2.0, please use `RuntimeOption.trt_option.enable_fp16 = False` instead."
        )
        self._option.trt_option.enable_fp16 = False

    def enable_pinned_memory(self):
        """Enable pinned memory. Pinned memory can be utilized to speedup the data transfer between CPU and GPU. Currently it's only supported in TRT backend and Paddle Inference backend."""
        return self._option.enable_pinned_memory()

    def disable_pinned_memory(self):
        """Disable pinned memory."""
        return self._option.disable_pinned_memory()

    def enable_paddle_to_trt(self):
        """While using TensorRT backend, enable_paddle_to_trt() will change to use Paddle Inference backend, and use its integrated TensorRT instead."""
        logging.warning(
            "`RuntimeOption.enable_paddle_to_trt` will be deprecated in v1.2.l0, if you want to run tensorrt with Paddle Inference backend, please use the following method, "
        )
        logging.warning("    ==============================================")
        logging.warning("    import ultra_infer as fd")
        logging.warning("    option = fd.RuntimeOption()")
        logging.warning("    option.use_gpu(0)")
        logging.warning("    option.use_paddle_infer_backend()")
        logging.warning("    option.paddle_infer_option.enable_trt = True")
        logging.warning("    ==============================================")
        self._option.use_paddle_backend()
        self._option.paddle_infer_option.enable_trt = True

    def set_trt_max_workspace_size(self, trt_max_workspace_size):
        """Set max workspace size while using TensorRT backend."""
        logging.warning(
            "`RuntimeOption.set_trt_max_workspace_size` will be deprecated in v1.2.0, please use `RuntimeOption.trt_option.max_workspace_size = {}` instead.".format(
                trt_max_workspace_size
            )
        )
        self._option.trt_option.max_workspace_size = trt_max_workspace_size

    def set_trt_max_batch_size(self, trt_max_batch_size):
        """Set max batch size while using TensorRT backend."""
        logging.warning(
            "`RuntimeOption.set_trt_max_batch_size` will be deprecated in v1.2.0, please use `RuntimeOption.trt_option.max_batch_size = {}` instead.".format(
                trt_max_batch_size
            )
        )
        self._option.trt_option.max_batch_size = trt_max_batch_size

    def enable_paddle_trt_collect_shape(self):
        """Enable collect subgraph shape information while using Paddle Inference with TensorRT"""
        logging.warning(
            "`RuntimeOption.enable_paddle_trt_collect_shape` will be deprecated in v1.2.0, please use `RuntimeOption.paddle_infer_option.collect_trt_shape = True` instead."
        )
        self._option.paddle_infer_option.collect_trt_shape = True

    def disable_paddle_trt_collect_shape(self):
        """Disable collect subgraph shape information while using Paddle Inference with TensorRT"""
        logging.warning(
            "`RuntimeOption.disable_paddle_trt_collect_shape` will be deprecated in v1.2.0, please use `RuntimeOption.paddle_infer_option.collect_trt_shape = False` instead."
        )
        self._option.paddle_infer_option.collect_trt_shape = False

    def delete_paddle_backend_pass(self, pass_name):
        """Delete pass by name in paddle backend"""
        logging.warning(
            "`RuntimeOption.delete_paddle_backend_pass` will be deprecated in v1.2.0, please use `RuntimeOption.paddle_infer_option.delete_pass` instead."
        )
        self._option.paddle_infer_option.delete_pass(pass_name)

    def disable_paddle_trt_ops(self, ops):
        """Disable some ops in paddle trt backend"""
        logging.warning(
            "`RuntimeOption.disable_paddle_trt_ops` will be deprecated in v1.2.0, please use `RuntimeOption.paddle_infer_option.disable_trt_ops()` instead."
        )
        self._option.disable_trt_ops(ops)

    def use_ipu(
        self,
        device_num=1,
        micro_batch_size=1,
        enable_pipelining=False,
        batches_per_step=1,
    ):
        return self._option.use_ipu(
            device_num, micro_batch_size, enable_pipelining, batches_per_step
        )

    def set_ipu_config(
        self,
        enable_fp16=False,
        replica_num=1,
        available_memory_proportion=1.0,
        enable_half_partial=False,
    ):
        logging.warning(
            "`RuntimeOption.set_ipu_config` will be deprecated in v1.2.0, please use `RuntimeOption.paddle_infer_option.set_ipu_config()` instead."
        )
        self._option.paddle_infer_option.set_ipu_config(
            enable_fp16, replica_num, available_memory_proportion, enable_half_partial
        )

    @property
    def poros_option(self):
        """Get PorosBackendOption object to configure Poros backend

        :return PorosBackendOption
        """
        return self._option.poros_option

    @property
    def paddle_lite_option(self):
        """Get LiteBackendOption object to configure Paddle Lite backend

        :return LiteBackendOption
        """
        return self._option.paddle_lite_option

    @property
    def openvino_option(self):
        """Get OpenVINOOption object to configure OpenVINO backend

        :return OpenVINOOption
        """
        return self._option.openvino_option

    @property
    def ort_option(self):
        """Get OrtBackendOption object to configure ONNX Runtime backend

        :return OrtBackendOption
        """
        return self._option.ort_option

    @property
    def trt_option(self):
        """Get TrtBackendOption object to configure TensorRT backend

        :return TrtBackendOption
        """
        return self._option.trt_option

    @property
    def paddle_infer_option(self):
        """Get PaddleBackendOption object to configure Paddle Inference backend

        :return PaddleBackendOption
        """
        return self._option.paddle_infer_option

    def enable_profiling(self, inclue_h2d_d2h=False, repeat=100, warmup=50):
        """Set the profile mode as 'true'.
        :param inclue_h2d_d2h Whether to include time of H2D_D2H for time of runtime.
        :param repeat Repeat times for runtime inference.
        :param warmup Warmup times for runtime inference.
        """
        return self._option.enable_profiling(inclue_h2d_d2h, repeat, warmup)

    def disable_profiling(self):
        """Set the profile mode as 'false'."""
        return self._option.disable_profiling()

    def set_external_raw_stream(self, cuda_stream):
        """Set the external raw stream used by ultra_infer runtime."""
        self._option.set_external_raw_stream(cuda_stream)

    def __repr__(self):
        attrs = dir(self._option)
        message = "RuntimeOption(\n"
        for attr in attrs:
            if attr.startswith("__"):
                continue
            if hasattr(getattr(self._option, attr), "__call__"):
                continue
            message += "  {} : {}\t\n".format(attr, getattr(self._option, attr))
        message.strip("\n")
        message += ")"
        return message
