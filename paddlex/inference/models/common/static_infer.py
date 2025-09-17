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

import abc
import subprocess
from os import PathLike
from pathlib import Path
from typing import List, Sequence, Union

import numpy as np

from ....utils import logging
from ....utils.deps import class_requires_deps
from ....utils.device import check_supported_device_type
from ....utils.flags import (
    DEBUG,
    DISABLE_MKLDNN_MODEL_BL,
    DISABLE_TRT_MODEL_BL,
    USE_PIR_TRT,
)
from ...utils.benchmark import benchmark, set_inference_operations
from ...utils.hpi import (
    HPIConfig,
    OMConfig,
    ONNXRuntimeConfig,
    OpenVINOConfig,
    TensorRTConfig,
    suggest_inference_backend_and_config,
)
from ...utils.mkldnn_blocklist import MKLDNN_BLOCKLIST
from ...utils.model_paths import get_model_paths
from ...utils.pp_option import PaddlePredictorOption, get_default_run_mode
from ...utils.trt_blocklist import TRT_BLOCKLIST
from ...utils.trt_config import DISABLE_TRT_HALF_OPS_CONFIG

CACHE_DIR = ".cache"

INFERENCE_OPERATIONS = [
    "PaddleInferChainLegacy",
    "MultiBackendInfer",
]
set_inference_operations(INFERENCE_OPERATIONS)


# XXX: Better use Paddle Inference API to do this
def _pd_dtype_to_np_dtype(pd_dtype):
    import paddle

    if pd_dtype == paddle.inference.DataType.FLOAT64:
        return np.float64
    elif pd_dtype == paddle.inference.DataType.FLOAT32:
        return np.float32
    elif pd_dtype == paddle.inference.DataType.INT64:
        return np.int64
    elif pd_dtype == paddle.inference.DataType.INT32:
        return np.int32
    elif pd_dtype == paddle.inference.DataType.UINT8:
        return np.uint8
    elif pd_dtype == paddle.inference.DataType.INT8:
        return np.int8
    else:
        raise TypeError(f"Unsupported data type: {pd_dtype}")


# old trt
def _collect_trt_shape_range_info(
    model_file,
    model_params,
    gpu_id,
    shape_range_info_path,
    dynamic_shapes,
    dynamic_shape_input_data,
):
    import paddle.inference

    dynamic_shape_input_data = dynamic_shape_input_data or {}

    config = paddle.inference.Config(model_file, model_params)
    config.enable_use_gpu(100, gpu_id)
    config.collect_shape_range_info(shape_range_info_path)
    # TODO: Add other needed options
    config.disable_glog_info()
    predictor = paddle.inference.create_predictor(config)

    input_names = predictor.get_input_names()
    for name in dynamic_shapes:
        if name not in input_names:
            raise ValueError(
                f"Invalid input name {repr(name)} found in `dynamic_shapes`"
            )
    for name in input_names:
        if name not in dynamic_shapes:
            raise ValueError(f"Input name {repr(name)} not found in `dynamic_shapes`")
    for name in dynamic_shape_input_data:
        if name not in input_names:
            raise ValueError(
                f"Invalid input name {repr(name)} found in `dynamic_shape_input_data`"
            )
    # It would be better to check if the shapes are valid.

    min_arrs, opt_arrs, max_arrs = {}, {}, {}
    for name, candidate_shapes in dynamic_shapes.items():
        # XXX: Currently we have no way to get the data type of the tensor
        # without creating an input handle.
        handle = predictor.get_input_handle(name)
        dtype = _pd_dtype_to_np_dtype(handle.type())
        min_shape, opt_shape, max_shape = candidate_shapes
        if name in dynamic_shape_input_data:
            min_arrs[name] = np.array(
                dynamic_shape_input_data[name][0], dtype=dtype
            ).reshape(min_shape)
            opt_arrs[name] = np.array(
                dynamic_shape_input_data[name][1], dtype=dtype
            ).reshape(opt_shape)
            max_arrs[name] = np.array(
                dynamic_shape_input_data[name][2], dtype=dtype
            ).reshape(max_shape)
        else:
            min_arrs[name] = np.ones(min_shape, dtype=dtype)
            opt_arrs[name] = np.ones(opt_shape, dtype=dtype)
            max_arrs[name] = np.ones(max_shape, dtype=dtype)

    # `opt_arrs` is used twice to ensure it is the most frequently used.
    for arrs in [min_arrs, opt_arrs, opt_arrs, max_arrs]:
        for name, arr in arrs.items():
            handle = predictor.get_input_handle(name)
            handle.reshape(arr.shape)
            handle.copy_from_cpu(arr)
        predictor.run()

    # HACK: The shape range info will be written to the file only when
    # `predictor` is garbage collected. It works in CPython, but it is
    # definitely a bad idea to count on the implementation-dependent behavior of
    # a garbage collector. Is there a more explicit and deterministic way to
    # handle this?

    # HACK: Manually delete the predictor to trigger its destructor, ensuring that the shape_range_info file would be saved.
    del predictor


# pir trt
def _convert_trt(
    trt_cfg_setting,
    pp_model_file,
    pp_params_file,
    trt_save_path,
    device_id,
    dynamic_shapes,
    dynamic_shape_input_data,
):
    import paddle.inference
    from paddle.tensorrt.export import Input, TensorRTConfig, convert

    def _set_trt_config():
        for attr_name in trt_cfg_setting:
            assert hasattr(
                trt_config, attr_name
            ), f"The `{type(trt_config)}` don't have the attribute `{attr_name}`!"
            setattr(trt_config, attr_name, trt_cfg_setting[attr_name])

    def _get_predictor(model_file, params_file):
        # HACK
        config = paddle.inference.Config(str(model_file), str(params_file))
        config.enable_use_gpu(100, device_id)
        # NOTE: Disable oneDNN to circumvent a bug in Paddle Inference
        config.disable_mkldnn()
        config.disable_glog_info()
        return paddle.inference.create_predictor(config)

    dynamic_shape_input_data = dynamic_shape_input_data or {}

    predictor = _get_predictor(pp_model_file, pp_params_file)
    input_names = predictor.get_input_names()
    for name in dynamic_shapes:
        if name not in input_names:
            raise ValueError(
                f"Invalid input name {repr(name)} found in `dynamic_shapes`"
            )
    for name in input_names:
        if name not in dynamic_shapes:
            raise ValueError(f"Input name {repr(name)} not found in `dynamic_shapes`")
    for name in dynamic_shape_input_data:
        if name not in input_names:
            raise ValueError(
                f"Invalid input name {repr(name)} found in `dynamic_shape_input_data`"
            )

    trt_inputs = []
    for name, candidate_shapes in dynamic_shapes.items():
        # XXX: Currently we have no way to get the data type of the tensor
        # without creating an input handle.
        handle = predictor.get_input_handle(name)
        dtype = _pd_dtype_to_np_dtype(handle.type())
        min_shape, opt_shape, max_shape = candidate_shapes
        if name in dynamic_shape_input_data:
            min_arr = np.array(dynamic_shape_input_data[name][0], dtype=dtype).reshape(
                min_shape
            )
            opt_arr = np.array(dynamic_shape_input_data[name][1], dtype=dtype).reshape(
                opt_shape
            )
            max_arr = np.array(dynamic_shape_input_data[name][2], dtype=dtype).reshape(
                max_shape
            )
        else:
            min_arr = np.ones(min_shape, dtype=dtype)
            opt_arr = np.ones(opt_shape, dtype=dtype)
            max_arr = np.ones(max_shape, dtype=dtype)

        # refer to: https://github.com/PolaKuma/Paddle/blob/3347f225bc09f2ec09802a2090432dd5cb5b6739/test/tensorrt/test_converter_model_resnet50.py
        trt_input = Input((min_arr, opt_arr, max_arr))
        trt_inputs.append(trt_input)

    # Create TensorRTConfig
    trt_config = TensorRTConfig(inputs=trt_inputs)
    _set_trt_config()
    trt_config.save_model_dir = str(trt_save_path)
    pp_model_path = str(pp_model_file.with_suffix(""))
    convert(pp_model_path, trt_config)


def _sort_inputs(inputs, names):
    # NOTE: Adjust input tensors to match the sorted sequence.
    indices = sorted(range(len(names)), key=names.__getitem__)
    inputs = [inputs[indices.index(i)] for i in range(len(inputs))]
    return inputs


# FIXME: Name might be misleading
@benchmark.timeit
class PaddleInferChainLegacy:
    def __init__(self, predictor):
        self.predictor = predictor
        input_names = self.predictor.get_input_names()
        self.input_handles = []
        self.output_handles = []
        for input_name in input_names:
            input_handle = self.predictor.get_input_handle(input_name)
            self.input_handles.append(input_handle)
        output_names = self.predictor.get_output_names()
        for output_name in output_names:
            output_handle = self.predictor.get_output_handle(output_name)
            self.output_handles.append(output_handle)

    def __call__(self, x):
        for input_, input_handle in zip(x, self.input_handles):
            input_handle.reshape(input_.shape)
            input_handle.copy_from_cpu(input_)
        self.predictor.run()
        outputs = [o.copy_to_cpu() for o in self.output_handles]
        return outputs


class StaticInfer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, x: Sequence[np.ndarray]) -> List[np.ndarray]:
        raise NotImplementedError


class PaddleInfer(StaticInfer):
    def __init__(
        self,
        model_name: str,
        model_dir: Union[str, PathLike],
        model_file_prefix: str,
        option: PaddlePredictorOption,
    ) -> None:
        super().__init__()
        self._model_name = model_name
        self.model_dir = Path(model_dir)
        self.model_file_prefix = model_file_prefix
        self._option = option
        self.predictor = self._create()
        self.infer = PaddleInferChainLegacy(self.predictor)

    def __call__(self, x: Sequence[np.ndarray]) -> List[np.ndarray]:
        names = self.predictor.get_input_names()
        if len(names) != len(x):
            raise ValueError(
                f"The number of inputs does not match the model: {len(names)} vs {len(x)}"
            )
        # TODO:
        # Ensure that input tensors follow the model's input sequence without sorting.
        x = _sort_inputs(x, names)
        x = list(map(np.ascontiguousarray, x))
        pred = self.infer(x)
        return pred

    def _check_run_mode(self):
        # TODO: Check if trt is available
        # check avaliable for trt
        if (
            not DISABLE_TRT_MODEL_BL
            and self._option.run_mode.startswith("trt")
            and self._model_name in TRT_BLOCKLIST
            and self._option.device_type == "gpu"
        ):
            logging.warning(
                f"The model({self._model_name}) is not supported to run in trt mode! Using `paddle` instead!"
            )
            self._option.run_mode = "paddle"

        # check avaliable for mkldnn
        elif (
            not DISABLE_MKLDNN_MODEL_BL
            and self._option.run_mode.startswith("mkldnn")
            and self._model_name in MKLDNN_BLOCKLIST
            and self._option.device_type == "cpu"
        ):
            logging.warning(
                f"The model({self._model_name}) is not supported to run in MKLDNN mode! Using `paddle` instead!"
            )
            self._option.run_mode = "paddle"
            return "paddle"

        # check avaliable for model
        if self._model_name == "LaTeX_OCR_rec" and self._option.device_type == "cpu":
            import cpuinfo

            if (
                "GenuineIntel" in cpuinfo.get_cpu_info().get("vendor_id_raw", "")
                and self._option.run_mode != "mkldnn"
            ):
                logging.warning(
                    "Now, the `LaTeX_OCR_rec` model only support `mkldnn` mode when running on Intel CPU devices. So using `mkldnn` instead."
                )
            self._option.run_mode = "mkldnn"

    def _create(
        self,
    ):
        """_create"""
        import paddle
        import paddle.inference

        model_paths = get_model_paths(self.model_dir, self.model_file_prefix)
        if "paddle" not in model_paths:
            raise RuntimeError("No valid PaddlePaddle model found")

        check_supported_device_type(self._option.device_type, self._model_name)
        self._check_run_mode()

        model_file, params_file = model_paths["paddle"]

        if self._option.device_type == "cpu" and self._option.device_id is not None:
            self._option.device_id = None
            logging.debug("`device_id` has been set to None")

        if (
            self._option.device_type in ("gpu", "dcu", "npu", "mlu", "gcu", "xpu")
            and self._option.device_id is None
        ):
            self._option.device_id = 0
            logging.debug("`device_id` has been set to 0")

        # for TRT
        if self._option.run_mode.startswith("trt"):
            assert self._option.device_type.lower() == "gpu", (
                f"`{self._option.run_mode}` is only available on GPU devices, "
                f"but got device_type='{self._option.device_type}'."
            )
            cache_dir = self.model_dir / CACHE_DIR / "paddle"
            config = self._configure_trt(
                model_file,
                params_file,
                cache_dir,
            )
            config.exp_disable_mixed_precision_ops({"feed", "fetch"})
            config.enable_use_gpu(100, self._option.device_id)
        # for Native Paddle and MKLDNN
        else:
            config = paddle.inference.Config(str(model_file), str(params_file))
            if self._option.device_type == "gpu":
                config.exp_disable_mixed_precision_ops({"feed", "fetch"})
                from paddle.inference import PrecisionType

                precision = (
                    PrecisionType.Half
                    if self._option.run_mode == "paddle_fp16"
                    else PrecisionType.Float32
                )
                config.disable_mkldnn()
                config.enable_use_gpu(100, self._option.device_id, precision)
                if hasattr(config, "enable_new_ir"):
                    config.enable_new_ir(self._option.enable_new_ir)
                    if self._option.enable_new_ir and self._option.enable_cinn:
                        config.enable_cinn()
                if hasattr(config, "enable_new_executor"):
                    config.enable_new_executor()
                config.set_optimization_level(3)
            elif self._option.device_type == "npu":
                config.enable_custom_device("npu", self._option.device_id)
                if hasattr(config, "enable_new_ir"):
                    config.enable_new_ir(self._option.enable_new_ir)
                if hasattr(config, "enable_new_executor"):
                    config.enable_new_executor()
            elif self._option.device_type == "xpu":
                config.enable_xpu()
                config.set_xpu_device_id(self._option.device_id)
                if hasattr(config, "enable_new_ir"):
                    config.enable_new_ir(self._option.enable_new_ir)
                if hasattr(config, "enable_new_executor"):
                    config.enable_new_executor()
                config.delete_pass("conv2d_bn_xpu_fuse_pass")
                config.delete_pass("transfer_layout_pass")
            elif self._option.device_type == "mlu":
                config.enable_custom_device("mlu", self._option.device_id)
                if hasattr(config, "enable_new_ir"):
                    config.enable_new_ir(self._option.enable_new_ir)
                if hasattr(config, "enable_new_executor"):
                    config.enable_new_executor()
            elif self._option.device_type == "gcu":
                from paddle_custom_device.gcu import passes as gcu_passes

                gcu_passes.setUp()
                config.enable_custom_device("gcu", self._option.device_id)
                if hasattr(config, "enable_new_ir"):
                    config.enable_new_ir()
                if hasattr(config, "enable_new_executor"):
                    config.enable_new_executor()
                else:
                    pass_builder = config.pass_builder()
                    name = "PaddleX_" + self._option.model_name
                    gcu_passes.append_passes_for_legacy_ir(pass_builder, name)
            elif self._option.device_type == "dcu":
                if hasattr(config, "enable_new_ir"):
                    config.enable_new_ir(self._option.enable_new_ir)
                config.enable_use_gpu(100, self._option.device_id)
                config.disable_mkldnn()
                if hasattr(config, "enable_new_executor"):
                    config.enable_new_executor()
                # XXX: is_compiled_with_rocm() must be True on dcu platform ?
                if paddle.is_compiled_with_rocm():
                    # Delete unsupported passes in dcu
                    config.delete_pass("conv2d_add_act_fuse_pass")
                    config.delete_pass("conv2d_add_fuse_pass")
            elif self._option.device_type == "iluvatar_gpu":
                config.enable_custom_device("iluvatar_gpu", int(self._option.device_id))
                if hasattr(config, "enable_new_ir"):
                    config.enable_new_ir(self._option.enable_new_ir)
                if hasattr(config, "enable_new_executor"):
                    config.enable_new_executor()
            else:
                assert self._option.device_type == "cpu"
                config.disable_gpu()
                if "mkldnn" in self._option.run_mode:
                    config.enable_mkldnn()
                    if "bf16" in self._option.run_mode:
                        config.enable_mkldnn_bfloat16()
                    config.set_mkldnn_cache_capacity(self._option.mkldnn_cache_capacity)
                else:
                    if hasattr(config, "disable_mkldnn"):
                        config.disable_mkldnn()
                config.set_cpu_math_library_num_threads(self._option.cpu_threads)

                if hasattr(config, "enable_new_ir"):
                    config.enable_new_ir(self._option.enable_new_ir)
                if hasattr(config, "enable_new_executor"):
                    config.enable_new_executor()
                config.set_optimization_level(3)

        config.enable_memory_optim()
        for del_p in self._option.delete_pass:
            config.delete_pass(del_p)

        # Disable paddle inference logging
        if not DEBUG:
            config.disable_glog_info()

        predictor = paddle.inference.create_predictor(config)

        return predictor

    def _configure_trt(self, model_file, params_file, cache_dir):
        # TODO: Support calibration
        import paddle.inference

        if USE_PIR_TRT:
            if self._option.trt_dynamic_shapes is None:
                raise RuntimeError("No dynamic shape information provided")
            trt_save_path = cache_dir / "trt" / self.model_file_prefix
            trt_model_file = trt_save_path.with_suffix(".json")
            trt_params_file = trt_save_path.with_suffix(".pdiparams")
            if not trt_model_file.exists() or not trt_params_file.exists():
                _convert_trt(
                    self._option.trt_cfg_setting,
                    model_file,
                    params_file,
                    trt_save_path,
                    self._option.device_id,
                    self._option.trt_dynamic_shapes,
                    self._option.trt_dynamic_shape_input_data,
                )
            else:
                logging.debug(
                    f"Use TRT cache files(`{trt_model_file}` and `{trt_params_file}`)."
                )
            config = paddle.inference.Config(str(trt_model_file), str(trt_params_file))
        else:
            config = paddle.inference.Config(str(model_file), str(params_file))
            config.set_optim_cache_dir(str(cache_dir / "optim_cache"))
            # call enable_use_gpu() first to use TensorRT engine
            config.enable_use_gpu(100, self._option.device_id)
            for func_name in self._option.trt_cfg_setting:
                assert hasattr(
                    config, func_name
                ), f"The `{type(config)}` don't have function `{func_name}`!"
                args = self._option.trt_cfg_setting[func_name]
                if isinstance(args, list):
                    getattr(config, func_name)(*args)
                else:
                    getattr(config, func_name)(**args)

            if self._option.trt_use_dynamic_shapes:
                if self._option.trt_dynamic_shapes is None:
                    raise RuntimeError("No dynamic shape information provided")
                if self._option.trt_collect_shape_range_info:
                    # NOTE: We always use a shape range info file.
                    if self._option.trt_shape_range_info_path is not None:
                        trt_shape_range_info_path = Path(
                            self._option.trt_shape_range_info_path
                        )
                    else:
                        trt_shape_range_info_path = cache_dir / "shape_range_info.pbtxt"
                    should_collect_shape_range_info = True
                    if not trt_shape_range_info_path.exists():
                        trt_shape_range_info_path.parent.mkdir(
                            parents=True, exist_ok=True
                        )
                        logging.info(
                            f"Shape range info will be collected into {trt_shape_range_info_path}"
                        )
                    elif self._option.trt_discard_cached_shape_range_info:
                        trt_shape_range_info_path.unlink()
                        logging.info(
                            f"The shape range info file ({trt_shape_range_info_path}) has been removed, and the shape range info will be re-collected."
                        )
                    else:
                        logging.info(
                            f"A shape range info file ({trt_shape_range_info_path}) already exists. There is no need to collect the info again."
                        )
                        should_collect_shape_range_info = False
                    if should_collect_shape_range_info:
                        _collect_trt_shape_range_info(
                            str(model_file),
                            str(params_file),
                            self._option.device_id,
                            str(trt_shape_range_info_path),
                            self._option.trt_dynamic_shapes,
                            self._option.trt_dynamic_shape_input_data,
                        )
                    if (
                        self._option.model_name in DISABLE_TRT_HALF_OPS_CONFIG
                        and self._option.run_mode == "trt_fp16"
                    ):
                        paddle.inference.InternalUtils.disable_tensorrt_half_ops(
                            config, DISABLE_TRT_HALF_OPS_CONFIG[self._option.model_name]
                        )
                    config.enable_tuned_tensorrt_dynamic_shape(
                        str(trt_shape_range_info_path),
                        self._option.trt_allow_rebuild_at_runtime,
                    )
                else:
                    min_shapes, opt_shapes, max_shapes = {}, {}, {}
                    for (
                        key,
                        shapes,
                    ) in self._option.trt_dynamic_shapes.items():
                        min_shapes[key] = shapes[0]
                        opt_shapes[key] = shapes[1]
                        max_shapes[key] = shapes[2]
                        config.set_trt_dynamic_shape_info(
                            min_shapes, max_shapes, opt_shapes
                        )

        return config


# FIXME: Name might be misleading
@benchmark.timeit
@class_requires_deps("ultra-infer")
class MultiBackendInfer(object):
    def __init__(self, ui_runtime):
        super().__init__()
        self.ui_runtime = ui_runtime

    # The time consumed by the wrapper code will also be taken into account.
    def __call__(self, x):
        outputs = self.ui_runtime.infer(x)
        return outputs


# TODO: It would be better to refactor the code to make `HPInfer` a higher-level
# class that uses `PaddleInfer`.
@class_requires_deps("ultra-infer")
class HPInfer(StaticInfer):
    def __init__(
        self,
        model_dir: Union[str, PathLike],
        model_file_prefix: str,
        config: HPIConfig,
    ) -> None:
        super().__init__()
        self._model_dir = Path(model_dir)
        self._model_file_prefix = model_file_prefix
        self._config = config
        backend, backend_config = self._determine_backend_and_config()
        if backend == "paddle":
            self._use_paddle = True
            self._paddle_infer = self._build_paddle_infer(backend_config)
        else:
            self._use_paddle = False
            ui_runtime = self._build_ui_runtime(backend, backend_config)
            self._multi_backend_infer = MultiBackendInfer(ui_runtime)
            num_inputs = ui_runtime.num_inputs()
            self._input_names = [
                ui_runtime.get_input_info(i).name for i in range(num_inputs)
            ]

    @property
    def model_dir(self) -> Path:
        return self._model_dir

    @property
    def model_file_prefix(self) -> str:
        return self._model_file_prefix

    @property
    def config(self) -> HPIConfig:
        return self._config

    def __call__(self, x: Sequence[np.ndarray]) -> List[np.ndarray]:
        if self._use_paddle:
            return self._call_paddle_infer(x)
        else:
            return self._call_multi_backend_infer(x)

    def _call_paddle_infer(self, x):
        return self._paddle_infer(x)

    def _call_multi_backend_infer(self, x):
        num_inputs = len(self._input_names)
        if len(x) != num_inputs:
            raise ValueError(f"Expected {num_inputs} inputs but got {len(x)} instead")
        x = _sort_inputs(x, self._input_names)
        inputs = {}
        for name, input_ in zip(self._input_names, x):
            inputs[name] = np.ascontiguousarray(input_)
        return self._multi_backend_infer(inputs)

    def _determine_backend_and_config(self):
        if self._config.auto_config:
            # Should we use the strategy pattern here to allow extensible
            # strategies?
            model_paths = get_model_paths(self._model_dir, self._model_file_prefix)
            ret = suggest_inference_backend_and_config(
                self._config,
                model_paths,
            )
            if ret[0] is None:
                # Should I use a custom exception?
                raise RuntimeError(
                    f"No inference backend and configuration could be suggested. Reason: {ret[1]}"
                )
            backend, backend_config = ret
        else:
            backend = self._config.backend
            if backend is None:
                raise RuntimeError(
                    "When automatic configuration is not used, the inference backend must be specified manually."
                )
            backend_config = self._config.backend_config or {}

        if backend == "paddle":
            if not backend_config:
                is_default_config = True
            elif backend_config.keys() != {"run_mode"}:
                is_default_config = False
            else:
                is_default_config = backend_config["run_mode"] == get_default_run_mode(
                    self._config.pdx_model_name, self._config.device_type
                )
            if is_default_config:
                logging.warning(
                    "The Paddle Inference backend is selected with the default configuration. This may not provide optimal performance."
                )

        return backend, backend_config

    def _build_paddle_infer(self, backend_config):
        kwargs = {
            "device_type": self._config.device_type,
            "device_id": self._config.device_id,
            **backend_config,
        }
        # TODO: This is probably redundant. Can we reuse the code in the
        # predictor class?
        paddle_info = None
        if self._config.hpi_info:
            hpi_info = self._config.hpi_info
            if hpi_info.backend_configs:
                paddle_info = hpi_info.backend_configs.paddle_infer
        if paddle_info is not None:
            if (
                kwargs.get("trt_dynamic_shapes") is None
                and paddle_info.trt_dynamic_shapes is not None
            ):
                trt_dynamic_shapes = paddle_info.trt_dynamic_shapes
                logging.debug("TensorRT dynamic shapes set to %s", trt_dynamic_shapes)
                kwargs["trt_dynamic_shapes"] = trt_dynamic_shapes
            if (
                kwargs.get("trt_dynamic_shape_input_data") is None
                and paddle_info.trt_dynamic_shape_input_data is not None
            ):
                trt_dynamic_shape_input_data = paddle_info.trt_dynamic_shape_input_data
                logging.debug(
                    "TensorRT dynamic shape input data set to %s",
                    trt_dynamic_shape_input_data,
                )
                kwargs["trt_dynamic_shape_input_data"] = trt_dynamic_shape_input_data
        pp_option = PaddlePredictorOption(**kwargs)
        pp_option.setdefault_by_model_name(model_name=self._config.pdx_model_name)
        logging.info("Using Paddle Inference backend")
        logging.info("Paddle predictor option: %s", pp_option)
        return PaddleInfer(
            self._config.pdx_model_name,
            self._model_dir,
            self._model_file_prefix,
            option=pp_option,
        )

    def _build_ui_runtime(self, backend, backend_config, ui_option=None):
        # TODO: Validate the compatibility of backends with device types

        from ultra_infer import ModelFormat, Runtime, RuntimeOption

        if ui_option is None:
            ui_option = RuntimeOption()

        if self._config.device_type == "cpu":
            pass
        elif self._config.device_type == "gpu":
            ui_option.use_gpu(self._config.device_id or 0)
        elif self._config.device_type == "npu":
            ui_option.use_ascend(self._config.device_id or 0)
        else:
            raise RuntimeError(
                f"Unsupported device type {repr(self._config.device_type)}"
            )

        model_paths = get_model_paths(self._model_dir, self.model_file_prefix)
        if backend in ("openvino", "onnxruntime", "tensorrt"):
            # XXX: This introduces side effects.
            if "onnx" not in model_paths:
                if self._config.auto_paddle2onnx:
                    if "paddle" not in model_paths:
                        raise RuntimeError("PaddlePaddle model required")
                    # The CLI is used here since there is currently no API.
                    logging.info(
                        "Automatically converting PaddlePaddle model to ONNX format"
                    )
                    try:
                        subprocess.run(
                            [
                                "paddlex",
                                "--paddle2onnx",
                                "--paddle_model_dir",
                                str(self._model_dir),
                                "--onnx_model_dir",
                                str(self._model_dir),
                            ],
                            capture_output=True,
                            check=True,
                            text=True,
                        )
                    except subprocess.CalledProcessError as e:
                        raise RuntimeError(
                            f"PaddlePaddle-to-ONNX conversion failed:\n{e.stderr}"
                        ) from e
                    model_paths = get_model_paths(
                        self._model_dir, self.model_file_prefix
                    )
                    assert "onnx" in model_paths
                else:
                    raise RuntimeError("ONNX model required")
            ui_option.set_model_path(str(model_paths["onnx"]), "", ModelFormat.ONNX)
        elif backend == "om":
            if "om" not in model_paths:
                raise RuntimeError("OM model required")
            ui_option.set_model_path(str(model_paths["om"]), "", ModelFormat.OM)
        else:
            raise ValueError(f"Unsupported inference backend {repr(backend)}")

        if backend == "openvino":
            backend_config = OpenVINOConfig.model_validate(backend_config)
            ui_option.use_openvino_backend()
            ui_option.set_cpu_thread_num(backend_config.cpu_num_threads)
        elif backend == "onnxruntime":
            backend_config = ONNXRuntimeConfig.model_validate(backend_config)
            ui_option.use_ort_backend()
            ui_option.set_cpu_thread_num(backend_config.cpu_num_threads)
        elif backend == "tensorrt":
            if (
                backend_config.get("use_dynamic_shapes", True)
                and backend_config.get("dynamic_shapes") is None
            ):
                trt_info = None
                if self._config.hpi_info:
                    hpi_info = self._config.hpi_info
                    if hpi_info.backend_configs:
                        trt_info = hpi_info.backend_configs.tensorrt
                if trt_info is not None and trt_info.dynamic_shapes is not None:
                    trt_dynamic_shapes = trt_info.dynamic_shapes
                    logging.debug(
                        "TensorRT dynamic shapes set to %s", trt_dynamic_shapes
                    )
                    backend_config = {
                        **backend_config,
                        "dynamic_shapes": trt_dynamic_shapes,
                    }
            backend_config = TensorRTConfig.model_validate(backend_config)
            ui_option.use_trt_backend()
            cache_dir = self._model_dir / CACHE_DIR / "tensorrt"
            cache_dir.mkdir(parents=True, exist_ok=True)
            ui_option.trt_option.serialize_file = str(cache_dir / "trt_serialized.trt")
            if backend_config.precision == "fp16":
                ui_option.trt_option.enable_fp16 = True
            if not backend_config.use_dynamic_shapes:
                raise RuntimeError(
                    "TensorRT static shape inference is currently not supported"
                )
            if backend_config.dynamic_shapes is not None:
                if not Path(ui_option.trt_option.serialize_file).exists():
                    for name, shapes in backend_config.dynamic_shapes.items():
                        ui_option.trt_option.set_shape(name, *shapes)
                else:
                    logging.info(
                        "TensorRT dynamic shapes will be loaded from the file."
                    )
        elif backend == "om":
            backend_config = OMConfig.model_validate(backend_config)
            ui_option.use_om_backend()
        else:
            raise ValueError(f"Unsupported inference backend {repr(backend)}")

        logging.info("Inference backend: %s", backend)
        logging.info("Inference backend config: %s", backend_config)

        ui_runtime = Runtime(ui_option)

        return ui_runtime
