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

from typing import Sequence, List
from pathlib import Path

import lazy_paddle
import numpy as np

from ....utils import logging
from ....utils.device import constr_device
from ....utils.flags import (
    DEBUG,
    USE_PIR_TRT,
    INFER_BENCHMARK_USE_NEW_INFER_API,
)
from ...utils.benchmark import benchmark, set_inference_operations
from ...utils.hpi import get_model_paths
from ...utils.pp_option import PaddlePredictorOption
from ...utils.trt_config import DISABLE_TRT_HALF_OPS_CONFIG


CACHE_DIR = ".cache"

if INFER_BENCHMARK_USE_NEW_INFER_API:
    INFERENCE_OPERATIONS = [
        "PaddleCopyToDevice",
        "PaddleCopyToHost",
        "PaddleModelInfer",
    ]
else:
    INFERENCE_OPERATIONS = ["PaddleInferChainLegacy"]
set_inference_operations(INFERENCE_OPERATIONS)


# XXX: Better use Paddle Inference API to do this
def _pd_dtype_to_np_dtype(pd_dtype):
    if pd_dtype == lazy_paddle.inference.DataType.FLOAT64:
        return np.float64
    elif pd_dtype == lazy_paddle.inference.DataType.FLOAT32:
        return np.float32
    elif pd_dtype == lazy_paddle.inference.DataType.INT64:
        return np.int64
    elif pd_dtype == lazy_paddle.inference.DataType.INT32:
        return np.int32
    elif pd_dtype == lazy_paddle.inference.DataType.UINT8:
        return np.uint8
    elif pd_dtype == lazy_paddle.inference.DataType.INT8:
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

    dynamic_shape_input_data = dynamic_shape_input_data or {}

    config = lazy_paddle.inference.Config(model_file, model_params)
    config.enable_use_gpu(100, gpu_id)
    config.collect_shape_range_info(shape_range_info_path)
    # TODO: Add other needed options
    config.disable_glog_info()
    predictor = lazy_paddle.inference.create_predictor(config)

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
    from lazy_paddle.tensorrt.export import (
        Input,
        TensorRTConfig,
        convert,
    )

    def _set_trt_config():
        for attr_name in trt_cfg_setting:
            assert hasattr(
                trt_config, attr_name
            ), f"The `{type(trt_config)}` don't have the attribute `{attr_name}`!"
            setattr(trt_config, attr_name, trt_cfg_setting[attr_name])

    def _get_predictor(model_file, params_file):
        # HACK
        config = lazy_paddle.inference.Config(str(model_file), str(params_file))
        config.enable_use_gpu(100, device_id)
        # NOTE: Disable oneDNN to circumvent a bug in Paddle Inference
        config.disable_mkldnn()
        config.disable_glog_info()
        return lazy_paddle.inference.create_predictor(config)

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


def _concatenate(*callables):
    def _chain(x):
        for c in callables:
            x = c(x)
        return x

    return _chain


@benchmark.timeit
class PaddleCopyToDevice:
    def __init__(self, device_type, device_id):
        self.device_type = device_type
        self.device_id = device_id

    def __call__(self, arrs):
        device_id = [self.device_id] if self.device_id is not None else self.device_id
        device = constr_device(self.device_type, device_id)
        paddle_tensors = [lazy_paddle.to_tensor(i, place=device) for i in arrs]
        return paddle_tensors


@benchmark.timeit
class PaddleCopyToHost:
    def __call__(self, paddle_tensors):
        arrs = [i.numpy() for i in paddle_tensors]
        return arrs


@benchmark.timeit
class PaddleModelInfer:
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def __call__(self, x):
        return self.predictor.run(x)


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


class StaticInfer(object):
    def __init__(
        self,
        model_dir: str,
        model_prefix: str,
        option: PaddlePredictorOption,
    ) -> None:
        super().__init__()
        self.model_dir = model_dir
        self.model_file_prefix = model_prefix
        self._option = option
        self.predictor = self._create()
        if self._use_new_inference_api:
            device_type = self._option.device_type
            device_type = "gpu" if device_type == "dcu" else device_type
            copy_to_device = PaddleCopyToDevice(device_type, self._option.device_id)
            copy_to_host = PaddleCopyToHost()
            model_infer = PaddleModelInfer(self.predictor)
            self.infer = _concatenate(copy_to_device, model_infer, copy_to_host)
        else:
            self.infer = PaddleInferChainLegacy(self.predictor)

    @property
    def _use_new_inference_api(self):
        # HACK: Temp fallback to legacy API via env var
        return INFER_BENCHMARK_USE_NEW_INFER_API

        # return self._option.device_type in ("cpu", "gpu", "dcu")

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

    def _create(
        self,
    ):
        """_create"""
        model_paths = get_model_paths(self.model_dir, self.model_file_prefix)
        if "paddle" not in model_paths:
            raise RuntimeError("No valid Paddle model found")
        model_file, params_file = model_paths["paddle"]

        if (
            self._option.model_name == "LaTeX_OCR_rec"
            and self._option.device_type == "cpu"
        ):
            import cpuinfo

            if (
                "GenuineIntel" in cpuinfo.get_cpu_info().get("vendor_id_raw", "")
                and self._option.run_mode != "mkldnn"
            ):
                logging.warning(
                    "Now, the `LaTeX_OCR_rec` model only support `mkldnn` mode when running on Intel CPU devices. So using `mkldnn` instead."
                )
            self._option.run_mode = "mkldnn"
            logging.debug("`run_mode` updated to 'mkldnn'")

        if self._option.device_type == "cpu" and self._option.device_id is not None:
            self._option.device_id = None
            logging.debug("`device_id` has been set to None")

        if (
            self._option.device_type in ("gpu", "dcu")
            and self._option.device_id is None
        ):
            self._option.device_id = 0
            logging.debug("`device_id` has been set to 0")

        # for TRT
        if self._option.run_mode.startswith("trt"):
            assert self._option.device_type == "gpu"
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
            config = lazy_paddle.inference.Config(str(model_file), str(params_file))
            if self._option.device_type == "gpu":
                config.exp_disable_mixed_precision_ops({"feed", "fetch"})
                from lazy_paddle.inference import PrecisionType

                precision = (
                    PrecisionType.Half
                    if self._option.run_mode == "paddle_fp16"
                    else PrecisionType.Float32
                )
                config.enable_use_gpu(100, self._option.device_id, precision)
                if hasattr(config, "enable_new_ir"):
                    config.enable_new_ir(self._option.enable_new_ir)
                if hasattr(config, "enable_new_executor"):
                    config.enable_new_executor()
                config.set_optimization_level(3)
            elif self._option.device_type == "npu":
                config.enable_custom_device("npu")
                if hasattr(config, "enable_new_executor"):
                    config.enable_new_executor()
            elif self._option.device_type == "xpu":
                if hasattr(config, "enable_new_executor"):
                    config.enable_new_executor()
            elif self._option.device_type == "mlu":
                config.enable_custom_device("mlu")
                if hasattr(config, "enable_new_executor"):
                    config.enable_new_executor()
            elif self._option.device_type == "gcu":
                from paddle_custom_device.gcu import passes as gcu_passes

                gcu_passes.setUp()
                config.enable_custom_device("gcu")
                if hasattr(config, "enable_new_executor"):
                    config.enable_new_ir()
                    config.enable_new_executor()
                else:
                    pass_builder = config.pass_builder()
                    name = "PaddleX_" + self._option.model_name
                    gcu_passes.append_passes_for_legacy_ir(pass_builder, name)
            elif self._option.device_type == "dcu":
                config.enable_use_gpu(100, self._option.device_id)
                if hasattr(config, "enable_new_executor"):
                    config.enable_new_executor()
                # XXX: is_compiled_with_rocm() must be True on dcu platform ?
                if lazy_paddle.is_compiled_with_rocm():
                    # Delete unsupported passes in dcu
                    config.delete_pass("conv2d_add_act_fuse_pass")
                    config.delete_pass("conv2d_add_fuse_pass")
            else:
                assert self._option.device_type == "cpu"
                config.disable_gpu()
                if "mkldnn" in self._option.run_mode:
                    try:
                        config.enable_mkldnn()
                        if "bf16" in self._option.run_mode:
                            config.enable_mkldnn_bfloat16()
                    except Exception as e:
                        logging.warning(
                            "MKL-DNN is not available. We will disable MKL-DNN."
                        )
                    config.set_mkldnn_cache_capacity(-1)
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

        predictor = lazy_paddle.inference.create_predictor(config)

        return predictor

    def _configure_trt(self, model_file, params_file, cache_dir):
        # TODO: Support calibration
        if USE_PIR_TRT:
            trt_save_path = cache_dir / "trt" / self.model_file_prefix
            _convert_trt(
                self._option.trt_cfg_setting,
                model_file,
                params_file,
                trt_save_path,
                self._option.device_id,
                self._option.trt_dynamic_shapes,
                self._option.trt_dynamic_shape_input_data,
            )
            model_file = trt_save_path.with_suffix(".json")
            params_file = trt_save_path.with_suffix(".pdiparams")
            config = lazy_paddle.inference.Config(str(model_file), str(params_file))
        else:
            config = lazy_paddle.inference.Config(str(model_file), str(params_file))
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
                    if self._option.model_name in DISABLE_TRT_HALF_OPS_CONFIG:
                        lazy_paddle.inference.InternalUtils.disable_tensorrt_half_ops(
                            config, DISABLE_TRT_HALF_OPS_CONFIG[self._option.model_name]
                        )
                    config.enable_tuned_tensorrt_dynamic_shape(
                        str(trt_shape_range_info_path),
                        self._option.trt_allow_rebuild_at_runtime,
                    )
                else:
                    if self._option.trt_dynamic_shapes is not None:
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
                    else:
                        raise RuntimeError("No dynamic shape information provided")

        return config
