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

"""Paddle Inference runner."""

from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from pydantic import BaseModel, ConfigDict

from .....utils import logging
from .....utils.deps import class_requires_deps
from .....utils.device import check_supported_device_type
from .....utils.flags import (
    DEBUG,
    DISABLE_MKLDNN_MODEL_BL,
    DISABLE_TRT_MODEL_BL,
    USE_PIR_TRT,
)
from ....utils.benchmark import add_inference_operations, benchmark
from ....utils.mkldnn_blocklist import MKLDNN_BLOCKLIST
from ....utils.model_paths import get_model_paths
from ....utils.pp_option import PaddlePredictorOption
from ....utils.trt_blocklist import TRT_BLOCKLIST
from ....utils.trt_config import DISABLE_TRT_HALF_OPS_CONFIG
from .utils import sort_inputs

CACHE_DIR = ".cache"


class PaddleStaticRunnerConfig(BaseModel):
    """Engine config for paddle_static inference."""

    model_config = ConfigDict(extra="forbid")

    run_mode: Optional[str] = None
    device_type: Optional[str] = None
    device_id: Optional[int] = None
    cpu_threads: Optional[int] = None
    delete_pass: Optional[List[str]] = None
    enable_new_ir: Optional[bool] = None
    enable_cinn: Optional[bool] = None
    trt_cfg_setting: Optional[Dict[str, Any]] = None
    trt_use_dynamic_shapes: Optional[bool] = None
    trt_collect_shape_range_info: Optional[bool] = None
    trt_discard_cached_shape_range_info: Optional[bool] = None
    trt_dynamic_shapes: Optional[Dict[str, List[List[int]]]] = None
    trt_dynamic_shape_input_data: Optional[Dict[str, List[List[float]]]] = None
    trt_shape_range_info_path: Optional[str] = None
    trt_allow_rebuild_at_runtime: Optional[bool] = None
    mkldnn_cache_capacity: Optional[int] = None


INFERENCE_OPERATIONS = [
    "PaddleRunnerChainLegacy",
]
add_inference_operations(*INFERENCE_OPERATIONS)


def resolve_paddle_static_engine_config(
    model_name: str,
    engine_config: Dict,
) -> Dict:
    """Resolve engine config with defaults. Returns dict for PaddleStaticRunner."""
    # TODO: Do not rely on `PaddlePredictorOption` and remove it eventually
    pp = PaddlePredictorOption()
    pp._cfg["model_name"] = model_name
    pp.setdefault_by_model_name(model_name)
    for k, v in (engine_config or {}).items():
        if hasattr(pp, k):
            setattr(pp, k, v)
    return pp._cfg.copy()


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
    config.delete_pass("matmul_add_act_fuse_pass")
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


@benchmark.timeit
class PaddleRunnerChainLegacy:
    """Legacy Paddle Inference chain wrapper."""

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


@class_requires_deps("paddlepaddle")
class PaddleStaticRunner:
    """Paddle static graph inference runner (Paddle Inference API)."""

    def __init__(
        self,
        model_name: str,
        model_dir: Union[str, PathLike],
        model_file_prefix: str,
        config: Dict[str, Any],
    ) -> None:
        super().__init__()
        self._model_name = model_name
        self.model_dir = Path(model_dir)
        self.model_file_prefix = model_file_prefix
        self._config = resolve_paddle_static_engine_config(model_name, config)
        self.predictor = self._create()
        self.infer = PaddleRunnerChainLegacy(self.predictor)

    def __call__(self, x: Sequence[np.ndarray]) -> List[np.ndarray]:
        names = self.predictor.get_input_names()
        if len(names) != len(x):
            raise ValueError(
                f"The number of inputs does not match the model: {len(names)} vs {len(x)}"
            )
        # TODO:
        # Ensure that input tensors follow the model's input sequence without sorting.
        x = sort_inputs(x, names)
        x = list(map(np.ascontiguousarray, x))
        pred = self.infer(x)
        return pred

    def _check_run_mode(self):
        # TODO: Check if trt is available
        # check avaliable for trt
        run_mode = self._config.get("run_mode", "paddle")
        device_type = self._config.get("device_type", "cpu")
        if (
            not DISABLE_TRT_MODEL_BL
            and run_mode.startswith("trt")
            and self._model_name in TRT_BLOCKLIST
            and device_type == "gpu"
        ):
            logging.warning(
                f"The model({self._model_name}) is not supported to run in trt mode! Using `paddle` instead!"
            )
            self._config["run_mode"] = "paddle"

        # check avaliable for mkldnn
        elif (
            not DISABLE_MKLDNN_MODEL_BL
            and run_mode.startswith("mkldnn")
            and self._model_name in MKLDNN_BLOCKLIST
            and device_type == "cpu"
        ):
            logging.warning(
                f"The model({self._model_name}) is not supported to run in MKLDNN mode! Using `paddle` instead!"
            )
            self._config["run_mode"] = "paddle"
            return "paddle"

        # check avaliable for model
        if self._model_name == "LaTeX_OCR_rec" and device_type == "cpu":
            import cpuinfo

            if (
                "GenuineIntel" in cpuinfo.get_cpu_info().get("vendor_id_raw", "")
                and run_mode != "mkldnn"
            ):
                logging.warning(
                    "Now, the `LaTeX_OCR_rec` model only support `mkldnn` mode when running on Intel CPU devices. So using `mkldnn` instead."
                )
            self._config["run_mode"] = "mkldnn"

    def _create(
        self,
    ):
        """_create"""
        import paddle
        import paddle.inference

        model_paths = get_model_paths(self.model_dir, self.model_file_prefix)
        if "paddle" not in model_paths:
            raise RuntimeError("No valid PaddlePaddle model found")

        check_supported_device_type(self._config["device_type"], self._model_name)
        self._check_run_mode()

        model_file, params_file = model_paths["paddle"]

        if (
            self._config["device_type"] == "cpu"
            and self._config.get("device_id") is not None
        ):
            self._config["device_id"] = None
            logging.debug("`device_id` has been set to None")

        if (
            self._config["device_type"]
            in ("gpu", "dcu", "npu", "mlu", "gcu", "xpu", "iluvatar_gpu", "metax_gpu")
            and self._config.get("device_id") is None
        ):
            self._config["device_id"] = 0
            logging.debug("`device_id` has been set to 0")

        # for TRT
        if self._config.get("run_mode", "paddle").startswith("trt"):
            assert self._config["device_type"].lower() == "gpu", (
                f"`{self._config.get('run_mode')}` is only available on GPU devices, "
                f"but got device_type='{self._config['device_type']}'."
            )
            cache_dir = self.model_dir / CACHE_DIR / "paddle"
            config = self._configure_trt(
                model_file,
                params_file,
                cache_dir,
            )
            config.exp_disable_mixed_precision_ops({"feed", "fetch"})
            config.enable_use_gpu(100, self._config.get("device_id", 0))
        # for Native Paddle and MKLDNN
        else:
            config = paddle.inference.Config(str(model_file), str(params_file))
            if self._config["device_type"] == "gpu":
                config.exp_disable_mixed_precision_ops({"feed", "fetch"})
                from paddle.inference import PrecisionType

                precision = (
                    PrecisionType.Half
                    if self._config.get("run_mode") == "paddle_fp16"
                    else PrecisionType.Float32
                )
                config.disable_mkldnn()
                config.enable_use_gpu(100, self._config.get("device_id", 0), precision)
                if hasattr(config, "enable_new_ir"):
                    config.enable_new_ir(self._config.get("enable_new_ir", True))
                    if self._config.get("enable_new_ir") and self._config.get(
                        "enable_cinn"
                    ):
                        config.enable_cinn()
                if hasattr(config, "enable_new_executor"):
                    config.enable_new_executor()
                config.set_optimization_level(3)
                # TODO(changdazhou): use a black list instead
                if self._model_name == "PP-DocLayoutV3":
                    config.delete_pass("matmul_add_act_fuse_pass")
                # ROCm does not support fused_conv2d_add_act kernel, delete the fuse passes
                if paddle.is_compiled_with_rocm():
                    config.delete_pass("conv2d_add_act_fuse_pass")
                    config.delete_pass("conv2d_add_fuse_pass")
            elif self._config["device_type"] == "npu":
                config.enable_custom_device("npu", self._config.get("device_id", 0))
                if hasattr(config, "enable_new_ir"):
                    config.enable_new_ir(self._config.get("enable_new_ir", True))
                if hasattr(config, "enable_new_executor"):
                    config.enable_new_executor()
            elif self._config["device_type"] == "xpu":
                config.enable_xpu()
                config.set_xpu_device_id(self._config.get("device_id", 0))
                if hasattr(config, "enable_new_ir"):
                    config.enable_new_ir(self._config.get("enable_new_ir", True))
                if hasattr(config, "enable_new_executor"):
                    config.enable_new_executor()
                config.delete_pass("conv2d_bn_xpu_fuse_pass")
                config.delete_pass("transfer_layout_pass")
            elif self._config["device_type"] == "mlu":
                config.enable_custom_device("mlu", self._config.get("device_id", 0))
                if hasattr(config, "enable_new_ir"):
                    config.enable_new_ir(self._config.get("enable_new_ir", True))
                if hasattr(config, "enable_new_executor"):
                    config.enable_new_executor()
            elif self._config["device_type"] == "metax_gpu":
                config.enable_custom_device(
                    "metax_gpu", self._config.get("device_id", 0)
                )
                if hasattr(config, "enable_new_ir"):
                    config.enable_new_ir(self._config.get("enable_new_ir", True))
                if hasattr(config, "enable_new_executor"):
                    config.enable_new_executor()
            elif self._config["device_type"] == "gcu":
                from paddle_custom_device.gcu import passes as gcu_passes

                gcu_passes.setUp()
                config.enable_custom_device("gcu", self._config.get("device_id", 0))
                if hasattr(config, "enable_new_ir"):
                    config.enable_new_ir()
                if hasattr(config, "enable_new_executor"):
                    config.enable_new_executor()
                else:
                    pass_builder = config.pass_builder()
                    name = "PaddleX_" + self._config.get("model_name", self._model_name)
                    gcu_passes.append_passes_for_legacy_ir(pass_builder, name)
            elif self._config["device_type"] == "dcu":
                if hasattr(config, "enable_new_ir"):
                    config.enable_new_ir(self._config.get("enable_new_ir", True))
                    if self._config.get("enable_new_ir") and self._config.get(
                        "enable_cinn"
                    ):
                        config.enable_cinn()
                config.enable_use_gpu(100, self._config.get("device_id", 0))
                config.disable_mkldnn()
                if hasattr(config, "enable_new_executor"):
                    config.enable_new_executor()
                # XXX: is_compiled_with_rocm() must be True on dcu platform ?
                if paddle.is_compiled_with_rocm():
                    # Delete unsupported passes in dcu
                    config.delete_pass("conv2d_add_act_fuse_pass")
                    config.delete_pass("conv2d_add_fuse_pass")
            elif self._config["device_type"] == "iluvatar_gpu":
                config.enable_custom_device(
                    "iluvatar_gpu", int(self._config.get("device_id", 0))
                )
                if hasattr(config, "enable_new_ir"):
                    config.enable_new_ir(self._config.get("enable_new_ir", True))
                if hasattr(config, "enable_new_executor"):
                    config.enable_new_executor()
            else:
                assert self._config["device_type"] == "cpu"
                config.disable_gpu()
                run_mode = self._config.get("run_mode", "paddle")
                if "mkldnn" in run_mode:
                    config.enable_mkldnn()
                    if "bf16" in run_mode:
                        config.enable_mkldnn_bfloat16()
                    config.set_mkldnn_cache_capacity(
                        self._config.get("mkldnn_cache_capacity", 10)
                    )
                else:
                    if hasattr(config, "disable_mkldnn"):
                        config.disable_mkldnn()
                config.set_cpu_math_library_num_threads(
                    self._config.get("cpu_threads", 10)
                )

                if hasattr(config, "enable_new_ir"):
                    config.enable_new_ir(self._config.get("enable_new_ir", True))
                if hasattr(config, "enable_new_executor"):
                    config.enable_new_executor()
                config.set_optimization_level(3)
                if paddle.is_compiled_with_rocm():
                    config.delete_pass("conv2d_add_act_fuse_pass")
                    config.delete_pass("conv2d_add_fuse_pass")
        config.enable_memory_optim()
        for del_p in self._config.get("delete_pass", []):
            config.delete_pass(del_p)

        # Disable paddle inference logging
        if not DEBUG:
            config.disable_glog_info()
        # ROCm does not support fused_conv2d_add_act kernel, delete the fuse passes
        if paddle.is_compiled_with_rocm():
            config.delete_pass("conv2d_add_act_fuse_pass")
            config.delete_pass("conv2d_add_fuse_pass")

        predictor = paddle.inference.create_predictor(config)

        return predictor

    def _configure_trt(self, model_file, params_file, cache_dir):
        # TODO: Support calibration
        import paddle.inference

        if USE_PIR_TRT:
            if self._config.get("trt_dynamic_shapes") is None:
                raise RuntimeError("No dynamic shape information provided")
            trt_save_path = cache_dir / "trt" / self.model_file_prefix
            trt_model_file = trt_save_path.with_suffix(".json")
            trt_params_file = trt_save_path.with_suffix(".pdiparams")
            if not trt_model_file.exists() or not trt_params_file.exists():
                _convert_trt(
                    self._config.get("trt_cfg_setting", {}),
                    model_file,
                    params_file,
                    trt_save_path,
                    self._config.get("device_id", 0),
                    self._config["trt_dynamic_shapes"],
                    self._config.get("trt_dynamic_shape_input_data"),
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
            config.enable_use_gpu(100, self._config.get("device_id", 0))
            for func_name in self._config.get("trt_cfg_setting", {}):
                assert hasattr(
                    config, func_name
                ), f"The `{type(config)}` don't have function `{func_name}`!"
                args = self._config["trt_cfg_setting"][func_name]
                if isinstance(args, list):
                    getattr(config, func_name)(*args)
                else:
                    getattr(config, func_name)(**args)

            if self._config.get("trt_use_dynamic_shapes", True):
                if self._config.get("trt_dynamic_shapes") is None:
                    raise RuntimeError("No dynamic shape information provided")
                if self._config.get("trt_collect_shape_range_info", True):
                    # NOTE: We always use a shape range info file.
                    if self._config.get("trt_shape_range_info_path") is not None:
                        trt_shape_range_info_path = Path(
                            self._config["trt_shape_range_info_path"]
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
                    elif self._config.get("trt_discard_cached_shape_range_info", False):
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
                            self._config.get("device_id", 0),
                            str(trt_shape_range_info_path),
                            self._config["trt_dynamic_shapes"],
                            self._config.get("trt_dynamic_shape_input_data"),
                        )
                    model_name = self._config.get("model_name", self._model_name)
                    if (
                        model_name in DISABLE_TRT_HALF_OPS_CONFIG
                        and self._config.get("run_mode") == "trt_fp16"
                    ):
                        paddle.inference.InternalUtils.disable_tensorrt_half_ops(
                            config, DISABLE_TRT_HALF_OPS_CONFIG[model_name]
                        )
                    config.enable_tuned_tensorrt_dynamic_shape(
                        str(trt_shape_range_info_path),
                        self._config.get("trt_allow_rebuild_at_runtime", True),
                    )
                else:
                    min_shapes, opt_shapes, max_shapes = {}, {}, {}
                    for (
                        key,
                        shapes,
                    ) in self._config["trt_dynamic_shapes"].items():
                        min_shapes[key] = shapes[0]
                        opt_shapes[key] = shapes[1]
                        max_shapes[key] = shapes[2]
                        config.set_trt_dynamic_shape_info(
                            min_shapes, max_shapes, opt_shapes
                        )

        return config
