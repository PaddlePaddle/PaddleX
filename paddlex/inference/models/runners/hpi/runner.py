# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""HPI (High Performance Inference) runner."""

import subprocess
from pathlib import Path
from typing import List, Sequence, Union

import numpy as np

from paddlex.inference.models.runners.paddle_static import (
    CACHE_DIR,
    PaddleStaticRunner,
    PaddleStaticRunnerConfig,
)
from paddlex.inference.models.runners.paddle_static.config import get_default_run_mode
from paddlex.inference.models.runners.utils import sort_inputs
from paddlex.inference.models.utils.model_paths import get_model_paths
from paddlex.utils import logging
from paddlex.utils.deps import class_requires_deps, require_hpip

from ..inference_runner import InferenceRunner
from .backend import suggest_inference_backend_and_config
from .config import (
    HPIConfig,
    OMConfig,
    ONNXRuntimeConfig,
    OpenVINOConfig,
    TensorRTConfig,
)


@class_requires_deps("ultra-infer")
class MultiBackendInfer(object):
    def __init__(self, ui_runtime):
        super().__init__()
        self.ui_runtime = ui_runtime

    def __call__(self, x):
        return self.ui_runtime.infer(x)


@class_requires_deps("ultra-infer")
class HPIRunner(InferenceRunner):
    """HPI runner supporting multiple backends (Paddle, ONNX, TensorRT, etc.)."""

    def __init__(
        self,
        model_dir: Union[str, Path],
        model_file_prefix: str,
        config: HPIConfig,
    ) -> None:
        require_hpip()
        super().__init__()
        self._model_dir = Path(model_dir)
        self._model_file_prefix = model_file_prefix
        self._config = config
        backend, backend_config = self._determine_backend_and_config()
        if backend == "paddle":
            self._use_paddle = True
            self._paddle_runner = self._build_paddle_static_runner(backend_config)
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
            return self._paddle_runner(x)
        num_inputs = len(self._input_names)
        if len(x) != num_inputs:
            raise ValueError(f"Expected {num_inputs} inputs but got {len(x)} instead")
        x = sort_inputs(x, self._input_names)
        inputs = {
            name: np.ascontiguousarray(input_)
            for name, input_ in zip(self._input_names, x)
        }
        return self._multi_backend_infer(inputs)

    def close(self) -> None:
        pass

    def _determine_backend_and_config(self):
        if self._config.auto_config:
            model_paths = get_model_paths(self._model_dir, self._model_file_prefix)
            ret = suggest_inference_backend_and_config(self._config, model_paths)
            if ret[0] is None:
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

    def _build_paddle_static_runner(self, backend_config):
        kwargs = {
            "device_type": self._config.device_type,
            "device_id": self._config.device_id,
            **backend_config,
        }
        paddle_info = None
        if self._config.hpi_info and self._config.hpi_info.backend_configs:
            paddle_info = self._config.hpi_info.backend_configs.paddle_infer
        if paddle_info is not None:
            if (
                kwargs.get("trt_dynamic_shapes") is None
                and paddle_info.trt_dynamic_shapes is not None
            ):
                kwargs["trt_dynamic_shapes"] = paddle_info.trt_dynamic_shapes
            if (
                kwargs.get("trt_dynamic_shape_input_data") is None
                and paddle_info.trt_dynamic_shape_input_data is not None
            ):
                kwargs["trt_dynamic_shape_input_data"] = (
                    paddle_info.trt_dynamic_shape_input_data
                )
        valid_keys = set(PaddleStaticRunnerConfig.model_fields.keys())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        engine_config = PaddleStaticRunnerConfig.model_validate(
            filtered_kwargs
        ).model_dump(exclude_none=True)
        logging.info("Using Paddle Inference backend")
        logging.info("Paddle engine config: %s", engine_config)
        return PaddleStaticRunner(
            self._config.pdx_model_name,
            self._model_dir,
            self._model_file_prefix,
            config=engine_config,
        )

    def _build_ui_runtime(self, backend, backend_config, ui_option=None):
        from ultra_infer import ModelFormat, Runtime, RuntimeOption

        if ui_option is None:
            ui_option = RuntimeOption()

        if self._config.device_type == "gpu":
            ui_option.use_gpu(self._config.device_id or 0)
        elif self._config.device_type == "npu":
            ui_option.use_ascend(self._config.device_id or 0)
        elif self._config.device_type != "cpu":
            raise RuntimeError(
                f"Unsupported device type {repr(self._config.device_type)}"
            )

        model_paths = get_model_paths(self._model_dir, self.model_file_prefix)
        if backend in ("openvino", "onnxruntime", "tensorrt"):
            if "onnx" not in model_paths:
                if self._config.auto_paddle2onnx:
                    if "paddle" not in model_paths:
                        raise RuntimeError("PaddlePaddle model required")
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
                if self._config.hpi_info and self._config.hpi_info.backend_configs:
                    trt_info = self._config.hpi_info.backend_configs.tensorrt
                if trt_info is not None and trt_info.dynamic_shapes is not None:
                    backend_config = {
                        **backend_config,
                        "dynamic_shapes": trt_info.dynamic_shapes,
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

        return Runtime(ui_option)
