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

import ctypes.util
import importlib.resources
import importlib.util
import json
import platform
from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypeAlias

from ...utils import logging
from ...utils.deps import function_requires_deps, is_paddle2onnx_plugin_available
from ...utils.env import get_paddle_cuda_version, get_paddle_version
from ...utils.flags import USE_PIR_TRT
from .misc import is_mkldnn_available
from .model_paths import ModelPaths


class PaddleInferenceInfo(BaseModel):
    trt_dynamic_shapes: Optional[Dict[str, List[List[int]]]] = None
    trt_dynamic_shape_input_data: Optional[Dict[str, List[List[float]]]] = None


class TensorRTInfo(BaseModel):
    dynamic_shapes: Optional[Dict[str, List[List[int]]]] = None


class InferenceBackendInfoCollection(BaseModel):
    paddle_infer: Optional[PaddleInferenceInfo] = None
    tensorrt: Optional[TensorRTInfo] = None


# Does using `TypedDict` make things more convenient?
class HPIInfo(BaseModel):
    backend_configs: Optional[InferenceBackendInfoCollection] = None


# For multi-backend inference only
InferenceBackend: TypeAlias = Literal[
    "paddle", "openvino", "onnxruntime", "tensorrt", "om"
]


class OpenVINOConfig(BaseModel):
    cpu_num_threads: int = 10


class ONNXRuntimeConfig(BaseModel):
    cpu_num_threads: int = 10


class TensorRTConfig(BaseModel):
    precision: Literal["fp32", "fp16"] = "fp32"
    use_dynamic_shapes: bool = True
    dynamic_shapes: Optional[Dict[str, List[List[int]]]] = None
    # TODO: Control caching behavior


class OMConfig(BaseModel):
    pass


class HPIConfig(BaseModel):
    pdx_model_name: Annotated[str, Field(alias="model_name")]
    device_type: str
    device_id: Optional[int] = None
    auto_config: bool = True
    backend: Optional[InferenceBackend] = None
    backend_config: Optional[Dict[str, Any]] = None
    hpi_info: Optional[HPIInfo] = None
    auto_paddle2onnx: bool = True
    # TODO: Add more validation logic here


class ModelInfo(BaseModel):
    name: str
    hpi_info: Optional[HPIInfo] = None


ModelFormat: TypeAlias = Literal["paddle", "onnx", "om"]


@lru_cache(1)
def _get_hpi_model_info_collection():
    with importlib.resources.open_text(
        __package__, "hpi_model_info_collection.json", encoding="utf-8"
    ) as f:
        hpi_model_info_collection = json.load(f)
    return hpi_model_info_collection


@function_requires_deps("ultra-infer")
def suggest_inference_backend_and_config(
    hpi_config: HPIConfig,
    model_paths: ModelPaths,
) -> Union[Tuple[InferenceBackend, Dict[str, Any]], Tuple[None, str]]:
    # TODO: The current strategy is naive. It would be better to consider
    # additional important factors, such as NVIDIA GPU compute capability and
    # device manufacturers. We should also allow users to provide hints.

    from ultra_infer import (
        is_built_with_om,
        is_built_with_openvino,
        is_built_with_ort,
        is_built_with_trt,
    )

    is_onnx_model_available = "onnx" in model_paths
    # TODO: Give a warning if the Paddle2ONNX plugin is not available but
    # can be used to select a better backend.
    if hpi_config.auto_paddle2onnx and is_paddle2onnx_plugin_available():
        is_onnx_model_available = is_onnx_model_available or "paddle" in model_paths
    available_backends = []
    if "paddle" in model_paths:
        available_backends.append("paddle")
    if (
        is_built_with_openvino()
        and is_onnx_model_available
        and hpi_config.device_type == "cpu"
    ):
        available_backends.append("openvino")
    if (
        is_built_with_ort()
        and is_onnx_model_available
        and hpi_config.device_type in ("cpu", "gpu")
    ):
        available_backends.append("onnxruntime")
    if (
        is_built_with_trt()
        and is_onnx_model_available
        and hpi_config.device_type == "gpu"
    ):
        available_backends.append("tensorrt")
    if is_built_with_om() and "om" in model_paths and hpi_config.device_type == "npu":
        available_backends.append("om")

    if not available_backends:
        return None, "No inference backends are available."

    if hpi_config.backend is not None and hpi_config.backend not in available_backends:
        return None, f"Inference backend {repr(hpi_config.backend)} is unavailable."

    paddle_version = get_paddle_version()

    if paddle_version[:3] >= (3, 1, 0):
        logging.debug(
            "Paddle version %s is not supported yet. The prior knowledge of Paddle 3.1.1 will be used.",
            paddle_version,
        )
        paddle_version = (3, 1, 1, None)

    if (3, 0) <= paddle_version[:2] <= (3, 1) and paddle_version[3] is None:
        if paddle_version[2] == 0:
            paddle_version = f"paddle{paddle_version[0]}{paddle_version[1]}"
        else:
            paddle_version = (
                f"paddle{paddle_version[0]}{paddle_version[1]}{paddle_version[2]}"
            )
    else:
        return (
            None,
            f"{paddle_version} is not a supported Paddle version.",
        )

    if hpi_config.device_type == "cpu":
        uname = platform.uname()
        arch = uname.machine.lower()
        if arch == "x86_64":
            key = "cpu_x64"
        else:
            return None, f"{repr(arch)} is not a supported architecture."
    elif hpi_config.device_type == "gpu":
        # TODO: Is it better to also check the runtime versions of CUDA and
        # cuDNN, and the versions of CUDA and cuDNN used to build `ultra-infer`?
        cuda_version = get_paddle_cuda_version()
        if not cuda_version:
            return None, "No CUDA version was found."
        cuda_version = cuda_version[0]
        key = f"gpu_cuda{cuda_version}"
    else:
        return None, f"{repr(hpi_config.device_type)} is not a supported device type."

    hpi_model_info_collection = _get_hpi_model_info_collection()

    if key not in hpi_model_info_collection:
        return None, "No prior knowledge can be utilized."
    hpi_model_info_collection_for_env = hpi_model_info_collection[key][paddle_version]

    if hpi_config.pdx_model_name not in hpi_model_info_collection_for_env:
        return None, f"{repr(hpi_config.pdx_model_name)} is not a known model."
    supported_pseudo_backends = hpi_model_info_collection_for_env[
        hpi_config.pdx_model_name
    ].copy()

    if not (is_mkldnn_available() and hpi_config.device_type == "cpu"):
        for pb in supported_pseudo_backends[:]:
            if pb.startswith("paddle_mkldnn"):
                supported_pseudo_backends.remove(pb)

    # XXX
    if not (
        USE_PIR_TRT
        and importlib.util.find_spec("tensorrt")
        and ctypes.util.find_library("nvinfer")
        and hpi_config.device_type == "gpu"
    ):
        for pb in supported_pseudo_backends[:]:
            if pb.startswith("paddle_tensorrt"):
                supported_pseudo_backends.remove(pb)

    supported_backends = []
    backend_to_pseudo_backends = defaultdict(list)
    for pb in supported_pseudo_backends:
        if pb.startswith("paddle"):
            backend = "paddle"
        elif pb.startswith("tensorrt"):
            backend = "tensorrt"
        else:
            backend = pb
        if available_backends is not None and backend not in available_backends:
            continue
        supported_backends.append(backend)
        backend_to_pseudo_backends[backend].append(pb)

    if not supported_backends:
        return None, "No inference backend can be selected."

    if hpi_config.backend is not None:
        if hpi_config.backend not in supported_backends:
            return (
                None,
                f"{repr(hpi_config.backend)} is not a supported inference backend.",
            )
        suggested_backend = hpi_config.backend
    else:
        # Prefer the first one.
        suggested_backend = supported_backends[0]

    pseudo_backends = backend_to_pseudo_backends[suggested_backend]

    if hpi_config.backend_config is not None:
        requested_base_pseudo_backend = None
        if suggested_backend == "paddle":
            if "run_mode" in hpi_config.backend_config:
                if hpi_config.backend_config["run_mode"].startswith("mkldnn"):
                    requested_base_pseudo_backend = "paddle_mkldnn"
                elif hpi_config.backend_config["run_mode"].startswith("trt"):
                    requested_base_pseudo_backend = "paddle_tensorrt"
        if requested_base_pseudo_backend:
            for pb in pseudo_backends:
                if pb.startswith(requested_base_pseudo_backend):
                    break
            else:
                return None, "Unsupported backend configuration."
    pseudo_backend = pseudo_backends[0]

    suggested_backend_config = {}
    if suggested_backend == "paddle":
        assert pseudo_backend in (
            "paddle",
            "paddle_fp16",
            "paddle_mkldnn",
            "paddle_tensorrt",
            "paddle_tensorrt_fp16",
        ), pseudo_backend
        if pseudo_backend == "paddle":
            suggested_backend_config.update({"run_mode": "paddle"})
        elif pseudo_backend == "paddle_fp16":
            suggested_backend_config.update({"run_mode": "paddle_fp16"})
        elif pseudo_backend == "paddle_mkldnn":
            suggested_backend_config.update({"run_mode": "mkldnn"})
        elif pseudo_backend == "paddle_tensorrt":
            suggested_backend_config.update({"run_mode": "trt_fp32"})
        elif pseudo_backend == "paddle_tensorrt_fp16":
            # TODO: Check if the target device supports FP16.
            suggested_backend_config.update({"run_mode": "trt_fp16"})
    elif suggested_backend == "tensorrt":
        assert pseudo_backend in ("tensorrt", "tensorrt_fp16"), pseudo_backend
        if pseudo_backend == "tensorrt_fp16":
            suggested_backend_config.update({"precision": "fp16"})

    if hpi_config.backend_config is not None:
        suggested_backend_config.update(hpi_config.backend_config)

    return suggested_backend, suggested_backend_config
