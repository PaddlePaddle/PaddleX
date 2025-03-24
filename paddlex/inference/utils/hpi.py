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

import importlib.resources
import json
import platform
from functools import lru_cache
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Final, List, Literal, Optional, Tuple, TypedDict, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypeAlias

from ...utils.flags import FLAGS_json_format_model


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
    cpu_num_threads: int = 8


class ONNXRuntimeConfig(BaseModel):
    cpu_num_threads: int = 8


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


class ModelPaths(TypedDict, total=False):
    paddle: Tuple[Path, Path]
    onnx: Path
    om: Path


def get_model_paths(
    model_dir: Union[str, PathLike], model_file_prefix: str
) -> ModelPaths:
    model_dir = Path(model_dir)
    model_paths: ModelPaths = {}
    pd_model_path = None
    if FLAGS_json_format_model:
        if (model_dir / f"{model_file_prefix}.json").exists():
            pd_model_path = model_dir / f"{model_file_prefix}.json"
    else:
        if (model_dir / f"{model_file_prefix}.json").exists():
            pd_model_path = model_dir / f"{model_file_prefix}.json"
        elif (model_dir / f"{model_file_prefix}.pdmodel").exists():
            pd_model_path = model_dir / f"{model_file_prefix}.pdmodel"
    if pd_model_path and (model_dir / f"{model_file_prefix}.pdiparams").exists():
        model_paths["paddle"] = (
            pd_model_path,
            model_dir / f"{model_file_prefix}.pdiparams",
        )
    if (model_dir / f"{model_file_prefix}.onnx").exists():
        model_paths["onnx"] = model_dir / f"{model_file_prefix}.onnx"
    if (model_dir / f"{model_file_prefix}.om").exists():
        model_paths["om"] = model_dir / f"{model_file_prefix}.om"
    return model_paths


@lru_cache(1)
def _get_hpi_model_info_collection():
    with importlib.resources.open_text(
        __package__, "hpi_model_info_collection.json", encoding="utf-8"
    ) as f:
        hpi_model_info_collection = json.load(f)
    return hpi_model_info_collection


def suggest_inference_backend_and_config(
    hpi_config: HPIConfig,
    available_backends: Optional[List[InferenceBackend]] = None,
) -> Union[Tuple[InferenceBackend, Dict[str, Any]], Tuple[None, str]]:
    # TODO: The current strategy is naive. It would be better to consider
    # additional important factors, such as NVIDIA GPU compute capability and
    # device manufacturers. We should also allow users to provide hints.

    import lazy_paddle as paddle

    if available_backends is not None and not available_backends:
        return None, "No inference backends are available."

    paddle_version = paddle.__version__
    if paddle_version != "3.0.0-rc0":
        return None, f"{repr(paddle_version)} is not a supported Paddle version."

    if hpi_config.device_type == "cpu":
        uname = platform.uname()
        arch = uname.machine.lower()
        if arch == "x86_64":
            key = "cpu_x64"
        else:
            return None, f"{repr(arch)} is not a supported architecture."
    elif hpi_config.device_type == "gpu":
        # FIXME: We should not rely on the PaddlePaddle library to detemine CUDA
        # and cuDNN versions.
        # Should we inject environment info from the outside?
        import lazy_paddle.version

        cuda_version = lazy_paddle.version.cuda()
        cuda_version = cuda_version.replace(".", "")
        cudnn_version = lazy_paddle.version.cudnn().rsplit(".", 1)[0]
        cudnn_version = cudnn_version.replace(".", "")
        key = f"gpu_cuda{cuda_version}_cudnn{cudnn_version}"
    else:
        return None, f"{repr(hpi_config.device_type)} is not a supported device type."

    hpi_model_info_collection = _get_hpi_model_info_collection()

    if key not in hpi_model_info_collection:
        return None, "No prior knowledge can be utilized."
    hpi_model_info_collection_for_env = hpi_model_info_collection[key]

    if hpi_config.pdx_model_name not in hpi_model_info_collection_for_env:
        return None, f"{repr(hpi_config.pdx_model_name)} is not a known model."
    supported_pseudo_backends = hpi_model_info_collection_for_env[
        hpi_config.pdx_model_name
    ]

    candidate_backends = []
    backend_to_pseudo_backend = {}
    for pb in supported_pseudo_backends:
        if pb.startswith("paddle"):
            backend = "paddle"
        elif pb.startswith("tensorrt"):
            backend = "tensorrt"
        else:
            backend = pb
        if available_backends is not None and backend not in available_backends:
            continue
        candidate_backends.append(backend)
        backend_to_pseudo_backend[backend] = pb

    if not candidate_backends:
        return None, "No inference backend can be selected."

    if hpi_config.backend is not None:
        if hpi_config.backend not in candidate_backends:
            return (
                None,
                f"{repr(hpi_config.backend)} is not a supported inference backend.",
            )
        suggested_backend = hpi_config.backend
    else:
        # The first backend is the preferred one.
        suggested_backend = candidate_backends[0]

    suggested_backend_config = {}
    if suggested_backend == "paddle":
        pseudo_backend = backend_to_pseudo_backend["paddle"]
        assert pseudo_backend in (
            "paddle",
            "paddle_tensorrt_fp32",
            "paddle_tensorrt_fp16",
        ), pseudo_backend
        if pseudo_backend == "paddle_tensorrt_fp32":
            suggested_backend_config.update({"run_mode": "trt_fp32"})
        elif pseudo_backend == "paddle_tensorrt_fp16":
            # TODO: Check if the target device supports FP16.
            suggested_backend_config.update({"run_mode": "trt_fp16"})
    elif suggested_backend == "tensorrt":
        pseudo_backend = backend_to_pseudo_backend["tensorrt"]
        assert pseudo_backend in ("tensorrt", "tensorrt_fp16"), pseudo_backend
        if pseudo_backend == "tensorrt_fp16":
            suggested_backend_config.update({"precision": "fp16"})

    if hpi_config.backend_config is not None:
        suggested_backend_config.update(hpi_config.backend_config)

    return suggested_backend, suggested_backend_config
