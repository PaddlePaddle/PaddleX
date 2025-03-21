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

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, Union

import ultra_infer as ui
from paddlex_hpi._model_info import get_model_info
from paddlex_hpi._strategy import SelectFirstStrategy, SelectSpecificStrategy
from paddlex_hpi._utils.typing import Backend, DeviceType
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import Annotated, TypeAlias, TypedDict, assert_never

from paddlex.utils import logging


class _BackendConfig(BaseModel):
    def update_ui_option(self, option: ui.RuntimeOption, model_dir: Path) -> None:
        raise NotImplementedError


class PaddleInferConfig(_BackendConfig):
    cpu_num_threads: int = 8
    enable_mkldnn: bool = True
    enable_trt: bool = False
    trt_dynamic_shapes: Optional[Dict[str, List[List[int]]]] = None
    trt_dynamic_shape_input_data: Optional[Dict[str, List[List[float]]]] = None
    trt_precision: Literal["FP32", "FP16"] = "FP32"
    enable_log_info: bool = False

    def update_ui_option(self, option: ui.RuntimeOption, model_dir: Path) -> None:
        option.use_paddle_infer_backend()
        option.set_cpu_thread_num(self.cpu_num_threads)
        option.paddle_infer_option.enable_mkldnn = self.enable_mkldnn
        option.paddle_infer_option.enable_trt = self.enable_trt
        if self.enable_trt:
            option.trt_option.serialize_file = str(model_dir / "trt_serialized.trt")
            option.paddle_infer_option.collect_trt_shape = True
            option.paddle_infer_option.collect_trt_shape_by_device = True
            if self.trt_dynamic_shapes is not None:
                for name, shapes in self.trt_dynamic_shapes.items():
                    option.trt_option.set_shape(name, *shapes)
            if self.trt_dynamic_shape_input_data is not None:
                for name, data in self.trt_dynamic_shape_input_data.items():
                    option.trt_option.set_input_data(name, *data)
            if self.trt_precision == "FP16":
                option.trt_option.enable_fp16 = True
        option.paddle_infer_option.enable_log_info = self.enable_log_info


class OpenVINOConfig(_BackendConfig):
    cpu_num_threads: int = 8

    def update_ui_option(self, option: ui.RuntimeOption, model_dir: Path) -> None:
        option.use_openvino_backend()
        option.set_cpu_thread_num(self.cpu_num_threads)


class ONNXRuntimeConfig(_BackendConfig):
    cpu_num_threads: int = 8

    def update_ui_option(self, option: ui.RuntimeOption, model_dir: Path) -> None:
        option.use_ort_backend()
        option.set_cpu_thread_num(self.cpu_num_threads)


class TensorRTConfig(_BackendConfig):
    precision: Literal["FP32", "FP16"] = "FP32"
    dynamic_shapes: Optional[Dict[str, List[List[int]]]] = None

    def update_ui_option(self, option: ui.RuntimeOption, model_dir: Path) -> None:
        option.use_trt_backend()
        option.trt_option.serialize_file = str(
            model_dir / f"trt_serialized_{self.precision}.trt"
        )
        if self.precision == "FP16":
            option.trt_option.enable_fp16 = True
        if self.dynamic_shapes is not None and not os.path.exists(
            option.trt_option.serialize_file
        ):
            for name, shapes in self.dynamic_shapes.items():
                option.trt_option.set_shape(name, *shapes)


class PaddleTensorRTConfig(_BackendConfig):
    dynamic_shapes: Dict[str, List[List[int]]]
    dynamic_shape_input_data: Optional[Dict[str, List[List[float]]]] = None
    enable_log_info: bool = False

    def update_ui_option(self, option: ui.RuntimeOption, model_dir: Path) -> None:
        option.use_paddle_infer_backend()
        option.paddle_infer_option.enable_trt = True
        option.trt_option.serialize_file = str(model_dir / "trt_serialized.trt")
        if self.dynamic_shapes is not None:
            option.paddle_infer_option.collect_trt_shape = True
            # TODO: Support setting collect_trt_shape_by_device
            for name, shapes in self.dynamic_shapes.items():
                option.trt_option.set_shape(name, *shapes)
        if self.dynamic_shape_input_data is not None:
            for name, data in self.dynamic_shape_input_data.items():
                option.trt_option.set_input_data(name, *data)
        option.paddle_infer_option.enable_log_info = self.enable_log_info


# Should we use tagged unions?
BackendConfig: TypeAlias = Union[
    PaddleInferConfig,
    OpenVINOConfig,
    ONNXRuntimeConfig,
    TensorRTConfig,
]


def get_backend_config_type(backend: Backend, /) -> Type[BackendConfig]:
    backend_config_type: Type[BackendConfig]
    if backend == "paddle_infer":
        backend_config_type = PaddleInferConfig
    elif backend == "openvino":
        backend_config_type = OpenVINOConfig
    elif backend == "onnx_runtime":
        backend_config_type = ONNXRuntimeConfig
    elif backend == "tensorrt":
        backend_config_type = TensorRTConfig
    else:
        assert_never(backend)
    return backend_config_type


# Can I create this dynamically and automatically?
class BackendConfigs(TypedDict, total=False):
    paddle_infer: PaddleInferConfig
    openvino: OpenVINOConfig
    onnx_runtime: ONNXRuntimeConfig
    tensorrt: TensorRTConfig
    paddle_tensorrt: PaddleTensorRTConfig


class HPIConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    selected_backends: Optional[Dict[DeviceType, Backend]] = None
    # For backward compatilibity
    backend_configs: Annotated[
        Optional[BackendConfigs], Field(validation_alias="backend_config")
    ] = None

    def get_backend_and_config(
        self, model_name: str, device_type: DeviceType, onnx_format: bool
    ) -> Tuple[Backend, BackendConfig]:
        # Do we need an extensible selector?
        model_info = get_model_info(model_name, device_type)
        if model_info:
            backend_config_pairs = model_info["backend_config_pairs"]
        else:
            backend_config_pairs = []

        use_specific_backend = (
            self.selected_backends and device_type in self.selected_backends
        )
        if use_specific_backend:
            specified_backend = self.selected_backends[device_type]
            strategy = SelectSpecificStrategy(onnx_format, specified_backend)
        else:
            strategy = SelectFirstStrategy(onnx_format)

        backend, config_dict = strategy.select_backend_and_config(backend_config_pairs)
        if self.backend_configs and backend in self.backend_configs:
            config_dict.update(
                self.backend_configs[backend].model_dump(exclude_unset=True)
            )
        backend_config_type = get_backend_config_type(backend)
        backend_config = backend_config_type.model_validate(config_dict)
        return backend, backend_config

    # XXX: For backward compatilibity
    @field_validator("selected_backends", mode="before")
    @classmethod
    def _hack_selected_backends(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            new_data = dict(data)
            for device_type in new_data:
                if new_data[device_type] == "paddle_tensorrt":
                    warnings.warn(
                        "`paddle_tensorrt` is deprecated. Please use `paddle_infer` instead.",
                        FutureWarning,
                    )
                    new_data[device_type] = "paddle_infer"
        return new_data

    @field_validator("backend_configs", mode="before")
    @classmethod
    def _hack_backend_configs(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            new_data = dict(data)
            if new_data and "paddle_tensorrt" in new_data:
                warnings.warn(
                    "`paddle_tensorrt` is deprecated. Please use `paddle_infer` instead.",
                    FutureWarning,
                )
                if "paddle_infer" not in new_data:
                    new_data["paddle_infer"] = {}
                pptrt_cfg = new_data["paddle_tensorrt"]
                logging.warning("`paddle_infer.enable_trt` will be set to `True`.")
                new_data["paddle_infer"]["enable_trt"] = True
                new_data["paddle_infer"]["trt_dynamic_shapes"] = pptrt_cfg[
                    "dynamic_shapes"
                ]
                if "dynamic_shape_input_data" in pptrt_cfg:
                    new_data["paddle_infer"]["trt_dynamic_shape_input_data"] = (
                        pptrt_cfg["dynamic_shape_input_data"]
                    )
                logging.warning("`paddle_tensorrt.enable_log_info` will be ignored.")
        return new_data
