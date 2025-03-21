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
from typing import Dict, Final, List, Tuple, Union


class BackendSelectionStrategy(metaclass=abc.ABCMeta):
    ONNX_ALLOWED_BACKENDS: Final[List[str]] = ["onnx_runtime", "tensorrt", "openvino"]
    PADDLE_ALLOWED_BACKENDS: Final[List[str]] = ONNX_ALLOWED_BACKENDS + ["paddle_infer"]

    onnx_format: bool

    def backend_filter(self, backend_config_pairs: List[Tuple[str, Dict]]) -> List[str]:
        allowed_backends = (
            self.ONNX_ALLOWED_BACKENDS
            if self.onnx_format
            else self.PADDLE_ALLOWED_BACKENDS
        )
        filtered_backends = [
            backend
            for backend in backend_config_pairs
            if backend[0] in allowed_backends
        ]

        return filtered_backends

    @abc.abstractmethod
    def select_backend_and_config(
        self, backend_config_pairs: List[Tuple[str, Dict]]
    ) -> Union[str, Dict]:
        raise NotImplementedError


class SelectSpecificStrategy(BackendSelectionStrategy):

    def __init__(self, onnx_format: bool, specified_backend: str):
        self.onnx_format = onnx_format
        self.specified_backend = specified_backend

    def select_backend_and_config(
        self, backend_config_pairs: List[Tuple[str, Dict]]
    ) -> Union[str, Dict]:
        filtered_backends = self.backend_filter(backend_config_pairs)

        for backend, config_dict in filtered_backends:
            if backend == self.specified_backend:
                return backend, config_dict

        if self.onnx_format:
            raise ValueError(
                f"Unspported backend: {self.specified_backend}. Supported backends are: {', '.join(i[0] for i in filtered_backends)}"
            )
        else:
            return "paddle_infer", {}


class SelectFirstStrategy(BackendSelectionStrategy):

    def __init__(self, onnx_format: bool):
        self.onnx_format = onnx_format

    def select_backend_and_config(
        self, backend_config_pairs: List[Tuple[str, Dict]]
    ) -> Union[str, Dict]:
        filtered_backends = self.backend_filter(backend_config_pairs)

        if filtered_backends:
            backend, config_dict = filtered_backends[0]
        else:
            if self.onnx_format:
                raise ValueError(
                    "There is no supported backend for the ONNX model. Please use Paddle model instead."
                )
            else:
                backend, config_dict = (
                    filtered_backends[0] if filtered_backends else ["paddle_infer", {}]
                )

        return backend, config_dict
