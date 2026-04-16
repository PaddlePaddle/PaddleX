#!/usr/bin/env python3
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

"""Inference engine registry."""

# Import engine modules so subclasses register themselves.
from . import (  # noqa: F401
    flexible,
    genai_client,
    hpi,
    onnxruntime,
    paddle,
    transformers,
)
from ._base import InferenceEngine, RunnerEngine
from .transformers import TransformersEngineConfig

__all__ = [
    "InferenceEngine",
    "RunnerEngine",
    "TransformersEngineConfig",
]
