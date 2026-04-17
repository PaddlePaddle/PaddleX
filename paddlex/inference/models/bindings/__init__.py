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

"""Binding registry: model_name × engine → predictor + structured binding data."""

from .registry import (
    Binding,
    BindingRegistration,
    BindingRegistry,
    ModelRegistryLookupError,
    RunnerBinding,
    UnknownModelError,
    UnsupportedEngineError,
    create_binding_registration,
    default_registry,
    get_binding,
    get_predictor_cls,
    get_supported_engines,
    register_predictor_binding_map,
    try_get_binding,
    try_get_supported_engines,
)

__all__ = [
    "Binding",
    "BindingRegistration",
    "BindingRegistry",
    "ModelRegistryLookupError",
    "RunnerBinding",
    "UnknownModelError",
    "UnsupportedEngineError",
    "create_binding_registration",
    "default_registry",
    "get_binding",
    "get_predictor_cls",
    "get_supported_engines",
    "register_predictor_binding_map",
    "try_get_binding",
    "try_get_supported_engines",
]
