# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from importlib import metadata as _metadata

from .request import triton_request

__all__ = ["__version__", "triton_request"]

# Ref: https://github.com/langchain-ai/langchain/blob/493e474063817b9a4c2521586b2dbc34d20b4cf1/libs/core/langchain_core/__init__.py
try:
    __version__ = _metadata.version(__package__)
except _metadata.PackageNotFoundError:
    __version__ = ""
