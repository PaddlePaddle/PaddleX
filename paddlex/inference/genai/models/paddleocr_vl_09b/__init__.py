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


def get_network_class(backend):
    if backend == "vllm":
        from ._vllm import PaddleOCRVLForConditionalGeneration

        return PaddleOCRVLForConditionalGeneration
    elif backend == "sglang":
        from ._sglang import PaddleOCRVLForConditionalGeneration

        return PaddleOCRVLForConditionalGeneration
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def get_processor_class(backend):
    if backend == "sglang":
        from ._sglang import PaddleOCRVLImageProcessor

        return PaddleOCRVLImageProcessor
    else:
        raise ValueError(f"Unsupported backend: {backend}")
