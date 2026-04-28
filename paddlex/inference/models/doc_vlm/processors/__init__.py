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
from .....utils.lazy_loader import LazyLoader
from .qwen2_5_vl import PPDocBee2Processor, Qwen2_5_VLImageProcessor
from .qwen2_vl import PPDocBeeProcessor, Qwen2VLImageProcessor

_GOT_ocr_2_0 = LazyLoader("_GOT_ocr_2_0", globals(), __name__ + ".GOT_ocr_2_0")
_paddleocr_vl = LazyLoader("_paddleocr_vl", globals(), __name__ + ".paddleocr_vl")

__all__ = [
    "GOTImageProcessor",
    "PPChart2TableProcessor",
    "PaddleOCRVLProcessor",
    "SiglipImageProcessor",
    "PPDocBee2Processor",
    "Qwen2_5_VLImageProcessor",
    "PPDocBeeProcessor",
    "Qwen2VLImageProcessor",
]

_LAZY_SYMBOL_TO_MODULE = {
    "GOTImageProcessor": _GOT_ocr_2_0,
    "PPChart2TableProcessor": _GOT_ocr_2_0,
    "PaddleOCRVLProcessor": _paddleocr_vl,
    "SiglipImageProcessor": _paddleocr_vl,
}


def __getattr__(name):
    module = _LAZY_SYMBOL_TO_MODULE.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(module, name)
    globals()[name] = value
    return value
