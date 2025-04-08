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

import paddle
from paddle.utils.cpp_extension import CppExtension, CUDAExtension, setup

from paddlex.ops import custom_ops

for op_name, op_dict in custom_ops.items():
    sources = op_dict.pop("sources", [])
    flags = None

    if paddle.device.is_compiled_with_cuda():
        extension = CUDAExtension
        flags = {"cxx": ["-DPADDLE_WITH_CUDA"]}
        if "extra_cuda_cflags" in op_dict:
            flags["nvcc"] = op_dict.pop("extra_cuda_cflags")
    else:
        sources = filter(lambda x: x.endswith("cu"), sources)
        extension = CppExtension

    if len(sources) == 0:
        continue

    extension = extension(sources=sources, extra_compile_args=flags)
    setup(name=op_name, ext_modules=extension)
