// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifdef WITH_GPU
#include <cuda_runtime_api.h>
#endif

#include "ultra_infer/core/allocate.h"

namespace ultra_infer {

bool FDHostAllocator::operator()(void **ptr, size_t size) const {
  *ptr = malloc(size);
  return *ptr != nullptr;
}

void FDHostFree::operator()(void *ptr) const { free(ptr); }

#ifdef WITH_GPU

bool FDDeviceAllocator::operator()(void **ptr, size_t size) const {
  return cudaMalloc(ptr, size) == cudaSuccess;
}

void FDDeviceFree::operator()(void *ptr) const { cudaFree(ptr); }

bool FDDeviceHostAllocator::operator()(void **ptr, size_t size) const {
  return cudaMallocHost(ptr, size) == cudaSuccess;
}

void FDDeviceHostFree::operator()(void *ptr) const { cudaFreeHost(ptr); }

#endif

} // namespace ultra_infer
