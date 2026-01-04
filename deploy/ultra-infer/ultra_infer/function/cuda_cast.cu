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
#include "ultra_infer/function/cuda_cast.h"
namespace ultra_infer {
namespace function {
template <typename T_IN, typename T_OUT>
__global__ void CudaCastKernel(const T_IN *in, T_OUT *out, int edge) {
  int position = blockDim.x * blockIdx.x + threadIdx.x;
  if (position >= edge)
    return;
  out[position] = (T_OUT)in[position];
}

void CudaCast(const FDTensor &in, FDTensor *out, cudaStream_t stream) {
  int jobs = in.Numel();
  int threads = 256;
  int blocks = ceil(jobs / (float)threads);
  if (in.dtype == FDDataType::INT64 && out->dtype == FDDataType::INT32) {
    CudaCastKernel<int64_t, int32_t><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<int64_t *>(const_cast<void *>(in.Data())),
        reinterpret_cast<int32_t *>(out->MutableData()), jobs);
  } else if (in.dtype == FDDataType::INT32 && out->dtype == FDDataType::INT64) {
    CudaCastKernel<int32_t, int64_t><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<int32_t *>(const_cast<void *>(in.Data())),
        reinterpret_cast<int64_t *>(out->MutableData()), jobs);
  } else {
    FDASSERT(false, "CudaCast only support input INT64, output INT32.");
  }
}

} // namespace function
} // namespace ultra_infer
#endif
