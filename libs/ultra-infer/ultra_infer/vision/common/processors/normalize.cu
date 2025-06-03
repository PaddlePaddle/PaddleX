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
#include "ultra_infer/vision/common/processors/normalize.h"

namespace ultra_infer {
namespace vision {

__global__ void NormalizeKernel(const uint8_t *src, float *dst,
                                const float *alpha, const float *beta,
                                int num_channel, bool swap_rb, int batch_size,
                                int edge) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= edge)
    return;

  int img_size = edge / batch_size;
  int n = idx / img_size;       // batch index
  int p = idx - (n * img_size); // pixel index within the image

  for (int i = 0; i < num_channel; ++i) {
    int j = i;
    if (swap_rb) {
      j = 2 - i;
    }
    dst[num_channel * idx + j] =
        src[num_channel * idx + j] * alpha[i] + beta[i];
  }
}

bool Normalize::ImplByCuda(FDMat *mat) {
  if (mat->layout != Layout::HWC) {
    FDERROR << "The input data must be NHWC format!" << std::endl;
    return false;
  }

  // Prepare input tensor
  FDTensor *src = CreateCachedGpuInputTensor(mat);
  src->ExpandDim(0);
  FDMatBatch mat_batch;
  mat_batch.SetTensor(src);
  mat_batch.mat_type = ProcLib::CUDA;
  mat_batch.input_cache = mat->input_cache;
  mat_batch.output_cache = mat->output_cache;

  bool ret = ImplByCuda(&mat_batch);

  FDTensor *dst = mat_batch.Tensor();
  dst->Squeeze(0);
  mat->SetTensor(dst);
  mat->mat_type = ProcLib::CUDA;
  return true;
}

bool Normalize::ImplByCuda(FDMatBatch *mat_batch) {
  if (mat_batch->layout != FDMatBatchLayout::NHWC) {
    FDERROR << "The input data must be NHWC format!" << std::endl;
    return false;
  }
  // Prepare input tensor
  FDTensor *src = CreateCachedGpuInputTensor(mat_batch);

  // Prepare output tensor
  mat_batch->output_cache->Resize(src->Shape(), FDDataType::FP32,
                                  "batch_output_cache", Device::GPU);

  // Copy alpha and beta to GPU
  gpu_alpha_.Resize({1, 1, static_cast<int>(alpha_.size())}, FDDataType::FP32,
                    "alpha", Device::GPU);
  cudaMemcpy(gpu_alpha_.Data(), alpha_.data(), gpu_alpha_.Nbytes(),
             cudaMemcpyHostToDevice);

  gpu_beta_.Resize({1, 1, static_cast<int>(beta_.size())}, FDDataType::FP32,
                   "beta", Device::GPU);
  cudaMemcpy(gpu_beta_.Data(), beta_.data(), gpu_beta_.Nbytes(),
             cudaMemcpyHostToDevice);

  int jobs =
      mat_batch->output_cache->Numel() / mat_batch->output_cache->shape[3];
  int threads = 256;
  int blocks = ceil(jobs / (float)threads);
  NormalizeKernel<<<blocks, threads, 0, mat_batch->Stream()>>>(
      reinterpret_cast<uint8_t *>(src->Data()),
      reinterpret_cast<float *>(mat_batch->output_cache->Data()),
      reinterpret_cast<float *>(gpu_alpha_.Data()),
      reinterpret_cast<float *>(gpu_beta_.Data()),
      mat_batch->output_cache->shape[3], swap_rb_,
      mat_batch->output_cache->shape[0], jobs);

  mat_batch->SetTensor(mat_batch->output_cache);
  mat_batch->mat_type = ProcLib::CUDA;
  return true;
}

#ifdef ENABLE_CVCUDA
bool Normalize::ImplByCvCuda(FDMat *mat) { return ImplByCuda(mat); }

bool Normalize::ImplByCvCuda(FDMatBatch *mat_batch) {
  return ImplByCuda(mat_batch);
}
#endif

} // namespace vision
} // namespace ultra_infer
#endif
