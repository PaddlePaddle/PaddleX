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
#include "ultra_infer/vision/common/processors/mat_batch.h"

namespace ultra_infer {
namespace vision {

#ifdef WITH_GPU
void FDMatBatch::SetStream(cudaStream_t s) {
  stream = s;
  for (size_t i = 0; i < mats->size(); ++i) {
    (*mats)[i].SetStream(s);
  }
}
#endif

FDTensor *FDMatBatch::Tensor() {
  if (has_batched_tensor) {
    return fd_tensor.get();
  }
  FDASSERT(mats != nullptr, "Failed to get batched tensor, Mats are empty.");
  FDASSERT(CheckShapeConsistency(mats), "Mats shapes are not consistent.");
  // Each mat has its own tensor,
  // to get a batched tensor, we need copy these tensors to a batched tensor
  FDTensor *src = (*mats)[0].Tensor();
  device = src->device;
  auto new_shape = src->Shape();
  new_shape.insert(new_shape.begin(), mats->size());
  input_cache->Resize(new_shape, src->Dtype(), "batch_input_cache", device);
  for (size_t i = 0; i < mats->size(); ++i) {
    FDASSERT(device == (*mats)[i].Tensor()->device,
             "Mats and MatBatch are not on the same device");
    uint8_t *p = reinterpret_cast<uint8_t *>(input_cache->Data());
    int num_bytes = (*mats)[i].Tensor()->Nbytes();
    FDTensor::CopyBuffer(p + i * num_bytes, (*mats)[i].Tensor()->Data(),
                         num_bytes, device, false);
  }
  SetTensor(input_cache);
  return fd_tensor.get();
}

void FDMatBatch::SetTensor(FDTensor *tensor) {
  fd_tensor->SetExternalData(tensor->Shape(), tensor->Dtype(), tensor->Data(),
                             tensor->device, tensor->device_id);
  device = tensor->device;
  has_batched_tensor = true;
}

FDTensor *CreateCachedGpuInputTensor(FDMatBatch *mat_batch) {
#ifdef WITH_GPU
  // Get the batched tensor
  FDTensor *src = mat_batch->Tensor();
  // Need to make sure the returned tensor is pointed to the input_cache.
  if (src->Data() == mat_batch->output_cache->Data()) {
    std::swap(mat_batch->input_cache, mat_batch->output_cache);
    std::swap(mat_batch->input_cache->name, mat_batch->output_cache->name);
  }
  if (src->device == Device::GPU) {
    return src;
  } else if (src->device == Device::CPU) {
    // Batched tensor on CPU, we need copy it to GPU
    mat_batch->output_cache->Resize(src->Shape(), src->Dtype(), "output_cache",
                                    Device::GPU);
    FDASSERT(cudaMemcpyAsync(mat_batch->output_cache->Data(), src->Data(),
                             src->Nbytes(), cudaMemcpyHostToDevice,
                             mat_batch->Stream()) == 0,
             "[ERROR] Error occurs while copy memory from CPU to GPU.");
    std::swap(mat_batch->input_cache, mat_batch->output_cache);
    std::swap(mat_batch->input_cache->name, mat_batch->output_cache->name);
    return mat_batch->input_cache;
  } else {
    FDASSERT(false, "FDMatBatch is on unsupported device: %d", src->device);
  }
#else
  FDASSERT(false, "UltraInfer didn't compile with WITH_GPU.");
#endif
  return nullptr;
}

} // namespace vision
} // namespace ultra_infer
