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
#include "ultra_infer/vision/common/processors/manager.h"

namespace ultra_infer {
namespace vision {

ProcessorManager::~ProcessorManager() {
#ifdef WITH_GPU
  if (stream_)
    cudaStreamDestroy(stream_);
#endif
}

void ProcessorManager::UseCuda(bool enable_cv_cuda, int gpu_id) {
#ifdef WITH_GPU
  if (gpu_id >= 0) {
    device_id_ = gpu_id;
    FDASSERT(cudaSetDevice(device_id_) == cudaSuccess,
             "[ERROR] Error occurs while setting cuda device.");
  }
  FDASSERT(cudaStreamCreate(&stream_) == cudaSuccess,
           "[ERROR] Error occurs while creating cuda stream.");
  proc_lib_ = ProcLib::CUDA;
#else
  FDASSERT(false, "UltraInfer didn't compile with WITH_GPU.");
#endif

  if (enable_cv_cuda) {
#ifdef ENABLE_CVCUDA
    proc_lib_ = ProcLib::CVCUDA;
#else
    FDASSERT(false, "UltraInfer didn't compile with CV-CUDA.");
#endif
  }
}

bool ProcessorManager::CudaUsed() {
  return (proc_lib_ == ProcLib::CUDA || proc_lib_ == ProcLib::CVCUDA);
}

void ProcessorManager::PreApply(FDMatBatch *image_batch) {
  FDASSERT(image_batch->mats != nullptr, "The mats is empty.");
  FDASSERT(image_batch->mats->size() > 0,
           "The size of input images should be greater than 0.");

  if (image_batch->mats->size() > input_caches_.size()) {
    input_caches_.resize(image_batch->mats->size());
    output_caches_.resize(image_batch->mats->size());
  }
  image_batch->input_cache = &batch_input_cache_;
  image_batch->output_cache = &batch_output_cache_;
  image_batch->proc_lib = proc_lib_;
  if (CudaUsed()) {
    SetStream(image_batch);
  }

  for (size_t i = 0; i < image_batch->mats->size(); ++i) {
    FDMat *mat = &(image_batch->mats->at(i));
    mat->input_cache = &input_caches_[i];
    mat->output_cache = &output_caches_[i];
    mat->proc_lib = proc_lib_;
    if (mat->mat_type == ProcLib::CUDA) {
      // Make a copy of the input data ptr, so that the original data ptr of
      // FDMat won't be modified.
      auto fd_tensor = std::make_shared<FDTensor>();
      fd_tensor->SetExternalData(mat->Tensor()->shape, mat->Tensor()->Dtype(),
                                 mat->Tensor()->Data(), mat->Tensor()->device,
                                 mat->Tensor()->device_id);
      mat->SetTensor(fd_tensor);
    }
  }
}

void ProcessorManager::PostApply() {
  if (CudaUsed()) {
    SyncStream();
  }
}

bool ProcessorManager::Run(std::vector<FDMat> *images,
                           std::vector<FDTensor> *outputs) {
  FDMatBatch image_batch(images);
  PreApply(&image_batch);
  bool ret = Apply(&image_batch, outputs);
  PostApply();
  return ret;
}

} // namespace vision
} // namespace ultra_infer
