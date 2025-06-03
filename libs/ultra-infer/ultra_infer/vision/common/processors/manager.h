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

#pragma once

#include "ultra_infer/utils/utils.h"
#include "ultra_infer/vision/common/processors/base.h"
#include "ultra_infer/vision/common/processors/mat.h"
#include "ultra_infer/vision/common/processors/mat_batch.h"

namespace ultra_infer {
namespace vision {

/*! @brief ProcessorManager for Preprocess
 */
class ULTRAINFER_DECL ProcessorManager {
public:
  ~ProcessorManager();

  /** \brief Use CUDA to boost the performance of processors
   *
   * \param[in] enable_cv_cuda true: use CV-CUDA, false: use CUDA only
   * \param[in] gpu_id GPU device id
   * \return true if the preprocess succeeded, otherwise false
   */
  void UseCuda(bool enable_cv_cuda = false, int gpu_id = -1);

  bool CudaUsed();

#ifdef WITH_GPU
  cudaStream_t Stream() const { return stream_; }
#endif

  void SetStream(FDMat *mat) {
#ifdef WITH_GPU
    mat->SetStream(stream_);
#endif
  }

  void SetStream(FDMatBatch *mat_batch) {
#ifdef WITH_GPU
    mat_batch->SetStream(stream_);
#endif
  }

  void SyncStream() {
#ifdef WITH_GPU
    FDASSERT(cudaStreamSynchronize(stream_) == cudaSuccess,
             "[ERROR] Error occurs while sync cuda stream.");
#endif
  }

  int DeviceId() { return device_id_; }

  /** \brief Process the input images and prepare input tensors for runtime
   *
   * \param[in] images The input image data list, all the elements are returned
   * by cv::imread() \param[in] outputs The output tensors which will feed in
   * runtime \return true if the preprocess succeeded, otherwise false
   */
  bool Run(std::vector<FDMat> *images, std::vector<FDTensor> *outputs);

  /** \brief Apply() is the body of Run() function, it needs to be implemented
   * by a derived class
   *
   * \param[in] image_batch The input image batch
   * \param[in] outputs The output tensors which will feed in runtime
   * \return true if the preprocess succeeded, otherwise false
   */
  virtual bool Apply(FDMatBatch *image_batch,
                     std::vector<FDTensor> *outputs) = 0;

  void PreApply(FDMatBatch *image_batch);

  void PostApply();

protected:
  ProcLib proc_lib_ = ProcLib::DEFAULT;

private:
#ifdef WITH_GPU
  cudaStream_t stream_ = nullptr;
#endif
  int device_id_ = -1;

  std::vector<FDTensor> input_caches_;
  std::vector<FDTensor> output_caches_;
  FDTensor batch_input_cache_;
  FDTensor batch_output_cache_;
};

} // namespace vision
} // namespace ultra_infer
