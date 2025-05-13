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

#include "ultra_infer/vision/common/processors/base.h"

namespace ultra_infer {
namespace vision {
/*! @brief Processor for Normalize images with given parameters.
 */
class ULTRAINFER_DECL Normalize : public Processor {
public:
  Normalize(const std::vector<float> &mean, const std::vector<float> &std,
            bool is_scale = true,
            const std::vector<float> &min = std::vector<float>(),
            const std::vector<float> &max = std::vector<float>(),
            bool swap_rb = false);
  bool ImplByOpenCV(Mat *mat);
#ifdef ENABLE_FLYCV
  bool ImplByFlyCV(Mat *mat);
#endif
#ifdef WITH_GPU
  bool ImplByCuda(FDMat *mat);
  bool ImplByCuda(FDMatBatch *mat_batch);
#endif
#ifdef ENABLE_CVCUDA
  bool ImplByCvCuda(FDMat *mat);
  bool ImplByCvCuda(FDMatBatch *mat_batch);
#endif
  std::string Name() { return "Normalize"; }

  // While use normalize, it is more recommend not use this function
  // this function will need to compute result = ((mat / 255) - mean) / std
  // if we use the following method
  // ```
  // auto norm = Normalize(...)
  // norm(mat)
  // ```
  // There will be some precomputation in construct function
  // and the `norm(mat)` only need to compute result = mat * alpha + beta
  // which will reduce lots of time
  /** \brief Process the input images
   *
   * \param[in] mat The input image data, `result = mat * alpha + beta`
   * \param[in] mean target mean vector of output images
   * \param[in] std target std vector of output images
   * \param[in] max max value vector to be in target image
   * \param[in] min min value vector to be in target image
   * \param[in] lib to define OpenCV or FlyCV or CVCUDA will be used.
   * \param[in] swap_rb to define whether to swap r and b channel order
   * \return true if the process successed, otherwise false
   */
  static bool Run(Mat *mat, const std::vector<float> &mean,
                  const std::vector<float> &std, bool is_scale = true,
                  const std::vector<float> &min = std::vector<float>(),
                  const std::vector<float> &max = std::vector<float>(),
                  ProcLib lib = ProcLib::DEFAULT, bool swap_rb = false);

  std::vector<float> GetAlpha() const { return alpha_; }
  std::vector<float> GetBeta() const { return beta_; }

  bool GetSwapRB() { return swap_rb_; }

  /** \brief Process the input images
   *
   * \param[in] swap_rb set the value of the swap_rb parameter
   */
  void SetSwapRB(bool swap_rb) { swap_rb_ = swap_rb; }

private:
  std::vector<float> alpha_;
  std::vector<float> beta_;
  FDTensor gpu_alpha_;
  FDTensor gpu_beta_;
  bool swap_rb_;
};
} // namespace vision
} // namespace ultra_infer
