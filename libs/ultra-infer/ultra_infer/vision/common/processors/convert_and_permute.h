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
/*! @brief Processor for convert images with given parameters and permute images
 * from HWC to CHW.
 */
class ULTRAINFER_DECL ConvertAndPermute : public Processor {
public:
  ConvertAndPermute(const std::vector<float> &alpha = std::vector<float>(),
                    const std::vector<float> &beta = std::vector<float>(),
                    bool swap_rb = false);
  bool ImplByOpenCV(FDMat *mat);
#ifdef ENABLE_FLYCV
  bool ImplByFlyCV(FDMat *mat);
#endif
  std::string Name() { return "ConvertAndPermute"; }

  /** \brief Process the input images
   *
   * \param[in] mat The input image data，`result = mat * alpha + beta`
   * \param[in] alpha The alpha channel data
   * \param[in] beta The beta channel data
   * \param[in] lib to define OpenCV or FlyCV or CVCUDA will be used.
   * \return true if the process successed, otherwise false
   */
  static bool Run(FDMat *mat, const std::vector<float> &alpha,
                  const std::vector<float> &beta, bool swap_rb = false,
                  ProcLib lib = ProcLib::DEFAULT);

  std::vector<float> GetAlpha() const { return alpha_; }

  /** \brief Process the input images
   *
   * \param[in] alpha set the value of the alpha parameter
   */
  void SetAlpha(const std::vector<float> &alpha) {
    alpha_.clear();
    std::vector<float>().swap(alpha_);
    alpha_.assign(alpha.begin(), alpha.end());
  }

  std::vector<float> GetBeta() const { return beta_; }

  /** \brief Process the input images
   *
   * \param[in] beta set the value of the beta parameter
   */
  void SetBeta(const std::vector<float> &beta) {
    beta_.clear();
    std::vector<float>().swap(beta_);
    beta_.assign(beta.begin(), beta.end());
  }

  bool GetSwapRB() { return swap_rb_; }

  /** \brief Process the input images
   *
   * \param[in] swap_rb set the value of the swap_rb parameter
   */
  void SetSwapRB(bool swap_rb) { swap_rb_ = swap_rb; }

private:
  std::vector<float> alpha_;
  std::vector<float> beta_;
  bool swap_rb_;
};
} // namespace vision
} // namespace ultra_infer
