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
#include "ultra_infer/vision/common/processors/transform.h"
#include "ultra_infer/vision/common/result.h"

namespace ultra_infer {
namespace vision {

namespace faceid {
/*! @brief Preprocessor object for AdaFace serials model.
 */
class ULTRAINFER_DECL AdaFacePreprocessor {
public:
  /** \brief Create a preprocessor instance for AdaFace serials model
   */
  AdaFacePreprocessor();

  /** \brief Process the input image and prepare input tensors for runtime
   *
   * \param[in] images The input image data list, all the elements are returned
   * by cv::imread() \param[in] outputs The output tensors which will feed in
   * runtime \return true if the preprocess successed, otherwise false
   */
  bool Run(std::vector<FDMat> *images, std::vector<FDTensor> *outputs);

  /// Get Size
  std::vector<int> GetSize() { return size_; }

  /// Set size.
  void SetSize(std::vector<int> &size) { size_ = size; }

  /// Get alpha
  std::vector<float> GetAlpha() { return alpha_; }

  /// Set alpha.
  void SetAlpha(std::vector<float> &alpha) { alpha_ = alpha; }

  /// Get beta
  std::vector<float> GetBeta() { return beta_; }

  /// Set beta.
  void SetBeta(std::vector<float> &beta) { beta_ = beta; }

  bool GetPermute() { return permute_; }

  /// Set permute.
  void SetPermute(bool permute) { permute_ = permute; }

protected:
  bool Preprocess(FDMat *mat, FDTensor *output);
  // Argument for image preprocessing step, tuple of (width, height),
  // decide the target size after resize, default (112, 112)
  std::vector<int> size_;
  // Argument for image preprocessing step, alpha values for normalization,
  // default alpha = {1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f};
  std::vector<float> alpha_;
  // Argument for image preprocessing step, beta values for normalization,
  // default beta = {-1.f, -1.f, -1.f}
  std::vector<float> beta_;
  // Argument for image preprocessing step, whether to swap the B and R channel,
  // such as BGR->RGB, default true.
  bool permute_;
};

} // namespace faceid
} // namespace vision
} // namespace ultra_infer
