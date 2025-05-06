// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

namespace classification {
/*! @brief Postprocessor object for PP-ShiTuV2 Recognizer model.
 */
class ULTRAINFER_DECL PPShiTuV2RecognizerPostprocessor {
public:
  PPShiTuV2RecognizerPostprocessor() = default;

  /** \brief Process the result of runtime and fill to ClassifyResult structure
   *
   * \param[in] tensors The inference result from runtime
   * \param[in] result The output result of feature vector (see
   * ClassifyResult.feature member) \return true if the postprocess successed,
   * otherwise false
   */
  bool Run(const std::vector<FDTensor> &tensors,
           std::vector<ClassifyResult> *results);
  /// Set the value of feature_norm_ for Postprocessor
  void SetFeatureNorm(bool feature_norm) { feature_norm_ = feature_norm; }
  /// Get the value of feature_norm_ from Postprocessor, default to true.
  bool GetFeatureNorm() { return feature_norm_; }

private:
  void FeatureNorm(std::vector<float> &feature);
  bool feature_norm_ = true;
};

} // namespace classification
} // namespace vision
} // namespace ultra_infer
