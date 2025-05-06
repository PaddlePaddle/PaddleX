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

namespace detection {
/*! @brief Postprocessor object for FastestDet serials model.
 */
class ULTRAINFER_DECL FastestDetPostprocessor {
public:
  /** \brief Create a postprocessor instance for FastestDet serials model
   */
  FastestDetPostprocessor();

  /** \brief Process the result of runtime and fill to DetectionResult structure
   *
   * \param[in] tensors The inference result from runtime
   * \param[in] result The output result of detection
   * \param[in] ims_info The shape info list, record input_shape and
   * output_shape \return true if the postprocess successed, otherwise false
   */
  bool
  Run(const std::vector<FDTensor> &tensors,
      std::vector<DetectionResult> *results,
      const std::vector<std::map<std::string, std::array<float, 2>>> &ims_info);

  /// Set conf_threshold, default 0.65
  void SetConfThreshold(const float &conf_threshold) {
    conf_threshold_ = conf_threshold;
  }

  /// Get conf_threshold, default 0.65
  float GetConfThreshold() const { return conf_threshold_; }

  /// Set nms_threshold, default 0.45
  void SetNMSThreshold(const float &nms_threshold) {
    nms_threshold_ = nms_threshold;
  }

  /// Get nms_threshold, default 0.45
  float GetNMSThreshold() const { return nms_threshold_; }

protected:
  float conf_threshold_;
  float nms_threshold_;
  float Sigmoid(float x);
  float Tanh(float x);
};

} // namespace detection
} // namespace vision
} // namespace ultra_infer
