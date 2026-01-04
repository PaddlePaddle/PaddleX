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
/*! @brief Postprocessor object for InsightFaceRecognition serials model.
 */
class ULTRAINFER_DECL InsightFaceRecognitionPostprocessor {
public:
  /** \brief Create a postprocessor instance for InsightFaceRecognition serials
   * model
   */
  InsightFaceRecognitionPostprocessor();

  /** \brief Process the result of runtime and fill to FaceRecognitionResult
   * structure
   *
   * \param[in] tensors The inference result from runtime
   * \param[in] result The output result of FaceRecognitionResult
   * \return true if the postprocess succeeded, otherwise false
   */
  bool Run(std::vector<FDTensor> &infer_result,
           std::vector<FaceRecognitionResult> *results);

  void SetL2Normalize(bool &l2_normalize) { l2_normalize_ = l2_normalize; }

  bool GetL2Normalize() { return l2_normalize_; }

private:
  bool l2_normalize_;
};

} // namespace faceid
} // namespace vision
} // namespace ultra_infer
