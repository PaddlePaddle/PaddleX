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

namespace perception {
/*! @brief Postprocessor object for Centerpoint serials model.
 */
class ULTRAINFER_DECL CenterpointPostprocessor {
public:
  /** \brief Create a postprocessor instance for Centerpoint serials model
   */
  CenterpointPostprocessor();

  /** \brief Process the result of runtime and fill to PerceptionResult
   * structure
   *
   * \param[in] tensors The inference result from runtime
   * \param[in] result The output result of detection
   * \param[in] ims_info The shape info list, record input_shape and
   * output_shape \return true if the postprocess succeeded, otherwise false
   */
  bool Run(const std::vector<FDTensor> &tensors, PerceptionResult *results);

protected:
  float conf_threshold_;
};

} // namespace perception
} // namespace vision
} // namespace ultra_infer
