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
#include "ultra_infer/function/functions.h"
#include "ultra_infer/vision/common/processors/transform.h"

namespace ultra_infer {
namespace vision {

namespace generation {
/*! @brief Postprocessor object for AnimeGAN serials model.
 */
class ULTRAINFER_DECL AnimeGANPostprocessor {
public:
  /** \brief Create a postprocessor instance for AnimeGAN serials model
   */
  AnimeGANPostprocessor() {}

  /** \brief Process the result of runtime
   *
   * \param[in] infer_results The inference results from runtime
   * \param[in] results The output results of style transfer
   * \return true if the postprocess succeeded, otherwise false
   */
  bool Run(std::vector<FDTensor> &infer_results, std::vector<cv::Mat> *results);
};

} // namespace generation
} // namespace vision
} // namespace ultra_infer
