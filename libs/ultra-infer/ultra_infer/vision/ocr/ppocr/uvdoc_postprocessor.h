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

namespace ocr {
/*! @brief Postprocessor object for UVDoc serials model.
 */
class ULTRAINFER_DECL UVDocPostprocessor {
public:
  UVDocPostprocessor() {}
  /** \brief Process the result of runtime and fill to UVDocResult
   *
   * \param[in] tensors The inference result from runtime
   * \param[in] results The output text results of UVDoc
   * \return true if the postprocess successed, otherwise false
   */
  bool Run(const std::vector<FDTensor> &tensors,
           std::vector<FDTensor> *results);
};

} // namespace ocr
} // namespace vision
} // namespace ultra_infer
