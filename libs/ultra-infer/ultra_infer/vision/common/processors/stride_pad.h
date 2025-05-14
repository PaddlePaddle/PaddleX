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
#ifdef ENABLE_CVCUDA
#include <cvcuda/OpCopyMakeBorder.hpp>

#include "ultra_infer/vision/common/processors/cvcuda_utils.h"
#endif

namespace ultra_infer {
namespace vision {

/*! @brief Processor for padding images with stride.
 */
class ULTRAINFER_DECL StridePad : public Processor {
public:
  // only support pad with left-top padding mode
  StridePad(int stride, const std::vector<float> &value) {
    stride_ = stride;
    value_ = value;
  }
  bool ImplByOpenCV(Mat *mat);
#ifdef ENABLE_FLYCV
  bool ImplByFlyCV(Mat *mat);
#endif
#ifdef ENABLE_CVCUDA
  bool ImplByCvCuda(FDMat *mat);
#endif
  std::string Name() { return "StridePad"; }

  /** \brief Process the input images
   *
   * \param[in] mat The input image data, `result = mat * alpha + beta`
   * \param[in] stride stride of the padding.
   * \param[in] value value vector used by padding of the output image.
   * \param[in] lib to define OpenCV or FlyCV or CVCUDA will be used.
   * \return true if the process succeeded, otherwise false
   */
  static bool Run(Mat *mat, int stride,
                  const std::vector<float> &value = std::vector<float>(),
                  ProcLib lib = ProcLib::DEFAULT);

private:
  int stride_ = 32;
  std::vector<float> value_;
#ifdef ENABLE_CVCUDA
  cvcuda::CopyMakeBorder cvcuda_pad_op_;
#endif
};
} // namespace vision
} // namespace ultra_infer
