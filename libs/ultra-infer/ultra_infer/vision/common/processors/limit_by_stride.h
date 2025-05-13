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

/*! @brief Processor for LimitByStride images with given parameters.
 */
class ULTRAINFER_DECL LimitByStride : public Processor {
public:
  explicit LimitByStride(int stride = 32, int interp = 1) {
    stride_ = stride;
    interp_ = interp;
  }

  // Resize Mat* mat to make the size divisible by stride_.
  bool ImplByOpenCV(Mat *mat);
#ifdef ENABLE_FLYCV
  bool ImplByFlyCV(Mat *mat);
#endif
  std::string Name() { return "LimitByStride"; }

  /** \brief Process the input images
   *
   * \param[in] mat The input image data
   * \param[in] stride limit image stride, default is 32
   * \param[in] interp interpolation method, default is 1
   * \param[in] lib to define OpenCV or FlyCV or CVCUDA will be used.
   * \return true if the process successed, otherwise false
   */
  static bool Run(Mat *mat, int stride = 32, int interp = 1,
                  ProcLib lib = ProcLib::DEFAULT);

private:
  int interp_;
  int stride_;
};
} // namespace vision
} // namespace ultra_infer
