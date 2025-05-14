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

/*! @brief Processor for padding images.
 */
class ULTRAINFER_DECL Pad : public Processor {
public:
  Pad(int top, int bottom, int left, int right,
      const std::vector<float> &value) {
    top_ = top;
    bottom_ = bottom;
    left_ = left;
    right_ = right;
    value_ = value;
  }
  bool ImplByOpenCV(Mat *mat);
#ifdef ENABLE_FLYCV
  bool ImplByFlyCV(Mat *mat);
#endif
#ifdef ENABLE_CVCUDA
  bool ImplByCvCuda(FDMat *mat);
#endif
  std::string Name() { return "Pad"; }

  /** \brief Process the input images
   *
   * \param[in] mat The input image data, `result = mat * alpha + beta`
   * \param[in] top top pad size of the output image.
   * \param[in] bottom bottom pad size of the output image.
   * \param[in] left left pad size of the output image.
   * \param[in] right right pad size of the output image.
   * \param[in] value value vector used by padding of the output image.
   * \param[in] lib to define OpenCV or FlyCV or CVCUDA will be used.
   * \return true if the process successed, otherwise false
   */
  static bool Run(Mat *mat, const int &top, const int &bottom, const int &left,
                  const int &right, const std::vector<float> &value,
                  ProcLib lib = ProcLib::DEFAULT);

  /** \brief Process the input images
   *
   * \param[in] top set the value of the top parameter
   * \param[in] bottom set the value of the bottom parameter
   * \param[in] left set the value of the left parameter
   * \param[in] right set the value of the right parameter
   */
  bool SetPaddingSize(int top, int bottom, int left, int right) {
    top_ = top;
    bottom_ = bottom;
    left_ = left;
    right_ = right;
    return true;
  }

private:
  int top_;
  int bottom_;
  int left_;
  int right_;
  std::vector<float> value_;
#ifdef ENABLE_CVCUDA
  cvcuda::CopyMakeBorder cvcuda_pad_op_;
#endif
};
} // namespace vision
} // namespace ultra_infer
