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

/*! @brief Processor for crop images with given parameters.
 */
class ULTRAINFER_DECL Crop : public Processor {
public:
  Crop(int offset_w, int offset_h, int width, int height) {
    offset_w_ = offset_w;
    offset_h_ = offset_h;
    width_ = width;
    height_ = height;
  }

  bool ImplByOpenCV(Mat *mat);

#ifdef ENABLE_FLYCV
  bool ImplByFlyCV(Mat *mat);
#endif
  std::string Name() { return "Crop"; }

  /** \brief Process the input images
   *
   * \param[in] mat The input image data
   * \param[in] offset_w The offset of width.
   * \param[in] offset_h The offset of height.
   * \param[in] width The width of the output image.
   * \param[in] height The height of the output image.
   * \param[in] lib to define OpenCV or FlyCV or CVCUDA will be used.
   * \return true if the process succeeded, otherwise false
   */
  static bool Run(Mat *mat, int offset_w, int offset_h, int width, int height,
                  ProcLib lib = ProcLib::DEFAULT);

private:
  int offset_w_;
  int offset_h_;
  int height_;
  int width_;
};

} // namespace vision
} // namespace ultra_infer
