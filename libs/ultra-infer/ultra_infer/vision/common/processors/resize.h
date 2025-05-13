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
#include <cvcuda/OpResize.hpp>

#include "ultra_infer/vision/common/processors/cvcuda_utils.h"
#endif

namespace ultra_infer {
namespace vision {

/*! @brief Processor for Resize images.
 */
class ULTRAINFER_DECL Resize : public Processor {
public:
  Resize(int width, int height, float scale_w = -1.0, float scale_h = -1.0,
         int interp = 1, bool use_scale = false) {
    width_ = width;
    height_ = height;
    scale_w_ = scale_w;
    scale_h_ = scale_h;
    interp_ = interp;
    use_scale_ = use_scale;
  }

  bool ImplByOpenCV(FDMat *mat);
#ifdef ENABLE_FLYCV
  bool ImplByFlyCV(FDMat *mat);
#endif
#ifdef ENABLE_CVCUDA
  bool ImplByCvCuda(FDMat *mat);
#endif
  std::string Name() { return "Resize"; }

  /** \brief Process the input images
   *
   * \param[in] mat The input image data, `result = mat * alpha + beta`
   * \param[in] width width of the output image.
   * \param[in] height height of the output image.
   * \param[in] scale_w scale of width, default is -1.0.
   * \param[in] scale_h scale of height, default is -1.0.
   * \param[in] interp interpolation method, default is 1.
   * \param[in] use_scale to define whether to scale the image, default is
   * true. \param[in] lib to define OpenCV or FlyCV or CVCUDA will be used.
   * \return true if the process successed, otherwise false
   */
  static bool Run(FDMat *mat, int width, int height, float scale_w = -1.0,
                  float scale_h = -1.0, int interp = 1, bool use_scale = false,
                  ProcLib lib = ProcLib::DEFAULT);

  /** \brief Process the input images
   *
   * \param[in] width set the value of the width parameter
   * \param[in] height set the value of the height parameter
   */
  bool SetWidthAndHeight(int width, int height) {
    width_ = width;
    height_ = height;
    return true;
  }

  std::tuple<int, int> GetWidthAndHeight() {
    return std::make_tuple(width_, height_);
  }

private:
  int width_;
  int height_;
  float scale_w_ = -1.0;
  float scale_h_ = -1.0;
  int interp_ = 1;
  bool use_scale_ = false;
#ifdef ENABLE_CVCUDA
  cvcuda::Resize cvcuda_resize_op_;
#endif
};
} // namespace vision
} // namespace ultra_infer
