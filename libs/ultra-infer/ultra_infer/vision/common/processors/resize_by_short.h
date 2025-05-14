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

/*! @brief Processor for resize images by short edge.
 */
class ULTRAINFER_DECL ResizeByShort : public Processor {
public:
  ResizeByShort(int target_size, int interp = 1, bool use_scale = true,
                const std::vector<int> &max_hw = std::vector<int>()) {
    target_size_ = target_size;
    max_hw_ = max_hw;
    interp_ = interp;
    use_scale_ = use_scale;
  }
  bool ImplByOpenCV(FDMat *mat);
#ifdef ENABLE_FLYCV
  bool ImplByFlyCV(FDMat *mat);
#endif
#ifdef ENABLE_CVCUDA
  bool ImplByCvCuda(FDMat *mat);
  bool ImplByCvCuda(FDMatBatch *mat_batch);
#endif
  std::string Name() { return "ResizeByShort"; }

  /** \brief Process the input images
   *
   * \param[in] mat The input image data, `result = mat * alpha + beta`
   * \param[in] target_size target size of the output image.
   * \param[in] interp interpolation method, default is 1.
   * \param[in] use_scale to define whether to scale the image, default is
   * true. \param[in] max_hw max HW of output image. \param[in] lib to define
   * OpenCV or FlyCV or CVCUDA will be used. \return true if the process
   * succeeded, otherwise false
   */
  static bool Run(FDMat *mat, int target_size, int interp = 1,
                  bool use_scale = true,
                  const std::vector<int> &max_hw = std::vector<int>(),
                  ProcLib lib = ProcLib::DEFAULT);

private:
  double GenerateScale(const int origin_w, const int origin_h);
  int target_size_;
  std::vector<int> max_hw_;
  int interp_;
  bool use_scale_;
#ifdef ENABLE_CVCUDA
  cvcuda::Resize cvcuda_resize_op_;
#endif
};
} // namespace vision
} // namespace ultra_infer
