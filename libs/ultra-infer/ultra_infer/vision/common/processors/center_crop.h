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
#include <cvcuda/OpCustomCrop.hpp>

#include "ultra_infer/vision/common/processors/cvcuda_utils.h"
#endif

namespace ultra_infer {
namespace vision {

/*! @brief Processor for crop images in center with given type default is
 * float.
 */
class ULTRAINFER_DECL CenterCrop : public Processor {
public:
  CenterCrop(int width, int height) : height_(height), width_(width) {}
  bool ImplByOpenCV(FDMat *mat);
#ifdef ENABLE_FLYCV
  bool ImplByFlyCV(FDMat *mat);
#endif
#ifdef ENABLE_CVCUDA
  bool ImplByCvCuda(FDMat *mat);
  bool ImplByCvCuda(FDMatBatch *mat_batch);
#endif
  std::string Name() { return "CenterCrop"; }

  /** \brief Process the input images
   *
   * \param[in] mat The input image data
   * \param[in] width width of data will be croped to
   * \param[in] height height of data will be croped to
   * \param[in] lib to define OpenCV or FlyCV or CVCUDA will be used.
   * \return true if the process successed, otherwise false
   */
  static bool Run(FDMat *mat, const int &width, const int &height,
                  ProcLib lib = ProcLib::DEFAULT);

private:
  int height_;
  int width_;
#ifdef ENABLE_CVCUDA
  cvcuda::CustomCrop cvcuda_crop_op_;
#endif
};

} // namespace vision
} // namespace ultra_infer
