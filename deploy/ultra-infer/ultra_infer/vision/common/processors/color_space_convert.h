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

/*! @brief Processor for transform images from BGR to RGB.
 */
class ULTRAINFER_DECL BGR2RGB : public Processor {
public:
  bool ImplByOpenCV(FDMat *mat);
#ifdef ENABLE_FLYCV
  bool ImplByFlyCV(FDMat *mat);
#endif
  virtual std::string Name() { return "BGR2RGB"; }

  /** \brief Process the input images
   *
   * \param[in] mat The input image data
   * \param[in] lib to define OpenCV or FlyCV or CVCUDA will be used.
   * \return true if the process succeeded, otherwise false
   */
  static bool Run(FDMat *mat, ProcLib lib = ProcLib::DEFAULT);
};

/*! @brief Processor for transform images from RGB to BGR.
 */
class ULTRAINFER_DECL RGB2BGR : public Processor {
public:
  bool ImplByOpenCV(FDMat *mat);
#ifdef ENABLE_FLYCV
  bool ImplByFlyCV(FDMat *mat);
#endif
  std::string Name() { return "RGB2BGR"; }

  /** \brief Process the input images
   *
   * \param[in] mat The input image data
   * \param[in] lib to define OpenCV or FlyCV or CVCUDA will be used.
   * \return true if the process succeeded, otherwise false
   */
  static bool Run(FDMat *mat, ProcLib lib = ProcLib::DEFAULT);
};

/*! @brief Processor for transform images from BGR to GRAY.
 */
class ULTRAINFER_DECL BGR2GRAY : public Processor {
public:
  bool ImplByOpenCV(FDMat *mat);
#ifdef ENABLE_FLYCV
  bool ImplByFlyCV(FDMat *mat);
#endif
  virtual std::string Name() { return "BGR2GRAY"; }

  /** \brief Process the input images
   *
   * \param[in] mat The input image data
   * \param[in] lib to define OpenCV or FlyCV or CVCUDA will be used.
   * \return true if the process succeeded, otherwise false
   */
  static bool Run(FDMat *mat, ProcLib lib = ProcLib::DEFAULT);
};

/*! @brief Processor for transform images from RGB to GRAY.
 */
class ULTRAINFER_DECL RGB2GRAY : public Processor {
public:
  bool ImplByOpenCV(FDMat *mat);
#ifdef ENABLE_FLYCV
  bool ImplByFlyCV(FDMat *mat);
#endif
  std::string Name() { return "RGB2GRAY"; }

  /** \brief Process the input images
   *
   * \param[in] mat The input image data
   * \param[in] lib to define OpenCV or FlyCV or CVCUDA will be used.
   * \return true if the process succeeded, otherwise false
   */
  static bool Run(FDMat *mat, ProcLib lib = ProcLib::DEFAULT);
};

} // namespace vision
} // namespace ultra_infer
