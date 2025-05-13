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

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "ultra_infer/utils/utils.h"
#include "ultra_infer/vision/common/processors/mat.h"
#include "ultra_infer/vision/common/processors/mat_batch.h"
#include <unordered_map>

namespace ultra_infer {
namespace vision {

/*! @brief Enable using FlyCV to process image while deploy vision models.
 * Currently, FlyCV in only available on ARM(Linux aarch64), so will
 * fallback to using OpenCV in other platform
 */
ULTRAINFER_DECL void EnableFlyCV();

/// Disable using FlyCV to process image while deploy vision models.
ULTRAINFER_DECL void DisableFlyCV();

/*! @brief Set the cpu num threads of ProcLib.
 */
ULTRAINFER_DECL void SetProcLibCpuNumThreads(int threads);

/*! @brief Processor base class for processors in
 * ultra_infer/vision/common/processors
 */
class ULTRAINFER_DECL Processor {
public:
  // default_lib has the highest priority
  // all the function in `processor` will force to use
  // default_lib if this flag is set.
  // DEFAULT means this flag is not set
  // static ProcLib default_lib;

  virtual std::string Name() = 0;

  virtual bool ImplByOpenCV(FDMat *mat);
  virtual bool ImplByOpenCV(FDMatBatch *mat_batch);

  virtual bool ImplByFlyCV(FDMat *mat);
  virtual bool ImplByFlyCV(FDMatBatch *mat_batch);

  virtual bool ImplByCuda(FDMat *mat);
  virtual bool ImplByCuda(FDMatBatch *mat_batch);

  virtual bool ImplByCvCuda(FDMat *mat);
  virtual bool ImplByCvCuda(FDMatBatch *mat_batch);

  /*! @brief operator `()` for calling processor in this way: `processor(mat)`
   *
   * \param[in] mat: The input mat
   * \return true if the process successed, otherwise false
   */
  virtual bool operator()(FDMat *mat);

  /*! @brief operator `()` for calling processor in this way: `processor(mat,
   * lib)` This function is for backward compatibility, will be removed in the
   * near future, please use operator()(FDMat* mat) instead and set proc_lib in
   * mat.
   *
   * \param[in] mat: The input mat
   * \param[in] lib: The processing library, opencv, cv-cuda, flycv, etc.
   * \return true if the process successed, otherwise false
   */
  virtual bool operator()(FDMat *mat, ProcLib lib);

  /*! @brief operator `()` for calling processor in this way:
   * `processor(mat_batch)`
   *
   * \param[in] mat_batch: The input mat batch
   * \return true if the process successed, otherwise false
   */
  virtual bool operator()(FDMatBatch *mat_batch);
};

} // namespace vision
} // namespace ultra_infer
