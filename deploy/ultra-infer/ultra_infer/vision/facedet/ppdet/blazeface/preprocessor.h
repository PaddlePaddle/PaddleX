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
#include "ultra_infer/vision/common/processors/transform.h"
#include "ultra_infer/vision/common/result.h"
#include "ultra_infer/vision/detection/ppdet/preprocessor.h"

namespace ultra_infer {

namespace vision {

namespace facedet {

class ULTRAINFER_DECL BlazeFacePreprocessor
    : public ultra_infer::vision::detection::PaddleDetPreprocessor {
public:
  /** \brief Create a preprocessor instance for BlazeFace serials model
   */
  BlazeFacePreprocessor() = default;

  /** \brief Create a preprocessor instance for Blazeface serials model
   *
   * \param[in] config_file Path of configuration file for deployment, e.g
   * ppyoloe/infer_cfg.yml
   */
  explicit BlazeFacePreprocessor(const std::string &config_file);

  /** \brief Process the input image and prepare input tensors for runtime
   *
   * \param[in] images The input image data list, all the elements are returned
   * by cv::imread() \param[in] outputs The output tensors which will feed in
   * runtime \param[in] ims_info The shape info list, record input_shape and
   * output_shape \ret
   */
  bool Run(std::vector<FDMat> *images, std::vector<FDTensor> *outputs,
           std::vector<std::map<std::string, std::array<float, 2>>> *ims_info);

private:
  bool BuildPreprocessPipelineFromConfig();

  // if is_scale_up is false, the input image only can be zoom out,
  // the maximum resize scale cannot exceed 1.0
  bool is_scale_;

  std::vector<float> normalize_mean_;

  std::vector<float> normalize_std_;

  std::vector<std::shared_ptr<Processor>> processors_;
  // read config file
  std::string config_file_;
};

} // namespace facedet

} // namespace vision

} // namespace ultra_infer
