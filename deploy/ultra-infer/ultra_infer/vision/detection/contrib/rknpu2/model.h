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
#include "ultra_infer/vision/detection/contrib/rknpu2/rkyolo.h"
namespace ultra_infer {
namespace vision {
namespace detection {

class ULTRAINFER_DECL RKYOLOV5 : public RKYOLO {
public:
  /** \brief Set path of model file and configuration file, and the
   * configuration of runtime
   *
   * \param[in] model_file Path of model file, e.g picodet/model.pdmodel
   * \param[in] custom_option RuntimeOption for inference, the default will use
   * cpu, and choose the backend defined in `valid_cpu_backends` \param[in]
   * model_format Model format of the loaded model, default is Paddle format
   */
  RKYOLOV5(const std::string &model_file,
           const RuntimeOption &custom_option = RuntimeOption(),
           const ModelFormat &model_format = ModelFormat::RKNN)
      : RKYOLO(model_file, custom_option, model_format) {
    valid_cpu_backends = {};
    valid_gpu_backends = {};
    valid_rknpu_backends = {Backend::RKNPU2};
    std::vector<int> anchors = {10, 13, 16,  30,  33, 23,  30,  61,  62,
                                45, 59, 119, 116, 90, 156, 198, 373, 326};
    int anchor_per_branch_ = 3;
    GetPostprocessor().SetAnchor(anchors);
    GetPostprocessor().SetAnchorPerBranch(anchor_per_branch_);
  }

  virtual std::string ModelName() const { return "RKYOLOV5"; }
};

class ULTRAINFER_DECL RKYOLOV7 : public RKYOLO {
public:
  /** \brief Set path of model file and configuration file, and the
   * configuration of runtime
   *
   * \param[in] model_file Path of model file, e.g picodet/model.pdmodel
   * \param[in] custom_option RuntimeOption for inference, the default will use
   * cpu, and choose the backend defined in `valid_cpu_backends` \param[in]
   * model_format Model format of the loaded model, default is Paddle format
   */
  RKYOLOV7(const std::string &model_file,
           const RuntimeOption &custom_option = RuntimeOption(),
           const ModelFormat &model_format = ModelFormat::RKNN)
      : RKYOLO(model_file, custom_option, model_format) {
    valid_cpu_backends = {};
    valid_gpu_backends = {};
    valid_rknpu_backends = {Backend::RKNPU2};
    std::vector<int> anchors = {12, 16, 19,  36,  40,  28,  36,  75,  76,
                                55, 72, 146, 142, 110, 192, 243, 459, 401};
    int anchor_per_branch_ = 3;
    GetPostprocessor().SetAnchor(anchors);
    GetPostprocessor().SetAnchorPerBranch(anchor_per_branch_);
  }

  virtual std::string ModelName() const { return "RKYOLOV7"; }
};

class ULTRAINFER_DECL RKYOLOX : public RKYOLO {
public:
  /** \brief Set path of model file and configuration file, and the
   * configuration of runtime
   *
   * \param[in] model_file Path of model file, e.g picodet/model.pdmodel
   * \param[in] custom_option RuntimeOption for inference, the default will use
   * cpu, and choose the backend defined in `valid_cpu_backends` \param[in]
   * model_format Model format of the loaded model, default is Paddle format
   */
  RKYOLOX(const std::string &model_file,
          const RuntimeOption &custom_option = RuntimeOption(),
          const ModelFormat &model_format = ModelFormat::RKNN)
      : RKYOLO(model_file, custom_option, model_format) {
    valid_cpu_backends = {};
    valid_gpu_backends = {};
    valid_rknpu_backends = {Backend::RKNPU2};
    std::vector<int> anchors = {1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1};
    int anchor_per_branch_ = 1;
    GetPostprocessor().SetAnchor(anchors);
    GetPostprocessor().SetAnchorPerBranch(anchor_per_branch_);
  }

  virtual std::string ModelName() const { return "RKYOLOV7"; }
};

} // namespace detection
} // namespace vision
} // namespace ultra_infer
