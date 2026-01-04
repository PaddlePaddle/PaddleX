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
#include "ultra_infer/vision/common/processors/manager.h"
#include "ultra_infer/vision/common/processors/transform.h"
#include "ultra_infer/vision/common/result.h"

namespace ultra_infer {
namespace vision {

namespace perception {
/*! @brief Preprocessor object for Caddn serials model.
 */
class ULTRAINFER_DECL CaddnPreprocessor : public ProcessorManager {
public:
  CaddnPreprocessor() = default;
  /** \brief Create a preprocessor instance for Caddn model
   *
   * \param[in] config_file Path of configuration file for deployment, e.g
   * Caddn/infer_cfg.yml
   */
  explicit CaddnPreprocessor(const std::string &config_file);

  bool Run(std::vector<FDMat> *images, std::vector<float> &input_cam_data,
           std::vector<float> &input_lidar_data,
           std::vector<FDTensor> *outputs);

  /** \brief Process the input image and prepare input tensors for runtime
   *
   * \param[in] images The input image data list, all the elements are returned
   * by cv::imread() \param[in] outputs The output tensors which will feed in
   * runtime \param[in] ims_info The shape info list, record input_shape and
   * output_shape \return true if the preprocess succeeded, otherwise false
   */
  bool Apply(FDMatBatch *image_batch, std::vector<FDTensor> *outputs) {
    FDERROR << "CaddnPreprocessor should input cam and lidar datas"
            << std::endl;
    return 0;
  };
  bool Apply(FDMatBatch *image_batch, std::vector<float> &input_cam_data,
             std::vector<float> &input_lidar_data,
             std::vector<FDTensor> *outputs);

protected:
  bool BuildPreprocessPipeline();
  std::vector<std::shared_ptr<Processor>> processors_;

  bool disable_permute_ = false;

  bool initialized_ = false;

  std::string config_file_;
};

} // namespace perception
} // namespace vision
} // namespace ultra_infer
