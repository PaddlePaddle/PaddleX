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
namespace segmentation {
/*! @brief Preprocessor object for PaddleSeg serials model.
 */
class ULTRAINFER_DECL PaddleSegPreprocessor : public ProcessorManager {
public:
  /** \brief Create a preprocessor instance for PaddleSeg serials model
   *
   * \param[in] config_file Path of configuration file for deployment, e.g
   * ppliteseg/deploy.yaml
   */
  explicit PaddleSegPreprocessor(const std::string &config_file);

  /** \brief Implement the virtual function of ProcessorManager, Apply() is the
   *  body of Run(). Apply() contains the main logic of preprocessing, Run() is
   *  called by users to execute preprocessing
   *
   * \param[in] image_batch The input image batch
   * \param[in] outputs The output tensors which will feed in runtime
   * \return true if the preprocess succeeded, otherwise false
   */
  virtual bool Apply(FDMatBatch *image_batch, std::vector<FDTensor> *outputs);

  /// Get is_vertical_screen property of PP-HumanSeg model, default is false
  bool GetIsVerticalScreen() const { return is_vertical_screen_; }

  /// Set is_vertical_screen value, bool type required
  void SetIsVerticalScreen(bool value) { is_vertical_screen_ = value; }

  /// This function will disable normalize in preprocessing step.
  void DisableNormalize();
  /// This function will disable hwc2chw in preprocessing step.
  void DisablePermute();
  /// This function will set imgs_info_ in PaddleSegPreprocessor
  void SetImgsInfo(
      std::map<std::string, std::vector<std::array<int, 2>>> *imgs_info) {
    imgs_info_ = imgs_info;
  }
  /// This function will get imgs_info_ in PaddleSegPreprocessor
  std::map<std::string, std::vector<std::array<int, 2>>> *GetImgsInfo() {
    return imgs_info_;
  }

private:
  virtual bool BuildPreprocessPipelineFromConfig();
  std::vector<std::shared_ptr<Processor>> processors_;
  std::string config_file_;

  /** \brief For PP-HumanSeg model, set true if the input image is vertical
   * image(height > width), default value is false
   */
  bool is_vertical_screen_ = false;

  // for recording the switch of hwc2chw
  bool disable_permute_ = false;
  // for recording the switch of normalize
  bool disable_normalize_ = false;

  bool is_contain_resize_op_ = false;

  bool initialized_ = false;

  std::map<std::string, std::vector<std::array<int, 2>>> *imgs_info_;
  std::shared_ptr<Resize> pre_resize_op_ = std::make_shared<Resize>(0, 0);
};

} // namespace segmentation
} // namespace vision
} // namespace ultra_infer
