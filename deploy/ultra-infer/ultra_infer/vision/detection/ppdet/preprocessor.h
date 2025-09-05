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

namespace detection {
/*! @brief Preprocessor object for PaddleDet serials model.
 */
class ULTRAINFER_DECL PaddleDetPreprocessor : public ProcessorManager {
public:
  PaddleDetPreprocessor() = default;
  /** \brief Create a preprocessor instance for PaddleDet serials model
   *
   * \param[in] config_file Path of configuration file for deployment, e.g
   * ppyoloe/infer_cfg.yml
   */
  explicit PaddleDetPreprocessor(const std::string &config_file);

  /** \brief Implement the virtual function of ProcessorManager, Apply() is the
   *  body of Run(). Apply() contains the main logic of preprocessing, Run() is
   *  called by users to execute preprocessing
   *
   * \param[in] image_batch The input image batch
   * \param[in] outputs The output tensors which will feed in runtime
   * \return true if the preprocess succeeded, otherwise false
   */
  virtual bool Apply(FDMatBatch *image_batch, std::vector<FDTensor> *outputs);

  /// This function will disable normalize in preprocessing step.
  void DisableNormalize();
  /// This function will disable hwc2chw in preprocessing step.
  void DisablePermute();

  std::string GetArch() { return arch_; }

private:
  bool BuildPreprocessPipelineFromConfig();
  std::vector<std::shared_ptr<Processor>> processors_;
  std::shared_ptr<PadToSize> pad_op_ =
      std::make_shared<PadToSize>(0, 0, std::vector<float>(3, 0));
  bool initialized_ = false;
  // for recording the switch of hwc2chw
  bool disable_permute_ = false;
  // for recording the switch of normalize
  bool disable_normalize_ = false;
  // read config file
  std::string config_file_;
  // read arch_ for postprocess
  std::string arch_;
};

} // namespace detection
} // namespace vision
} // namespace ultra_infer
