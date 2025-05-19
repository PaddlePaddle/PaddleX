// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.  //NOLINT
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

#include "ultra_infer/ultra_infer_model.h"
#include "ultra_infer/vision/perception/paddle3d/petr/postprocessor.h"
#include "ultra_infer/vision/perception/paddle3d/petr/preprocessor.h"

namespace ultra_infer {
namespace vision {
namespace perception {
/*! @brief petr model object used when to load a petr model exported by petr.
 */
class ULTRAINFER_DECL Petr : public UltraInferModel {
public:
  /** \brief  Set path of model file and the configuration of runtime.
   *
   * \param[in] model_file Path of model file, e.g petr/model.pdiparams
   * \param[in] params_file Path of parameter file, e.g petr/model.pdiparams, if
   * the model format is ONNX, this parameter will be ignored \param[in]
   * custom_option RuntimeOption for inference, the default will use cpu, and
   * choose the backend defined in "valid_cpu_backends" \param[in] model_format
   * Model format of the loaded model, default is Paddle format
   */
  Petr(const std::string &model_file, const std::string &params_file,
       const std::string &config_file,
       const RuntimeOption &custom_option = RuntimeOption(),
       const ModelFormat &model_format = ModelFormat::PADDLE);

  std::string ModelName() const { return "Paddle3D/petr"; }

  /** \brief Predict the perception result for an input image
   *
   * \param[in] img The input image data, comes from cv::imread(), is a 3-D
   * array with layout HWC, BGR format \param[in] result The output perception
   * result will be written to this structure \return true if the prediction
   * succeeded, otherwise false
   */
  virtual bool Predict(const cv::Mat &img, PerceptionResult *result);

  /** \brief Predict the perception results for a batch of input images
   *
   * \param[in] imgs, The input image list, each element comes from cv::imread()
   * \param[in] results The output perception result list
   * \return true if the prediction succeeded, otherwise false
   */
  virtual bool BatchPredict(const std::vector<cv::Mat> &imgs,
                            std::vector<PerceptionResult> *results);

  /// Get preprocessor reference of Petr
  virtual PetrPreprocessor &GetPreprocessor() { return preprocessor_; }

  /// Get postprocessor reference of Petr
  virtual PetrPostprocessor &GetPostprocessor() { return postprocessor_; }

protected:
  bool Initialize();
  PetrPreprocessor preprocessor_;
  PetrPostprocessor postprocessor_;
  bool initialized_ = false;
};

} // namespace perception
} // namespace vision
} // namespace ultra_infer
