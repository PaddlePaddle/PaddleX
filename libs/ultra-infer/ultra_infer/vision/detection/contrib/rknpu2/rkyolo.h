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
#include "ultra_infer/vision/detection/contrib/rknpu2/postprocessor.h"
#include "ultra_infer/vision/detection/contrib/rknpu2/preprocessor.h"

namespace ultra_infer {
namespace vision {
namespace detection {

class ULTRAINFER_DECL RKYOLO : public UltraInferModel {
public:
  RKYOLO(const std::string &model_file,
         const RuntimeOption &custom_option = RuntimeOption(),
         const ModelFormat &model_format = ModelFormat::RKNN);

  std::string ModelName() const { return "RKYOLO"; }

  /** \brief Predict the detection result for an input image
   *
   * \param[in] img The input image data, comes from cv::imread(), is a 3-D
   * array with layout HWC, BGR format \param[in] result The output detection
   * result will be written to this structure \return true if the prediction
   * successed, otherwise false
   */
  virtual bool Predict(const cv::Mat &img, DetectionResult *result);

  /** \brief Predict the detection results for a batch of input images
   *
   * \param[in] imgs, The input image list, each element comes from cv::imread()
   * \param[in] results The output detection result list
   * \return true if the prediction successed, otherwise false
   */
  virtual bool BatchPredict(const std::vector<cv::Mat> &imgs,
                            std::vector<DetectionResult> *results);

  /// Get preprocessor reference of YOLOv5
  RKYOLOPreprocessor &GetPreprocessor() { return preprocessor_; }

  /// Get postprocessor reference of YOLOv5
  RKYOLOPostprocessor &GetPostprocessor() { return postprocessor_; }

protected:
  bool Initialize();
  RKYOLOPreprocessor preprocessor_;
  RKYOLOPostprocessor postprocessor_;
};

} // namespace detection
} // namespace vision
} // namespace ultra_infer
