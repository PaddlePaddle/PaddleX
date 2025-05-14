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

#include "ultra_infer/ultra_infer_model.h"
#include "ultra_infer/vision/common/result.h"
#include "ultra_infer/vision/detection/ppdet/model.h"
#include "ultra_infer/vision/keypointdet/pptinypose/pptinypose.h"

namespace ultra_infer {
/** \brief All pipeline model APIs are defined inside this namespace
 *
 */
namespace pipeline {

/*! @brief PPTinyPose Pipeline object used when to load a detection model +
 * pptinypose model
 */
class ULTRAINFER_DECL PPTinyPose {
public:
  /** \brief Set initialized detection model object and pptinypose model object
   *
   * \param[in] det_model Initialized detection model object
   * \param[in] pptinypose_model Initialized pptinypose model object
   */
  PPTinyPose(
      ultra_infer::vision::detection::PicoDet *det_model,
      ultra_infer::vision::keypointdetection::PPTinyPose *pptinypose_model);

  /** \brief Predict the keypoint detection result for an input image
   *
   * \param[in] img The input image data, comes from cv::imread()
   * \param[in] result The output keypoint detection result will be written to
   * this structure \return true if the prediction succeeded, otherwise false
   */
  virtual bool Predict(cv::Mat *img,
                       ultra_infer::vision::KeyPointDetectionResult *result);

  /* \brief The score threshold for detectin model to filter bbox before
   * inputting pptinypose model
   */
  float detection_model_score_threshold = 0;

protected:
  ultra_infer::vision::detection::PicoDet *detector_ = nullptr;
  ultra_infer::vision::keypointdetection::PPTinyPose *pptinypose_model_ =
      nullptr;

  virtual bool Detect(cv::Mat *img,
                      ultra_infer::vision::DetectionResult *result);
  virtual bool
  KeypointDetect(cv::Mat *img,
                 ultra_infer::vision::KeyPointDetectionResult *result,
                 ultra_infer::vision::DetectionResult &detection_result);
};

} // namespace pipeline
} // namespace ultra_infer
