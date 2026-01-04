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

#include "ultra_infer/pipeline/pptinypose/pipeline.h"

namespace ultra_infer {
namespace pipeline {
PPTinyPose::PPTinyPose(
    ultra_infer::vision::detection::PicoDet *det_model,
    ultra_infer::vision::keypointdetection::PPTinyPose *pptinypose_model)
    : detector_(det_model), pptinypose_model_(pptinypose_model) {}

bool PPTinyPose::Detect(cv::Mat *img,
                        ultra_infer::vision::DetectionResult *detection_res) {
  if (!detector_->Predict(img, detection_res)) {
    FDERROR << "There's a error while detectiong human box in image."
            << std::endl;
    return false;
  }
  return true;
}

bool PPTinyPose::KeypointDetect(
    cv::Mat *img, ultra_infer::vision::KeyPointDetectionResult *result,
    ultra_infer::vision::DetectionResult &detection_result) {
  if (!pptinypose_model_->Predict(img, result, detection_result)) {
    FDERROR << "There's a error while detecting keypoint in image "
            << std::endl;
    return false;
  }
  return true;
}

bool PPTinyPose::Predict(cv::Mat *img,
                         ultra_infer::vision::KeyPointDetectionResult *result) {
  result->Clear();
  ultra_infer::vision::DetectionResult detection_res;
  if (nullptr != detector_ && !Detect(img, &detection_res)) {
    FDERROR << "Failed to detect image." << std::endl;
    return false;
  }
  ultra_infer::vision::DetectionResult filter_detection_res;
  for (size_t i = 0; i < detection_res.boxes.size(); ++i) {
    if (detection_res.scores[i] > detection_model_score_threshold) {
      filter_detection_res.boxes.push_back(detection_res.boxes[i]);
      filter_detection_res.scores.push_back(detection_res.scores[i]);
      filter_detection_res.label_ids.push_back(detection_res.label_ids[i]);
    }
  }
  if (nullptr != pptinypose_model_ &&
      !KeypointDetect(img, result, filter_detection_res)) {
    FDERROR << "Failed to detect keypoint in image " << std::endl;
    return false;
  }
  return true;
};

} // namespace pipeline
} // namespace ultra_infer
