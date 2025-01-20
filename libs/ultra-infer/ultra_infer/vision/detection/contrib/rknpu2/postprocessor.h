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
#include "ultra_infer/vision/detection/contrib/rknpu2/utils.h"
#include <array>
namespace ultra_infer {
namespace vision {
namespace detection {
/*! @brief Postprocessor object for YOLOv5 serials model.
 */
class ULTRAINFER_DECL RKYOLOPostprocessor {
public:
  /** \brief Create a postprocessor instance for YOLOv5 serials model
   */
  RKYOLOPostprocessor();

  /** \brief Process the result of runtime and fill to DetectionResult structure
   *
   * \param[in] tensors The inference result from runtime
   * \param[in] result The output result of detection
   * \param[in] ims_info The shape info list, record input_shape and
   * output_shape \return true if the postprocess successed, otherwise false
   */
  bool Run(const std::vector<FDTensor> &tensors,
           std::vector<DetectionResult> *results);

  /// Set nms_threshold, default 0.45
  void SetNMSThreshold(float nms_threshold) { nms_threshold_ = nms_threshold; }

  /// Set conf_threshold, default 0.25
  void SetConfThreshold(float conf_threshold) {
    conf_threshold_ = conf_threshold;
  }

  /// Get conf_threshold, default 0.25
  const float GetConfThreshold() { return conf_threshold_; }

  /// Get nms_threshold, default 0.45
  const float GetNMSThreshold() { return nms_threshold_; }

  /// Set height and weight
  void SetHeightAndWeight(int height, int width) {
    height_ = height;
    width_ = width;
  }

  /// Set pad_hw_values
  void SetPadHWValues(const std::vector<std::vector<int>> &pad_hw_values) {
    pad_hw_values_ = pad_hw_values;
  }

  /// Set scale
  void SetScale(const std::vector<float> &scale) { scale_ = scale; }

  /// Get Anchor
  const std::vector<int> &GetAnchor() { return anchors_; }

  /// Set Anchor
  void SetAnchor(const std::vector<int> &anchors) { anchors_ = anchors; }

  void SetAnchorPerBranch(int anchor_per_branch) {
    anchor_per_branch_ = anchor_per_branch;
  }

  /// Set the number of class
  void SetClassNum(int num) {
    obj_class_num_ = num;
    prob_box_size_ = obj_class_num_ + 5;
  }
  /// Get the number of class
  int GetClassNum() { return obj_class_num_; }

private:
  std::vector<int> anchors_ = {10, 13, 16,  30,  33, 23,  30,  61,  62,
                               45, 59, 119, 116, 90, 156, 198, 373, 326};
  int strides_[3] = {8, 16, 32};
  int height_ = 0;
  int width_ = 0;
  int anchor_per_branch_ = 0;

  int ProcessFP16(float *input, int *anchor, int grid_h, int grid_w, int stride,
                  std::vector<float> &boxes, std::vector<float> &boxScores,
                  std::vector<int> &classId, float threshold);
  // Model
  int QuickSortIndiceInverse(std::vector<float> &input, int left, int right,
                             std::vector<int> &indices);

  // post_process values
  std::vector<std::vector<int>> pad_hw_values_;
  std::vector<float> scale_;
  float nms_threshold_ = 0.45;
  float conf_threshold_ = 0.25;
  int prob_box_size_ = 85;
  int obj_class_num_ = 80;
  int obj_num_bbox_max_size = 200;
};

} // namespace detection
} // namespace vision
} // namespace ultra_infer
