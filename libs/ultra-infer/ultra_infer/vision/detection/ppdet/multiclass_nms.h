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
#include <map>
#include <string>
#include <vector>

namespace ultra_infer {
namespace vision {
namespace detection {
/** \brief Config for PaddleMultiClassNMS
 * \param[in] background_label the value of background label
 * \param[in] keep_top_k the value of keep_top_k
 * \param[in] nms_eta the value of nms_eta
 * \param[in] nms_threshold a dict that contains the arguments of nms operations
 * \param[in] nms_top_k if there are more than max_num bboxes after NMS, only
 * top max_num will be kept. \param[in] normalized Determine whether normalized
 * is required \param[in] score_threshold bbox threshold, bboxes with scores
 * lower than it will not be considered.
 */
struct NMSOption {
  NMSOption() = default;
  int64_t background_label = -1;
  int64_t keep_top_k = 100;
  float nms_eta = 1.0;
  float nms_threshold = 0.5;
  int64_t nms_top_k = 1000;
  bool normalized = true;
  float score_threshold = 0.3;
};

struct PaddleMultiClassNMS {
  int64_t background_label = -1;
  int64_t keep_top_k = -1;
  float nms_eta;
  float nms_threshold = 0.7;
  int64_t nms_top_k;
  bool normalized;
  float score_threshold;

  std::vector<int32_t> out_num_rois_data;
  std::vector<int32_t> out_index_data;
  std::vector<float> out_box_data;
  void FastNMS(const float *boxes, const float *scores, const int &num_boxes,
               std::vector<int> *keep_indices);
  int NMSForEachSample(const float *boxes, const float *scores, int num_boxes,
                       int num_classes,
                       std::map<int, std::vector<int>> *keep_indices);
  void Compute(const float *boxes, const float *scores,
               const std::vector<int64_t> &boxes_dim,
               const std::vector<int64_t> &scores_dim);

  void SetNMSOption(const struct NMSOption &nms_option) {
    background_label = nms_option.background_label;
    keep_top_k = nms_option.keep_top_k;
    nms_eta = nms_option.nms_eta;
    nms_threshold = nms_option.nms_threshold;
    nms_top_k = nms_option.nms_top_k;
    normalized = nms_option.normalized;
    score_threshold = nms_option.score_threshold;
  }
};
} // namespace detection
} // namespace vision
} // namespace ultra_infer
