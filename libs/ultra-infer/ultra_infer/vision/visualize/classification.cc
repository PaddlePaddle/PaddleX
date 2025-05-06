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

#include <algorithm>

#include "opencv2/imgproc/imgproc.hpp"
#include "ultra_infer/vision/visualize/visualize.h"

namespace ultra_infer {
namespace vision {

cv::Mat VisClassification(const cv::Mat &im, const ClassifyResult &result,
                          int top_k, float score_threshold, float font_size) {
  int h = im.rows;
  int w = im.cols;
  auto vis_im = im.clone();
  int h_sep = h / 30;
  int w_sep = w / 10;
  if (top_k > result.scores.size()) {
    top_k = result.scores.size();
  }
  for (int i = 0; i < top_k; ++i) {
    if (result.scores[i] < score_threshold) {
      continue;
    }
    std::string id = std::to_string(result.label_ids[i]);
    std::string score = std::to_string(result.scores[i]);
    if (score.size() > 4) {
      score = score.substr(0, 4);
    }
    std::string text = id + "," + score;
    int font = cv::FONT_HERSHEY_SIMPLEX;
    cv::Point origin;
    origin.x = w_sep;
    origin.y = h_sep * (i + 1);
    cv::putText(vis_im, text, origin, font, font_size,
                cv::Scalar(255, 255, 255), 1);
  }
  return vis_im;
}

// Visualize ClassifyResult with custom labels.
cv::Mat VisClassification(const cv::Mat &im, const ClassifyResult &result,
                          const std::vector<std::string> &labels, int top_k,
                          float score_threshold, float font_size) {
  int h = im.rows;
  int w = im.cols;
  auto vis_im = im.clone();
  int h_sep = h / 30;
  int w_sep = w / 10;
  if (top_k > result.scores.size()) {
    top_k = result.scores.size();
  }
  for (int i = 0; i < top_k; ++i) {
    if (result.scores[i] < score_threshold) {
      continue;
    }
    std::string id = std::to_string(result.label_ids[i]);
    std::string score = std::to_string(result.scores[i]);
    if (score.size() > 4) {
      score = score.substr(0, 4);
    }
    std::string text = id + "," + score;
    if (labels.size() > result.label_ids[i]) {
      text = labels[result.label_ids[i]] + "," + text;
    } else {
      FDWARNING << "The label_id: " << result.label_ids[i]
                << " in DetectionResult should be less than length of labels:"
                << labels.size() << "." << std::endl;
    }
    if (text.size() > 16) {
      text = text.substr(0, 16);
    }
    int font = cv::FONT_HERSHEY_SIMPLEX;
    cv::Point origin;
    origin.x = w_sep;
    origin.y = h_sep * (i + 1);
    cv::putText(vis_im, text, origin, font, font_size,
                cv::Scalar(255, 255, 255), 1);
  }
  return vis_im;
}

} // namespace vision
} // namespace ultra_infer
