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

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "ultra_infer/vision/visualize/visualize.h"

namespace ultra_infer {
namespace vision {

cv::Mat RemoveSmallConnectedArea(const cv::Mat &alpha_pred, float threshold) {
  cv::Mat gray, binary;
  alpha_pred.convertTo(gray, CV_8UC1, 255.f);
  cv::Mat alpha_pred_clone = alpha_pred.clone();
  // 255 * 0.05 ~ 13
  unsigned int binary_threshold = static_cast<unsigned int>(255.f * threshold);
  cv::threshold(gray, binary, binary_threshold, 255, cv::THRESH_BINARY);
  // morphologyEx with OPEN operation to remove noise first.
  auto kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3),
                                          cv::Point(-1, -1));
  cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
  // Computationally connected domain
  cv::Mat labels = cv::Mat::zeros(alpha_pred_clone.size(), CV_32S);
  cv::Mat stats, centroids;
  int num_labels =
      cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8, 4);
  if (num_labels <= 1) {
    // no noise, skip.
    return alpha_pred;
  }
  // find max connected area, 0 is background
  int max_connected_id = 1; // 1,2,...
  int max_connected_area = stats.at<int>(max_connected_id, cv::CC_STAT_AREA);
  for (int i = 1; i < num_labels; ++i) {
    int tmp_connected_area = stats.at<int>(i, cv::CC_STAT_AREA);
    if (tmp_connected_area > max_connected_area) {
      max_connected_area = tmp_connected_area;
      max_connected_id = i;
    }
  }
  const int h = alpha_pred_clone.rows;
  const int w = alpha_pred_clone.cols;
  // remove small connected area.
  for (int i = 0; i < h; ++i) {
    int *label_row_ptr = labels.ptr<int>(i);
    float *alpha_row_ptr = alpha_pred_clone.ptr<float>(i);
    for (int j = 0; j < w; ++j) {
      if (label_row_ptr[j] != max_connected_id)
        alpha_row_ptr[j] = 0.f;
    }
  }
  return alpha_pred_clone;
}

cv::Mat Visualize::RemoveSmallConnectedArea(const cv::Mat &alpha_pred,
                                            float threshold) {
  cv::Mat gray, binary;
  alpha_pred.convertTo(gray, CV_8UC1, 255.f);
  cv::Mat alpha_pred_clone = alpha_pred.clone();
  // 255 * 0.05 ~ 13
  unsigned int binary_threshold = static_cast<unsigned int>(255.f * threshold);
  cv::threshold(gray, binary, binary_threshold, 255, cv::THRESH_BINARY);
  // morphologyEx with OPEN operation to remove noise first.
  auto kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3),
                                          cv::Point(-1, -1));
  cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
  // Computationally connected domain
  cv::Mat labels = cv::Mat::zeros(alpha_pred_clone.size(), CV_32S);
  cv::Mat stats, centroids;
  int num_labels =
      cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8, 4);
  if (num_labels <= 1) {
    // no noise, skip.
    return alpha_pred;
  }
  // find max connected area, 0 is background
  int max_connected_id = 1; // 1,2,...
  int max_connected_area = stats.at<int>(max_connected_id, cv::CC_STAT_AREA);
  for (int i = 1; i < num_labels; ++i) {
    int tmp_connected_area = stats.at<int>(i, cv::CC_STAT_AREA);
    if (tmp_connected_area > max_connected_area) {
      max_connected_area = tmp_connected_area;
      max_connected_id = i;
    }
  }
  const int h = alpha_pred_clone.rows;
  const int w = alpha_pred_clone.cols;
  // remove small connected area.
  for (int i = 0; i < h; ++i) {
    int *label_row_ptr = labels.ptr<int>(i);
    float *alpha_row_ptr = alpha_pred_clone.ptr<float>(i);
    for (int j = 0; j < w; ++j) {
      if (label_row_ptr[j] != max_connected_id)
        alpha_row_ptr[j] = 0.f;
    }
  }
  return alpha_pred_clone;
}

} // namespace vision
} // namespace ultra_infer
