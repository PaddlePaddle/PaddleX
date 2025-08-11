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
#include "ultra_infer/vision/visualize/visualize.h"

namespace ultra_infer {
namespace vision {

cv::Mat VisKeypointDetection(const cv::Mat &im,
                             const KeyPointDetectionResult &results,
                             float conf_threshold) {
  const int edge[][2] = {{0, 1},   {0, 2},  {1, 3},   {2, 4},   {3, 5},
                         {4, 6},   {5, 7},  {6, 8},   {7, 9},   {8, 10},
                         {5, 11},  {6, 12}, {11, 13}, {12, 14}, {13, 15},
                         {14, 16}, {11, 12}};
  auto colormap = GenerateColorMap();
  cv::Mat vis_img = im.clone();
  int detection_nums = results.keypoints.size() / 17;
  for (int i = 0; i < detection_nums; i++) {
    int index = i * 17;
    bool is_over_threshold = true;
    for (int j = 0; j < results.num_joints; j++) {
      if (results.scores[index + j] < conf_threshold) {
        is_over_threshold = false;
        break;
      }
    }
    if (is_over_threshold) {
      for (int k = 0; k < results.num_joints; k++) {
        int x_coord = int(results.keypoints[index + k][0]);
        int y_coord = int(results.keypoints[index + k][1]);
        cv::circle(vis_img, cv::Point2d(x_coord, y_coord), 1,
                   cv::Scalar(0, 0, 255), 2);
        int x_start = int(results.keypoints[index + edge[k][0]][0]);
        int y_start = int(results.keypoints[index + edge[k][0]][1]);
        int x_end = int(results.keypoints[index + edge[k][1]][0]);
        int y_end = int(results.keypoints[index + edge[k][1]][1]);
        cv::line(vis_img, cv::Point2d(x_start, y_start),
                 cv::Point2d(x_end, y_end), colormap[k], 1);
      }
    }
  }
  return vis_img;
}

} // namespace vision
} // namespace ultra_infer
