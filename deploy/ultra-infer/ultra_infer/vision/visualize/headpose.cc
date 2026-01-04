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

#include "opencv2/imgproc/imgproc.hpp"
#include "ultra_infer/vision/visualize/visualize.h"

namespace ultra_infer {

namespace vision {

cv::Mat VisHeadPose(const cv::Mat &im, const HeadPoseResult &result, int size,
                    int line_size) {
  const float PI = 3.1415926535;
  auto vis_im = im.clone();
  int h = im.rows;
  int w = im.cols;
  // vis headpose
  float pitch = result.euler_angles[0] * PI / 180.f;
  float yaw = -result.euler_angles[1] * PI / 180.f;
  float roll = result.euler_angles[2] * PI / 180.f;

  int tdx = w / 2;
  int tdy = h / 2;

  // X-Axis | drawn in red
  int x1 = static_cast<int>(size * std::cos(yaw) * std::cos(roll)) + tdx;
  int y1 = static_cast<int>(
               size * (std::cos(pitch) * std::sin(roll) +
                       std::cos(roll) * std::sin(pitch) * std::sin(yaw))) +
           tdy;
  // Y-Axis | drawn in green
  int x2 = static_cast<int>(-size * std::cos(yaw) * std::sin(roll)) + tdx;
  int y2 = static_cast<int>(
               size * (std::cos(pitch) * std::cos(roll) -
                       std::sin(pitch) * std::sin(yaw) * std::sin(roll))) +
           tdy;
  // Z-Axis | drawn in blue
  int x3 = static_cast<int>(size * std::sin(yaw)) + tdx;
  int y3 = static_cast<int>(-size * std::cos(yaw) * std::sin(pitch)) + tdy;

  cv::line(vis_im, cv::Point2i(tdx, tdy), cv::Point2i(x1, y1),
           cv::Scalar(0, 0, 255), line_size);
  cv::line(vis_im, cv::Point2i(tdx, tdy), cv::Point2i(x2, y2),
           cv::Scalar(0, 255, 0), line_size);
  cv::line(vis_im, cv::Point2i(tdx, tdy), cv::Point2i(x3, y3),
           cv::Scalar(255, 0, 0), line_size);
  return vis_im;
}

} // namespace vision
} // namespace ultra_infer
