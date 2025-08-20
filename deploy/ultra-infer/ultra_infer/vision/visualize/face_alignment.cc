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

cv::Mat VisFaceAlignment(const cv::Mat &im, const FaceAlignmentResult &result,
                         int line_size) {
  auto vis_im = im.clone();
  // vis landmarks
  cv::Scalar landmark_color = cv::Scalar(0, 255, 0);
  for (size_t i = 0; i < result.landmarks.size(); ++i) {
    cv::Point landmark;
    landmark.x = static_cast<int>(result.landmarks[i][0]);
    landmark.y = static_cast<int>(result.landmarks[i][1]);
    cv::circle(vis_im, landmark, line_size, landmark_color, -1);
  }
  return vis_im;
}

} // namespace vision
} // namespace ultra_infer
