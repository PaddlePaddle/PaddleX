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

#include <iomanip>

#include "ultra_infer/vision/visualize/visualize.h"

namespace ultra_infer {
namespace vision {

cv::Scalar GetMOTBoxColor(int idx) {
  idx = idx * 3;
  cv::Scalar color =
      cv::Scalar((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255);
  return color;
}

cv::Mat VisMOT(const cv::Mat &img, const MOTResult &results,
               float score_threshold, tracking::TrailRecorder *recorder) {
  cv::Mat vis_img = img.clone();
  int im_h = img.rows;
  int im_w = img.cols;
  float text_scale = std::max(1, static_cast<int>(im_w / 1600.));
  float text_thickness = 2.;
  float line_thickness = std::max(1, static_cast<int>(im_w / 500.));
  for (int i = 0; i < results.boxes.size(); ++i) {
    if (results.scores[i] < score_threshold) {
      continue;
    }
    const int obj_id = results.ids[i];
    const float score = results.scores[i];
    cv::Scalar color = GetMOTBoxColor(obj_id);
    if (recorder != nullptr) {
      int id = results.ids[i];
      auto iter = recorder->records.find(id);
      if (iter != recorder->records.end()) {
        for (int j = 0; j < iter->second.size(); j++) {
          cv::Point center(iter->second[j][0], iter->second[j][1]);
          cv::circle(vis_img, center, text_thickness, color);
        }
      }
    }
    cv::Point pt1 = cv::Point(results.boxes[i][0], results.boxes[i][1]);
    cv::Point pt2 = cv::Point(results.boxes[i][2], results.boxes[i][3]);
    cv::Point id_pt = cv::Point(results.boxes[i][0], results.boxes[i][1] + 10);
    cv::Point score_pt =
        cv::Point(results.boxes[i][0], results.boxes[i][1] - 10);
    cv::rectangle(vis_img, pt1, pt2, color, line_thickness);
    std::ostringstream idoss;
    idoss << std::setiosflags(std::ios::fixed) << std::setprecision(4);
    idoss << obj_id;
    std::string id_text = idoss.str();

    cv::putText(vis_img, id_text, id_pt, cv::FONT_HERSHEY_PLAIN, text_scale,
                color, text_thickness);

    std::ostringstream soss;
    soss << std::setiosflags(std::ios::fixed) << std::setprecision(2);
    soss << score;
    std::string score_text = soss.str();

    cv::putText(vis_img, score_text, score_pt, cv::FONT_HERSHEY_PLAIN,
                text_scale, color, text_thickness);
  }
  return vis_img;
}
} // namespace vision
} // namespace ultra_infer
