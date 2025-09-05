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

cv::Mat VisFaceDetection(const cv::Mat &im, const FaceDetectionResult &result,
                         int line_size, float font_size) {
  auto color_map = GenerateColorMap();
  int h = im.rows;
  int w = im.cols;

  auto vis_im = im.clone();
  bool vis_landmarks = false;
  if ((result.landmarks_per_face > 0) &&
      (result.boxes.size() * result.landmarks_per_face ==
       result.landmarks.size())) {
    vis_landmarks = true;
  }
  for (size_t i = 0; i < result.boxes.size(); ++i) {
    cv::Rect rect(result.boxes[i][0], result.boxes[i][1],
                  result.boxes[i][2] - result.boxes[i][0],
                  result.boxes[i][3] - result.boxes[i][1]);
    int color_id = i % 333;
    int c0 = color_map[3 * color_id + 0];
    int c1 = color_map[3 * color_id + 1];
    int c2 = color_map[3 * color_id + 2];
    cv::Scalar rect_color = cv::Scalar(c0, c1, c2);
    std::string text = std::to_string(result.scores[i]);
    if (text.size() > 4) {
      text = text.substr(0, 4);
    }
    int font = cv::FONT_HERSHEY_SIMPLEX;
    cv::Size text_size = cv::getTextSize(text, font, font_size, 1, nullptr);
    cv::Point origin;
    origin.x = rect.x;
    origin.y = rect.y;
    cv::Rect text_background =
        cv::Rect(result.boxes[i][0], result.boxes[i][1] - text_size.height,
                 text_size.width, text_size.height);
    cv::rectangle(vis_im, rect, rect_color, line_size);
    cv::putText(vis_im, text, origin, font, font_size,
                cv::Scalar(255, 255, 255), 1);
    // vis landmarks (if have)
    if (vis_landmarks) {
      cv::Scalar landmark_color = rect_color;
      for (size_t j = 0; j < result.landmarks_per_face; ++j) {
        cv::Point landmark;
        landmark.x = static_cast<int>(
            result.landmarks[i * result.landmarks_per_face + j][0]);
        landmark.y = static_cast<int>(
            result.landmarks[i * result.landmarks_per_face + j][1]);
        cv::circle(vis_im, landmark, line_size, landmark_color, -1);
      }
    }
  }
  return vis_im;
}

// Default only support visualize num_classes <= 1000
// If need to visualize num_classes > 1000
// Please call Visualize::GetColorMap(num_classes) first
cv::Mat Visualize::VisFaceDetection(const cv::Mat &im,
                                    const FaceDetectionResult &result,
                                    int line_size, float font_size) {
  FDWARNING
      << "DEPRECATED: ultra_infer::vision::Visualize::VisFaceDetection is "
         "deprecated, please use ultra_infer::vision:VisFaceDetection "
         "function instead."
      << std::endl;
  auto color_map = GetColorMap();
  int h = im.rows;
  int w = im.cols;

  auto vis_im = im.clone();
  bool vis_landmarks = false;
  if ((result.landmarks_per_face > 0) &&
      (result.boxes.size() * result.landmarks_per_face ==
       result.landmarks.size())) {
    vis_landmarks = true;
  }
  for (size_t i = 0; i < result.boxes.size(); ++i) {
    cv::Rect rect(result.boxes[i][0], result.boxes[i][1],
                  result.boxes[i][2] - result.boxes[i][0],
                  result.boxes[i][3] - result.boxes[i][1]);
    int color_id = i % 333;
    int c0 = color_map[3 * color_id + 0];
    int c1 = color_map[3 * color_id + 1];
    int c2 = color_map[3 * color_id + 2];
    cv::Scalar rect_color = cv::Scalar(c0, c1, c2);
    std::string text = std::to_string(result.scores[i]);
    if (text.size() > 4) {
      text = text.substr(0, 4);
    }
    int font = cv::FONT_HERSHEY_SIMPLEX;
    cv::Size text_size = cv::getTextSize(text, font, font_size, 1, nullptr);
    cv::Point origin;
    origin.x = rect.x;
    origin.y = rect.y;
    cv::Rect text_background =
        cv::Rect(result.boxes[i][0], result.boxes[i][1] - text_size.height,
                 text_size.width, text_size.height);
    cv::rectangle(vis_im, rect, rect_color, line_size);
    cv::putText(vis_im, text, origin, font, font_size,
                cv::Scalar(255, 255, 255), 1);
    // vis landmarks (if have)
    if (vis_landmarks) {
      cv::Scalar landmark_color = rect_color;
      for (size_t j = 0; j < result.landmarks_per_face; ++j) {
        cv::Point landmark;
        landmark.x = static_cast<int>(
            result.landmarks[i * result.landmarks_per_face + j][0]);
        landmark.y = static_cast<int>(
            result.landmarks[i * result.landmarks_per_face + j][1]);
        cv::circle(vis_im, landmark, line_size, landmark_color, -1);
      }
    }
  }
  return vis_im;
}

} // namespace vision
} // namespace ultra_infer
