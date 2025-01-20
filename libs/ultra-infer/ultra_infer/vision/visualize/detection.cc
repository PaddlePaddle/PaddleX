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

cv::Mat VisDetection(const cv::Mat &im, const DetectionResult &result,
                     float score_threshold, int line_size, float font_size) {
  if (result.boxes.empty() && result.rotated_boxes.empty()) {
    return im;
  }
  if (result.contain_masks) {
    FDASSERT(result.boxes.size() == result.masks.size(),
             "The size of masks must be equal to the size of boxes, but now "
             "%zu != %zu.",
             result.boxes.size(), result.masks.size());
  }
  int max_label_id =
      *std::max_element(result.label_ids.begin(), result.label_ids.end());
  std::vector<int> color_map = GenerateColorMap(max_label_id);

  int h = im.rows;
  int w = im.cols;
  auto vis_im = im.clone();
  for (size_t i = 0; i < result.rotated_boxes.size(); ++i) {
    if (result.scores[i] < score_threshold) {
      continue;
    }

    int c0 = color_map[3 * result.label_ids[i] + 0];
    int c1 = color_map[3 * result.label_ids[i] + 1];
    int c2 = color_map[3 * result.label_ids[i] + 2];
    cv::Scalar rect_color = cv::Scalar(c0, c1, c2);
    std::string id = std::to_string(result.label_ids[i]);
    std::string score = std::to_string(result.scores[i]);
    if (score.size() > 4) {
      score = score.substr(0, 4);
    }
    std::string text = id + ", " + score;
    int font = cv::FONT_HERSHEY_SIMPLEX;
    cv::Size text_size = cv::getTextSize(text, font, font_size, 1, nullptr);

    for (int j = 0; j < 4; j++) {
      auto start = cv::Point(
          static_cast<int>(round(result.rotated_boxes[i][2 * j])),
          static_cast<int>(round(result.rotated_boxes[i][2 * j + 1])));

      cv::Point end;
      if (j != 3) {
        end = cv::Point(
            static_cast<int>(round(result.rotated_boxes[i][2 * (j + 1)])),
            static_cast<int>(round(result.rotated_boxes[i][2 * (j + 1) + 1])));
      } else {
        end = cv::Point(static_cast<int>(round(result.rotated_boxes[i][0])),
                        static_cast<int>(round(result.rotated_boxes[i][1])));
        cv::putText(vis_im, text, end, font, font_size,
                    cv::Scalar(255, 255, 255), 1);
      }
      cv::line(vis_im, start, end, cv::Scalar(255, 255, 255), 3, cv::LINE_AA,
               0);
    }
  }

  for (size_t box_i = 0; box_i < result.boxes.size(); ++box_i) {
    if (result.scores[box_i] < score_threshold) {
      continue;
    }
    int x1 = static_cast<int>(round(result.boxes[box_i][0]));
    int y1 = static_cast<int>(round(result.boxes[box_i][1]));
    int x2 = static_cast<int>(round(result.boxes[box_i][2]));
    int y2 = static_cast<int>(round(result.boxes[box_i][3]));
    int box_h = y2 - y1;
    int box_w = x2 - x1;
    int c0 = color_map[3 * result.label_ids[box_i] + 0];
    int c1 = color_map[3 * result.label_ids[box_i] + 1];
    int c2 = color_map[3 * result.label_ids[box_i] + 2];
    cv::Scalar rect_color = cv::Scalar(c0, c1, c2);
    std::string id = std::to_string(result.label_ids[box_i]);
    std::string score = std::to_string(result.scores[box_i]);
    if (score.size() > 4) {
      score = score.substr(0, 4);
    }
    std::string text = id + ", " + score;
    int font = cv::FONT_HERSHEY_SIMPLEX;
    cv::Size text_size = cv::getTextSize(text, font, font_size, 1, nullptr);
    cv::Point origin;
    origin.x = x1;
    origin.y = y1;
    cv::Rect rect(x1, y1, box_w, box_h);
    cv::rectangle(vis_im, rect, rect_color, line_size);
    cv::putText(vis_im, text, origin, font, font_size,
                cv::Scalar(255, 255, 255), 1);
    if (result.contain_masks) {
      int mask_h = static_cast<int>(result.masks[box_i].shape[0]);
      int mask_w = static_cast<int>(result.masks[box_i].shape[1]);
      // non-const pointer for cv:Mat constructor
      uint32_t *mask_raw_data = const_cast<uint32_t *>(
          static_cast<const uint32_t *>(result.masks[box_i].Data()));
      // only reference to mask data (zero copy)
      cv::Mat mask(mask_h, mask_w, CV_32SC1, mask_raw_data);
      if ((mask_h != box_h) || (mask_w != box_w)) {
        cv::resize(mask, mask, cv::Size(box_w, box_h));
      }
      // use a bright color for instance mask
      int mc0 = 255 - c0 >= 127 ? 255 - c0 : 127;
      int mc1 = 255 - c1 >= 127 ? 255 - c1 : 127;
      int mc2 = 255 - c2 >= 127 ? 255 - c2 : 127;
      uint32_t *mask_data = reinterpret_cast<uint32_t *>(mask.data);
      // inplace blending (zero copy)
      uchar *vis_im_data = static_cast<uchar *>(vis_im.data);
      for (size_t i = y1; i < y2; ++i) {
        for (size_t j = x1; j < x2; ++j) {
          if (mask_data[(i - y1) * mask_w + (j - x1)] != 0) {
            vis_im_data[i * w * 3 + j * 3 + 0] = cv::saturate_cast<uchar>(
                static_cast<float>(mc0) * 0.5f +
                static_cast<float>(vis_im_data[i * w * 3 + j * 3 + 0]) * 0.5f);
            vis_im_data[i * w * 3 + j * 3 + 1] = cv::saturate_cast<uchar>(
                static_cast<float>(mc1) * 0.5f +
                static_cast<float>(vis_im_data[i * w * 3 + j * 3 + 1]) * 0.5f);
            vis_im_data[i * w * 3 + j * 3 + 2] = cv::saturate_cast<uchar>(
                static_cast<float>(mc2) * 0.5f +
                static_cast<float>(vis_im_data[i * w * 3 + j * 3 + 2]) * 0.5f);
          }
        }
      }
    }
  }
  return vis_im;
}

// Visualize DetectionResult with custom labels.
cv::Mat VisDetection(const cv::Mat &im, const DetectionResult &result,
                     const std::vector<std::string> &labels,
                     float score_threshold, int line_size, float font_size,
                     std::vector<int> font_color, int font_thickness) {
  if (result.boxes.empty()) {
    return im;
  }
  if (result.contain_masks) {
    FDASSERT(result.boxes.size() == result.masks.size(),
             "The size of masks must be equal to the size of boxes, but now "
             "%zu != %zu.",
             result.boxes.size(), result.masks.size());
  }
  int max_label_id =
      *std::max_element(result.label_ids.begin(), result.label_ids.end());
  std::vector<int> color_map = GenerateColorMap(max_label_id);

  int h = im.rows;
  int w = im.cols;
  auto vis_im = im.clone();
  auto font_color_ = cv::Scalar(font_color[0], font_color[1], font_color[2]);
  for (size_t i = 0; i < result.rotated_boxes.size(); ++i) {
    if (result.scores[i] < score_threshold) {
      continue;
    }

    int c0 = color_map[3 * result.label_ids[i] + 0];
    int c1 = color_map[3 * result.label_ids[i] + 1];
    int c2 = color_map[3 * result.label_ids[i] + 2];
    cv::Scalar rect_color = cv::Scalar(c0, c1, c2);
    std::string id = std::to_string(result.label_ids[i]);
    std::string score = std::to_string(result.scores[i]);
    if (score.size() > 4) {
      score = score.substr(0, 4);
    }
    std::string text = id + ", " + score;
    int font = cv::FONT_HERSHEY_SIMPLEX;
    cv::Size text_size = cv::getTextSize(text, font, font_size, 1, nullptr);

    for (int j = 0; j < 4; j++) {
      auto start = cv::Point(
          static_cast<int>(round(result.rotated_boxes[i][2 * j])),
          static_cast<int>(round(result.rotated_boxes[i][2 * j + 1])));

      cv::Point end;
      if (j == 3) {
        end = cv::Point(
            static_cast<int>(round(result.rotated_boxes[i][2 * j])),
            static_cast<int>(round(result.rotated_boxes[i][2 * j + 1])));
      } else {
        end = cv::Point(static_cast<int>(round(result.rotated_boxes[i][0])),
                        static_cast<int>(round(result.rotated_boxes[i][1])));
        cv::putText(vis_im, text, end, font, font_size, font_color_,
                    font_thickness);
      }
      cv::line(vis_im, start, end, cv::Scalar(255, 255, 255), 3, cv::LINE_AA,
               0);
    }
  }
  for (size_t i = 0; i < result.boxes.size(); ++i) {
    if (result.scores[i] < score_threshold) {
      continue;
    }
    int x1 = static_cast<int>(result.boxes[i][0]);
    int y1 = static_cast<int>(result.boxes[i][1]);
    int x2 = static_cast<int>(result.boxes[i][2]);
    int y2 = static_cast<int>(result.boxes[i][3]);
    int box_h = y2 - y1;
    int box_w = x2 - x1;
    int c0 = color_map[3 * result.label_ids[i] + 0];
    int c1 = color_map[3 * result.label_ids[i] + 1];
    int c2 = color_map[3 * result.label_ids[i] + 2];
    cv::Scalar rect_color = cv::Scalar(c0, c1, c2);
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
    cv::Size text_size = cv::getTextSize(text, font, font_size, 1, nullptr);
    cv::Point origin;
    origin.x = x1;
    origin.y = y1;
    cv::Rect rect(x1, y1, box_w, box_h);
    cv::rectangle(vis_im, rect, rect_color, line_size);
    cv::putText(vis_im, text, origin, font, font_size, font_color_,
                font_thickness);
    if (result.contain_masks) {
      int mask_h = static_cast<int>(result.masks[i].shape[0]);
      int mask_w = static_cast<int>(result.masks[i].shape[1]);
      // non-const pointer for cv:Mat constructor
      int32_t *mask_raw_data = const_cast<int32_t *>(
          static_cast<const int32_t *>(result.masks[i].Data()));
      // only reference to mask data (zero copy)
      cv::Mat mask(mask_h, mask_w, CV_32SC1, mask_raw_data);
      if ((mask_h != box_h) || (mask_w != box_w)) {
        cv::resize(mask, mask, cv::Size(box_w, box_h));
      }
      // use a bright color for instance mask
      int mc0 = 255 - c0 >= 127 ? 255 - c0 : 127;
      int mc1 = 255 - c1 >= 127 ? 255 - c1 : 127;
      int mc2 = 255 - c2 >= 127 ? 255 - c2 : 127;
      int32_t *mask_data = reinterpret_cast<int32_t *>(mask.data);
      // inplace blending (zero copy)
      uchar *vis_im_data = static_cast<uchar *>(vis_im.data);
      for (size_t i = y1; i < y2; ++i) {
        for (size_t j = x1; j < x2; ++j) {
          if (mask_data[(i - y1) * mask_w + (j - x1)] != 0) {
            vis_im_data[i * w * 3 + j * 3 + 0] = cv::saturate_cast<uchar>(
                static_cast<float>(mc0) * 0.5f +
                static_cast<float>(vis_im_data[i * w * 3 + j * 3 + 0]) * 0.5f);
            vis_im_data[i * w * 3 + j * 3 + 1] = cv::saturate_cast<uchar>(
                static_cast<float>(mc1) * 0.5f +
                static_cast<float>(vis_im_data[i * w * 3 + j * 3 + 1]) * 0.5f);
            vis_im_data[i * w * 3 + j * 3 + 2] = cv::saturate_cast<uchar>(
                static_cast<float>(mc2) * 0.5f +
                static_cast<float>(vis_im_data[i * w * 3 + j * 3 + 2]) * 0.5f);
          }
        }
      }
    }
  }
  return vis_im;
}

// Default only support visualize num_classes <= 1000
// If need to visualize num_classes > 1000
// Please call Visualize::GetColorMap(num_classes) first
cv::Mat Visualize::VisDetection(const cv::Mat &im,
                                const DetectionResult &result,
                                float score_threshold, int line_size,
                                float font_size) {
  if (result.boxes.empty()) {
    return im;
  }
  FDWARNING << "DEPRECATED: ultra_infer::vision::Visualize::VisDetection is "
               "deprecated, please use ultra_infer::vision:VisDetection "
               "function instead."
            << std::endl;
  if (result.contain_masks) {
    FDASSERT(result.boxes.size() == result.masks.size(),
             "The size of masks must be equal the size of boxes!");
  }
  auto color_map = GetColorMap();
  int h = im.rows;
  int w = im.cols;
  auto vis_im = im.clone();
  for (size_t i = 0; i < result.boxes.size(); ++i) {
    if (result.scores[i] < score_threshold) {
      continue;
    }
    int x1 = static_cast<int>(result.boxes[i][0]);
    int y1 = static_cast<int>(result.boxes[i][1]);
    int x2 = static_cast<int>(result.boxes[i][2]);
    int y2 = static_cast<int>(result.boxes[i][3]);
    int box_h = y2 - y1;
    int box_w = x2 - x1;
    int c0 = color_map[3 * result.label_ids[i] + 0];
    int c1 = color_map[3 * result.label_ids[i] + 1];
    int c2 = color_map[3 * result.label_ids[i] + 2];
    cv::Scalar rect_color = cv::Scalar(c0, c1, c2);
    std::string id = std::to_string(result.label_ids[i]);
    std::string score = std::to_string(result.scores[i]);
    if (score.size() > 4) {
      score = score.substr(0, 4);
    }
    std::string text = id + "," + score;
    int font = cv::FONT_HERSHEY_SIMPLEX;
    cv::Size text_size = cv::getTextSize(text, font, font_size, 1, nullptr);
    cv::Point origin;
    origin.x = x1;
    origin.y = y1;
    cv::Rect rect(x1, y1, box_w, box_h);
    cv::rectangle(vis_im, rect, rect_color, line_size);
    cv::putText(vis_im, text, origin, font, font_size,
                cv::Scalar(255, 255, 255), 1);
    if (result.contain_masks) {
      int mask_h = static_cast<int>(result.masks[i].shape[0]);
      int mask_w = static_cast<int>(result.masks[i].shape[1]);
      // non-const pointer for cv:Mat constructor
      int32_t *mask_raw_data = const_cast<int32_t *>(
          static_cast<const int32_t *>(result.masks[i].Data()));
      // only reference to mask data (zero copy)
      cv::Mat mask(mask_h, mask_w, CV_32SC1, mask_raw_data);
      if ((mask_h != box_h) || (mask_w != box_w)) {
        cv::resize(mask, mask, cv::Size(box_w, box_h));
      }
      // use a bright color for instance mask
      int mc0 = 255 - c0 >= 127 ? 255 - c0 : 127;
      int mc1 = 255 - c1 >= 127 ? 255 - c1 : 127;
      int mc2 = 255 - c2 >= 127 ? 255 - c2 : 127;
      int32_t *mask_data = reinterpret_cast<int32_t *>(mask.data);
      // inplace blending (zero copy)
      uchar *vis_im_data = static_cast<uchar *>(vis_im.data);
      for (size_t i = y1; i < y2; ++i) {
        for (size_t j = x1; j < x2; ++j) {
          if (mask_data[(i - y1) * mask_w + (j - x1)] != 0) {
            vis_im_data[i * w * 3 + j * 3 + 0] = cv::saturate_cast<uchar>(
                static_cast<float>(mc0) * 0.5f +
                static_cast<float>(vis_im_data[i * w * 3 + j * 3 + 0]) * 0.5f);
            vis_im_data[i * w * 3 + j * 3 + 1] = cv::saturate_cast<uchar>(
                static_cast<float>(mc1) * 0.5f +
                static_cast<float>(vis_im_data[i * w * 3 + j * 3 + 1]) * 0.5f);
            vis_im_data[i * w * 3 + j * 3 + 2] = cv::saturate_cast<uchar>(
                static_cast<float>(mc2) * 0.5f +
                static_cast<float>(vis_im_data[i * w * 3 + j * 3 + 2]) * 0.5f);
          }
        }
      }
    }
  }
  return vis_im;
}

} // namespace vision
} // namespace ultra_infer
