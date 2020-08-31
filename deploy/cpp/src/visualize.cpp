//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "include/paddlex/visualize.h"

namespace PaddleX {
std::vector<int> GenerateColorMap(int num_class) {
  auto colormap = std::vector<int>(3 * num_class, 0);
  for (int i = 0; i < num_class; ++i) {
    int j = 0;
    int lab = i;
    while (lab) {
      colormap[i * 3] |= (((lab >> 0) & 1) << (7 - j));
      colormap[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j));
      colormap[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j));
      ++j;
      lab >>= 3;
    }
  }
  return colormap;
}

cv::Mat Visualize(const cv::Mat& img,
                     const DetResult& result,
                     const std::map<int, std::string>& labels,
                     float threshold) {
  auto colormap = GenerateColorMap(labels.size());
  cv::Mat vis_img = img.clone();
  auto boxes = result.boxes;
  for (int i = 0; i < boxes.size(); ++i) {
    if (boxes[i].score < threshold) {
      continue;
    }
    cv::Rect roi = cv::Rect(boxes[i].coordinate[0],
                            boxes[i].coordinate[1],
                            boxes[i].coordinate[2],
                            boxes[i].coordinate[3]);

    // draw box and title
    std::string text = boxes[i].category;
    int c1 = colormap[3 * boxes[i].category_id + 0];
    int c2 = colormap[3 * boxes[i].category_id + 1];
    int c3 = colormap[3 * boxes[i].category_id + 2];
    cv::Scalar roi_color = cv::Scalar(c1, c2, c3);
    text += std::to_string(static_cast<int>(boxes[i].score * 100)) + "%";
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.5f;
    float thickness = 0.5;
    cv::Size text_size =
        cv::getTextSize(text, font_face, font_scale, thickness, nullptr);
    cv::Point origin;
    origin.x = roi.x;
    origin.y = roi.y;

    // background
    cv::Rect text_back = cv::Rect(boxes[i].coordinate[0],
                                  boxes[i].coordinate[1] - text_size.height,
                                  text_size.width,
                                  text_size.height);

    // draw
    cv::rectangle(vis_img, roi, roi_color, 2);
    cv::rectangle(vis_img, text_back, roi_color, -1);
    cv::putText(vis_img,
                text,
                origin,
                font_face,
                font_scale,
                cv::Scalar(255, 255, 255),
                thickness);

    // mask
    if (boxes[i].mask.data.size() == 0) {
      continue;
    }
    std::vector<float> mask_data;
    mask_data.assign(boxes[i].mask.data.begin(), boxes[i].mask.data.end());
    cv::Mat bin_mask(boxes[i].mask.shape[1],
                     boxes[i].mask.shape[0],
                     CV_32FC1,
                     boxes[i].mask.data.data());
    cv::Mat full_mask = cv::Mat::zeros(vis_img.size(), CV_8UC1);
    bin_mask.copyTo(full_mask(roi));
    cv::Mat mask_ch[3];
    mask_ch[0] = full_mask * c1;
    mask_ch[1] = full_mask * c2;
    mask_ch[2] = full_mask * c3;
    cv::Mat mask;
    cv::merge(mask_ch, 3, mask);
    cv::addWeighted(vis_img, 1, mask, 0.5, 0, vis_img);
  }
  return vis_img;
}

cv::Mat Visualize(const cv::Mat& img,
                     const SegResult& result,
                     const std::map<int, std::string>& labels) {
  auto colormap = GenerateColorMap(labels.size());
  std::vector<uint8_t> label_map(result.label_map.data.begin(),
                                 result.label_map.data.end());
  cv::Mat mask(result.label_map.shape[0],
               result.label_map.shape[1],
               CV_8UC1,
               label_map.data());
  cv::Mat color_mask = cv::Mat::zeros(
      result.label_map.shape[0], result.label_map.shape[1], CV_8UC3);
  int rows = img.rows;
  int cols = img.cols;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      int category_id = static_cast<int>(mask.at<uchar>(i, j));
      color_mask.at<cv::Vec3b>(i, j)[0] = colormap[3 * category_id + 0];
      color_mask.at<cv::Vec3b>(i, j)[1] = colormap[3 * category_id + 1];
      color_mask.at<cv::Vec3b>(i, j)[2] = colormap[3 * category_id + 2];
    }
  }
  return color_mask;
}

std::string generate_save_path(const std::string& save_dir,
                               const std::string& file_path) {
  if (access(save_dir.c_str(), 0) < 0) {
#ifdef _WIN32
    mkdir(save_dir.c_str());
#else
    if (mkdir(save_dir.c_str(), S_IRWXU) < 0) {
      std::cerr << "Fail to create " << save_dir << "directory." << std::endl;
    }
#endif
  }
  int pos = file_path.find_last_of(OS_PATH_SEP);
  std::string image_name(file_path.substr(pos + 1));
  return save_dir + OS_PATH_SEP + image_name;
}
}  // namespace PaddleX
