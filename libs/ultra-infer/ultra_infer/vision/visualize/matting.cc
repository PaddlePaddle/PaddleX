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

cv::Mat VisMatting(const cv::Mat &im, const MattingResult &result,
                   bool transparent_background, float transparent_threshold,
                   bool remove_small_connected_area) {
  FDASSERT((!im.empty()), "im can't be empty!");
  FDASSERT((im.channels() == 3), "Only support 3 channels mat!");
  auto vis_img = im.clone();
  cv::Mat transparent_vis_mat;
  int channel = im.channels();
  int out_h = static_cast<int>(result.shape[0]);
  int out_w = static_cast<int>(result.shape[1]);
  int height = im.rows;
  int width = im.cols;
  std::vector<float> alpha_copy;
  alpha_copy.assign(result.alpha.begin(), result.alpha.end());
  float *alpha_ptr = static_cast<float *>(alpha_copy.data());
  cv::Mat alpha(out_h, out_w, CV_32FC1, alpha_ptr);
  if (remove_small_connected_area) {
    alpha = RemoveSmallConnectedArea(alpha, 0.05f);
  }
  if ((out_h != height) || (out_w != width)) {
    cv::resize(alpha, alpha, cv::Size(width, height));
  }

  if ((vis_img).type() != CV_8UC3) {
    (vis_img).convertTo((vis_img), CV_8UC3);
  }

  if (transparent_background) {
    if (vis_img.channels() != 4) {
      cv::cvtColor(vis_img, transparent_vis_mat, cv::COLOR_BGR2BGRA);
      vis_img = transparent_vis_mat;
      channel = 4;
    }
  }

  uchar *vis_data = static_cast<uchar *>(vis_img.data);
  uchar *im_data = static_cast<uchar *>(im.data);
  float *alpha_data = reinterpret_cast<float *>(alpha.data);

  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      float alpha_val = alpha_data[i * width + j];
      if (transparent_background) {
        if (alpha_val < transparent_threshold) {
          vis_data[i * width * channel + j * channel + 3] =
              cv::saturate_cast<uchar>(0.f);
        } else {
          vis_data[i * width * channel + j * channel + 0] =
              cv::saturate_cast<uchar>(
                  static_cast<float>(im_data[i * width * 3 + j * 3 + 0]));
          vis_data[i * width * channel + j * channel + 1] =
              cv::saturate_cast<uchar>(
                  static_cast<float>(im_data[i * width * 3 + j * 3 + 1]));
          vis_data[i * width * channel + j * channel + 2] =
              cv::saturate_cast<uchar>(
                  static_cast<float>(im_data[i * width * 3 + j * 3 + 2]));
        }
      } else {
        vis_data[i * width * channel + j * channel + 0] =
            cv::saturate_cast<uchar>(
                static_cast<float>(im_data[i * width * 3 + j * 3 + 0]) *
                    alpha_val +
                (1.f - alpha_val) * 153.f);
        vis_data[i * width * channel + j * channel + 1] =
            cv::saturate_cast<uchar>(
                static_cast<float>(im_data[i * width * 3 + j * 3 + 1]) *
                    alpha_val +
                (1.f - alpha_val) * 255.f);
        vis_data[i * width * channel + j * channel + 2] =
            cv::saturate_cast<uchar>(
                static_cast<float>(im_data[i * width * 3 + j * 3 + 2]) *
                    alpha_val +
                (1.f - alpha_val) * 120.f);
      }
    }
  }
  return vis_img;
}

cv::Mat Visualize::VisMattingAlpha(const cv::Mat &im,
                                   const MattingResult &result,
                                   bool remove_small_connected_area) {
  FDWARNING << "DEPRECATED: ultra_infer::vision::Visualize::VisMattingAlpha is "
               "deprecated, please use ultra_infer::vision:VisMatting function "
               "instead."
            << std::endl;
  FDASSERT((!im.empty()), "im can't be empty!");
  FDASSERT((im.channels() == 3), "Only support 3 channels mat!");

  auto vis_img = im.clone();
  int out_h = static_cast<int>(result.shape[0]);
  int out_w = static_cast<int>(result.shape[1]);
  int height = im.rows;
  int width = im.cols;
  std::vector<float> alpha_copy;
  alpha_copy.assign(result.alpha.begin(), result.alpha.end());
  float *alpha_ptr = static_cast<float *>(alpha_copy.data());
  cv::Mat alpha(out_h, out_w, CV_32FC1, alpha_ptr);
  if (remove_small_connected_area) {
    alpha = RemoveSmallConnectedArea(alpha, 0.05f);
  }
  if ((out_h != height) || (out_w != width)) {
    cv::resize(alpha, alpha, cv::Size(width, height));
  }

  if ((vis_img).type() != CV_8UC3) {
    (vis_img).convertTo((vis_img), CV_8UC3);
  }

  uchar *vis_data = static_cast<uchar *>(vis_img.data);
  uchar *im_data = static_cast<uchar *>(im.data);
  float *alpha_data = reinterpret_cast<float *>(alpha.data);

  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      float alpha_val = alpha_data[i * width + j];
      vis_data[i * width * 3 + j * 3 + 0] = cv::saturate_cast<uchar>(
          static_cast<float>(im_data[i * width * 3 + j * 3 + 0]) * alpha_val +
          (1.f - alpha_val) * 153.f);
      vis_data[i * width * 3 + j * 3 + 1] = cv::saturate_cast<uchar>(
          static_cast<float>(im_data[i * width * 3 + j * 3 + 1]) * alpha_val +
          (1.f - alpha_val) * 255.f);
      vis_data[i * width * 3 + j * 3 + 2] = cv::saturate_cast<uchar>(
          static_cast<float>(im_data[i * width * 3 + j * 3 + 2]) * alpha_val +
          (1.f - alpha_val) * 120.f);
    }
  }
  return vis_img;
}

} // namespace vision
} // namespace ultra_infer
