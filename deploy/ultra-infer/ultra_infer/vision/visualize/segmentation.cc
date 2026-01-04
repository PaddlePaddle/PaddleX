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
#include "ultra_infer/vision/visualize/segmentation_arm.h"
#include "ultra_infer/vision/visualize/visualize.h"

namespace ultra_infer {
namespace vision {

static cv::Mat VisSegmentationCommonCpu(const cv::Mat &im,
                                        const SegmentationResult &result,
                                        float weight) {
  // Use the native c++ version without any optimization.
  auto color_map = GenerateColorMap(1000);
  int64_t height = result.shape[0];
  int64_t width = result.shape[1];
  auto vis_img = cv::Mat(height, width, CV_8UC3);

  int64_t index = 0;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int category_id = result.label_map[index++];
      if (category_id == 0) {
        vis_img.at<cv::Vec3b>(i, j)[0] = im.at<cv::Vec3b>(i, j)[0];
        vis_img.at<cv::Vec3b>(i, j)[1] = im.at<cv::Vec3b>(i, j)[1];
        vis_img.at<cv::Vec3b>(i, j)[2] = im.at<cv::Vec3b>(i, j)[2];
      } else {
        vis_img.at<cv::Vec3b>(i, j)[0] = color_map[3 * category_id + 0];
        vis_img.at<cv::Vec3b>(i, j)[1] = color_map[3 * category_id + 1];
        vis_img.at<cv::Vec3b>(i, j)[2] = color_map[3 * category_id + 2];
      }
    }
  }
  cv::addWeighted(im, 1.0 - weight, vis_img, weight, 0, vis_img);
  return vis_img;
}

cv::Mat VisSegmentation(const cv::Mat &im, const SegmentationResult &result,
                        float weight) {
  // TODO: Support SSE/AVX on x86_64 platforms
#ifdef __ARM_NEON
  return VisSegmentationNEON(im, result, weight, true);
#else
  return VisSegmentationCommonCpu(im, result, weight);
#endif
}

cv::Mat Visualize::VisSegmentation(const cv::Mat &im,
                                   const SegmentationResult &result) {
  FDWARNING << "DEPRECATED: ultra_infer::vision::Visualize::VisSegmentation is "
               "deprecated, please use ultra_infer::vision:VisSegmentation "
               "function instead."
            << std::endl;
#ifdef __ARM_NEON
  return VisSegmentationNEON(im, result, 0.5f, true);
#else
  return VisSegmentationCommonCpu(im, result, 0.5f);
#endif
}

} // namespace vision
} // namespace ultra_infer
