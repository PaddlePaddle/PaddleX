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

#include "ultra_infer/vision/utils/utils.h"

namespace ultra_infer {
namespace vision {
namespace utils {

bool CropImageByBox(Mat &src_im, Mat *dst_im, const std::vector<float> &box,
                    std::vector<float> *center, std::vector<float> *scale,
                    const float expandratio) {
  const cv::Mat *img = src_im.GetOpenCVMat();
  cv::Mat *crop_img = dst_im->GetOpenCVMat();
  int xmin = static_cast<int>(box[0]);
  int ymin = static_cast<int>(box[1]);
  int xmax = static_cast<int>(box[2]);
  int ymax = static_cast<int>(box[3]);
  float centerx = (xmin + xmax) / 2.0f;
  float centery = (ymin + ymax) / 2.0f;
  float half_h = (ymax - ymin) * (1 + expandratio) / 2.0f;
  float half_w = (xmax - xmin) * (1 + expandratio) / 2.0f;
  // adjust h or w to keep image ratio, expand the shorter edge
  if (half_h * 3 > half_w * 4) {
    half_w = half_h * 0.75;
  }
  int crop_xmin = std::max(0, static_cast<int>(centerx - half_w));
  int crop_ymin = std::max(0, static_cast<int>(centery - half_h));
  int crop_xmax = std::min(img->cols - 1, static_cast<int>(centerx + half_w));
  int crop_ymax = std::min(img->rows - 1, static_cast<int>(centery + half_h));

  crop_img->create(crop_ymax - crop_ymin, crop_xmax - crop_xmin, img->type());
  *crop_img =
      (*img)(cv::Range(crop_ymin, crop_ymax), cv::Range(crop_xmin, crop_xmax));
  center->clear();
  center->emplace_back((crop_xmin + crop_xmax) / 2.0f);
  center->emplace_back((crop_ymin + crop_ymax) / 2.0f);

  scale->clear();
  scale->emplace_back((crop_xmax - crop_xmin));
  scale->emplace_back((crop_ymax - crop_ymin));

  dst_im->SetWidth(crop_img->cols);
  dst_im->SetHeight(crop_img->rows);
  return true;
}

} // namespace utils
} // namespace vision
} // namespace ultra_infer
