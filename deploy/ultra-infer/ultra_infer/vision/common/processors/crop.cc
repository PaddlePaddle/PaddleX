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

#include "ultra_infer/vision/common/processors/crop.h"

namespace ultra_infer {
namespace vision {

bool Crop::ImplByOpenCV(Mat *mat) {
  cv::Mat *im = mat->GetOpenCVMat();
  int height = static_cast<int>(im->rows);
  int width = static_cast<int>(im->cols);
  if (height < height_ + offset_h_ || width < width_ + offset_w_) {
    FDERROR << "[Crop] Cannot crop [" << height_ << ", " << width_
            << "] from the input image [" << height << ", " << width
            << "], with offset [" << offset_h_ << ", " << offset_w_ << "]."
            << std::endl;
    return false;
  }
  cv::Rect crop_roi(offset_w_, offset_h_, width_, height_);
  cv::Mat new_im = (*im)(crop_roi).clone();
  mat->SetMat(new_im);
  mat->SetWidth(width_);
  mat->SetHeight(height_);
  return true;
}

#ifdef ENABLE_FLYCV
bool Crop::ImplByFlyCV(Mat *mat) {
  fcv::Mat *im = mat->GetFlyCVMat();
  int height = static_cast<int>(im->height());
  int width = static_cast<int>(im->width());
  if (height < height_ + offset_h_ || width < width_ + offset_w_) {
    FDERROR << "[Crop] Cannot crop [" << height_ << ", " << width_
            << "] from the input image [" << height << ", " << width
            << "], with offset [" << offset_h_ << ", " << offset_w_ << "]."
            << std::endl;
    return false;
  }
  fcv::Rect crop_roi(offset_w_, offset_h_, width_, height_);
  fcv::Mat new_im;
  fcv::crop(*im, new_im, crop_roi);
  mat->SetMat(new_im);
  mat->SetWidth(width_);
  mat->SetHeight(height_);
  return true;
}
#endif

bool Crop::Run(Mat *mat, int offset_w, int offset_h, int width, int height,
               ProcLib lib) {
  auto c = Crop(offset_w, offset_h, width, height);
  return c(mat, lib);
}

} // namespace vision
} // namespace ultra_infer
