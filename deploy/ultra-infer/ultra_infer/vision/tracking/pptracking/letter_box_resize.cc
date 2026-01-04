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

#include "ultra_infer/vision/tracking/pptracking/letter_box_resize.h"
#include "ultra_infer/vision/common/processors/transform.h"

namespace ultra_infer {
namespace vision {

bool LetterBoxResize::ImplByOpenCV(Mat *mat) {
  if (mat->Channels() != color_.size()) {
    FDERROR << "LetterBoxResize: Require input channels equals to size of "
               "color value, "
               "but now channels = "
            << mat->Channels()
            << ", the size of color values = " << color_.size() << "."
            << std::endl;
    return false;
  }
  // generate scale_factor
  int origin_w = mat->Width();
  int origin_h = mat->Height();
  int target_h = target_size_[0];
  int target_w = target_size_[1];
  float ratio_h = static_cast<float>(target_h) / static_cast<float>(origin_h);
  float ratio_w = static_cast<float>(target_w) / static_cast<float>(origin_w);
  float resize_scale = std::min(ratio_h, ratio_w);
  // get_resized_shape
  int new_shape_w = std::round(origin_w * resize_scale);
  int new_shape_h = std::round(origin_h * resize_scale);
  // calculate pad
  float padw = (target_size_[1] - new_shape_w) / 2.;
  float padh = (target_size_[0] - new_shape_h) / 2.;
  int top = std::round(padh - 0.1);
  int bottom = std::round(padh + 0.1);
  int left = std::round(padw - 0.1);
  int right = std::round(padw + 0.1);
  Resize::Run(mat, new_shape_w, new_shape_h, -1.0, -1.0, 3, false);
  Pad::Run(mat, top, bottom, left, right, color_);
  return true;
}

#ifdef ENABLE_FLYCV
bool LetterBoxResize::ImplByFlyCV(Mat *mat) {
  if (mat->Channels() != color_.size()) {
    FDERROR << "LetterBoxResize: Require input channels equals to size of "
               "color value, "
               "but now channels = "
            << mat->Channels()
            << ", the size of color values = " << color_.size() << "."
            << std::endl;
    return false;
  }
  // generate scale_factor
  int origin_w = mat->Width();
  int origin_h = mat->Height();
  int target_h = target_size_[0];
  int target_w = target_size_[1];
  float ratio_h = static_cast<float>(target_h) / static_cast<float>(origin_h);
  float ratio_w = static_cast<float>(target_w) / static_cast<float>(origin_w);
  float resize_scale = std::min(ratio_h, ratio_w);
  // get_resized_shape
  int new_shape_w = std::round(origin_w * resize_scale);
  int new_shape_h = std::round(origin_h * resize_scale);
  // calculate pad
  float padw = (target_size_[1] - new_shape_w) / 2.;
  float padh = (target_size_[0] - new_shape_h) / 2.;
  int top = std::round(padh - 0.1);
  int bottom = std::round(padh + 0.1);
  int left = std::round(padw - 0.1);
  int right = std::round(padw + 0.1);
  Resize::Run(mat, new_shape_w, new_shape_h, -1.0, -1.0, 3, false,
              ProcLib::FLYCV);
  Pad::Run(mat, top, bottom, left, right, color_, ProcLib::FLYCV);
  return true;
}
#endif

#ifdef ENABLE_CVCUDA
bool LetterBoxResize::ImplByCvCuda(Mat *mat) {
  if (mat->Channels() != color_.size()) {
    FDERROR << "LetterBoxResize: Require input channels equals to size of "
               "color value, "
               "but now channels = "
            << mat->Channels()
            << ", the size of color values = " << color_.size() << "."
            << std::endl;
    return false;
  }
  // generate scale_factor
  int origin_w = mat->Width();
  int origin_h = mat->Height();
  int target_h = target_size_[0];
  int target_w = target_size_[1];
  float ratio_h = static_cast<float>(target_h) / static_cast<float>(origin_h);
  float ratio_w = static_cast<float>(target_w) / static_cast<float>(origin_w);
  float resize_scale = std::min(ratio_h, ratio_w);
  // get_resized_shape
  int new_shape_w = std::round(origin_w * resize_scale);
  int new_shape_h = std::round(origin_h * resize_scale);
  // calculate pad
  float padw = (target_size_[1] - new_shape_w) / 2.;
  float padh = (target_size_[0] - new_shape_h) / 2.;
  int top = std::round(padh - 0.1);
  int bottom = std::round(padh + 0.1);
  int left = std::round(padw - 0.1);
  int right = std::round(padw + 0.1);
  Resize::Run(mat, new_shape_w, new_shape_h, -1.0, -1.0, 3, false,
              ProcLib::CVCUDA);
  Pad::Run(mat, top, bottom, left, right, color_, ProcLib::CVCUDA);
  return true;
}
#endif

#ifdef ENABLE_CUDA
bool LetterBoxResize::ImplByCuda(Mat *mat) {
  if (mat->Channels() != color_.size()) {
    FDERROR << "LetterBoxResize: Require input channels equals to size of "
               "color value, "
               "but now channels = "
            << mat->Channels()
            << ", the size of color values = " << color_.size() << "."
            << std::endl;
    return false;
  }
  // generate scale_factor
  int origin_w = mat->Width();
  int origin_h = mat->Height();
  int target_h = target_size_[0];
  int target_w = target_size_[1];
  float ratio_h = static_cast<float>(target_h) / static_cast<float>(origin_h);
  float ratio_w = static_cast<float>(target_w) / static_cast<float>(origin_w);
  float resize_scale = std::min(ratio_h, ratio_w);
  // get_resized_shape
  int new_shape_w = std::round(origin_w * resize_scale);
  int new_shape_h = std::round(origin_h * resize_scale);
  // calculate pad
  float padw = (target_size_[1] - new_shape_w) / 2.;
  float padh = (target_size_[0] - new_shape_h) / 2.;
  int top = std::round(padh - 0.1);
  int bottom = std::round(padh + 0.1);
  int left = std::round(padw - 0.1);
  int right = std::round(padw + 0.1);
  Resize::Run(mat, new_shape_w, new_shape_h, -1.0, -1.0, 3, false,
              ProcLib::CUDA);
  Pad::Run(mat, top, bottom, left, right, color_, ProcLib::CUDA);
  return true;
}
#endif

bool LetterBoxResize::Run(Mat *mat, const std::vector<int> &target_size,
                          const std::vector<float> &color, ProcLib lib) {
  auto l = LetterBoxResize(target_size, color);
  return l(mat, lib);
}

} // namespace vision
} // namespace ultra_infer
