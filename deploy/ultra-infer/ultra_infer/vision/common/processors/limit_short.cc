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

#include "ultra_infer/vision/common/processors/limit_short.h"

namespace ultra_infer {
namespace vision {

bool LimitShort::ImplByOpenCV(Mat *mat) {
  cv::Mat *im = mat->GetOpenCVMat();
  int origin_w = im->cols;
  int origin_h = im->rows;
  int im_size_min = std::min(origin_w, origin_h);
  int target = im_size_min;
  if (max_short_ > 0 && im_size_min > max_short_) {
    target = max_short_;
  } else if (min_short_ > 0 && im_size_min < min_short_) {
    target = min_short_;
  }
  double scale = -1.f;
  if (target != im_size_min) {
    scale = static_cast<double>(target) / static_cast<double>(im_size_min);
  }
  if (fabs(scale - 1.0) > 1e-06) {
    cv::resize(*im, *im, cv::Size(), scale, scale, interp_);
    mat->SetWidth(im->cols);
    mat->SetHeight(im->rows);
  }
  return true;
}

#ifdef ENABLE_FLYCV
bool LimitShort::ImplByFlyCV(Mat *mat) {
  fcv::Mat *im = mat->GetFlyCVMat();
  int origin_w = im->width();
  int origin_h = im->height();
  int im_size_min = std::min(origin_w, origin_h);
  int target = im_size_min;
  if (max_short_ > 0 && im_size_min > max_short_) {
    target = max_short_;
  } else if (min_short_ > 0 && im_size_min < min_short_) {
    target = min_short_;
  }
  double scale = -1.f;
  if (target != im_size_min) {
    scale = static_cast<double>(target) / static_cast<double>(im_size_min);
  }
  if (fabs(scale - 1.0) > 1e-06) {
    auto interp_method = fcv::InterpolationType::INTER_LINEAR;
    if (interp_ == 0) {
      interp_method = fcv::InterpolationType::INTER_NEAREST;
    } else if (interp_ == 1) {
      interp_method = fcv::InterpolationType::INTER_LINEAR;
    } else if (interp_ == 2) {
      interp_method = fcv::InterpolationType::INTER_CUBIC;
    } else if (interp_ == 3) {
      interp_method = fcv::InterpolationType::INTER_AREA;
    } else {
      FDERROR
          << "LimitByShort: Only support interp_ be 0/1/2/3 with FlyCV, but "
             "now it's "
          << interp_ << "." << std::endl;
      return false;
    }

    fcv::Mat new_im;
    fcv::resize(*im, new_im, fcv::Size(), scale, scale, interp_method);
    mat->SetMat(new_im);
    mat->SetWidth(new_im.width());
    mat->SetHeight(new_im.height());
  }
  return true;
}
#endif

bool LimitShort::Run(Mat *mat, int max_short, int min_short, int interp,
                     ProcLib lib) {
  auto l = LimitShort(max_short, min_short, interp);
  return l(mat, lib);
}
} // namespace vision
} // namespace ultra_infer
