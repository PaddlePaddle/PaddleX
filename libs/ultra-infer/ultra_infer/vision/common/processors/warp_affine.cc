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

#include "ultra_infer/vision/common/processors/warp_affine.h"

namespace ultra_infer {
namespace vision {

bool WarpAffine::ImplByOpenCV(Mat *mat) {
  if (mat->layout != Layout::HWC) {
    FDERROR << "WarpAffine: The format of input is not HWC." << std::endl;
    return false;
  }
  cv::Mat *im = mat->GetOpenCVMat();
  if (width_ > 0 && height_ > 0) {
    cv::warpAffine(*im, *im, trans_matrix_, cv::Size(width_, height_), interp_,
                   border_mode_, borderValue_);
  } else {
    FDERROR
        << "WarpAffine: the parameters must satisfy (width > 0 && height > 0) ."
        << std::endl;
    return false;
  }
  mat->SetWidth(im->cols);
  mat->SetHeight(im->rows);

  return true;
}

bool WarpAffine::Run(Mat *mat, const cv::Mat &trans_matrix, int width,
                     int height, int interp, int border_mode,
                     const cv::Scalar &borderValue, ProcLib lib) {
  auto r =
      WarpAffine(trans_matrix, width, height, interp, border_mode, borderValue);
  return r(mat, lib);
}

} // namespace vision
} // namespace ultra_infer
