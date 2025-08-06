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

#include "ultra_infer/vision/common/processors/resize.h"

namespace ultra_infer {
namespace vision {

bool Resize::ImplByOpenCV(FDMat *mat) {
  if (mat->layout != Layout::HWC) {
    FDERROR << "Resize: The format of input is not HWC." << std::endl;
    return false;
  }
  cv::Mat *im = mat->GetOpenCVMat();
  int origin_w = im->cols;
  int origin_h = im->rows;

  if (width_ == origin_w && height_ == origin_h) {
    return true;
  }
  if (fabs(scale_w_ - 1.0) < 1e-06 && fabs(scale_h_ - 1.0) < 1e-06) {
    return true;
  }

  if (width_ > 0 && height_ > 0) {
    if (use_scale_) {
      float scale_w = width_ * 1.0 / origin_w;
      float scale_h = height_ * 1.0 / origin_h;
      cv::resize(*im, *im, cv::Size(0, 0), scale_w, scale_h, interp_);
    } else {
      cv::resize(*im, *im, cv::Size(width_, height_), 0, 0, interp_);
    }
  } else if (scale_w_ > 0 && scale_h_ > 0) {
    cv::resize(*im, *im, cv::Size(0, 0), scale_w_, scale_h_, interp_);
  } else {
    FDERROR << "Resize: the parameters must satisfy (width > 0 && height > 0) "
               "or (scale_w > 0 && scale_h > 0)."
            << std::endl;
    return false;
  }
  mat->SetWidth(im->cols);
  mat->SetHeight(im->rows);
  return true;
}

#ifdef ENABLE_FLYCV
bool Resize::ImplByFlyCV(FDMat *mat) {
  if (mat->layout != Layout::HWC) {
    FDERROR << "Resize: The format of input is not HWC." << std::endl;
    return false;
  }
  fcv::Mat *im = mat->GetFlyCVMat();
  int origin_w = im->width();
  int origin_h = im->height();

  if (width_ == origin_w && height_ == origin_h) {
    return true;
  }
  if (fabs(scale_w_ - 1.0) < 1e-06 && fabs(scale_h_ - 1.0) < 1e-06) {
    return true;
  }

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
    FDERROR << "Resize: Only support interp_ be 0/1/2/3 with FlyCV, but "
               "now it's "
            << interp_ << "." << std::endl;
    return false;
  }

  if (width_ > 0 && height_ > 0) {
    fcv::Mat new_im;
    if (use_scale_) {
      float scale_w = width_ * 1.0 / origin_w;
      float scale_h = height_ * 1.0 / origin_h;
      fcv::resize(*im, new_im, fcv::Size(), scale_w, scale_h, interp_method);
    } else {
      fcv::resize(*im, new_im, fcv::Size(width_, height_), 0, 0, interp_method);
    }
    mat->SetMat(new_im);
    mat->SetWidth(new_im.width());
    mat->SetHeight(new_im.height());
  } else if (scale_w_ > 0 && scale_h_ > 0) {
    fcv::Mat new_im;
    fcv::resize(*im, new_im, fcv::Size(0, 0), scale_w_, scale_h_,
                interp_method);
    mat->SetMat(new_im);
    mat->SetWidth(new_im.width());
    mat->SetHeight(new_im.height());
  } else {
    FDERROR << "Resize: the parameters must satisfy (width > 0 && height > 0) "
               "or (scale_w > 0 && scale_h > 0)."
            << std::endl;
    return false;
  }
  return true;
}
#endif

#ifdef ENABLE_CVCUDA
bool Resize::ImplByCvCuda(FDMat *mat) {
  if (width_ == mat->Width() && height_ == mat->Height()) {
    return true;
  }
  if (fabs(scale_w_ - 1.0) < 1e-06 && fabs(scale_h_ - 1.0) < 1e-06) {
    return true;
  }

  if (width_ > 0 && height_ > 0) {
  } else if (scale_w_ > 0 && scale_h_ > 0) {
    width_ = std::round(scale_w_ * mat->Width());
    height_ = std::round(scale_h_ * mat->Height());
  } else {
    FDERROR << "Resize: the parameters must satisfy (width > 0 && height > 0) "
               "or (scale_w > 0 && scale_h > 0)."
            << std::endl;
    return false;
  }

  // Prepare input tensor
  FDTensor *src = CreateCachedGpuInputTensor(mat);
  auto src_tensor = CreateCvCudaTensorWrapData(*src);

  // Prepare output tensor
  mat->output_cache->Resize({height_, width_, mat->Channels()}, mat->Type(),
                            "output_cache", Device::GPU);
  auto dst_tensor = CreateCvCudaTensorWrapData(*(mat->output_cache));

  // CV-CUDA Interp value is compatible with OpenCV
  cvcuda_resize_op_(mat->Stream(), *src_tensor, *dst_tensor,
                    CreateCvCudaInterp(interp_));

  mat->SetTensor(mat->output_cache);
  mat->SetWidth(width_);
  mat->SetHeight(height_);
  mat->device = Device::GPU;
  mat->mat_type = ProcLib::CVCUDA;
  return true;
}
#endif

bool Resize::Run(FDMat *mat, int width, int height, float scale_w,
                 float scale_h, int interp, bool use_scale, ProcLib lib) {
  if (mat->Height() == height && mat->Width() == width) {
    return true;
  }
  auto r = Resize(width, height, scale_w, scale_h, interp, use_scale);
  return r(mat, lib);
}

} // namespace vision
} // namespace ultra_infer
