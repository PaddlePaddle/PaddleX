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

#include "ultra_infer/vision/common/processors/color_space_convert.h"

namespace ultra_infer {
namespace vision {
bool BGR2RGB::ImplByOpenCV(FDMat *mat) {
  cv::Mat *im = mat->GetOpenCVMat();
  cv::Mat new_im;
  cv::cvtColor(*im, new_im, cv::COLOR_BGR2RGB);
  mat->SetMat(new_im);
  return true;
}

#ifdef ENABLE_FLYCV
bool BGR2RGB::ImplByFlyCV(FDMat *mat) {
  fcv::Mat *im = mat->GetFlyCVMat();
  if (im->channels() != 3) {
    FDERROR << "[BGR2RGB] The channel of input image must be 3, but not it's "
            << im->channels() << "." << std::endl;
    return false;
  }
  fcv::Mat new_im;
  fcv::cvt_color(*im, new_im, fcv::ColorConvertType::CVT_PA_BGR2PA_RGB);
  mat->SetMat(new_im);
  return true;
}
#endif

bool RGB2BGR::ImplByOpenCV(FDMat *mat) {
  cv::Mat *im = mat->GetOpenCVMat();
  cv::Mat new_im;
  cv::cvtColor(*im, new_im, cv::COLOR_RGB2BGR);
  mat->SetMat(new_im);
  return true;
}

#ifdef ENABLE_FLYCV
bool RGB2BGR::ImplByFlyCV(FDMat *mat) {
  fcv::Mat *im = mat->GetFlyCVMat();
  if (im->channels() != 3) {
    FDERROR << "[RGB2BGR] The channel of input image must be 3, but not it's "
            << im->channels() << "." << std::endl;
    return false;
  }
  fcv::Mat new_im;
  fcv::cvt_color(*im, new_im, fcv::ColorConvertType::CVT_PA_RGB2PA_BGR);
  mat->SetMat(new_im);
  return true;
}
#endif

bool BGR2GRAY::ImplByOpenCV(FDMat *mat) {
  cv::Mat *im = mat->GetOpenCVMat();
  cv::Mat new_im;
  cv::cvtColor(*im, new_im, cv::COLOR_BGR2GRAY);
  mat->SetMat(new_im);
  mat->SetChannels(1);
  return true;
}

#ifdef ENABLE_FLYCV
bool BGR2GRAY::ImplByFlyCV(FDMat *mat) {
  fcv::Mat *im = mat->GetFlyCVMat();
  if (im->channels() != 3) {
    FDERROR << "[BGR2GRAY] The channel of input image must be 3, but not it's "
            << im->channels() << "." << std::endl;
    return false;
  }
  fcv::Mat new_im;
  fcv::cvt_color(*im, new_im, fcv::ColorConvertType::CVT_PA_BGR2GRAY);
  mat->SetMat(new_im);
  return true;
}
#endif

bool RGB2GRAY::ImplByOpenCV(FDMat *mat) {
  cv::Mat *im = mat->GetOpenCVMat();
  cv::Mat new_im;
  cv::cvtColor(*im, new_im, cv::COLOR_RGB2GRAY);
  mat->SetMat(new_im);
  return true;
}

#ifdef ENABLE_FLYCV
bool RGB2GRAY::ImplByFlyCV(FDMat *mat) {
  fcv::Mat *im = mat->GetFlyCVMat();
  if (im->channels() != 3) {
    FDERROR << "[RGB2GRAY] The channel of input image must be 3, but not it's "
            << im->channels() << "." << std::endl;
    return false;
  }
  fcv::Mat new_im;
  fcv::cvt_color(*im, new_im, fcv::ColorConvertType::CVT_PA_RGB2GRAY);
  mat->SetMat(new_im);
  return true;
}
#endif

bool BGR2RGB::Run(FDMat *mat, ProcLib lib) {
  auto b = BGR2RGB();
  return b(mat, lib);
}

bool RGB2BGR::Run(FDMat *mat, ProcLib lib) {
  auto r = RGB2BGR();
  return r(mat, lib);
}

bool BGR2GRAY::Run(FDMat *mat, ProcLib lib) {
  auto b = BGR2GRAY();
  return b(mat, lib);
}

bool RGB2GRAY::Run(FDMat *mat, ProcLib lib) {
  auto r = RGB2GRAY();
  return r(mat, lib);
}

} // namespace vision
} // namespace ultra_infer
