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

#include "ultra_infer/vision/common/processors/pad.h"

namespace ultra_infer {
namespace vision {

bool Pad::ImplByOpenCV(Mat *mat) {
  if (mat->layout != Layout::HWC) {
    FDERROR << "Pad: The input data must be Layout::HWC format!" << std::endl;
    return false;
  }
  if (mat->Channels() > 4) {
    FDERROR << "Pad: Only support channels <= 4." << std::endl;
    return false;
  }
  if (mat->Channels() != value_.size()) {
    FDERROR << "Pad: Require input channels equals to size of padding value, "
               "but now channels = "
            << mat->Channels()
            << ", the size of padding values = " << value_.size() << "."
            << std::endl;
    return false;
  }
  cv::Mat *im = mat->GetOpenCVMat();
  cv::Scalar value;
  if (value_.size() == 1) {
    value = cv::Scalar(value_[0]);
  } else if (value_.size() == 2) {
    value = cv::Scalar(value_[0], value_[1]);
  } else if (value_.size() == 3) {
    value = cv::Scalar(value_[0], value_[1], value_[2]);
  } else {
    value = cv::Scalar(value_[0], value_[1], value_[2], value_[3]);
  }
  cv::copyMakeBorder(*im, *im, top_, bottom_, left_, right_,
                     cv::BORDER_CONSTANT, value);
  mat->SetHeight(im->rows);
  mat->SetWidth(im->cols);
  return true;
}

#ifdef ENABLE_FLYCV
bool Pad::ImplByFlyCV(Mat *mat) {
  if (mat->layout != Layout::HWC) {
    FDERROR << "Pad: The input data must be Layout::HWC format!" << std::endl;
    return false;
  }
  if (mat->Channels() > 4) {
    FDERROR << "Pad: Only support channels <= 4." << std::endl;
    return false;
  }
  if (mat->Channels() != value_.size()) {
    FDERROR << "Pad: Require input channels equals to size of padding value, "
               "but now channels = "
            << mat->Channels()
            << ", the size of padding values = " << value_.size() << "."
            << std::endl;
    return false;
  }
  fcv::Mat *im = mat->GetFlyCVMat();
  fcv::Scalar value;
  if (value_.size() == 1) {
    value = fcv::Scalar(value_[0]);
  } else if (value_.size() == 2) {
    value = fcv::Scalar(value_[0], value_[1]);
  } else if (value_.size() == 3) {
    value = fcv::Scalar(value_[0], value_[1], value_[2]);
  } else {
    value = fcv::Scalar(value_[0], value_[1], value_[2], value_[3]);
  }
  fcv::Mat new_im;
  fcv::copy_make_border(*im, new_im, top_, bottom_, left_, right_,
                        fcv::BorderType::BORDER_CONSTANT, value);
  mat->SetMat(new_im);
  mat->SetHeight(new_im.height());
  mat->SetWidth(new_im.width());
  return true;
}
#endif

#ifdef ENABLE_CVCUDA
bool Pad::ImplByCvCuda(FDMat *mat) {
  if (mat->layout != Layout::HWC) {
    FDERROR << "Pad: The input data must be Layout::HWC format!" << std::endl;
    return false;
  }
  if (mat->Channels() > 4) {
    FDERROR << "Pad: Only support channels <= 4." << std::endl;
    return false;
  }
  if (mat->Channels() != value_.size()) {
    FDERROR << "Pad: Require input channels equals to size of padding value, "
               "but now channels = "
            << mat->Channels()
            << ", the size of padding values = " << value_.size() << "."
            << std::endl;
    return false;
  }

  float4 value;
  if (value_.size() == 1) {
    value = make_float4(value_[0], 0.0f, 0.0f, 0.0f);
  } else if (value_.size() == 2) {
    value = make_float4(value_[0], value_[1], 0.0f, 0.0f);
  } else if (value_.size() == 3) {
    value = make_float4(value_[0], value_[1], value_[2], 0.0f);
  } else {
    value = make_float4(value_[0], value_[1], value_[2], value_[3]);
  }

  // Prepare input tensor
  FDTensor *src = CreateCachedGpuInputTensor(mat);
  auto src_tensor = CreateCvCudaTensorWrapData(*src);

  int height = mat->Height() + top_ + bottom_;
  int width = mat->Width() + left_ + right_;

  // Prepare output tensor
  mat->output_cache->Resize({height, width, mat->Channels()}, mat->Type(),
                            "output_cache", Device::GPU);
  auto dst_tensor = CreateCvCudaTensorWrapData(*(mat->output_cache));

  cvcuda_pad_op_(mat->Stream(), *src_tensor, *dst_tensor, top_, left_,
                 NVCV_BORDER_CONSTANT, value);

  mat->SetTensor(mat->output_cache);
  mat->mat_type = ProcLib::CVCUDA;
  return true;
}
#endif

bool Pad::Run(Mat *mat, const int &top, const int &bottom, const int &left,
              const int &right, const std::vector<float> &value, ProcLib lib) {
  auto p = Pad(top, bottom, left, right, value);
  return p(mat, lib);
}

} // namespace vision
} // namespace ultra_infer
