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

#include "ultra_infer/vision/common/processors/stride_pad.h"

namespace ultra_infer {
namespace vision {

bool StridePad::ImplByOpenCV(Mat *mat) {
  if (mat->layout != Layout::HWC) {
    FDERROR << "StridePad: The input data must be Layout::HWC format!"
            << std::endl;
    return false;
  }
  if (mat->Channels() > 4) {
    FDERROR << "StridePad: Only support channels <= 4." << std::endl;
    return false;
  }
  if (mat->Channels() != value_.size()) {
    FDERROR
        << "StridePad: Require input channels equals to size of padding value, "
           "but now channels = "
        << mat->Channels() << ", the size of padding values = " << value_.size()
        << "." << std::endl;
    return false;
  }
  int origin_w = mat->Width();
  int origin_h = mat->Height();

  int pad_h = (mat->Height() / stride_) * stride_ +
              (mat->Height() % stride_ != 0) * stride_ - mat->Height();
  int pad_w = (mat->Width() / stride_) * stride_ +
              (mat->Width() % stride_ != 0) * stride_ - mat->Width();
  if (pad_h == 0 && pad_w == 0) {
    return true;
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
  // top, bottom, left, right
  cv::copyMakeBorder(*im, *im, 0, pad_h, 0, pad_w, cv::BORDER_CONSTANT, value);
  mat->SetHeight(origin_h + pad_h);
  mat->SetWidth(origin_w + pad_w);
  return true;
}

#ifdef ENABLE_FLYCV
bool StridePad::ImplByFlyCV(Mat *mat) {
  if (mat->layout != Layout::HWC) {
    FDERROR << "StridePad: The input data must be Layout::HWC format!"
            << std::endl;
    return false;
  }
  if (mat->Channels() > 4) {
    FDERROR << "StridePad: Only support channels <= 4." << std::endl;
    return false;
  }
  if (mat->Channels() != value_.size()) {
    FDERROR
        << "StridePad: Require input channels equals to size of padding value, "
           "but now channels = "
        << mat->Channels() << ", the size of padding values = " << value_.size()
        << "." << std::endl;
    return false;
  }
  int origin_w = mat->Width();
  int origin_h = mat->Height();

  int pad_h = (mat->Height() / stride_) * stride_ +
              (mat->Height() % stride_ != 0) * stride_ - mat->Height();
  int pad_w = (mat->Width() / stride_) * stride_ +
              (mat->Width() % stride_ != 0) * stride_ - mat->Width();
  if (pad_h == 0 && pad_w == 0) {
    return true;
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
  // top, bottom, left, right
  fcv::copy_make_border(*im, new_im, 0, pad_h, 0, pad_w,
                        fcv::BorderType::BORDER_CONSTANT, value);
  mat->SetMat(new_im);
  mat->SetHeight(new_im.height());
  mat->SetWidth(new_im.width());
  return true;
}
#endif

#ifdef ENABLE_CVCUDA
bool StridePad::ImplByCvCuda(FDMat *mat) {
  if (mat->layout != Layout::HWC) {
    FDERROR << "StridePad: The input data must be Layout::HWC format!"
            << std::endl;
    return false;
  }
  if (mat->Channels() > 4) {
    FDERROR << "StridePad: Only support channels <= 4." << std::endl;
    return false;
  }
  if (mat->Channels() != value_.size()) {
    FDERROR
        << "StridePad: Require input channels equals to size of padding value, "
           "but now channels = "
        << mat->Channels() << ", the size of padding values = " << value_.size()
        << "." << std::endl;
    return false;
  }
  int origin_w = mat->Width();
  int origin_h = mat->Height();

  int pad_h = (mat->Height() / stride_) * stride_ +
              (mat->Height() % stride_ != 0) * stride_ - mat->Height();
  int pad_w = (mat->Width() / stride_) * stride_ +
              (mat->Width() % stride_ != 0) * stride_ - mat->Width();
  if (pad_h == 0 && pad_w == 0) {
    return true;
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

  int height = mat->Height() + pad_h;
  int width = mat->Width() + pad_w;

  // Prepare output tensor
  mat->output_cache->Resize({height, width, mat->Channels()}, mat->Type(),
                            "output_cache", Device::GPU);
  auto dst_tensor = CreateCvCudaTensorWrapData(*(mat->output_cache));

  cvcuda_pad_op_(mat->Stream(), *src_tensor, *dst_tensor, 0, 0,
                 NVCV_BORDER_CONSTANT, value);

  mat->SetTensor(mat->output_cache);
  mat->mat_type = ProcLib::CVCUDA;
  return true;
}
#endif

bool StridePad::Run(Mat *mat, int stride, const std::vector<float> &value,
                    ProcLib lib) {
  auto p = StridePad(stride, value);
  return p(mat, lib);
}

} // namespace vision
} // namespace ultra_infer
