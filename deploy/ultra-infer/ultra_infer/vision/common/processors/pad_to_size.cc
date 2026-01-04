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

#include "ultra_infer/vision/common/processors/pad_to_size.h"

#include "ultra_infer/vision/common/processors/utils.h"

namespace ultra_infer {
namespace vision {

static bool PadHWCByOpenCV(FDMat *mat, int width, int height,
                           const std::vector<float> &value) {
  int origin_w = mat->Width();
  int origin_h = mat->Height();
  cv::Mat *im = mat->GetOpenCVMat();
  cv::Scalar scalar;
  if (value.size() == 1) {
    scalar = cv::Scalar(value[0]);
  } else if (value.size() == 2) {
    scalar = cv::Scalar(value[0], value[1]);
  } else if (value.size() == 3) {
    scalar = cv::Scalar(value[0], value[1], value[2]);
  } else {
    scalar = cv::Scalar(value[0], value[1], value[2], value[3]);
  }
  // top, bottom, left, right
  cv::copyMakeBorder(*im, *im, 0, height - origin_h, 0, width - origin_w,
                     cv::BORDER_CONSTANT, scalar);
  mat->SetHeight(height);
  mat->SetWidth(width);
  return true;
}

static bool PadCHWByOpenCV(FDMat *mat, int width, int height,
                           const std::vector<float> &value) {
  int origin_w = mat->Width();
  int origin_h = mat->Height();
  cv::Mat *im = mat->GetOpenCVMat();
  cv::Mat new_im(height, width,
                 CreateOpenCVDataType(mat->Type(), mat->Channels()));

  for (int i = 0; i < mat->Channels(); ++i) {
    uint8_t *src_data =
        im->ptr() + i * origin_w * origin_h * FDDataTypeSize(mat->Type());
    cv::Mat src(origin_h, origin_w, CreateOpenCVDataType(mat->Type(), 1),
                src_data);

    uint8_t *dst_data =
        new_im.ptr() + i * width * height * FDDataTypeSize(mat->Type());
    cv::Mat dst(height, width, CreateOpenCVDataType(mat->Type(), 1), dst_data);

    cv::copyMakeBorder(src, dst, 0, height - origin_h, 0, width - origin_w,
                       cv::BORDER_CONSTANT, cv::Scalar(value[i]));
  }
  mat->SetMat(new_im);
  mat->SetHeight(height);
  mat->SetWidth(width);
  return true;
}

bool PadToSize::CheckArgs(FDMat *mat) {
  if (mat->Channels() > 4) {
    FDERROR << "PadToSize: Only support channels <= 4." << std::endl;
    return false;
  }
  if (mat->Channels() != value_.size()) {
    FDERROR
        << "PadToSize: Require input channels equals to size of padding value, "
           "but now channels = "
        << mat->Channels() << ", the size of padding values = " << value_.size()
        << "." << std::endl;
    return false;
  }
  if (mat->Width() > width_) {
    FDERROR << "PadToSize: the input width:" << mat->Width()
            << " is greater than the target width: " << width_ << "."
            << std::endl;
    return false;
  }
  if (mat->Height() > height_) {
    FDERROR << "PadToSize: the input height:" << mat->Height()
            << " is greater than the target height: " << height_ << "."
            << std::endl;
    return false;
  }
  return true;
}

bool PadToSize::ImplByOpenCV(FDMat *mat) {
  if (width_ == -1 || height_ == -1 ||
      (mat->Width() == width_ && mat->Height() == height_)) {
    return true;
  }
  if (CheckArgs(mat) == false) {
    return false;
  }
  if (mat->layout == Layout::HWC) {
    return PadHWCByOpenCV(mat, width_, height_, value_);
  } else if (mat->layout == Layout::CHW) {
    return PadCHWByOpenCV(mat, width_, height_, value_);
  }
  return false;
}

#ifdef ENABLE_FLYCV
static bool PadHWCByFlyCV(FDMat *mat, int width, int height,
                          const std::vector<float> &value) {
  int origin_w = mat->Width();
  int origin_h = mat->Height();
  fcv::Mat *im = mat->GetFlyCVMat();
  fcv::Scalar scalar;
  if (value.size() == 1) {
    scalar = fcv::Scalar(value[0]);
  } else if (value.size() == 2) {
    scalar = fcv::Scalar(value[0], value[1]);
  } else if (value.size() == 3) {
    scalar = fcv::Scalar(value[0], value[1], value[2]);
  } else {
    scalar = fcv::Scalar(value[0], value[1], value[2], value[3]);
  }
  fcv::Mat new_im;
  // top, bottom, left, right
  fcv::copy_make_border(*im, new_im, 0, height - origin_h, 0, width - origin_w,
                        fcv::BorderType::BORDER_CONSTANT, scalar);
  mat->SetMat(new_im);
  mat->SetHeight(height);
  mat->SetWidth(width);
  return true;
}

static bool PadCHWByFlyCV(FDMat *mat, int width, int height,
                          const std::vector<float> &value) {
  int origin_w = mat->Width();
  int origin_h = mat->Height();
  fcv::Mat new_im(height, width,
                  CreateFlyCVDataType(mat->Type(), mat->Channels()));
  for (int i = 0; i < mat->Channels(); ++i) {
    uint8_t *src_data = reinterpret_cast<uint8_t *>(mat->Data()) +
                        i * origin_w * origin_h * FDDataTypeSize(mat->Type());
    fcv::Mat src(origin_h, origin_w, CreateFlyCVDataType(mat->Type(), 1),
                 src_data);

    uint8_t *dst_data = reinterpret_cast<uint8_t *>(new_im.data()) +
                        i * width * height * FDDataTypeSize(mat->Type());
    fcv::Mat dst(height, width, CreateFlyCVDataType(mat->Type(), 1), dst_data);

    fcv::copy_make_border(src, dst, 0, height - origin_h, 0, width - origin_w,
                          fcv::BorderType::BORDER_CONSTANT,
                          fcv::Scalar(value[i]));
  }
  mat->SetMat(new_im);
  mat->SetHeight(height);
  mat->SetWidth(width);
  return true;
}

bool PadToSize::ImplByFlyCV(FDMat *mat) {
  if (width_ == -1 || height_ == -1 ||
      (mat->Width() == width_ && mat->Height() == height_)) {
    return true;
  }
  if (CheckArgs(mat) == false) {
    return false;
  }
  if (mat->layout == Layout::HWC) {
    return PadHWCByFlyCV(mat, width_, height_, value_);
  } else if (mat->layout == Layout::CHW) {
    return PadCHWByFlyCV(mat, width_, height_, value_);
  }
  return false;
}
#endif

#ifdef ENABLE_CVCUDA
static bool PadHWCByCvCuda(cvcuda::CopyMakeBorder &pad_op, FDMat *mat,
                           int width, int height,
                           const std::vector<float> &value) {
  float4 border_value;
  if (value.size() == 1) {
    border_value = make_float4(value[0], 0.0f, 0.0f, 0.0f);
  } else if (value.size() == 2) {
    border_value = make_float4(value[0], value[1], 0.0f, 0.0f);
  } else if (value.size() == 3) {
    border_value = make_float4(value[0], value[1], value[2], 0.0f);
  } else {
    border_value = make_float4(value[0], value[1], value[2], value[3]);
  }

  // Prepare input tensor
  FDTensor *src = CreateCachedGpuInputTensor(mat);
  auto src_tensor = CreateCvCudaTensorWrapData(*src);

  // Prepare output tensor
  mat->output_cache->Resize({height, width, mat->Channels()}, mat->Type(),
                            "output_cache", Device::GPU);
  auto dst_tensor = CreateCvCudaTensorWrapData(*(mat->output_cache));

  pad_op(mat->Stream(), *src_tensor, *dst_tensor, 0, 0, NVCV_BORDER_CONSTANT,
         border_value);

  mat->SetTensor(mat->output_cache);
  mat->mat_type = ProcLib::CVCUDA;
  return true;
}

static bool PadCHWByCvCuda(cvcuda::CopyMakeBorder &pad_op, FDMat *mat,
                           int width, int height,
                           const std::vector<float> &value) {
  float4 border_value = make_float4(value[0], 0.0f, 0.0f, 0.0f);
  FDTensor *input = CreateCachedGpuInputTensor(mat);
  int channels = input->shape[0];
  mat->output_cache->Resize({channels, height, width}, mat->Type(),
                            "output_cache", Device::GPU);
  for (int i = 0; i < channels; ++i) {
    uint8_t *src_data =
        reinterpret_cast<uint8_t *>(input->Data()) +
        i * mat->Width() * mat->Height() * FDDataTypeSize(mat->Type());
    FDTensor src;
    src.SetExternalData({mat->Height(), mat->Width(), 1}, input->Dtype(),
                        src_data, input->device, input->device_id);
    auto src_tensor = CreateCvCudaTensorWrapData(src);

    uint8_t *dst_data = reinterpret_cast<uint8_t *>(mat->output_cache->Data()) +
                        i * width * height * FDDataTypeSize(mat->Type());
    FDTensor dst;
    dst.SetExternalData({height, width, 1}, input->Dtype(), dst_data,
                        input->device, input->device_id);
    auto dst_tensor = CreateCvCudaTensorWrapData(dst);

    pad_op(mat->Stream(), (*src_tensor), (*dst_tensor), 0, 0,
           NVCV_BORDER_CONSTANT, border_value);
  }
  mat->SetTensor(mat->output_cache);
  mat->mat_type = ProcLib::CVCUDA;
  return true;
}
bool PadToSize::ImplByCvCuda(FDMat *mat) {
  if (width_ == -1 || height_ == -1 ||
      (mat->Width() == width_ && mat->Height() == height_)) {
    return true;
  }
  if (CheckArgs(mat) == false) {
    return false;
  }
  if (mat->layout == Layout::HWC) {
    return PadHWCByCvCuda(cvcuda_pad_op_, mat, width_, height_, value_);
  } else if (mat->layout == Layout::CHW) {
    return PadCHWByCvCuda(cvcuda_pad_op_, mat, width_, height_, value_);
  }
  return false;
}
#endif

bool PadToSize::Run(Mat *mat, int width, int height,
                    const std::vector<float> &value, ProcLib lib) {
  auto p = PadToSize(width, height, value);
  return p(mat, lib);
}

} // namespace vision
} // namespace ultra_infer
