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

#include "ultra_infer/vision/common/processors/center_crop.h"

namespace ultra_infer {
namespace vision {

bool CenterCrop::ImplByOpenCV(FDMat *mat) {
  cv::Mat *im = mat->GetOpenCVMat();
  int height = static_cast<int>(im->rows);
  int width = static_cast<int>(im->cols);
  if (height < height_ || width < width_) {
    FDERROR << "[CenterCrop] Image size less than crop size" << std::endl;
    return false;
  }
  int offset_x = static_cast<int>((width - width_) / 2);
  int offset_y = static_cast<int>((height - height_) / 2);
  cv::Rect crop_roi(offset_x, offset_y, width_, height_);
  cv::Mat new_im = (*im)(crop_roi).clone();
  mat->SetMat(new_im);
  mat->SetWidth(width_);
  mat->SetHeight(height_);
  return true;
}

#ifdef ENABLE_FLYCV
bool CenterCrop::ImplByFlyCV(FDMat *mat) {
  fcv::Mat *im = mat->GetFlyCVMat();
  int height = static_cast<int>(im->height());
  int width = static_cast<int>(im->width());
  if (height < height_ || width < width_) {
    FDERROR << "[CenterCrop] Image size less than crop size" << std::endl;
    return false;
  }
  int offset_x = static_cast<int>((width - width_) / 2);
  int offset_y = static_cast<int>((height - height_) / 2);
  fcv::Rect crop_roi(offset_x, offset_y, width_, height_);
  fcv::Mat new_im;
  fcv::crop(*im, new_im, crop_roi);
  mat->SetMat(new_im);
  mat->SetWidth(width_);
  mat->SetHeight(height_);
  return true;
}
#endif

#ifdef ENABLE_CVCUDA
bool CenterCrop::ImplByCvCuda(FDMat *mat) {
  // Prepare input tensor
  FDTensor *src = CreateCachedGpuInputTensor(mat);
  auto src_tensor = CreateCvCudaTensorWrapData(*src);

  // Prepare output tensor
  mat->output_cache->Resize({height_, width_, mat->Channels()}, src->Dtype(),
                            "output_cache", Device::GPU);
  auto dst_tensor = CreateCvCudaTensorWrapData(*(mat->output_cache));

  int offset_x = static_cast<int>((mat->Width() - width_) / 2);
  int offset_y = static_cast<int>((mat->Height() - height_) / 2);
  NVCVRectI crop_roi = {offset_x, offset_y, width_, height_};
  cvcuda_crop_op_(mat->Stream(), *src_tensor, *dst_tensor, crop_roi);

  mat->SetTensor(mat->output_cache);
  mat->SetWidth(width_);
  mat->SetHeight(height_);
  mat->device = Device::GPU;
  mat->mat_type = ProcLib::CVCUDA;
  return true;
}

bool CenterCrop::ImplByCvCuda(FDMatBatch *mat_batch) {
  for (size_t i = 0; i < mat_batch->mats->size(); ++i) {
    if (ImplByCvCuda(&((*(mat_batch->mats))[i])) != true) {
      return false;
    }
  }
  mat_batch->device = Device::GPU;
  mat_batch->mat_type = ProcLib::CVCUDA;
  return true;
}
#endif

bool CenterCrop::Run(FDMat *mat, const int &width, const int &height,
                     ProcLib lib) {
  auto c = CenterCrop(width, height);
  return c(mat, lib);
}

} // namespace vision
} // namespace ultra_infer
