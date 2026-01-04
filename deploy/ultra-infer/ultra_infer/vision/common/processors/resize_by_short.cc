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

#include "ultra_infer/vision/common/processors/resize_by_short.h"

namespace ultra_infer {
namespace vision {

bool ResizeByShort::ImplByOpenCV(FDMat *mat) {
  cv::Mat *im = mat->GetOpenCVMat();
  int origin_w = im->cols;
  int origin_h = im->rows;
  double scale = GenerateScale(origin_w, origin_h);
  if (use_scale_ && fabs(scale - 1.0) >= 1e-06) {
    cv::resize(*im, *im, cv::Size(), scale, scale, interp_);
  } else {
    int width = static_cast<int>(round(scale * im->cols));
    int height = static_cast<int>(round(scale * im->rows));
    if (width != origin_w || height != origin_h) {
      cv::resize(*im, *im, cv::Size(width, height), 0, 0, interp_);
    }
  }
  mat->SetWidth(im->cols);
  mat->SetHeight(im->rows);
  return true;
}

#ifdef ENABLE_FLYCV
bool ResizeByShort::ImplByFlyCV(FDMat *mat) {
  fcv::Mat *im = mat->GetFlyCVMat();
  int origin_w = im->width();
  int origin_h = im->height();
  double scale = GenerateScale(origin_w, origin_h);

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
    FDERROR << "LimitByShort: Only support interp_ be 0/1/2/3 with FlyCV, but "
               "now it's "
            << interp_ << "." << std::endl;
    return false;
  }

  if (use_scale_ && fabs(scale - 1.0) >= 1e-06) {
    fcv::Mat new_im;
    fcv::resize(*im, new_im, fcv::Size(), scale, scale, interp_method);
    mat->SetMat(new_im);
    mat->SetHeight(new_im.height());
    mat->SetWidth(new_im.width());
  } else {
    int width = static_cast<int>(round(scale * im->width()));
    int height = static_cast<int>(round(scale * im->height()));
    if (width != origin_w || height != origin_h) {
      fcv::Mat new_im;
      fcv::resize(*im, new_im, fcv::Size(width, height), 0, 0, interp_method);
      mat->SetMat(new_im);
      mat->SetHeight(new_im.height());
      mat->SetWidth(new_im.width());
    }
  }
  return true;
}
#endif

#ifdef ENABLE_CVCUDA
bool ResizeByShort::ImplByCvCuda(FDMat *mat) {
  // Prepare input tensor
  FDTensor *src = CreateCachedGpuInputTensor(mat);
  auto src_tensor = CreateCvCudaTensorWrapData(*src);

  double scale = GenerateScale(mat->Width(), mat->Height());
  int width = static_cast<int>(round(scale * mat->Width()));
  int height = static_cast<int>(round(scale * mat->Height()));

  // Prepare output tensor
  mat->output_cache->Resize({height, width, mat->Channels()}, mat->Type(),
                            "output_cache", Device::GPU);
  auto dst_tensor = CreateCvCudaTensorWrapData(*(mat->output_cache));

  cvcuda_resize_op_(mat->Stream(), *src_tensor, *dst_tensor,
                    CreateCvCudaInterp(interp_));

  mat->SetTensor(mat->output_cache);
  mat->SetWidth(width);
  mat->SetHeight(height);
  mat->device = Device::GPU;
  mat->mat_type = ProcLib::CVCUDA;
  return true;
}

bool ResizeByShort::ImplByCvCuda(FDMatBatch *mat_batch) {
  // TODO(wangxinyu): to support batched tensor as input
  FDASSERT(mat_batch->has_batched_tensor == false,
           "ResizeByShort doesn't support batched tensor as input for now.");
  // Prepare input batch
  std::string tensor_name = Name() + "_cvcuda_src";
  std::vector<FDTensor *> src_tensors;
  for (size_t i = 0; i < mat_batch->mats->size(); ++i) {
    FDTensor *src = CreateCachedGpuInputTensor(&(*(mat_batch->mats))[i]);
    src_tensors.push_back(src);
  }
  nvcv::ImageBatchVarShape src_batch(mat_batch->mats->size());
  CreateCvCudaImageBatchVarShape(src_tensors, src_batch);

  // Prepare output batch
  tensor_name = Name() + "_cvcuda_dst";
  std::vector<FDTensor *> dst_tensors;
  for (size_t i = 0; i < mat_batch->mats->size(); ++i) {
    FDMat *mat = &(*(mat_batch->mats))[i];
    double scale = GenerateScale(mat->Width(), mat->Height());
    int width = static_cast<int>(round(scale * mat->Width()));
    int height = static_cast<int>(round(scale * mat->Height()));
    mat->output_cache->Resize({height, width, mat->Channels()}, mat->Type(),
                              "output_cache", Device::GPU);
    dst_tensors.push_back(mat->output_cache);
  }
  nvcv::ImageBatchVarShape dst_batch(mat_batch->mats->size());
  CreateCvCudaImageBatchVarShape(dst_tensors, dst_batch);

  // CV-CUDA Interp value is compatible with OpenCV
  cvcuda_resize_op_(mat_batch->Stream(), src_batch, dst_batch,
                    CreateCvCudaInterp(interp_));

  for (size_t i = 0; i < mat_batch->mats->size(); ++i) {
    FDMat *mat = &(*(mat_batch->mats))[i];
    mat->SetTensor(dst_tensors[i]);
    mat->SetWidth(dst_tensors[i]->Shape()[1]);
    mat->SetHeight(dst_tensors[i]->Shape()[0]);
    mat->device = Device::GPU;
    mat->mat_type = ProcLib::CVCUDA;
  }
  mat_batch->device = Device::GPU;
  mat_batch->mat_type = ProcLib::CVCUDA;
  return true;
}
#endif

double ResizeByShort::GenerateScale(const int origin_w, const int origin_h) {
  int im_size_max = std::max(origin_w, origin_h);
  int im_size_min = std::min(origin_w, origin_h);
  double scale =
      static_cast<double>(target_size_) / static_cast<double>(im_size_min);

  if (max_hw_.size() > 0) {
    FDASSERT(max_hw_.size() == 2,
             "Require size of max_hw_ be 2, but now it's %zu.", max_hw_.size());
    FDASSERT(
        max_hw_[0] > 0 && max_hw_[1] > 0,
        "Require elements in max_hw_ greater than 0, but now it's [%d, %d].",
        max_hw_[0], max_hw_[1]);

    double scale_h =
        static_cast<double>(max_hw_[0]) / static_cast<double>(origin_h);
    double scale_w =
        static_cast<double>(max_hw_[1]) / static_cast<double>(origin_w);
    double min_scale = std::min(scale_h, scale_w);
    if (min_scale < scale) {
      scale = min_scale;
    }
  }
  return scale;
}

bool ResizeByShort::Run(FDMat *mat, int target_size, int interp, bool use_scale,
                        const std::vector<int> &max_hw, ProcLib lib) {
  auto r = ResizeByShort(target_size, interp, use_scale, max_hw);
  return r(mat, lib);
}
} // namespace vision
} // namespace ultra_infer
