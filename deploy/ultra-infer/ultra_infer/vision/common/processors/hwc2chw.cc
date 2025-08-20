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

#include "ultra_infer/vision/common/processors/hwc2chw.h"

#include "ultra_infer/function/transpose.h"

namespace ultra_infer {
namespace vision {
bool HWC2CHW::ImplByOpenCV(Mat *mat) {
  if (mat->layout != Layout::HWC) {
    FDERROR << "HWC2CHW: The input data is not Layout::HWC format!"
            << std::endl;
    return false;
  }
  cv::Mat *im = mat->GetOpenCVMat();
  cv::Mat im_clone = im->clone();
  int rh = im->rows;
  int rw = im->cols;
  int rc = im->channels();

  for (int i = 0; i < rc; ++i) {
    cv::extractChannel(
        im_clone,
        cv::Mat(rh, rw, im->type() % 8,
                im->ptr() + i * rh * rw * FDDataTypeSize(mat->Type())),
        i);
  }
  mat->layout = Layout::CHW;
  return true;
}

#ifdef ENABLE_FLYCV
bool HWC2CHW::ImplByFlyCV(Mat *mat) {
  if (mat->layout != Layout::HWC) {
    FDERROR << "HWC2CHW: The input data is not Layout::HWC format!"
            << std::endl;
    return false;
  }
  if (mat->Type() != FDDataType::FP32) {
    FDERROR << "HWC2CHW: Only support float data while use FlyCV, but now it's "
            << mat->Type() << "." << std::endl;
    return false;
  }
  fcv::Mat *im = mat->GetFlyCVMat();
  fcv::Mat new_im;
  fcv::normalize_to_submean_to_reorder(*im, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0},
                                       std::vector<uint32_t>(), new_im, false);
  mat->SetMat(new_im);
  mat->layout = Layout::CHW;
  return true;
}
#endif

#ifdef ENABLE_CVCUDA
bool HWC2CHW::ImplByCvCuda(FDMat *mat) {
  // Prepare input tensor
  FDTensor *src = CreateCachedGpuInputTensor(mat);
  auto src_tensor = CreateCvCudaTensorWrapData(*src);

  // Prepare output tensor
  mat->output_cache->Resize({mat->Channels(), mat->Height(), mat->Width()},
                            src->Dtype(), "output_cache", Device::GPU);
  auto dst_tensor =
      CreateCvCudaTensorWrapData(*(mat->output_cache), Layout::CHW);

  cvcuda_reformat_op_(mat->Stream(), *src_tensor, *dst_tensor);

  mat->layout = Layout::CHW;
  mat->SetTensor(mat->output_cache);
  mat->mat_type = ProcLib::CVCUDA;
  return true;
}
#endif

bool HWC2CHW::Run(Mat *mat, ProcLib lib) {
  auto h = HWC2CHW();
  return h(mat, lib);
}

} // namespace vision
} // namespace ultra_infer
