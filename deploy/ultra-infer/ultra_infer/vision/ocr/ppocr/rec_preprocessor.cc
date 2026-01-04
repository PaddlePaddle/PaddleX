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

#include "ultra_infer/vision/ocr/ppocr/rec_preprocessor.h"

#include "ultra_infer/function/concat.h"
#include "ultra_infer/utils/perf.h"
#include "ultra_infer/vision/ocr/ppocr/utils/ocr_utils.h"

namespace ultra_infer {
namespace vision {
namespace ocr {

RecognizerPreprocessor::RecognizerPreprocessor() {
  resize_op_ = std::make_shared<Resize>(-1, -1);

  std::vector<float> value = {127, 127, 127};
  pad_op_ = std::make_shared<Pad>(0, 0, 0, 0, value);

  std::vector<float> mean = {0.5f, 0.5f, 0.5f};
  std::vector<float> std = {0.5f, 0.5f, 0.5f};
  normalize_permute_op_ =
      std::make_shared<NormalizeAndPermute>(mean, std, true);
  normalize_op_ = std::make_shared<Normalize>(mean, std, true);
  hwc2chw_op_ = std::make_shared<HWC2CHW>();
  cast_op_ = std::make_shared<Cast>("float");
}

void RecognizerPreprocessor::OcrRecognizerResizeImage(
    FDMat *mat, float max_wh_ratio, const std::vector<int> &rec_image_shape,
    bool static_shape_infer) {
  int img_h, img_w;
  img_h = rec_image_shape[1];
  img_w = rec_image_shape[2];

  if (!static_shape_infer) {
    img_w = int(img_h * max_wh_ratio);
    float ratio = float(mat->Width()) / float(mat->Height());

    int resize_w;
    if (ceilf(img_h * ratio) > img_w) {
      resize_w = img_w;
    } else {
      resize_w = int(ceilf(img_h * ratio));
    }
    resize_op_->SetWidthAndHeight(resize_w, img_h);
    (*resize_op_)(mat);
    pad_op_->SetPaddingSize(0, 0, 0, int(img_w - mat->Width()));
    (*pad_op_)(mat);
  } else {
    if (mat->Width() >= img_w) {
      // Reszie W to 320
      resize_op_->SetWidthAndHeight(img_w, img_h);
      (*resize_op_)(mat);
    } else {
      resize_op_->SetWidthAndHeight(mat->Width(), img_h);
      (*resize_op_)(mat);
      // Pad to 320
      pad_op_->SetPaddingSize(0, 0, 0, int(img_w - mat->Width()));
      (*pad_op_)(mat);
    }
  }
}

bool RecognizerPreprocessor::Run(std::vector<FDMat> *images,
                                 std::vector<FDTensor> *outputs,
                                 size_t start_index, size_t end_index,
                                 const std::vector<int> &indices) {
  if (images->size() == 0 || end_index <= start_index ||
      end_index > images->size()) {
    FDERROR << "images->size() or index error. Correct is: 0 <= start_index < "
               "end_index <= images->size()"
            << std::endl;
    return false;
  }

  std::vector<FDMat> mats(end_index - start_index);
  for (size_t i = start_index; i < end_index; ++i) {
    size_t real_index = i;
    if (indices.size() != 0) {
      real_index = indices[i];
    }
    mats[i - start_index] = images->at(real_index);
  }
  return Run(&mats, outputs);
}

bool RecognizerPreprocessor::Apply(FDMatBatch *image_batch,
                                   std::vector<FDTensor> *outputs) {
  int img_h = rec_image_shape_[1];
  int img_w = rec_image_shape_[2];
  float max_wh_ratio = img_w * 1.0 / img_h;
  float ori_wh_ratio;

  for (size_t i = 0; i < image_batch->mats->size(); ++i) {
    FDMat *mat = &(image_batch->mats->at(i));
    ori_wh_ratio = mat->Width() * 1.0 / mat->Height();
    max_wh_ratio = std::max(max_wh_ratio, ori_wh_ratio);
  }

  for (size_t i = 0; i < image_batch->mats->size(); ++i) {
    FDMat *mat = &(image_batch->mats->at(i));
    OcrRecognizerResizeImage(mat, max_wh_ratio, rec_image_shape_,
                             static_shape_infer_);
  }

  if (!disable_normalize_ && !disable_permute_) {
    (*normalize_permute_op_)(image_batch);
  } else {
    if (!disable_normalize_) {
      (*normalize_op_)(image_batch);
    }
    if (!disable_permute_) {
      (*hwc2chw_op_)(image_batch);
      (*cast_op_)(image_batch);
    }
  }

  // Only have 1 output Tensor.
  outputs->resize(1);
  // Get the NCHW tensor
  FDTensor *tensor = image_batch->Tensor();
  (*outputs)[0].SetExternalData(tensor->Shape(), tensor->Dtype(),
                                tensor->Data(), tensor->device,
                                tensor->device_id);
  return true;
}

} // namespace ocr
} // namespace vision
} // namespace ultra_infer
