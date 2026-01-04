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

#include "ultra_infer/vision/ocr/ppocr/cls_preprocessor.h"

#include "ultra_infer/function/concat.h"
#include "ultra_infer/utils/perf.h"
#include "ultra_infer/vision/ocr/ppocr/utils/ocr_utils.h"

namespace ultra_infer {
namespace vision {
namespace ocr {

ClassifierPreprocessor::ClassifierPreprocessor() {
  resize_op_ = std::make_shared<Resize>(-1, -1);

  std::vector<float> value = {0, 0, 0};
  pad_op_ = std::make_shared<Pad>(0, 0, 0, 0, value);

  normalize_op_ =
      std::make_shared<Normalize>(std::vector<float>({0.5f, 0.5f, 0.5f}),
                                  std::vector<float>({0.5f, 0.5f, 0.5f}), true);
  hwc2chw_op_ = std::make_shared<HWC2CHW>();
}

void ClassifierPreprocessor::OcrClassifierResizeImage(
    FDMat *mat, const std::vector<int> &cls_image_shape) {
  int img_c = cls_image_shape[0];
  int img_h = cls_image_shape[1];
  int img_w = cls_image_shape[2];

  float ratio = float(mat->Width()) / float(mat->Height());

  int resize_w;
  if (ceilf(img_h * ratio) > img_w)
    resize_w = img_w;
  else
    resize_w = int(ceilf(img_h * ratio));

  resize_op_->SetWidthAndHeight(resize_w, img_h);
  (*resize_op_)(mat);
}

bool ClassifierPreprocessor::Run(std::vector<FDMat> *images,
                                 std::vector<FDTensor> *outputs,
                                 size_t start_index, size_t end_index) {
  if (images->size() == 0 || start_index < 0 || end_index <= start_index ||
      end_index > images->size()) {
    FDERROR << "images->size() or index error. Correct is: 0 <= start_index < "
               "end_index <= images->size()"
            << std::endl;
    return false;
  }

  std::vector<FDMat> mats(end_index - start_index);
  for (size_t i = start_index; i < end_index; ++i) {
    mats[i - start_index] = images->at(i);
  }
  return Run(&mats, outputs);
}

bool ClassifierPreprocessor::Apply(FDMatBatch *image_batch,
                                   std::vector<FDTensor> *outputs) {
  for (size_t i = 0; i < image_batch->mats->size(); ++i) {
    FDMat *mat = &(image_batch->mats->at(i));
    OcrClassifierResizeImage(mat, cls_image_shape_);
    if (!disable_normalize_) {
      (*normalize_op_)(mat);
    }
    std::vector<float> value = {0, 0, 0};
    if (mat->Width() < cls_image_shape_[2]) {
      pad_op_->SetPaddingSize(0, 0, 0, cls_image_shape_[2] - mat->Width());
      (*pad_op_)(mat);
    }
    if (!disable_permute_) {
      (*hwc2chw_op_)(mat);
    }
  }
  // Only have 1 output tensor.
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
