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

#include "ultra_infer/vision/ocr/ppocr/structurev2_table_preprocessor.h"

#include "ultra_infer/function/concat.h"
#include "ultra_infer/utils/perf.h"
#include "ultra_infer/vision/ocr/ppocr/utils/ocr_utils.h"

namespace ultra_infer {
namespace vision {
namespace ocr {

StructureV2TablePreprocessor::StructureV2TablePreprocessor() {
  resize_op_ = std::make_shared<Resize>(-1, -1);

  std::vector<float> value = {0, 0, 0};
  pad_op_ = std::make_shared<Pad>(0, 0, 0, 0, value);

  std::vector<float> mean = {0.485f, 0.456f, 0.406f};
  std::vector<float> std = {0.229f, 0.224f, 0.225f};
  normalize_op_ = std::make_shared<Normalize>(mean, std, true);
  hwc2chw_op_ = std::make_shared<HWC2CHW>();
}

void StructureV2TablePreprocessor::StructureV2TableResizeImage(FDMat *mat,
                                                               int batch_idx) {
  float img_h = float(rec_image_shape_[1]);
  float img_w = float(rec_image_shape_[2]);
  float width = float(mat->Width());
  float height = float(mat->Height());
  float ratio = float(float(max_len) / (std::max(height, width) * 1.0));
  int resize_h = int(height * ratio);
  int resize_w = int(width * ratio);

  resize_op_->SetWidthAndHeight(resize_w, resize_h);
  (*resize_op_)(mat);

  (*normalize_op_)(mat);
  pad_op_->SetPaddingSize(0, int(max_len - resize_h), 0,
                          int(max_len - resize_w));
  (*pad_op_)(mat);

  (*hwc2chw_op_)(mat);

  batch_det_img_info_[batch_idx] = {int(width),   int(height),  float(ratio),
                                    float(ratio), int(max_len), int(max_len)};
}

bool StructureV2TablePreprocessor::Run(std::vector<FDMat> *images,
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

bool StructureV2TablePreprocessor::Apply(FDMatBatch *image_batch,
                                         std::vector<FDTensor> *outputs) {
  batch_det_img_info_.clear();
  batch_det_img_info_.resize(image_batch->mats->size());
  for (size_t i = 0; i < image_batch->mats->size(); ++i) {
    FDMat *mat = &(image_batch->mats->at(i));
    StructureV2TableResizeImage(mat, i);
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
