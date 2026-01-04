// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include "ultra_infer/vision/ocr/ppocr/structurev2_layout_preprocessor.h"
#include "ultra_infer/vision/ocr/ppocr/utils/ocr_utils.h"

namespace ultra_infer {
namespace vision {
namespace ocr {

StructureV2LayoutPreprocessor::StructureV2LayoutPreprocessor() {
  // default width(608) and height(900)
  resize_op_ =
      std::make_shared<Resize>(layout_image_shape_[2], layout_image_shape_[1]);
  normalize_permute_op_ = std::make_shared<NormalizeAndPermute>(
      std::vector<float>({0.485f, 0.456f, 0.406f}),
      std::vector<float>({0.229f, 0.224f, 0.225f}), true);
}

std::array<int, 4> StructureV2LayoutPreprocessor::GetLayoutImgInfo(FDMat *img) {
  if (static_shape_infer_) {
    return {img->Width(), img->Height(), layout_image_shape_[2],
            layout_image_shape_[1]};
  } else {
    FDASSERT(false, "not support dynamic shape inference now!")
  }
  return {img->Width(), img->Height(), layout_image_shape_[2],
          layout_image_shape_[1]};
}

bool StructureV2LayoutPreprocessor::ResizeLayoutImage(FDMat *img, int resize_w,
                                                      int resize_h) {
  resize_op_->SetWidthAndHeight(resize_w, resize_h);
  (*resize_op_)(img);
  return true;
}

bool StructureV2LayoutPreprocessor::Apply(FDMatBatch *image_batch,
                                          std::vector<FDTensor> *outputs) {
  batch_layout_img_info_.clear();
  batch_layout_img_info_.resize(image_batch->mats->size());
  for (size_t i = 0; i < image_batch->mats->size(); ++i) {
    FDMat *mat = &(image_batch->mats->at(i));
    batch_layout_img_info_[i] = GetLayoutImgInfo(mat);
    ResizeLayoutImage(mat, batch_layout_img_info_[i][2],
                      batch_layout_img_info_[i][3]);
  }
  if (!disable_normalize_ && !disable_permute_) {
    (*normalize_permute_op_)(image_batch);
  }

  outputs->resize(1);
  FDTensor *tensor = image_batch->Tensor();
  (*outputs)[0].SetExternalData(tensor->Shape(), tensor->Dtype(),
                                tensor->Data(), tensor->device,
                                tensor->device_id);
  return true;
}

} // namespace ocr
} // namespace vision
} // namespace ultra_infer
