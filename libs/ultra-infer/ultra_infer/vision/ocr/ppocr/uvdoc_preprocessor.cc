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

#include "ultra_infer/vision/ocr/ppocr/uvdoc_preprocessor.h"
#include "ultra_infer/vision/ocr/ppocr/utils/ocr_utils.h"

namespace ultra_infer {
namespace vision {
namespace ocr {

UVDocPreprocessor::UVDocPreprocessor() {
  normalize_permute_op_ = std::make_shared<NormalizeAndPermute>(
      std::vector<float>({0.0f, 0.0f, 0.0f}),
      std::vector<float>({1.0f, 1.0f, 1.0f}), true);
}

bool UVDocPreprocessor::Apply(FDMatBatch *image_batch,
                              std::vector<FDTensor> *outputs) {

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
