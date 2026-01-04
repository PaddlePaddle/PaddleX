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

#include "ultra_infer/vision/faceid/contrib/insightface/preprocessor.h"

namespace ultra_infer {
namespace vision {
namespace faceid {

InsightFaceRecognitionPreprocessor::InsightFaceRecognitionPreprocessor() {
  // parameters for preprocess
  size_ = {112, 112};
  alpha_ = {1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f};
  beta_ = {-1.f, -1.f, -1.f}; // RGB
}

bool InsightFaceRecognitionPreprocessor::Preprocess(FDMat *mat,
                                                    FDTensor *output) {
  // face recognition model's preprocess steps in insightface
  // reference: insightface/recognition/arcface_torch/inference.py
  // 1. Resize
  // 2. BGR2RGB
  // 3. Convert(opencv style) or Normalize
  // 4. HWC2CHW
  int resize_w = size_[0];
  int resize_h = size_[1];
  if (resize_h != mat->Height() || resize_w != mat->Width()) {
    Resize::Run(mat, resize_w, resize_h);
  }

  if (!disable_permute_) {
    BGR2RGB::Run(mat);
  }

  if (!disable_normalize_) {
    Convert::Run(mat, alpha_, beta_);
    HWC2CHW::Run(mat);
    Cast::Run(mat, "float");
  }

  mat->ShareWithTensor(output);
  output->ExpandDim(0); // reshape to n, c, h, w
  return true;
}

bool InsightFaceRecognitionPreprocessor::Run(std::vector<FDMat> *images,
                                             std::vector<FDTensor> *outputs) {
  if (images->empty()) {
    FDERROR << "The size of input images should be greater than 0."
            << std::endl;
    return false;
  }
  FDASSERT(images->size() == 1, "Only support batch = 1 now.");
  outputs->resize(1);
  // Concat all the preprocessed data to a batch tensor
  std::vector<FDTensor> tensors(images->size());
  for (size_t i = 0; i < images->size(); ++i) {
    if (!Preprocess(&(*images)[i], &tensors[i])) {
      FDERROR << "Failed to preprocess input image." << std::endl;
      return false;
    }
  }
  (*outputs)[0] = std::move(tensors[0]);
  return true;
}
} // namespace faceid
} // namespace vision
} // namespace ultra_infer
