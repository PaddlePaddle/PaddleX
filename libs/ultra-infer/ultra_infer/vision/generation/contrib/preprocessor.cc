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

#include "ultra_infer/vision/generation/contrib/preprocessor.h"

namespace ultra_infer {
namespace vision {
namespace generation {

bool AnimeGANPreprocessor::Run(std::vector<Mat> &images,
                               std::vector<FDTensor> *outputs) {
  // 1. BGR2RGB
  // 2. Convert(opencv style) or Normalize
  for (size_t i = 0; i < images.size(); ++i) {
    auto ret = BGR2RGB::Run(&images[i]);
    if (!ret) {
      FDERROR << "Failed to processs image:" << i << " in "
              << "BGR2RGB"
              << "." << std::endl;
      return false;
    }
    ret = Cast::Run(&images[i], "float");
    if (!ret) {
      FDERROR << "Failed to processs image:" << i << " in "
              << "Cast"
              << "." << std::endl;
      return false;
    }
    std::vector<float> mean{1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f};
    std::vector<float> std{-1.f, -1.f, -1.f};
    ret = Convert::Run(&images[i], mean, std);
    if (!ret) {
      FDERROR << "Failed to processs image:" << i << " in "
              << "Cast"
              << "." << std::endl;
      return false;
    }
  }
  outputs->resize(1);
  // Concat all the preprocessed data to a batch tensor
  std::vector<FDTensor> tensors(images.size());
  for (size_t i = 0; i < images.size(); ++i) {
    images[i].ShareWithTensor(&(tensors[i]));
    tensors[i].ExpandDim(0);
  }
  if (tensors.size() == 1) {
    (*outputs)[0] = std::move(tensors[0]);
  } else {
    function::Concat(tensors, &((*outputs)[0]), 0);
  }
  return true;
}

} // namespace generation
} // namespace vision
} // namespace ultra_infer
