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

#include "ultra_infer/vision/classification/ppshitu/ppshituv2_rec_postprocessor.h"
#include "ultra_infer/vision/utils/utils.h"
#include <cmath>
#include <numeric>

namespace ultra_infer {
namespace vision {
namespace classification {

bool PPShiTuV2RecognizerPostprocessor::Run(
    const std::vector<FDTensor> &tensors,
    std::vector<ClassifyResult> *results) {
  int batch = tensors[0].shape[0]; // e.g [batch, 512]
  int num_feature = tensors[0].shape[1];
  const float *tensor_data = reinterpret_cast<const float *>(tensors[0].Data());

  results->resize(batch);

  // post processing per batch=1
  for (int i = 0; i < batch; ++i) {
    (*results)[i].feature.resize(num_feature);
    const float *tensor_data_i_start = tensor_data + i * num_feature;
    std::memcpy((*results)[i].feature.data(), tensor_data_i_start,
                num_feature * sizeof(float));
    if (feature_norm_) {
      FeatureNorm((*results)[i].feature);
    }
  }

  return true;
}

void PPShiTuV2RecognizerPostprocessor::FeatureNorm(
    std::vector<float> &feature) {
  float feature_sqrt = std::sqrt(std::inner_product(
      feature.begin(), feature.end(), feature.begin(), 0.0f));
  for (int i = 0; i < feature.size(); ++i) {
    feature[i] /= feature_sqrt;
  }
}

} // namespace classification
} // namespace vision
} // namespace ultra_infer
