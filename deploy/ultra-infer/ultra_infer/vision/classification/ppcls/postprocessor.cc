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

#include "ultra_infer/vision/classification/ppcls/postprocessor.h"
#include "ultra_infer/vision/utils/utils.h"

namespace ultra_infer {
namespace vision {
namespace classification {

PaddleClasPostprocessor::PaddleClasPostprocessor(int topk) {
  topk_ = topk;
  initialized_ = true;
}

bool PaddleClasPostprocessor::Run(const std::vector<FDTensor> &infer_result,
                                  std::vector<ClassifyResult> *results) {
  if (!initialized_) {
    FDERROR << "Postprocessor is not initialized." << std::endl;
    return false;
  }

  int batch = infer_result[0].shape[0];
  int num_classes = infer_result[0].shape[1];
  const float *infer_result_data =
      reinterpret_cast<const float *>(infer_result[0].Data());

  results->resize(batch);

  int topk = std::min(num_classes, topk_);
  for (int i = 0; i < batch; ++i) {
    (*results)[i].label_ids = utils::TopKIndices(
        infer_result_data + i * num_classes, num_classes, topk);
    (*results)[i].scores.resize(topk);
    for (int j = 0; j < topk; ++j) {
      (*results)[i].scores[j] =
          infer_result_data[i * num_classes + (*results)[i].label_ids[j]];
    }
  }

  return true;
}

} // namespace classification
} // namespace vision
} // namespace ultra_infer
