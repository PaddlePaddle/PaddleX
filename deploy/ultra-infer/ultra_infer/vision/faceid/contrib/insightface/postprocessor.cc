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

#include "ultra_infer/vision/faceid/contrib/insightface/postprocessor.h"
#include "ultra_infer/vision/utils/utils.h"

namespace ultra_infer {
namespace vision {
namespace faceid {

InsightFaceRecognitionPostprocessor::InsightFaceRecognitionPostprocessor() {
  l2_normalize_ = false;
}

bool InsightFaceRecognitionPostprocessor::Run(
    std::vector<FDTensor> &infer_result,
    std::vector<FaceRecognitionResult> *results) {
  if (infer_result[0].dtype != FDDataType::FP32) {
    FDERROR << "Only support post process with float32 data." << std::endl;
    return false;
  }
  if (infer_result.size() != 1) {
    FDERROR << "The default number of output tensor "
               "must be 1 according to insightface."
            << std::endl;
  }
  int batch = infer_result[0].shape[0];
  results->resize(batch);
  for (size_t bs = 0; bs < batch; ++bs) {
    FDTensor &embedding_tensor = infer_result.at(bs);
    FDASSERT((embedding_tensor.shape[0] == 1), "Only support batch = 1 now.");
    if (embedding_tensor.dtype != FDDataType::FP32) {
      FDERROR << "Only support post process with float32 data." << std::endl;
      return false;
    }
    (*results)[bs].Clear();
    (*results)[bs].Resize(embedding_tensor.Numel());

    // Copy the raw embedding vector directly without L2 normalize
    // post process. Let the user decide whether to normalize or not.
    // Will call utils::L2Normlize() method to perform L2
    // normalize if l2_normalize was set as 'true'.
    std::memcpy((*results)[bs].embedding.data(), embedding_tensor.Data(),
                embedding_tensor.Nbytes());
    if (l2_normalize_) {
      auto norm_embedding = utils::L2Normalize((*results)[bs].embedding);
      std::memcpy((*results)[bs].embedding.data(), norm_embedding.data(),
                  embedding_tensor.Nbytes());
    }
  }
  return true;
}

} // namespace faceid
} // namespace vision
} // namespace ultra_infer
