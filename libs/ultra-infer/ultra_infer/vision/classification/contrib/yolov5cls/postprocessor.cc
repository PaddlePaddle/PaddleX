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

#include "ultra_infer/vision/classification/contrib/yolov5cls/postprocessor.h"
#include "ultra_infer/vision/utils/utils.h"

namespace ultra_infer {
namespace vision {
namespace classification {

YOLOv5ClsPostprocessor::YOLOv5ClsPostprocessor() { topk_ = 1; }

bool YOLOv5ClsPostprocessor::Run(
    const std::vector<FDTensor> &tensors, std::vector<ClassifyResult> *results,
    const std::vector<std::map<std::string, std::array<float, 2>>> &ims_info) {
  int batch = tensors[0].shape[0];
  FDTensor infer_result = tensors[0];
  FDTensor infer_result_softmax;
  function::Softmax(infer_result, &infer_result_softmax, 1);
  results->resize(batch);

  for (size_t bs = 0; bs < batch; ++bs) {
    (*results)[bs].Clear();
    // output (1,1000) score classnum 1000
    int num_classes = infer_result_softmax.shape[1];
    const float *infer_result_buffer =
        reinterpret_cast<const float *>(infer_result_softmax.Data()) +
        bs * infer_result_softmax.shape[1];
    topk_ = std::min(num_classes, topk_);
    (*results)[bs].label_ids =
        utils::TopKIndices(infer_result_buffer, num_classes, topk_);
    (*results)[bs].scores.resize(topk_);
    for (int i = 0; i < topk_; ++i) {
      (*results)[bs].scores[i] =
          *(infer_result_buffer + (*results)[bs].label_ids[i]);
    }

    if ((*results)[bs].label_ids.size() == 0) {
      return true;
    }
  }
  return true;
}

} // namespace classification
} // namespace vision
} // namespace ultra_infer
