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

#include "ultra_infer/vision/ocr/ppocr/cls_postprocessor.h"
#include "ultra_infer/utils/perf.h"
#include "ultra_infer/vision/ocr/ppocr/utils/ocr_utils.h"

namespace ultra_infer {
namespace vision {
namespace ocr {

bool SingleBatchPostprocessor(const float *out_data, const size_t &length,
                              int *cls_label, float *cls_score) {

  *cls_label = std::distance(&out_data[0],
                             std::max_element(&out_data[0], &out_data[length]));

  *cls_score = float(*std::max_element(&out_data[0], &out_data[length]));
  return true;
}

bool ClassifierPostprocessor::Run(const std::vector<FDTensor> &tensors,
                                  std::vector<int32_t> *cls_labels,
                                  std::vector<float> *cls_scores) {
  size_t total_size = tensors[0].shape[0];
  return Run(tensors, cls_labels, cls_scores, 0, total_size);
}

bool ClassifierPostprocessor::Run(const std::vector<FDTensor> &tensors,
                                  std::vector<int32_t> *cls_labels,
                                  std::vector<float> *cls_scores,
                                  size_t start_index, size_t total_size) {
  // Classifier have only 1 output tensor.
  const FDTensor &tensor = tensors[0];

  // For Classifier, the output tensor shape = [batch,2]
  size_t batch = tensor.shape[0];
  size_t length = accumulate(tensor.shape.begin() + 1, tensor.shape.end(), 1,
                             std::multiplies<int>());

  if (batch <= 0) {
    FDERROR << "The infer outputTensor.shape[0] <=0, wrong infer result."
            << std::endl;
    return false;
  }
  if (start_index < 0 || total_size <= 0) {
    FDERROR << "start_index or total_size error. Correct is: 0 <= start_index "
               "< total_size"
            << std::endl;
    return false;
  }
  if ((start_index + batch) > total_size) {
    FDERROR << "start_index or total_size error. Correct is: start_index + "
               "batch(outputTensor.shape[0]) <= total_size"
            << std::endl;
    return false;
  }

  cls_labels->resize(total_size);
  cls_scores->resize(total_size);
  const float *tensor_data = reinterpret_cast<const float *>(tensor.Data());
  for (int i_batch = 0; i_batch < batch; ++i_batch) {
    SingleBatchPostprocessor(tensor_data + i_batch * length, length,
                             &cls_labels->at(i_batch + start_index),
                             &cls_scores->at(i_batch + start_index));
  }

  return true;
}

} // namespace ocr
} // namespace vision
} // namespace ultra_infer
