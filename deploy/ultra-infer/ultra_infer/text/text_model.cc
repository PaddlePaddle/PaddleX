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

#include "ultra_infer/text/text_model.h"
#include "ultra_infer/text/common/option.h"
#include "ultra_infer/text/common/result.h"
#include "ultra_infer/text/postprocessor/postprocessor.h"
#include "ultra_infer/text/preprocessor/preprocessor.h"

namespace ultra_infer {
namespace text {

bool TextModel::Predict(const std::string &raw_text, Result *result,
                        const PredictionOption &option) {
  // Preprocess
  std::vector<FDTensor> input_tensor;
  std::vector<FDTensor> output_tensor;
  if (!preprocessor_->Encode(raw_text, &input_tensor)) {
    FDERROR << "Failed to preprocess input data while using model:"
            << ModelName() << "." << std::endl;
    return false;
  }

  // Inference Runtime
  if (!Infer(input_tensor, &output_tensor)) {
    FDERROR << "Failed to inference while using model:" << ModelName() << "."
            << std::endl;
    return false;
  }

  // Postprocess
  if (postprocessor_->Decode(output_tensor, result)) {
    FDERROR << "Failed to postprocess while using model:" << ModelName() << "."
            << std::endl;
    return false;
  }
  return true;
}

bool TextModel::PredictBatch(const std::vector<std::string> &raw_text_array,
                             Result *results, const PredictionOption &option) {
  // Preprocess
  std::vector<FDTensor> input_tensor;
  std::vector<FDTensor> output_tensor;
  if (!preprocessor_->EncodeBatch(raw_text_array, &input_tensor)) {
    FDERROR << "Failed to preprocess input data while using model:"
            << ModelName() << "." << std::endl;
    return false;
  }

  // Inference Runtime
  if (!Infer(input_tensor, &output_tensor)) {
    FDERROR << "Failed to inference while using model:" << ModelName() << "."
            << std::endl;
    return false;
  }

  // Postprocess
  if (postprocessor_->DecodeBatch(output_tensor, results)) {
    FDERROR << "Failed to postprocess while using model:" << ModelName() << "."
            << std::endl;
    return false;
  }
  return true;
}

} // namespace text
} // namespace ultra_infer
