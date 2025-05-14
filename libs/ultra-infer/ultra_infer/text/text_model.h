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

#pragma once
#include <memory>

#include "ultra_infer/ultra_infer_model.h"
#include "ultra_infer/utils/unique_ptr.h"

namespace ultra_infer {
namespace text {

class Preprocessor;
class Postprocessor;
class Result;
class PredictionOption;

class ULTRAINFER_DECL TextModel : public UltraInferModel {
public:
  virtual std::string ModelName() const { return "TextModel"; }
  virtual bool Predict(const std::string &raw_text, Result *result,
                       const PredictionOption &option);
  virtual bool PredictBatch(const std::vector<std::string> &raw_text_array,
                            Result *result, const PredictionOption &option);
  template <typename T, typename... Args> void SetPreprocessor(Args &&...args) {
    preprocessor_ = utils::make_unique<T>(std::forward<Args>(args)...);
  }
  template <typename T, typename... Args>
  void SetPostprocessor(Args &&...args) {
    postprocessor_ = utils::make_unique<T>(std::forward<Args>(args)...);
  }

private:
  std::unique_ptr<Preprocessor> preprocessor_;
  std::unique_ptr<Postprocessor> postprocessor_;
};

} // namespace text
} // namespace ultra_infer
