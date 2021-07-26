// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <vector>

#include <inference_engine.hpp>

#include "model_deploy/common/include/base_model.h"
#include "model_deploy/common/include/output_struct.h"
#include "model_deploy/engine/include/engine.h"

namespace PaddleDeploy {

class OpenVinoEngine : public InferEngine {
 public:
  virtual bool Init(const InferenceConfig& engine_config);

  virtual bool Infer(const std::vector<DataBlob>& inputs,
                     std::vector<DataBlob>* outputs);

 private:
  bool GetDtype(const InferenceEngine::TensorDesc &output_blob,
                DataBlob *output);

  InferenceEngine::CNNNetwork network_;
  InferenceEngine::ExecutableNetwork executable_network_;
};

}  // namespace PaddleDeploy
