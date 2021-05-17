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

#include <iostream>
#include <string>
#include <vector>
#include <memory>

#include "./common.h"
#include "./http_client.h"
#include "rapidjson/document.h"
#include "rapidjson/rapidjson.h"
#include "rapidjson/error/en.h"

#include "model_deploy/common/include/output_struct.h"
#include "model_deploy/engine/include/engine.h"
#include "model_deploy/common/include/base_model.h"

namespace nic = nvidia::inferenceserver::client;

namespace PaddleDeploy {

class TritonInferenceEngine : public InferEngine {
 public:
  std::unique_ptr<nic::InferenceServerHttpClient> client_;

  TritonInferenceEngine() : options_("") {}

  bool Init(const InferenceConfig& engine_configs);

  bool Infer(const std::vector<DataBlob>& input_blobs,
             std::vector<DataBlob>* output_blobs);

 private:
  nic::InferOptions options_;
  nic::Headers headers_;
  nic::Parameters query_params_;

  void ParseConfigs(const TritonInferenceConfigs& configs);

  void CreateInput(const std::vector<DataBlob>& input_blobs,
                   std::vector<nic::InferInput* >* inputs);

  void CreateOutput(const rapidjson::Document& model_metadata,
                    std::vector<const nic::InferRequestedOutput* >* outputs);

  nic::Error GetModelMetaData(rapidjson::Document* model_metadata);
};

}  // namespace PaddleDeploy
