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

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "paddle_api.h" // NOLINT

#include "ultra_infer/runtime/backends/backend.h"
#include "ultra_infer/runtime/backends/lite/option.h"
#include "ultra_infer/runtime/runtime_option.h"

namespace ultra_infer {

class LiteBackend : public BaseBackend {
public:
  LiteBackend() {}
  virtual ~LiteBackend() = default;

  bool Init(const RuntimeOption &option) override;

  bool Infer(std::vector<FDTensor> &inputs, std::vector<FDTensor> *outputs,
             bool copy_to_fd = true) override; // NOLINT

  int NumInputs() const override { return inputs_desc_.size(); }

  int NumOutputs() const override { return outputs_desc_.size(); }

  TensorInfo GetInputInfo(int index) override;
  TensorInfo GetOutputInfo(int index) override;
  std::vector<TensorInfo> GetInputInfos() override;
  std::vector<TensorInfo> GetOutputInfos() override;

private:
  // Build CxxConfig from option for Paddle Lite full api.
  void BuildOption(const LiteBackendOption &option);
  // Configure many hardwares for Paddle Lite full api.
  void ConfigureCpu(const LiteBackendOption &option);
  void ConfigureGpu(const LiteBackendOption &option);
  void ConfigureTimvx(const LiteBackendOption &option);
  void ConfigureAscend(const LiteBackendOption &option);
  void ConfigureKunlunXin(const LiteBackendOption &option);
  void ConfigureNNAdapter(const LiteBackendOption &option);

  paddle::lite_api::CxxConfig config_;
  std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor_;
  paddle::lite_api::MobileConfig mobile_config_;
  std::vector<TensorInfo> inputs_desc_;
  std::vector<TensorInfo> outputs_desc_;
  std::map<std::string, int> inputs_order_;
  LiteBackendOption option_;
};

// Convert data type from paddle lite to ultra_infer
FDDataType LiteDataTypeToFD(const paddle::lite_api::PrecisionType &dtype);

// Helper function to read file
bool ReadFile(const std::string &filename, std::vector<char> *contents,
              bool binary = true);

} // namespace ultra_infer
