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

#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "dnn/hb_dnn.h"
#include "ultra_infer/core/fd_tensor.h"
#include "ultra_infer/runtime/backends/backend.h"

namespace ultra_infer {
class HorizonBackend : public BaseBackend {
public:
  HorizonBackend() = default;
  ~HorizonBackend();

  // Horizon Backend implementation.
  bool Init(const RuntimeOption &runtime_option);

  int NumInputs() const override {
    return static_cast<int>(inputs_desc_.size());
  }

  int NumOutputs() const override {
    return static_cast<int>(outputs_desc_.size());
  }

  TensorInfo GetInputInfo(int index) override;
  TensorInfo GetOutputInfo(int index) override;
  std::vector<TensorInfo> GetInputInfos() override;
  std::vector<TensorInfo> GetOutputInfos() override;
  bool Infer(std::vector<FDTensor> &inputs, std::vector<FDTensor> *outputs,
             bool copy_to_fd = true) override;

private:
  hbPackedDNNHandle_t packed_dnn_handle_;
  hbDNNHandle_t dnn_handle_;
  hbDNNTensorProperties *input_properties_ = nullptr;
  hbDNNTensorProperties *output_properties_ = nullptr;
  hbDNNTensor *input_mems_;
  hbDNNTensor *output_mems_;

  bool infer_init_ = false;
  std::vector<TensorInfo> inputs_desc_;
  std::vector<TensorInfo> outputs_desc_;
  bool GetModelInputOutputInfos();
  bool LoadModel(const char *model);

  static FDDataType HorizonTensorTypeToFDDataType(int32_t type);
  static hbDNNDataType FDDataTypeToHorizonTensorType(FDDataType type);
};
} // namespace ultra_infer
