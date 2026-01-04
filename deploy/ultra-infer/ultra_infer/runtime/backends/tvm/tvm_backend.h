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

#include "ultra_infer/core/fd_tensor.h"
#include "ultra_infer/runtime/backends/backend.h"
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <unistd.h>

namespace ultra_infer {
class TVMBackend : public BaseBackend {
public:
  TVMBackend() = default;
  virtual ~TVMBackend() = default;
  bool Init(const RuntimeOption &runtime_option) override;
  int NumInputs() const override { return inputs_desc_.size(); }
  int NumOutputs() const override { return outputs_desc_.size(); }
  TensorInfo GetInputInfo(int index) override { return inputs_desc_[index]; }
  TensorInfo GetOutputInfo(int index) override { return outputs_desc_[index]; }
  std::vector<TensorInfo> GetInputInfos() override { return inputs_desc_; }
  std::vector<TensorInfo> GetOutputInfos() override { return outputs_desc_; }
  bool Infer(std::vector<FDTensor> &inputs, std::vector<FDTensor> *outputs,
             bool copy_to_fd = true) override;

private:
  DLDevice dev_{};
  tvm::runtime::Module gmod_;
  std::vector<TensorInfo> inputs_desc_;
  std::vector<TensorInfo> outputs_desc_;

  bool BuildDLDevice(Device device);
  bool BuildModel(const RuntimeOption &runtime_option);
  bool InitInputAndOutputTensor();

  std::vector<tvm::runtime::NDArray> input_tensor_;
  std::vector<tvm::runtime::NDArray> output_tensor_;

  FDDataType TVMTensorTypeToFDDataType(tvm::String type);
  DLDataType FDDataTypeToDLDataType(FDDataType dtype);
};
} // namespace ultra_infer
