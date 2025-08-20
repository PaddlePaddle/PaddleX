// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "acl/acl.h"
#include "ultra_infer/core/fd_tensor.h"
#include "ultra_infer/runtime/backends/backend.h"

namespace ultra_infer {
class OmBackend : public BaseBackend {
public:
  OmBackend() = default;
  virtual ~OmBackend();

  // OM Backend implementation.
  bool Init(const RuntimeOption &runtime_option) override;

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
  static bool aclInitFlag;

private:
  std::vector<TensorInfo> inputs_desc_;
  std::vector<TensorInfo> outputs_desc_;
  std::vector<void *> inputBuffer;
  std::vector<void *> outputBuffer;
  bool loadFlag_ = false; // model load flag
  int32_t deviceId_;
  uint32_t modelId_;
  size_t modelWorkSize_;   // model work memory buffer size
  size_t modelWeightSize_; // model weight memory buffer size
  void *modelWorkPtr_;     // model work memory buffer
  void *modelWeightPtr_;   // model weight memory buffer
  aclmdlDesc *modelDesc_;
  aclmdlDataset *input_;
  aclmdlDataset *output_;
  aclrtContext context_;
  aclrtStream stream_;

  bool LoadModel(const char *modelPath);
  bool Execute();
  bool CreateInput();
  void DestroyInput();
  bool CreateOutput();
  void DestroyOutput();
  void DestroyResource();
  bool CreateModelDesc();
  void FreeInputBuffer();
  void FreeOutputBuffer();
  bool InitResource();
};
} // namespace ultra_infer
