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
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h" // NOLINT
#include "ultra_infer/runtime/backends/backend.h"
#include "ultra_infer/runtime/backends/ort/option.h"

#ifdef WITH_DIRECTML
#include "dml_provider_factory.h" // NOLINT
#endif

namespace ultra_infer {

struct OrtValueInfo {
  std::string name;
  std::vector<int64_t> shape;
  ONNXTensorElementDataType dtype;
};

class OrtBackend : public BaseBackend {
public:
  OrtBackend() {}
  virtual ~OrtBackend() = default;

  bool BuildOption(const OrtBackendOption &option);

  bool Init(const RuntimeOption &option);

  bool Infer(std::vector<FDTensor> &inputs, std::vector<FDTensor> *outputs,
             bool copy_to_fd = true) override;

  int NumInputs() const override { return inputs_desc_.size(); }

  int NumOutputs() const override { return outputs_desc_.size(); }

  TensorInfo GetInputInfo(int index) override;
  TensorInfo GetOutputInfo(int index) override;
  std::vector<TensorInfo> GetInputInfos() override;
  std::vector<TensorInfo> GetOutputInfos() override;
  static std::vector<OrtCustomOp *> custom_operators_;
  void InitCustomOperators();

private:
  bool InitFromPaddle(const std::string &model_buffer,
                      const std::string &params_buffer,
                      const OrtBackendOption &option = OrtBackendOption(),
                      bool verbose = false);

  bool InitFromOnnx(const std::string &model_buffer,
                    const OrtBackendOption &option = OrtBackendOption());

  Ort::Env env_;
  Ort::Session session_{nullptr};
  Ort::SessionOptions session_options_;
  std::shared_ptr<Ort::IoBinding> binding_;
  std::vector<OrtValueInfo> inputs_desc_;
  std::vector<OrtValueInfo> outputs_desc_;

  // the ONNX model file name,
  // when ONNX is bigger than 2G, we will set this name
  std::string model_file_name;
  // recorded if the model has been converted to fp16
  bool converted_to_fp16 = false;

#ifndef NON_64_PLATFORM
  Ort::CustomOpDomain custom_op_domain_ = Ort::CustomOpDomain("Paddle");
#endif
  OrtBackendOption option_;
  void OrtValueToFDTensor(const Ort::Value &value, FDTensor *tensor,
                          const std::string &name, bool copy_to_fd);
};
} // namespace ultra_infer
