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

#include "rknn_api.h" // NOLINT
#include "ultra_infer/core/fd_tensor.h"
#include "ultra_infer/runtime/backends/backend.h"
#include "ultra_infer/runtime/backends/rknpu2/option.h"
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace ultra_infer {
class RKNPU2Backend : public BaseBackend {
public:
  /***************************** BaseBackend API *****************************/
  RKNPU2Backend() = default;
  virtual ~RKNPU2Backend();
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
  /***************************** BaseBackend API *****************************/

private:
  /*
   *  @name       RuntimeOptionIsApplicable
   *  @brief      This function is used to determine whether the RuntimeOption
   *              meets the operating conditions of RKNPU2.
   *  @param      None
   *  @return     bool
   *  @note       None
   */
  bool RuntimeOptionIsApplicable(const RuntimeOption &runtime_option);

  /*
   *  @name       LoadModel
   *  @brief      Read the model and initialize rknn context.
   *  @param      model: Binary data for the RKNN model or the path of RKNN
   * model.
   *  @return     bool
   *  @note       None
   */
  bool LoadModel(void *model);

  /*
   *  @name       GetSDKAndDeviceVersion
   *  @brief      Get RKNPU2 sdk and device version.
   *  @param      None
   *  @return     bool
   *  @note       The private variable ctx must be initialized to use this
   * function.
   */
  bool GetSDKAndDeviceVersion();

  /*
   *  @name      BuildOption
   *  @brief     Save option and set core mask.
   *  @param     RKNPU2BackendOption
   *  @note      None
   */
  void BuildOption(const RKNPU2BackendOption &option);

  /*
   *  @name       SetCoreMask
   *  @brief      Set NPU core for model
   *  @param      core_mask: The specification of NPU core setting.
   *  @return     bool
   *  @note       Only support RK3588
   */
  bool SetCoreMask(const rknpu2::CoreMask &core_mask) const;

  /*
   *  @name       InitInputAndOutputNumber
   *  @brief      Initialize io_num_.
   *  @param
   *  @return     bool
   *  @note       The private variable ctx must be initialized to use this
   * function.
   */
  bool InitInputAndOutputNumber();

  /*
   *  @name       InitRKNNTensorAddress
   *  @brief      Allocate memory for input_attrs_ and output_attrs_.
   *  @param      None
   *  @return     bool
   *  @note       None
   */
  bool InitRKNNTensorAddress();

  /*
   *  @name       InitInputAndOutputInformation
   *  @brief      Initialize inputs_desc_ and outputs_desc_.
   *  @param      None
   *  @return     bool
   *  @note       None
   */
  bool InitInputAndOutputInformation();

  /*
   *  @name       InitRKNNTensorMemory
   *  @brief      Allocate memory for input and output tensors.
   *  @param      std::vector<FDTensor>& inputs
   *  @return     None
   *  @note       None
   */
  bool InitRKNNTensorMemory(std::vector<FDTensor> &inputs);

  rknn_context ctx_{};
  rknn_sdk_version sdk_ver_{};

  rknn_input_output_num io_num_{0, 0};

  std::vector<TensorInfo> inputs_desc_;
  std::vector<TensorInfo> outputs_desc_;

  rknn_tensor_attr *input_attrs_ = nullptr;
  rknn_tensor_attr *output_attrs_ = nullptr;

  std::vector<rknn_tensor_mem *> input_mems_;
  std::vector<rknn_tensor_mem *> output_mems_;

  bool io_num_init_ = false;
  bool tensor_attrs_init_ = false;
  bool tensor_memory_init_ = false;

  RKNPU2BackendOption option_;

  /*
   *  @name       DumpTensorAttr
   *  @brief      Get the model's detailed inputs and outputs
   *  @param      rknn_tensor_attr
   *  @return     None
   *  @note       None
   */
  void DumpTensorAttr(rknn_tensor_attr &attr);

  /*
   *  @name       RknnTensorTypeToFDDataType
   *  @brief      Change RknnTensorType To FDDataType
   *  @param      rknn_tensor_type
   *  @return     None
   *  @note       Most post-processing does not support the fp16 format.
   *              Therefore, if the input is FP16, the output will be FP32.
   */
  FDDataType RknnTensorTypeToFDDataType(rknn_tensor_type type);

  /*
   *  @name       FDDataTypeToRknnTensorType
   *  @brief      Change FDDataType To RknnTensorType
   *  @param      FDDataType
   *  @return     None
   *  @note       None
   */
  rknn_tensor_type FDDataTypeToRknnTensorType(FDDataType type);
};
} // namespace ultra_infer
