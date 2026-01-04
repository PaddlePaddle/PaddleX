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

#include "ultra_infer/runtime/backends/horizon/horizon_backend.h"
namespace ultra_infer {

HorizonBackend::~HorizonBackend() {
  int ret = -1;
  // Release memory uniformly here
  if (input_properties_ != nullptr) {
    free(input_properties_);
  }
  if (output_properties_ != nullptr) {
    free(output_properties_);
  }
  if (input_mems_ == nullptr) {
    return;
  }
  for (int i = 0; i < NumInputs(); i++) {

    ret = hbSysFreeMem(&(input_mems_[i].sysMem[0]));

    if (ret != 0) {
      FDERROR << "release input mem fail! ret=" << ret << std::endl;
    }
    if (input_mems_ != nullptr) {
      free(input_mems_);
    }
  }

  for (int i = 0; i < NumOutputs(); i++) {
    ret = hbSysFreeMem(&(output_mems_[i].sysMem[0]));

    if (ret != 0) {
      FDERROR << "release output mem fail! ret=" << ret << std::endl;
    }
    if (output_mems_ != nullptr) {
      free(output_mems_);
    }
  }
  ret = hbDNNRelease(packed_dnn_handle_);
  if (ret != 0) {
    FDERROR << "hbDNNRelease  fail! ret=" << ret << std::endl;
  }
}

bool HorizonBackend::GetModelInputOutputInfos() {
  const char **model_name_list;
  int model_count = 0;
  int ret;
  // get model name
  ret =
      hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle_);
  if (ret != 0) {
    FDERROR << "get model name fail! ret=" << ret << std::endl;
    return false;
  }
  // get dnn handle
  ret =
      hbDNNGetModelHandle(&dnn_handle_, packed_dnn_handle_, model_name_list[0]);
  if (ret != 0) {
    FDERROR << "get dnn handle fail! ret=" << ret << std::endl;
    return false;
  }
  // get input infos
  // Get detailed input parameters
  int input_count = 0;
  ret = hbDNNGetInputCount(&input_count, dnn_handle_);
  if (ret != 0) {
    FDERROR << "get input count fail! ret=" << ret << std::endl;
    return false;
  }
  input_properties_ = (hbDNNTensorProperties *)malloc(
      sizeof(hbDNNTensorProperties) * input_count);
  memset(input_properties_, 0, input_count * sizeof(hbDNNTensorProperties));

  inputs_desc_.resize(input_count);

  // get input info and copy to input tensor info
  for (uint32_t i = 0; i < input_count; i++) {
    ret = hbDNNGetInputTensorProperties(&input_properties_[i], dnn_handle_, i);

    if (ret != 0) {
      FDERROR << "get input tensor properties fail! ret=" << ret << std::endl;
      return false;
    }

    if ((input_properties_[i].tensorLayout != HB_DNN_LAYOUT_NHWC)) {
      FDERROR << "horizon_backend only support input layout is NHWC"
              << std::endl;
    }
    if (input_properties_[i].tensorType != HB_DNN_IMG_TYPE_RGB) {
      FDERROR << "horizon_backend only support input format is RGB"
              << std::endl;
    }

    const char *name;

    ret = hbDNNGetInputName(&name, dnn_handle_, i);
    if (ret != 0) {
      FDERROR << "get input tensor name fail! ret=" << ret << std::endl;
      return false;
    }
    // copy input proper to input tensor info
    std::string temp_name = name;
    std::vector<int> temp_shape{};
    int n_dims = input_properties_[i].validShape.numDimensions;

    temp_shape.resize(n_dims);
    for (int j = 0; j < n_dims; j++) {
      temp_shape[j] = (int)input_properties_[i].validShape.dimensionSize[j];
    }

    // Only support RGB format, so input type is UINT8
    FDDataType temp_dtype = FDDataType::UINT8;
    TensorInfo temp_input_info = {temp_name, temp_shape, temp_dtype};
    inputs_desc_[i] = temp_input_info;
  }

  // get output infos
  // Get detailed output parameters
  int output_count = 0;
  ret = hbDNNGetOutputCount(&output_count, dnn_handle_);
  if (ret != 0) {
    FDERROR << "get output count fail! ret=" << ret << std::endl;
    return false;
  }
  output_properties_ = (hbDNNTensorProperties *)malloc(
      sizeof(hbDNNTensorProperties) * output_count);
  memset(output_properties_, 0, output_count * sizeof(hbDNNTensorProperties));

  outputs_desc_.resize(output_count);

  for (uint32_t i = 0; i < output_count; i++) {
    // get model output size
    ret =
        hbDNNGetOutputTensorProperties(&output_properties_[i], dnn_handle_, i);

    const char *name;
    ret = hbDNNGetOutputName(&name, dnn_handle_, i);
    if (ret != 0) {
      FDERROR << "get output tensor name fail! ret=" << ret << std::endl;
      return false;
    }

    // copy output proper to output tensor info
    std::string temp_name = name;
    std::vector<int> temp_shape{};
    int n_dims = output_properties_[i].validShape.numDimensions;

    if ((n_dims == 4) &&
        (output_properties_[i].validShape.dimensionSize[3] == 1)) {
      n_dims--;
    }
    temp_shape.resize(n_dims);
    for (int j = 0; j < n_dims; j++) {
      temp_shape[j] = (int)output_properties_[i].validShape.dimensionSize[j];
    }

    FDDataType temp_dtype =
        HorizonTensorTypeToFDDataType(output_properties_[i].tensorType);

    TensorInfo temp_input_info = {temp_name, temp_shape, temp_dtype};
    outputs_desc_[i] = temp_input_info;
  }

  return true;
}

TensorInfo HorizonBackend::GetInputInfo(int index) {
  FDASSERT(index < NumInputs(),
           "The index: %d should less than the number of inputs: %d.", index,
           NumInputs());
  return inputs_desc_[index];
}

std::vector<TensorInfo> HorizonBackend::GetInputInfos() { return inputs_desc_; }

TensorInfo HorizonBackend::GetOutputInfo(int index) {
  FDASSERT(index < NumOutputs(),
           "The index: %d should less than the number of outputs %d.", index,
           NumOutputs());

  return outputs_desc_[index];
}

std::vector<TensorInfo> HorizonBackend::GetOutputInfos() {
  return outputs_desc_;
}

bool HorizonBackend::LoadModel(const char *model) {
  int ret = -1;
  ret = hbDNNInitializeFromFiles(&packed_dnn_handle_, &model, 1);
  if (ret != 0) {
    FDERROR << "horizon_init fail! ret=" << ret << std::endl;
    return false;
  }
  return true;
}
bool HorizonBackend::Init(const RuntimeOption &runtime_option) {
  // Init model from file
  if (!LoadModel((char *)runtime_option.model_file.data())) {
    FDERROR << "load model failed" << std::endl;
    return false;
  }

  // GetModelInputOutputInfos
  if (!GetModelInputOutputInfos()) {
    FDERROR << "get model input output infos failed" << std::endl;
    return false;
  }

  return true;
}

bool HorizonBackend::Infer(std::vector<FDTensor> &inputs,
                           std::vector<FDTensor> *outputs, bool copy_to_fd) {

  // Judge whether the input and output size are the same
  if (inputs.size() != inputs_desc_.size()) {
    FDERROR << "[HorizonBackend] Size of the inputs(" << inputs.size()
            << ") should keep same with the inputs of this model("
            << inputs_desc_.size() << ")." << std::endl;
    return false;
  }
  RUNTIME_PROFILE_LOOP_H2D_D2H_BEGIN
  int ret = -1;
  if (!infer_init_) {
    // Create input tensor memory
    int input_count = NumInputs();
    int output_count = NumOutputs();

    input_mems_ = (hbDNNTensor *)malloc(sizeof(hbDNNTensor) * input_count);
    output_mems_ = (hbDNNTensor *)malloc(sizeof(hbDNNTensor) * output_count);

    for (uint32_t i = 0; i < input_count; i++) {
      input_mems_[i].properties = input_properties_[i];

      input_mems_[i].properties.alignedShape =
          input_mems_[i].properties.validShape;

      auto current_shape = GetInputInfo(i).shape;
      auto &mem = input_mems_[i].sysMem[0];
      int intput_memSize = input_properties_[i].alignedByteSize;

      ret = hbSysAllocCachedMem(&mem, intput_memSize);

      if (ret != 0) {
        FDERROR << "hbSysAllocCachedMem fails." << std::endl;
        return false;
      }
    }

    for (uint32_t i = 0; i < output_count; i++) {

      output_mems_[i].properties = output_properties_[i];

      auto current_shape = GetOutputInfo(i).shape;
      auto &mem = output_mems_[i].sysMem[0];
      int output_memSize = output_properties_[i].alignedByteSize;

      ret = hbSysAllocCachedMem(&mem, output_memSize);
      if (ret != 0) {
        FDERROR << "hbSysAllocCachedMem fails." << std::endl;
        return false;
      }
    }
    infer_init_ = true;
  }
  // Copy input data to input tensor memory
  for (uint32_t i = 0; i < NumInputs(); i++) {
    if (inputs[i].Data() == nullptr) {
      FDERROR << "inputs[i].Data is NULL." << std::endl;
      return false;
    }
    auto &mem = input_mems_[i].sysMem[0];

    memcpy(mem.virAddr, inputs[i].Data(), inputs[i].Nbytes());
    ret = hbSysFlushMem(&mem, HB_SYS_MEM_CACHE_CLEAN);
    if (ret != 0) {
      FDERROR << "hbSysFlushMem fails." << std::endl;
      return false;
    }
  }

  hbDNNTaskHandle_t task_handle = nullptr;
  hbDNNInferCtrlParam infer_ctrl_param;
  HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);

  RUNTIME_PROFILE_LOOP_BEGIN(1)
  ret = hbDNNInfer(&task_handle, &output_mems_, input_mems_, dnn_handle_,
                   &infer_ctrl_param);
  RUNTIME_PROFILE_LOOP_END
  if (ret != 0) {
    FDERROR << "hbDNNInference fails." << std::endl;
    return false;
  }
  ret = hbDNNWaitTaskDone(task_handle, 0);
  if (ret != 0) {
    FDERROR << "hbDNNWaitTaskDone fails." << std::endl;
    return false;
  }
  ret = hbDNNReleaseTask(task_handle);
  if (ret != 0) {
    FDERROR << "hbDNNReleaseTask fails." << std::endl;
    return false;
  }
  // get result
  outputs->resize(outputs_desc_.size());
  std::vector<int64_t> temp_shape(4);
  for (size_t i = 0; i < outputs_desc_.size(); ++i) {
    temp_shape.resize(outputs_desc_[i].shape.size());
    for (int j = 0; j < outputs_desc_[i].shape.size(); ++j) {
      temp_shape[j] = outputs_desc_[i].shape[j];
    }
    (*outputs)[i].Resize(temp_shape, outputs_desc_[i].dtype,
                         outputs_desc_[i].name);

    hbSysFlushMem(&(output_mems_[i].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    auto data = (float *)(output_mems_[i].sysMem[0].virAddr);

    auto shift = output_mems_[i].properties.shift.shiftData;
    auto scale = output_mems_[i].properties.scale.scaleData;

    for (int j = 0; j < (*outputs)[i].Nbytes(); j++) {
      if (output_mems_[i].properties.quantiType == SHIFT) {
        data[j] = data[j] / (1 << shift[j]);
      } else if (output_mems_[i].properties.quantiType == SCALE) {
        data[j] = data[j] * scale[j];
      }
    }

    memcpy((*outputs)[i].MutableData(),
           (float *)output_mems_[i].sysMem[0].virAddr, (*outputs)[i].Nbytes());
  }
  RUNTIME_PROFILE_LOOP_H2D_D2H_END
  return true;
}

FDDataType HorizonBackend::HorizonTensorTypeToFDDataType(int32_t type) {
  if (type == hbDNNDataType::HB_DNN_TENSOR_TYPE_F16) {
    return FDDataType::FP16;
  }
  if (type == hbDNNDataType::HB_DNN_TENSOR_TYPE_F32) {
    return FDDataType::FP32;
  }
  if (type == hbDNNDataType::HB_DNN_TENSOR_TYPE_S8) {
    return FDDataType::INT8;
  }
  if (type == hbDNNDataType::HB_DNN_TENSOR_TYPE_S16) {
    return FDDataType::INT16;
  }
  if (type == hbDNNDataType::HB_DNN_TENSOR_TYPE_S32) {
    return FDDataType::INT32;
  }
  if (type == hbDNNDataType::HB_DNN_TENSOR_TYPE_U8) {
    return FDDataType::UINT8;
  }

  FDERROR << "FDDataType don't support this type" << std::endl;
  return FDDataType::UNKNOWN1;
}

hbDNNDataType HorizonBackend::FDDataTypeToHorizonTensorType(FDDataType type) {
  if (type == FDDataType::FP16) {
    return hbDNNDataType::HB_DNN_TENSOR_TYPE_F16;
  }
  if (type == FDDataType::FP32) {
    return hbDNNDataType::HB_DNN_TENSOR_TYPE_F32;
  }
  if (type == FDDataType::INT8) {
    return hbDNNDataType::HB_DNN_TENSOR_TYPE_S8;
  }
  if (type == FDDataType::INT16) {
    return hbDNNDataType::HB_DNN_TENSOR_TYPE_S16;
  }
  if (type == FDDataType::INT32) {
    return hbDNNDataType::HB_DNN_TENSOR_TYPE_S32;
  }
  if (type == FDDataType::UINT8) {
    return hbDNNDataType::HB_DNN_TENSOR_TYPE_U8;
  }
  FDERROR << "horizon_tensor_type don't support this type" << std::endl;

  return hbDNNDataType::HB_DNN_TENSOR_TYPE_MAX;
}

} // namespace ultra_infer
