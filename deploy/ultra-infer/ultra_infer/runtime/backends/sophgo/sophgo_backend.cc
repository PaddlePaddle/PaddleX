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
#include "ultra_infer/runtime/backends/sophgo/sophgo_backend.h"

#include <assert.h>

namespace ultra_infer {
SophgoBackend::~SophgoBackend() { bm_dev_free(handle_); }
/***************************************************************
 *  @name       GetSDKAndDeviceVersion
 *  @brief      get Sophgo sdk and device version
 *  @param      None
 *  @return     bool
 *  @note       None
 ***************************************************************/
bool SophgoBackend::GetSDKAndDeviceVersion() { return true; }

/***************************************************************
 *  @name       Init
 *  @brief      Initialize Sophgo model
 *  @param      model_file: Binary data for the Sophgo model.
 *              params_file: None
 *              option: config
 *  @return     bool
 *  @note       None
 ***************************************************************/
bool SophgoBackend::Init(const RuntimeOption &option) {
  if (option.model_from_memory_) {
    FDERROR << "SophgoBackend doesn't support load model from memory, please "
               "load model from disk."
            << std::endl;
    return false;
  }
  if (option.model_format != ModelFormat::SOPHGO) {
    FDERROR << "SophgoBackend only supports model format SOPHGO, but now it's "
            << option.model_format << "." << std::endl;
    return false;
  }
  if (option.device != Device::SOPHGOTPUD) {
    FDERROR << "SophgoBackend only supports device::SOPHGOTPUD, but now it's "
            << option.device << "." << std::endl;
    return false;
  }

  std::string model_file = option.model_file;

  // LoadModel
  if (!this->LoadModel((char *)model_file.data())) {
    FDERROR << "load model failed" << std::endl;
    return false;
  }

  // GetSDKAndDeviceVersion
  if (!this->GetSDKAndDeviceVersion()) {
    FDERROR << "get SDK and device version failed" << std::endl;
    return false;
  }

  // GetModelInputOutputInfos
  if (!this->GetModelInputOutputInfos()) {
    FDERROR << "get model input output infos failed" << std::endl;
    return false;
  }

  return true;
}

/***************************************************************
 *  @name       LoadModel
 *  @brief      read Sophgo bmodel
 *  @param      model: Binary data for the Sophgo model.
 *  @return     bool
 *  @note       None
 ***************************************************************/
bool SophgoBackend::LoadModel(void *model) {
  unsigned int card_num = 0;
  bm_status_t status = bm_get_card_num(&card_num);
  status = bm_dev_request(&handle_, 0);
  p_bmrt_ = bmrt_create(handle_);
  assert(NULL != p_bmrt_);

  bool load_status = bmrt_load_bmodel(p_bmrt_, (char *)model);
  assert(load_status);

  int network_num = bmrt_get_network_number(p_bmrt_);

  const char **net_names = NULL;
  bmrt_get_network_names(p_bmrt_, &net_names);
  net_name_ = net_names[0];
  free(net_names);

  net_info_ = bmrt_get_network_info(p_bmrt_, net_name_.c_str());
  assert(NULL != net_info_);

  return true;
}

/***************************************************************
 *  @name       GetModelInputOutputInfos
 *  @brief      Get the detailed input and output infos of Model
 *  @param      None
 *  @return     bool
 *  @note       None
 ***************************************************************/
bool SophgoBackend::GetModelInputOutputInfos() {
  inputs_desc_.resize(net_info_->input_num);
  bm_shape_t *input_shapes = net_info_->stages->input_shapes;
  for (int idx = 0; idx < net_info_->input_num; idx++) {
    std::string temp_name = (net_info_->input_names)[idx];
    std::vector<int> temp_shape{};
    temp_shape.resize(input_shapes[idx].num_dims);
    for (int i = 0; i < input_shapes[idx].num_dims; i++) {
      temp_shape[i] = input_shapes[idx].dims[i];
    }
    bm_data_type_t *input_dtypes = net_info_->input_dtypes;
    // SophgoType to FDDataType
    FDDataType temp_dtype = SophgoTensorTypeToFDDataType(*input_dtypes);
    TensorInfo temp_input_info = {temp_name, temp_shape, temp_dtype};
    inputs_desc_[idx] = temp_input_info;
  }

  outputs_desc_.resize(net_info_->output_num);
  bm_shape_t *output_shapes = net_info_->stages->output_shapes;
  for (int idx = 0; idx < net_info_->output_num; idx++) {
    std::string temp_name1 = (net_info_->output_names)[idx];
    std::vector<int> temp_shape1{};
    temp_shape1.resize(output_shapes[idx].num_dims);
    for (int i = 0; i < output_shapes[idx].num_dims; i++) {
      temp_shape1[i] = output_shapes[idx].dims[i];
    }
    bm_data_type_t *output_dtypes = net_info_->output_dtypes;
    // SophgoType to FDDataType
    FDDataType temp_dtype1 = SophgoTensorTypeToFDDataType(*output_dtypes);
    TensorInfo temp_output_info = {temp_name1, temp_shape1, temp_dtype1};
    outputs_desc_[idx] = temp_output_info;
  }
  return true;
}

TensorInfo SophgoBackend::GetInputInfo(int index) {
  FDASSERT(index < NumInputs(),
           "The index: %d should less than the number of inputs: %d.", index,
           NumInputs())
  return inputs_desc_[index];
}

std::vector<TensorInfo> SophgoBackend::GetInputInfos() { return inputs_desc_; }

TensorInfo SophgoBackend::GetOutputInfo(int index) {
  FDASSERT(index < NumOutputs(),
           "The index: %d should less than the number of outputs %d.", index,
           NumOutputs())
  return outputs_desc_[index];
}

std::vector<TensorInfo> SophgoBackend::GetOutputInfos() {
  return outputs_desc_;
}

bool SophgoBackend::Infer(std::vector<FDTensor> &inputs,
                          std::vector<FDTensor> *outputs, bool copy_to_fd) {
  int input_size = inputs.size();
  assert(input_size != 0);
  assert(input_size == NumInputs());
  bm_tensor_t input_tensors[input_size];
  bm_status_t status = BM_SUCCESS;

  RUNTIME_PROFILE_LOOP_H2D_D2H_BEGIN
  bm_data_type_t *input_dtypes = net_info_->input_dtypes;
  for (int i = 0; i < input_size; i++) {
    status = bm_malloc_device_byte(handle_, &input_tensors[i].device_mem,
                                   net_info_->max_input_bytes[i]);
    assert(BM_SUCCESS == status);
    input_tensors[i].dtype = input_dtypes[i];
    input_tensors[i].st_mode = BM_STORE_1N;
    input_tensors[i].shape = net_info_->stages[0].input_shapes[i];
    unsigned int input_byte = bmrt_tensor_bytesize(&input_tensors[i]);
    bm_memcpy_s2d_partial(handle_, input_tensors[i].device_mem,
                          (void *)inputs[i].Data(),
                          bmrt_tensor_bytesize(&input_tensors[i]));
  }

  int output_size = NumOutputs();
  bm_tensor_t output_tensors[output_size];
  for (int i = 0; i < output_size; i++) {
    status = bm_malloc_device_byte(handle_, &output_tensors[i].device_mem,
                                   net_info_->max_output_bytes[i]);
    assert(BM_SUCCESS == status);
  }

  RUNTIME_PROFILE_LOOP_BEGIN(1)
  bool launch_status = bmrt_launch_tensor_ex(
      p_bmrt_, net_name_.c_str(), input_tensors, net_info_->input_num,
      output_tensors, net_info_->output_num, true, false);
  assert(launch_status);
  status = bm_thread_sync(handle_);
  assert(status == BM_SUCCESS);
  RUNTIME_PROFILE_LOOP_END

  outputs->resize(outputs_desc_.size());
  bm_data_type_t *output_dtypes = net_info_->output_dtypes;
  for (int i = 0; i < output_size; i++) {
    int temp_bytesize = bmrt_tensor_bytesize(&output_tensors[i]); // Byte
    float *temp_out = (float *)malloc(temp_bytesize);
    bm_memcpy_d2s_partial(handle_, temp_out, output_tensors[i].device_mem,
                          temp_bytesize);

    std::vector<int64_t> temp_shape;
    temp_shape.resize(outputs_desc_[i].shape.size());
    for (int j = 0; j < outputs_desc_[i].shape.size(); ++j) {
      temp_shape[j] = outputs_desc_[i].shape[j];
    }
    (*outputs)[i].Resize(temp_shape, outputs_desc_[i].dtype,
                         outputs_desc_[i].name);

    memcpy((*outputs)[i].MutableData(), temp_out, (*outputs)[i].Nbytes());
    free(temp_out);
  }

  for (int i = 0; i < input_size; i++) {
    bm_free_device(handle_, input_tensors[i].device_mem);
  }
  for (int i = 0; i < output_size; i++) {
    bm_free_device(handle_, output_tensors[i].device_mem);
  }
  RUNTIME_PROFILE_LOOP_H2D_D2H_END

  return true;
}

/***************************************************************
 *  @name       SophgoTensorTypeToFDDataType
 *  @brief      Change SophgoTensorType To FDDataType
 *  @param      bm_data_type_t
 *  @return     None
 *  @note       None
 ***************************************************************/
FDDataType SophgoBackend::SophgoTensorTypeToFDDataType(bm_data_type_t type) {
  if (type == BM_FLOAT16) {
    return FDDataType::FP32;
  }
  if (type == BM_FLOAT32) {
    return FDDataType::FP32;
  }
  if (type == BM_INT8) {
    return FDDataType::INT8;
  }
  if (type == BM_INT16) {
    return FDDataType::INT16;
  }
  if (type == BM_INT32) {
    return FDDataType::INT32;
  }
  if (type == BM_UINT8) {
    return FDDataType::UINT8;
  }
  FDERROR << "FDDataType don't support this type" << std::endl;
  return FDDataType::UNKNOWN1;
}

/***************************************************************
 *  @name       FDDataTypeToSophgoTensorType
 *  @brief      Change FDDataType To SophgoTensorType
 *  @param      FDDataType
 *  @return     None
 *  @note       None
 ***************************************************************/
// Sophgo_tensor_type
bm_data_type_t
SophgoBackend::FDDataTypeToSophgoTensorType(ultra_infer::FDDataType type) {
  if (type == FDDataType::FP16) {
    return BM_FLOAT16;
  }
  if (type == FDDataType::FP32) {
    return BM_FLOAT32;
  }
  if (type == FDDataType::INT8) {
    return BM_INT8;
  }
  if (type == FDDataType::INT16) {
    return BM_INT16;
  }
  if (type == FDDataType::INT32) {
    return BM_INT32;
  }
  if (type == FDDataType::UINT8) {
    return BM_UINT8;
  }
  FDERROR << "Sophgo_tensor_type don't support this type" << std::endl;
  return BM_FLOAT32;
}

} // namespace ultra_infer
