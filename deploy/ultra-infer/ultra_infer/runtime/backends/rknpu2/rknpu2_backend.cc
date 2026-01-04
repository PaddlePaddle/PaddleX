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
#include "ultra_infer/runtime/backends/rknpu2/rknpu2_backend.h"
namespace ultra_infer {
RKNPU2Backend::~RKNPU2Backend() {
  if (tensor_attrs_init_) {
    if (input_attrs_ != nullptr) {
      free(input_attrs_);
    }

    if (output_attrs_ != nullptr) {
      free(output_attrs_);
    }
  }

  if (tensor_memory_init_) {
    for (uint32_t i = 0; i < io_num_.n_input; i++) {
      rknn_destroy_mem(ctx_, input_mems_[i]);
    }

    for (uint32_t i = 0; i < io_num_.n_output; i++) {
      rknn_destroy_mem(ctx_, output_mems_[i]);
    }
  }
}

/*
 *  @name       RuntimeOptionIsApplicable
 *  @brief      This function is used to determine whether the RuntimeOption
 *              meets the operating conditions of RKNPU2.
 *  @param      None
 *  @return     bool
 *  @note       None
 */
bool RKNPU2Backend::RuntimeOptionIsApplicable(
    const RuntimeOption &runtime_option) {
  if (!Supported(runtime_option.model_format, Backend::RKNPU2)) {
    FDERROR << "The model format is not supported for RKNPU2." << std::endl;
    return false;
  }

  if (!Supported(runtime_option.device, Backend::RKNPU2)) {
    FDERROR << "The device is not supported for RKNPU2." << std::endl;
    return false;
  }

  if (runtime_option.model_from_memory_) {
    FDERROR << "RKNPU2 backend doesn't support load model from memory, please "
               "load model from disk."
            << std::endl;
    return false;
  }
  return true;
}

/*
 *  @name       GetSDKAndDeviceVersion
 *  @brief      Get RKNPU2 sdk and device version.
 *  @param      None
 *  @return     bool
 *  @note       The private variable ctx_ must be initialized.
 */
bool RKNPU2Backend::GetSDKAndDeviceVersion() {
  int ret;
  ret = rknn_query(ctx_, RKNN_QUERY_SDK_VERSION, &sdk_ver_, sizeof(sdk_ver_));
  if (ret != RKNN_SUCC) {
    FDERROR << "The function(rknn_query) failed! ret=" << ret << std::endl;
    return false;
  }
  FDINFO << "rknpu2 runtime version: " << sdk_ver_.api_version << std::endl;
  FDINFO << "rknpu2 driver version: " << sdk_ver_.drv_version << std::endl;
  return true;
}

/*
 *  @name      BuildOption
 *  @brief     Save option and set core mask.
 *  @param     RKNPU2BackendOption
 *  @note      None
 */
void RKNPU2Backend::BuildOption(const RKNPU2BackendOption &option) {
  option_ = option;

  // save cpu_name
  option_.cpu_name = option.cpu_name;

  // save context
  option_.core_mask = option.core_mask;

  // set core mask
  if (option_.cpu_name == rknpu2::CpuName::RK3588) {
    if (!SetCoreMask(option_.core_mask)) {
      FDERROR << "set core mask failed" << std::endl;
    }
  }
}

/***************************************************************
 *  @name       Init
 *  @brief      Initialize RKNN model
 *  @param      model_file: Binary data for the RKNN model or the path of RKNN
 *  @return     bool
 *  @note       None
 ***************************************************************/
bool RKNPU2Backend::Init(const RuntimeOption &runtime_option) {
  if (!RuntimeOptionIsApplicable(runtime_option)) {
    FDERROR << "Runtime option is not applicable." << std::endl;
    return false;
  }

  if (!LoadModel((char *)runtime_option.model_file.data())) {
    FDERROR << "Load model failed" << std::endl;
    return false;
  }

  if (!InitInputAndOutputNumber()) {
    FDERROR << "Init input and output number failed" << std::endl;
    return false;
  }

  if (!GetSDKAndDeviceVersion()) {
    FDERROR << "Get SDK and device version failed" << std::endl;
    return false;
  }

  BuildOption(runtime_option.rknpu2_option);

  if (!InitInputAndOutputInformation()) {
    FDERROR << "Get model input output information failed" << std::endl;
    return false;
  }

  return true;
}

/*
 *  @name       SetCoreMask
 *  @brief      Set NPU core for model
 *  @param      core_mask: The specification of NPU core setting.
 *  @return     bool
 *  @note       Only support RK3588
 */
bool RKNPU2Backend::SetCoreMask(const rknpu2::CoreMask &core_mask) const {
  if (option_.cpu_name != rknpu2::CpuName::RK3588) {
    FDINFO << "SetCoreMask only support when soc is RK3588." << std::endl;
    return false;
  }

  int ret = rknn_set_core_mask(ctx_, static_cast<rknn_core_mask>(core_mask));
  if (ret != RKNN_SUCC) {
    FDERROR << "The function(rknn_set_core_mask) failed! ret=" << ret
            << std::endl;
    return false;
  }
  return true;
}

/*
 *  @name       LoadModel
 *  @brief      Read the model and initialize rknn context.
 *  @param      model: Binary data for the RKNN model or the path of RKNN model.
 *  @return     bool
 *  @note       None
 */
bool RKNPU2Backend::LoadModel(void *model) {
  int ret = RKNN_SUCC;
  ret = rknn_init(&ctx_, model, 0, 0, nullptr);
  if (ret != RKNN_SUCC) {
    FDERROR << "The function(rknn_init) failed! ret=" << ret << std::endl;
    return false;
  }
  return true;
}

/*
 *  @name       InitInputAndOutputNumber
 *  @brief      Initialize io_num_.
 *  @param
 *  @return     bool
 *  @note       The private variable ctx must be initialized to use this
 * function.
 */
bool RKNPU2Backend::InitInputAndOutputNumber() {
  if (io_num_init_) {
    FDERROR << "The private variable io_num_ has been initialized."
            << std::endl;
    return false;
  }
  int ret = RKNN_SUCC;
  ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
  if (ret != RKNN_SUCC) {
    FDERROR << "The function(rknn_query) failed! ret=" << ret << std::endl;
    return false;
  }
  io_num_init_ = true;
  return true;
}

/*
 *  @name       InitRKNNTensorAddress
 *  @brief      Allocate memory for input_attrs_ and output_attrs_.
 *  @param      None
 *  @return     bool
 *  @note       None
 */
bool RKNPU2Backend::InitRKNNTensorAddress() {
  if (tensor_attrs_init_) {
    FDERROR << "Private variable input_attrs_ and output_attrs_ memory has "
               "been allocated. Please do not allocate memory repeatedly or "
               "memory leak may occur."
            << std::endl;
    return false;
  }

  if (!io_num_init_) {
    InitInputAndOutputNumber();
  }

  if (io_num_.n_input == 0) {
    FDERROR << "The number of input tensors is 0." << std::endl;
    return false;
  }

  if (io_num_.n_output == 0) {
    FDERROR << "The number of output tensors is 0." << std::endl;
    return false;
  }

  // Allocate memory for private variable input_attrs_.
  input_attrs_ =
      (rknn_tensor_attr *)malloc(sizeof(rknn_tensor_attr) * io_num_.n_input);
  memset(input_attrs_, 0, io_num_.n_input * sizeof(rknn_tensor_attr));
  for (uint32_t i = 0; i < io_num_.n_input; i++) {
    int ret = RKNN_SUCC;
    input_attrs_[i].index = i;
    ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &(input_attrs_[i]),
                     sizeof(rknn_tensor_attr));

    if (ret != RKNN_SUCC) {
      FDERROR << "The function(rknn_query) failed! ret=" << ret << std::endl;
      return false;
    }

    if ((input_attrs_[i].fmt != RKNN_TENSOR_NHWC) &&
        (input_attrs_[i].fmt != RKNN_TENSOR_UNDEFINED)) {
      FDERROR << "rknpu2_backend only support input format is NHWC or UNDEFINED"
              << std::endl;
      return false;
    }

    DumpTensorAttr(input_attrs_[i]);
  }

  // Allocate memory for private variable output_attrs_.
  output_attrs_ =
      (rknn_tensor_attr *)malloc(sizeof(rknn_tensor_attr) * io_num_.n_output);
  memset(output_attrs_, 0, io_num_.n_output * sizeof(rknn_tensor_attr));
  for (uint32_t i = 0; i < io_num_.n_output; i++) {
    int ret = RKNN_SUCC;
    output_attrs_[i].index = i;
    ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs_[i]),
                     sizeof(rknn_tensor_attr));

    if (ret != RKNN_SUCC) {
      FDERROR << "The function(rknn_query) failed! ret=" << ret << std::endl;
      return false;
    }

    // UltraInfer Only support postprocess when output type is fp32,
    // so output_attrs_.type needs to be fixed as RKNN_TENSOR_FLOAT32.
    output_attrs_[i].type = RKNN_TENSOR_FLOAT32;
    DumpTensorAttr(output_attrs_[i]);
  }
  tensor_attrs_init_ = true;
  return true;
}

/*
 *  @name       InitInputAndOutputInformation
 *  @brief      Get the detailed input and output information of Model
 *  @param      None
 *  @return     bool
 *  @note       None
 */
bool RKNPU2Backend::InitInputAndOutputInformation() {
  if (!io_num_init_) {
    InitInputAndOutputNumber();
  }

  if (!tensor_attrs_init_) {
    InitRKNNTensorAddress();
  }

  if (io_num_.n_input == 0) {
    FDERROR << "The number of input tensors is 0." << std::endl;
    return false;
  }

  if (io_num_.n_output == 0) {
    FDERROR << "The number of output tensors is 0." << std::endl;
    return false;
  }

  inputs_desc_.resize(io_num_.n_input);
  outputs_desc_.resize(io_num_.n_output);

  // Get input info and copy to input tensor info
  for (uint32_t i = 0; i < io_num_.n_input; i++) {
    // Copy input_attrs_ to input tensor info
    std::string temp_name = input_attrs_[i].name;
    std::vector<int> temp_shape{};
    temp_shape.resize(input_attrs_[i].n_dims);
    for (int j = 0; j < input_attrs_[i].n_dims; j++) {
      temp_shape[j] = (int)input_attrs_[i].dims[j];
    }
    FDDataType temp_dtype =
        ultra_infer::RKNPU2Backend::RknnTensorTypeToFDDataType(
            input_attrs_[i].type);
    TensorInfo temp_input_info = {temp_name, temp_shape, temp_dtype};
    inputs_desc_[i] = temp_input_info;
  }

  for (uint32_t i = 0; i < io_num_.n_output; i++) {
    // If the output dimension is 3, the runtime will automatically change it
    // to 4. Obviously, this is wrong, and manual correction is required here.
    int n_dims = static_cast<int>(output_attrs_[i].n_dims);
    if ((n_dims == 4) && (output_attrs_[i].dims[3] == 1)) {
      n_dims--;
    }

    // Copy output_attrs_ to output tensor
    std::string temp_name = output_attrs_[i].name;
    std::vector<int> temp_shape{};
    temp_shape.resize(n_dims);
    for (int j = 0; j < n_dims; j++) {
      temp_shape[j] = (int)output_attrs_[i].dims[j];
    }

    // The data type of output data is changed to FP32
    FDDataType temp_dtype = FDDataType::FP32;
    TensorInfo temp_input_info = {temp_name, temp_shape, temp_dtype};
    outputs_desc_[i] = temp_input_info;
  }
  return true;
}

/*
 *  @name       DumpTensorAttr
 *  @brief      Get the model's detailed inputs and outputs
 *  @param      rknn_tensor_attr
 *  @return     None
 *  @note       None
 */
void RKNPU2Backend::DumpTensorAttr(rknn_tensor_attr &attr) {
  printf("index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], "
         "n_elems=%d, size=%d, fmt=%s, type=%s, "
         "qnt_type=%s, zp=%d, scale=%f, pass_through=%d\n",
         attr.index, attr.name, attr.n_dims, attr.dims[0], attr.dims[1],
         attr.dims[2], attr.dims[3], attr.n_elems, attr.size,
         get_format_string(attr.fmt), get_type_string(attr.type),
         get_qnt_type_string(attr.qnt_type), attr.zp, attr.scale,
         attr.pass_through);
}

TensorInfo RKNPU2Backend::GetInputInfo(int index) {
  FDASSERT(index < NumInputs(),
           "The index: %d should less than the number of inputs: %d.", index,
           NumInputs())
  return inputs_desc_[index];
}

std::vector<TensorInfo> RKNPU2Backend::GetInputInfos() { return inputs_desc_; }

TensorInfo RKNPU2Backend::GetOutputInfo(int index) {
  FDASSERT(index < NumOutputs(),
           "The index: %d should less than the number of outputs %d.", index,
           NumOutputs())
  return outputs_desc_[index];
}

std::vector<TensorInfo> RKNPU2Backend::GetOutputInfos() {
  return outputs_desc_;
}

/*
 *  @name       InitRKNNTensorMemory
 *  @brief      Allocate memory for input and output tensors.
 *  @param      std::vector<FDTensor>& inputs
 *  @return     None
 *  @note       None
 */
bool RKNPU2Backend::InitRKNNTensorMemory(std::vector<FDTensor> &inputs) {
  if (tensor_memory_init_) {
    FDERROR << "Private variable input_mems_ and output_mems_ memory has "
               "been allocated. Please do not allocate memory repeatedly or "
               "memory leak may occur."
            << std::endl;
    return false;
  }
  int ret = RKNN_SUCC;
  input_mems_.resize(io_num_.n_input);
  output_mems_.resize(io_num_.n_output);
  for (uint32_t i = 0; i < io_num_.n_input; i++) {
    // Judge whether the input and output types are the same
    rknn_tensor_type input_type =
        ultra_infer::RKNPU2Backend::FDDataTypeToRknnTensorType(inputs[i].dtype);
    if (input_type != input_attrs_[i].type) {
      FDWARNING << "The input tensor type != model's inputs type."
                << "The input_type need "
                << get_type_string(input_attrs_[i].type) << ",but inputs[" << i
                << "].type is " << get_type_string(input_type) << std::endl;
    }

    // Create input tensor memory
    input_attrs_[i].type = input_type;
    input_attrs_[i].size = inputs[i].Nbytes();
    input_attrs_[i].size_with_stride = inputs[i].Nbytes();

    input_mems_[i] = rknn_create_mem(ctx_, inputs[i].Nbytes());
    if (input_mems_[i] == nullptr) {
      FDERROR << "The function(rknn_create_mem) failed! ret=" << ret
              << std::endl;
      return false;
    }

    // Set input tensor memory
    ret = rknn_set_io_mem(ctx_, input_mems_[i], &input_attrs_[i]);
    if (ret != RKNN_SUCC) {
      FDERROR << "The function(rknn_set_io_mem) failed! ret=" << ret
              << std::endl;
      return false;
    }
  }

  for (uint32_t i = 0; i < io_num_.n_output; ++i) {
    // Most post-processing does not support the fp16 format.
    uint32_t output_size = output_attrs_[i].n_elems * sizeof(float);
    output_mems_[i] = rknn_create_mem(ctx_, output_size);
    if (output_mems_[i] == nullptr) {
      FDERROR << "The function(rknn_create_mem) failed! ret=" << ret
              << std::endl;
      return false;
    }

    // Set output tensor memory
    ret = rknn_set_io_mem(ctx_, output_mems_[i], &output_attrs_[i]);
    if (ret != RKNN_SUCC) {
      FDERROR << "The function(rknn_set_io_mem) failed! ret=" << ret
              << std::endl;
      return false;
    }
  }

  tensor_memory_init_ = true;
  return true;
}

bool RKNPU2Backend::Infer(std::vector<FDTensor> &inputs,
                          std::vector<FDTensor> *outputs, bool copy_to_fd) {
  if (!tensor_memory_init_) {
    if (!InitRKNNTensorMemory(inputs)) {
      FDERROR << "Init tensor memory failed." << std::endl;
    }
  }

  int ret = RKNN_SUCC;
  // Judge whether the input and output size are the same
  if (inputs.size() != inputs_desc_.size()) {
    FDERROR << "[RKNPU2Backend] Size of the inputs(" << inputs.size()
            << ") should keep same with the inputs of this model("
            << inputs_desc_.size() << ")." << std::endl;
    return false;
  }

  // Copy input data to input tensor memory
  for (uint32_t i = 0; i < io_num_.n_input; i++) {
    uint32_t width = input_attrs_[i].dims[2];
    uint32_t stride = input_attrs_[i].w_stride;
    if (width == stride) {
      if (inputs[i].Data() == nullptr) {
        FDERROR << "inputs[0].Data is NULL." << std::endl;
        return false;
      }
      memcpy(input_mems_[i]->virt_addr, inputs[i].Data(), inputs[i].Nbytes());
    } else {
      FDERROR << "[RKNPU2Backend] only support width == stride." << std::endl;
      return false;
    }
  }

  // run rknn
  ret = rknn_run(ctx_, nullptr);
  if (ret != RKNN_SUCC) {
    FDERROR << "rknn run error! ret=" << ret << std::endl;
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
    memcpy((*outputs)[i].MutableData(), (float *)output_mems_[i]->virt_addr,
           (*outputs)[i].Nbytes());
  }

  return true;
}

/*
 *  @name       RknnTensorTypeToFDDataType
 *  @brief      Change RknnTensorType To FDDataType
 *  @param      rknn_tensor_type
 *  @return     None
 *  @note       Most post-processing does not support the fp16 format.
 *              Therefore, if the input is FP16, the output will be FP32.
 */
FDDataType RKNPU2Backend::RknnTensorTypeToFDDataType(rknn_tensor_type type) {
  if (type == rknn_tensor_type::RKNN_TENSOR_FLOAT16) {
    return FDDataType::FP32;
  }
  if (type == rknn_tensor_type::RKNN_TENSOR_FLOAT32) {
    return FDDataType::FP32;
  }
  if (type == rknn_tensor_type::RKNN_TENSOR_INT8) {
    return FDDataType::INT8;
  }
  if (type == rknn_tensor_type::RKNN_TENSOR_INT16) {
    return FDDataType::INT16;
  }
  if (type == rknn_tensor_type::RKNN_TENSOR_INT32) {
    return FDDataType::INT32;
  }
  if (type == rknn_tensor_type::RKNN_TENSOR_UINT8) {
    return FDDataType::UINT8;
  }
  if (type == rknn_tensor_type::RKNN_TENSOR_BOOL) {
    return FDDataType::BOOL;
  }
  FDERROR << "FDDataType don't support this type" << std::endl;
  return FDDataType::UNKNOWN1;
}

/*
 *  @name       FDDataTypeToRknnTensorType
 *  @brief      Change FDDataType To RknnTensorType
 *  @param      FDDataType
 *  @return     None
 *  @note       None
 */
rknn_tensor_type
RKNPU2Backend::FDDataTypeToRknnTensorType(ultra_infer::FDDataType type) {
  if (type == FDDataType::FP16) {
    return rknn_tensor_type::RKNN_TENSOR_FLOAT16;
  }
  if (type == FDDataType::FP32) {
    return rknn_tensor_type::RKNN_TENSOR_FLOAT32;
  }
  if (type == FDDataType::INT8) {
    return rknn_tensor_type::RKNN_TENSOR_INT8;
  }
  if (type == FDDataType::INT16) {
    return rknn_tensor_type::RKNN_TENSOR_INT16;
  }
  if (type == FDDataType::INT32) {
    return rknn_tensor_type::RKNN_TENSOR_INT32;
  }
  if (type == FDDataType::UINT8) {
    return rknn_tensor_type::RKNN_TENSOR_UINT8;
  }
  if (type == FDDataType::BOOL) {
    return rknn_tensor_type::RKNN_TENSOR_BOOL;
  }
  FDERROR << "rknn_tensor_type don't support this type" << std::endl;
  return RKNN_TENSOR_TYPE_MAX;
}
} // namespace ultra_infer
