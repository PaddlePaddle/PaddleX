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

#include "ultra_infer/runtime/backends/om/om_backend.h"

#include "acl/acl.h"
#include <chrono>
#include <sys/stat.h>

namespace ultra_infer {

bool OmBackend::aclInitFlag = false;

OmBackend::~OmBackend() {
  FreeInputBuffer();
  FreeOutputBuffer();
  DestroyInput();
  DestroyOutput();
  DestroyResource();
}

TensorInfo OmBackend::GetInputInfo(int index) {
  FDASSERT(index < NumInputs(),
           "The index: %d should less than the number of inputs: %d.", index,
           NumInputs());
  return inputs_desc_[index];
}

std::vector<TensorInfo> OmBackend::GetInputInfos() { return inputs_desc_; }

TensorInfo OmBackend::GetOutputInfo(int index) {
  FDASSERT(index < NumOutputs(),
           "The index: %d should less than the number of outputs %d.", index,
           NumOutputs());

  return outputs_desc_[index];
}

std::vector<TensorInfo> OmBackend::GetOutputInfos() { return outputs_desc_; }

bool OmBackend::Init(const RuntimeOption &runtime_option) {
  deviceId_ = runtime_option.device_id;
  // ACL init
  aclError ret = InitResource();
  if (ret != true) {
    FDERROR << "execute InitResource failed, errorCode = "
            << static_cast<int32_t>(ret);
    return false;
  }

  // model init;
  const char *omModelPath = (char *)runtime_option.model_file.data();
  FDINFO << "omModelPath = " << omModelPath;
  ret = LoadModel(omModelPath);
  if (ret != true) {
    FDERROR << "execute LoadModel failed";
    return false;
  }

  // build input/output info
  ret = CreateModelDesc();
  if (ret != true) {
    FDERROR << "execute CreateModelDesc failed";
    return false;
  }
  ret = CreateInput();
  if (ret != true) {
    FDERROR << "execute CreateInput failed";
    FreeInputBuffer();
    return false;
  }
  ret = CreateOutput();
  if (ret != true) {
    FDERROR << "execute CreateOutput failed";
    FreeInputBuffer();
    return false;
  }

  return true;
}

bool OmBackend::Infer(std::vector<FDTensor> &inputs,
                      std::vector<FDTensor> *outputs, bool copy_to_fd) {
  // set context
  aclError aclRet = aclrtSetCurrentContext(context_);
  if (aclRet != ACL_SUCCESS) {
    FDERROR << "aclrtSetCurrentContext failed"
            << ", errorCode is " << static_cast<int32_t>(aclRet);
    return false;
  }

  // Judge whether the input and output size are the same
  if (inputs.size() != inputs_desc_.size()) {
    FDERROR << "[OmBackend] Size of the inputs(" << inputs.size()
            << ") should keep same with the inputs of this model("
            << inputs_desc_.size() << ")." << std::endl;
    FreeInputBuffer();
    return false;
  }

  // cp input tensor to inputBuffer
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].Data() == nullptr) {
      FDERROR << "inputs[i].Data is NULL." << std::endl;
      return false;
    }
    size_t modelInputSize = aclmdlGetInputSizeByIndex(modelDesc_, i);
    aclRet = aclrtMemcpy(inputBuffer[i], modelInputSize, inputs[i].Data(),
                         inputs[i].Nbytes(), ACL_MEMCPY_DEVICE_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
      FDERROR << "memcpy d2d failed. buffer size is " << modelInputSize
              << ", inputs[i].Nbytes() is " << inputs[i].Nbytes()
              << ", errorCode is " << static_cast<int32_t>(aclRet);
      return false;
    }
  }

  bool ret = Execute();
  if (ret != true) {
    FDERROR << "execute inference failed";
    FreeInputBuffer();
    DestroyInput();
    DestroyOutput();
    return false;
  }

  // cp outputbuffer to outputs
  outputs->resize(outputs_desc_.size());
  std::vector<int64_t> temp_shape(4);
  for (size_t i = 0; i < outputs_desc_.size(); ++i) {
    temp_shape.resize(outputs_desc_[i].shape.size());
    for (int j = 0; j < outputs_desc_[i].shape.size(); ++j) {
      temp_shape[j] = outputs_desc_[i].shape[j];
    }
    (*outputs)[i].Resize(temp_shape, outputs_desc_[i].dtype,
                         outputs_desc_[i].name);
    size_t modelOutputSize = aclmdlGetOutputSizeByIndex(modelDesc_, i);
    if (modelOutputSize != (*outputs)[i].Nbytes()) {
      FDERROR << "output size is not match, index: " << i
              << ", modelOutputSize:" << modelOutputSize
              << ", (*outputs)[i].Nbytes():" << (*outputs)[i].Nbytes();
      return false;
    }
    aclError aclRet = aclrtMemcpy(
        (*outputs)[i].MutableData(), (*outputs)[i].Nbytes(), outputBuffer[i],
        (*outputs)[i].Nbytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_SUCCESS) {
      FDERROR << "memcpy h2d failed. buffer size is " << (*outputs)[i].Nbytes()
              << ", errorCode is " << static_cast<int32_t>(aclRet);
      return false;
    }
  }

  return true;
}

bool OmBackend::InitResource() {
  // ACL init
  aclError ret;
  if (aclInitFlag == false) {
    ret = aclInit(NULL);
    if (ret != ACL_SUCCESS) {
      FDERROR << "acl init failed, errorCode = " << static_cast<int32_t>(ret);
      return false;
    }
    aclInitFlag = true;
  }
  // set device
  ret = aclrtSetDevice(deviceId_);
  if (ret != ACL_SUCCESS) {
    FDERROR << "acl set device" << deviceId_
            << " failed, errorCode = " << static_cast<int32_t>(ret);
    return false;
  }

  // create context (set current)
  ret = aclrtCreateContext(&context_, deviceId_);
  if (ret != ACL_SUCCESS) {
    FDERROR << "acl create context failed, deviceId" << deviceId_
            << ", errorCode = " << static_cast<int32_t>(ret);
    return false;
  }

  // create stream
  ret = aclrtCreateStream(&stream_);
  if (ret != ACL_SUCCESS) {
    FDERROR << "acl create stream failed, deviceId" << deviceId_
            << ", errorCode = " << static_cast<int32_t>(ret);
    return false;
  }

  // get run mode
  // runMode is ACL_HOST which represents app is running in host
  // runMode is ACL_DEVICE which represents app is running in device
  aclrtRunMode runMode;
  ret = aclrtGetRunMode(&runMode);
  if (ret != ACL_SUCCESS) {
    FDERROR << "acl get run mode failed, errorCode = "
            << static_cast<int32_t>(ret);
    return false;
  }

  return true;
}

bool OmBackend::LoadModel(const char *modelPath) {
  if (loadFlag_) {
    FDERROR << "model has already been loaded";
    return false;
  }
  aclError ret = aclmdlQuerySize(modelPath, &modelWorkSize_, &modelWeightSize_);
  if (ret != ACL_SUCCESS) {
    FDERROR << "query model false, model file is" << modelPath
            << ", errorCode is " << static_cast<int32_t>(ret);
    return false;
  }
  // using ACL_MEM_MALLOC_HUGE_FIRST to malloc memory, huge memory is preferred
  // to use and huge memory can improve performance.
  ret = aclrtMalloc(&modelWorkPtr_, modelWorkSize_, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) {
    FDERROR << "malloc buffer for work failed, require size is "
            << modelWorkSize_ << ", errorCode is " << static_cast<int32_t>(ret);
    return false;
  }

  // using ACL_MEM_MALLOC_HUGE_FIRST to malloc memory, huge memory is preferred
  // to use and huge memory can improve performance.
  ret = aclrtMalloc(&modelWeightPtr_, modelWeightSize_,
                    ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) {
    FDERROR << "malloc buffer for weight failed, require size is "
            << modelWeightSize_ << ", errorCode is "
            << static_cast<int32_t>(ret);
    return false;
  }

  ret = aclmdlLoadFromFileWithMem(modelPath, &modelId_, modelWorkPtr_,
                                  modelWorkSize_, modelWeightPtr_,
                                  modelWeightSize_);
  if (ret != ACL_SUCCESS) {
    FDERROR << "load model from file failed, model file is " << modelPath
            << ", errorCode is " << static_cast<int32_t>(ret);
    return false;
  }

  loadFlag_ = true;
  FDINFO << "load model " << modelPath << " success";
  return true;
}

bool OmBackend::Execute() {
  aclError ret = aclmdlExecute(modelId_, input_, output_);
  if (ret != ACL_SUCCESS) {
    FDERROR << "execute model failed, modelId is " << modelId_
            << ", errorCode is " << static_cast<int32_t>(ret);
    return false;
  }
  return true;
}

bool OmBackend::CreateModelDesc() {
  modelDesc_ = aclmdlCreateDesc();
  if (modelDesc_ == nullptr) {
    FDERROR << "create model description failed";
    return false;
  }

  aclError ret = aclmdlGetDesc(modelDesc_, modelId_);
  if (ret != ACL_SUCCESS) {
    FDERROR << "get model description failed, modelId is " << modelId_
            << ", errorCode is " << static_cast<int32_t>(ret);
    return false;
  }
  return true;
}

bool OmBackend::CreateInput() {
  // om used in this sample has only one input
  if (modelDesc_ == nullptr) {
    FDERROR << "no model description, create input failed";
    return false;
  }

  // input:aclmdlDataset
  input_ = aclmdlCreateDataset();
  if (input_ == nullptr) {
    FDERROR << "can't create dataset, create input failed";
    return false;
  }

  // get input nums
  size_t inputNum = aclmdlGetNumInputs(modelDesc_);
  inputs_desc_.resize(inputNum);
  inputBuffer.resize(inputNum, nullptr);
  // inputBuffer = {nullptr};
  for (size_t i = 0; i < inputNum; ++i) {
    // get input size
    size_t modelInputSize = aclmdlGetInputSizeByIndex(modelDesc_, i);
    aclError ret =
        aclrtMalloc(&inputBuffer[i], modelInputSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      FDERROR << "can't malloc buffer, size is " << modelInputSize
              << ", errorCode is " << static_cast<int32_t>(ret);
      return false;
    }
    // inputData:aclDataBuffer
    aclDataBuffer *inputData =
        aclCreateDataBuffer(inputBuffer[i], modelInputSize);
    if (inputData == nullptr) {
      FDERROR << "can't create data buffer, create input failed";
      return false;
    }

    // add aclDataBuffer to input
    ret = aclmdlAddDatasetBuffer(input_, inputData);
    if (ret != ACL_SUCCESS) {
      FDERROR << "add input dataset buffer failed, errorCode is "
              << static_cast<int32_t>(ret);
      (void)aclDestroyDataBuffer(inputData);
      inputData = nullptr;
      return false;
    }

    // get name/shape/dtype of input to build inputs_desc_
    const char *name;
    name = aclmdlGetInputNameByIndex(modelDesc_, i);
    std::string temp_name = name;

    std::vector<int> temp_shape{};
    aclmdlIODims dims;
    ret = aclmdlGetInputDims(modelDesc_, i, &dims);
    if (ret != ACL_SUCCESS) {
      FDERROR << "get input tensor dims fail! ret=" << ret << std::endl;
      return false;
    }
    int n_dims = (int)dims.dimCount;
    temp_shape.resize(n_dims);
    for (int j = 0; j < n_dims; j++) {
      temp_shape[j] = (int)dims.dims[j];
    }

    aclDataType dtype = aclmdlGetInputDataType(modelDesc_, i);
    FDDataType temp_dtype;
    switch (dtype) {
    case ACL_BOOL:
      temp_dtype = FDDataType::BOOL;
      break;
    case ACL_UINT8:
      temp_dtype = FDDataType::UINT8;
      break;
    case ACL_INT8:
      temp_dtype = FDDataType::INT8;
      break;
    case ACL_INT16:
      temp_dtype = FDDataType::INT16;
      break;
    case ACL_INT32:
      temp_dtype = FDDataType::INT32;
      break;
    case ACL_INT64:
      temp_dtype = FDDataType::INT64;
      break;
    case ACL_FLOAT16:
      temp_dtype = FDDataType::FP16;
      break;
    case ACL_FLOAT:
      temp_dtype = FDDataType::FP32;
      break;
    case ACL_DOUBLE:
      temp_dtype = FDDataType::FP64;
      break;
    default:
      FDERROR << "unsupported input tensor dtype: " << (int)dtype;
      return false;
    }
    TensorInfo temp_input_info = {temp_name, temp_shape, temp_dtype};
    inputs_desc_[i] = temp_input_info;
  }
  return true;
}

bool OmBackend::CreateOutput() {
  if (modelDesc_ == nullptr) {
    FDERROR << "no model description, create output failed";
    return false;
  }

  output_ = aclmdlCreateDataset();
  if (output_ == nullptr) {
    FDERROR << "can't create dataset, create output failed";
    return false;
  }

  size_t outputSize = aclmdlGetNumOutputs(modelDesc_);
  outputs_desc_.resize(outputSize);
  outputBuffer.resize(outputSize, nullptr);
  for (size_t i = 0; i < outputSize; ++i) {
    size_t modelOutputSize = aclmdlGetOutputSizeByIndex(modelDesc_, i);
    aclError ret = aclrtMalloc(&outputBuffer[i], modelOutputSize,
                               ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      FDERROR << "can't malloc buffer, size is " << modelOutputSize
              << ", errorCode is " << static_cast<int32_t>(ret);
      return false;
    }

    aclDataBuffer *outputData =
        aclCreateDataBuffer(outputBuffer[i], modelOutputSize);
    if (outputData == nullptr) {
      FDERROR << "can't create data buffer, create output failed";
      return false;
    }

    ret = aclmdlAddDatasetBuffer(output_, outputData);
    if (ret != ACL_SUCCESS) {
      FDERROR << "add output dataset buffer failed, errorCode is "
              << static_cast<int32_t>(ret);
      (void)aclDestroyDataBuffer(outputData);
      return false;
    }

    const char *name;
    name = aclmdlGetOutputNameByIndex(modelDesc_, i);
    std::string temp_name = name;

    std::vector<int> temp_shape{};
    aclmdlIODims dims;
    ret = aclmdlGetOutputDims(modelDesc_, i, &dims);
    if (ret != ACL_SUCCESS) {
      FDERROR << "get output tensor dims fail! ret=" << ret << std::endl;
      return false;
    }
    int n_dims = (int)dims.dimCount;
    temp_shape.resize(n_dims);
    for (int j = 0; j < n_dims; j++) {
      temp_shape[j] = (int)dims.dims[j];
    }

    aclDataType dtype = aclmdlGetOutputDataType(modelDesc_, i);
    FDDataType temp_dtype;
    switch (dtype) {
    case ACL_BOOL:
      temp_dtype = FDDataType::BOOL;
      break;
    case ACL_UINT8:
      temp_dtype = FDDataType::UINT8;
      break;
    case ACL_INT8:
      temp_dtype = FDDataType::INT8;
      break;
    case ACL_INT16:
      temp_dtype = FDDataType::INT16;
      break;
    case ACL_INT32:
      temp_dtype = FDDataType::INT32;
      break;
    case ACL_INT64:
      temp_dtype = FDDataType::INT64;
      break;
    case ACL_FLOAT16:
      temp_dtype = FDDataType::FP16;
      break;
    case ACL_FLOAT:
      temp_dtype = FDDataType::FP32;
      break;
    case ACL_DOUBLE:
      temp_dtype = FDDataType::FP64;
      break;
    default:
      FDERROR << "unsupported output tensor dtype: " << (int)dtype;
      return false;
    }
    TensorInfo temp_output_info = {temp_name, temp_shape, temp_dtype};
    outputs_desc_[i] = temp_output_info;
  }
  return true;
}

void OmBackend::FreeInputBuffer() {
  for (int i = 0; i < (int)inputs_desc_.size(); ++i) {
    if (inputBuffer[i] != nullptr) {
      (void)aclrtFree(inputBuffer[i]);
      inputBuffer[i] = nullptr;
    }
  }
}

void OmBackend::FreeOutputBuffer() {
  for (int i = 0; i < (int)outputs_desc_.size(); ++i) {
    if (outputBuffer[i] != nullptr) {
      (void)aclrtFree(outputBuffer[i]);
      outputBuffer[i] = nullptr;
    }
  }
}

void OmBackend::DestroyInput() {
  if (input_ == nullptr) {
    return;
  }

  for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(input_); ++i) {
    aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(input_, i);
    (void)aclDestroyDataBuffer(dataBuffer);
  }
  (void)aclmdlDestroyDataset(input_);
  input_ = nullptr;
}

void OmBackend::DestroyOutput() {
  if (output_ == nullptr) {
    return;
  }

  for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output_); ++i) {
    aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(output_, i);
    void *data = aclGetDataBufferAddr(dataBuffer);
    (void)aclrtFree(data);
    (void)aclDestroyDataBuffer(dataBuffer);
  }

  (void)aclmdlDestroyDataset(output_);
  output_ = nullptr;
}

void OmBackend::DestroyResource() {
  // set context
  aclError ret = aclrtSetCurrentContext(context_);
  if (ret != ACL_SUCCESS) {
    FDERROR << "aclrtSetCurrentContext failed"
            << ", errorCode is " << static_cast<int32_t>(ret);
    return;
  }
  if (stream_ != nullptr) {
    ret = aclrtDestroyStream(stream_);
    if (ret != ACL_SUCCESS) {
      FDERROR << "destroy stream failed, errorCode = "
              << static_cast<int32_t>(ret);
    }
    stream_ = nullptr;
  }

  if (context_ != nullptr) {
    ret = aclrtDestroyContext(context_);
    if (ret != ACL_SUCCESS) {
      FDERROR << "destroy context failed, errorCode = "
              << static_cast<int32_t>(ret);
    }
    context_ = nullptr;
  }

  ret = aclrtResetDevice(deviceId_);
  if (ret != ACL_SUCCESS) {
    FDERROR << "reset device " << deviceId_
            << " failed, errorCode = " << static_cast<int32_t>(ret);
  }

  if (aclInitFlag == true) {
    ret = aclFinalize();
    if (ret != ACL_SUCCESS) {
      FDERROR << "finalize acl failed, errorCode = "
              << static_cast<int32_t>(ret);
    }
    aclInitFlag = false;
  }
}

} // namespace ultra_infer
