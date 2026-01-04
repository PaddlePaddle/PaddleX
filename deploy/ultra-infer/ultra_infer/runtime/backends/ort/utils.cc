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

#include "ultra_infer/runtime/backends/ort/utils.h"

#include "ultra_infer/utils/utils.h"

namespace ultra_infer {

ONNXTensorElementDataType GetOrtDtype(const FDDataType &fd_dtype) {
  if (fd_dtype == FDDataType::FP32) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  } else if (fd_dtype == FDDataType::FP64) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
  } else if (fd_dtype == FDDataType::INT32) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  } else if (fd_dtype == FDDataType::INT64) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  } else if (fd_dtype == FDDataType::UINT8) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  } else if (fd_dtype == FDDataType::INT8) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
  } else if (fd_dtype == FDDataType::FP16) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  }
  FDERROR << "Unrecognized fastdeply data type:" << Str(fd_dtype) << "."
          << std::endl;
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
}

FDDataType GetFdDtype(const ONNXTensorElementDataType &ort_dtype) {
  if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return FDDataType::FP32;
  } else if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
    return FDDataType::FP64;
  } else if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    return FDDataType::INT32;
  } else if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return FDDataType::INT64;
  } else if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    return FDDataType::FP16;
  } else if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
    return FDDataType::UINT8;
  } else if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) {
    return FDDataType::INT8;
  }
  FDERROR << "Unrecognized ort data type:" << ort_dtype << "." << std::endl;
  return FDDataType::FP32;
}

Ort::Value CreateOrtValue(FDTensor &tensor, bool is_backend_cuda) {
  FDASSERT(tensor.device == Device::GPU || tensor.device == Device::CPU,
           "Only support tensor which device is CPU or GPU for OrtBackend.");
  if (tensor.device == Device::GPU && is_backend_cuda) {
    Ort::MemoryInfo memory_info("Cuda", OrtDeviceAllocator, 0,
                                OrtMemTypeDefault);
    auto ort_value = Ort::Value::CreateTensor(
        memory_info, tensor.MutableData(), tensor.Nbytes(), tensor.shape.data(),
        tensor.shape.size(), GetOrtDtype(tensor.dtype));
    return ort_value;
  }
  Ort::MemoryInfo memory_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  auto ort_value = Ort::Value::CreateTensor(
      memory_info, tensor.Data(), tensor.Nbytes(), tensor.shape.data(),
      tensor.shape.size(), GetOrtDtype(tensor.dtype));
  return ort_value;
}

} // namespace ultra_infer
