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

#include "ultra_infer/core/float16.h"
#include "ultra_infer/runtime/backends/paddle/paddle_backend.h"

namespace ultra_infer {
paddle_infer::PlaceType ConvertFDDeviceToPlace(Device device) {
  if (device == Device::GPU) {
    return paddle_infer::PlaceType::kGPU;
  } else if (device == Device::KUNLUNXIN) {
    return paddle_infer::PlaceType::kXPU;
  }
  return paddle_infer::PlaceType::kCPU;
}

void ShareTensorFromFDTensor(paddle_infer::Tensor *tensor,
                             FDTensor &fd_tensor) {
  std::vector<int> shape(fd_tensor.shape.begin(), fd_tensor.shape.end());
  tensor->Reshape(shape);
  auto place = ConvertFDDeviceToPlace(fd_tensor.device);
  if (fd_tensor.dtype == FDDataType::FP32) {
    if (place == paddle_infer::PlaceType::kGPU) {
      tensor->ShareExternalData(static_cast<const float *>(fd_tensor.Data()),
                                shape, place);
    } else {
      tensor->CopyFromCpu(static_cast<const float *>(fd_tensor.Data()));
    }
    return;
  } else if (fd_tensor.dtype == FDDataType::INT32) {
    if (place == paddle_infer::PlaceType::kGPU) {
      tensor->ShareExternalData(static_cast<const int32_t *>(fd_tensor.Data()),
                                shape, place);
    } else {
      tensor->CopyFromCpu(static_cast<const int32_t *>(fd_tensor.Data()));
    }
    return;
  } else if (fd_tensor.dtype == FDDataType::INT64) {
    if (place == paddle_infer::PlaceType::kGPU) {
      tensor->ShareExternalData(static_cast<const int64_t *>(fd_tensor.Data()),
                                shape, place);
    } else {
      tensor->CopyFromCpu(static_cast<const int64_t *>(fd_tensor.Data()));
    }
    return;
  } else if (fd_tensor.dtype == FDDataType::INT8) {
    if (place == paddle_infer::PlaceType::kGPU) {
      tensor->ShareExternalData(static_cast<const int8_t *>(fd_tensor.Data()),
                                shape, place);
    } else {
      tensor->CopyFromCpu(static_cast<const int8_t *>(fd_tensor.Data()));
    }
    return;
  } else if (fd_tensor.dtype == FDDataType::UINT8) {
    if (place == paddle_infer::PlaceType::kGPU) {
      tensor->ShareExternalData(static_cast<const uint8_t *>(fd_tensor.Data()),
                                shape, place);
    } else {
      tensor->CopyFromCpu(static_cast<const uint8_t *>(fd_tensor.Data()));
    }
    return;
  }
  FDASSERT(false, "Unexpected data type(%s) while infer with PaddleBackend.",
           Str(fd_tensor.dtype).c_str());
}

void ShareOutTensorFromFDTensor(paddle_infer::Tensor *tensor,
                                FDTensor &fd_tensor) {
  std::vector<int> shape(fd_tensor.shape.begin(), fd_tensor.shape.end());
  auto place = ConvertFDDeviceToPlace(fd_tensor.device);
  if (fd_tensor.dtype == FDDataType::FP32) {
    if (place == paddle_infer::PlaceType::kGPU) {
      tensor->ShareExternalData(static_cast<float *>(fd_tensor.MutableData()),
                                shape, place);
    } else {
      tensor->CopyToCpu(static_cast<float *>(fd_tensor.MutableData()));
    }
    return;
  } else if (fd_tensor.dtype == FDDataType::INT32) {
    if (place == paddle_infer::PlaceType::kGPU) {
      tensor->ShareExternalData(static_cast<int32_t *>(fd_tensor.MutableData()),
                                shape, place);
    } else {
      tensor->CopyToCpu(static_cast<int32_t *>(fd_tensor.MutableData()));
    }
    return;
  } else if (fd_tensor.dtype == FDDataType::INT64) {
    if (place == paddle_infer::PlaceType::kGPU) {
      tensor->ShareExternalData(static_cast<int64_t *>(fd_tensor.MutableData()),
                                shape, place);
    } else {
      tensor->CopyToCpu(static_cast<int64_t *>(fd_tensor.MutableData()));
    }
    return;
  } else if (fd_tensor.dtype == FDDataType::INT8) {
    if (place == paddle_infer::PlaceType::kGPU) {
      tensor->ShareExternalData(static_cast<const int8_t *>(fd_tensor.Data()),
                                shape, place);
    } else {
      tensor->CopyFromCpu(static_cast<const int8_t *>(fd_tensor.Data()));
    }
    return;
  } else if (fd_tensor.dtype == FDDataType::UINT8) {
    if (place == paddle_infer::PlaceType::kGPU) {
      tensor->ShareExternalData(static_cast<const uint8_t *>(fd_tensor.Data()),
                                shape, place);
    } else {
      tensor->CopyFromCpu(static_cast<const uint8_t *>(fd_tensor.Data()));
    }
    return;
  }
  FDASSERT(false, "Unexpected data type(%s) while infer with PaddleBackend.",
           Str(fd_tensor.dtype).c_str());
}

void PaddleTensorToFDTensor(std::unique_ptr<paddle_infer::Tensor> &tensor,
                            FDTensor *fd_tensor, bool copy_to_fd) {
  auto fd_dtype = PaddleDataTypeToFD(tensor->type());
  std::vector<int64_t> shape;
  auto tmp_shape = tensor->shape();
  shape.assign(tmp_shape.begin(), tmp_shape.end());
  if (copy_to_fd) {
    fd_tensor->Resize(shape, fd_dtype, tensor->name());
    if (fd_tensor->dtype == FDDataType::FP32) {
      tensor->CopyToCpu(static_cast<float *>(fd_tensor->MutableData()));
      return;
    } else if (fd_tensor->dtype == FDDataType::INT32) {
      tensor->CopyToCpu(static_cast<int32_t *>(fd_tensor->MutableData()));
      return;
    } else if (fd_tensor->dtype == FDDataType::INT64) {
      tensor->CopyToCpu(static_cast<int64_t *>(fd_tensor->MutableData()));
      return;
    } else if (fd_tensor->dtype == FDDataType::INT8) {
      tensor->CopyToCpu(static_cast<int8_t *>(fd_tensor->MutableData()));
      return;
    } else if (fd_tensor->dtype == FDDataType::UINT8) {
      tensor->CopyToCpu(static_cast<uint8_t *>(fd_tensor->MutableData()));
      return;
    }
    FDASSERT(false, "Unexpected data type(%s) while infer with PaddleBackend.",
             Str(fd_tensor->dtype).c_str());
  } else {
    paddle_infer::PlaceType place;
    int size = 0;
    // TODO(liqi): The tensor->data interface of paddle don't return device id
    //               and don't support return void*.
    void *out_data = nullptr;
    if (fd_dtype == FDDataType::FP32) {
      out_data = tensor->data<float>(&place, &size);
    } else if (fd_dtype == FDDataType::INT32) {
      out_data = tensor->data<int>(&place, &size);
    } else if (fd_dtype == FDDataType::INT64) {
      out_data = tensor->data<int64_t>(&place, &size);
    } else if (fd_dtype == FDDataType::INT8) {
      out_data = tensor->data<int8_t>(&place, &size);
    } else if (fd_dtype == FDDataType::UINT8) {
      out_data = tensor->data<uint8_t>(&place, &size);
    } else {
      FDASSERT(
          false,
          "Unexpected data type(%s) while infer shared with PaddleBackend.",
          Str(fd_dtype).c_str());
    }
    Device device = Device::CPU;
    if (place == paddle_infer::PlaceType::kGPU) {
      device = Device::GPU;
    } else if (place == paddle_infer::PlaceType::kXPU) {
      device = Device::KUNLUNXIN;
      FDASSERT(false, "Currently, copy_to_fd=false, FDTensor SetExternalData "
                      "is not support for Device::KUNLUNXIN now!")
    }
    fd_tensor->name = tensor->name();
    fd_tensor->SetExternalData(shape, fd_dtype, out_data, device);
  }
}

FDDataType PaddleDataTypeToFD(const paddle_infer::DataType &dtype) {
  auto fd_dtype = FDDataType::FP32;
  if (dtype == paddle_infer::FLOAT32) {
    fd_dtype = FDDataType::FP32;
  } else if (dtype == paddle_infer::INT64) {
    fd_dtype = FDDataType::INT64;
  } else if (dtype == paddle_infer::INT32) {
    fd_dtype = FDDataType::INT32;
  } else if (dtype == paddle_infer::UINT8) {
    fd_dtype = FDDataType::UINT8;
  } else if (dtype == paddle_infer::INT8) {
    fd_dtype = FDDataType::INT8;
  } else if (dtype == paddle_infer::FLOAT16) {
    fd_dtype = FDDataType::FP16;
  } else {
    FDASSERT(
        false,
        "Unexpected data type: %d while call CopyTensorToCpu in PaddleBackend.",
        int(dtype));
  }
  return fd_dtype;
}

FDDataType ReaderDataTypeToFD(int32_t dtype) {
  auto fd_dtype = FDDataType::FP32;
  if (dtype == 0) {
    fd_dtype = FDDataType::FP32;
  } else if (dtype == 1) {
    fd_dtype = FDDataType::FP64;
  } else if (dtype == 2) {
    fd_dtype = FDDataType::UINT8;
  } else if (dtype == 3) {
    fd_dtype = FDDataType::INT8;
  } else if (dtype == 4) {
    fd_dtype = FDDataType::INT32;
  } else if (dtype == 5) {
    fd_dtype = FDDataType::INT64;
  } else if (dtype == 6) {
    fd_dtype = FDDataType::FP16;
  } else {
    FDASSERT(false,
             "Unexpected data type: %d while call ReaderDataTypeToFD in "
             "PaddleBackend.",
             dtype);
  }
  return fd_dtype;
}

} // namespace ultra_infer
