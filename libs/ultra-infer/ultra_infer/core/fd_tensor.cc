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
#include "ultra_infer/core/fd_tensor.h"

#include <algorithm>
#include <cstring>

#include "ultra_infer/core/float16.h"
#include "ultra_infer/utils/utils.h"
#ifdef WITH_GPU
#include <cuda_runtime_api.h>
#endif

namespace ultra_infer {

void *FDTensor::MutableData() {
  if (external_data_ptr != nullptr) {
    return external_data_ptr;
  }
  return buffer_;
}

void *FDTensor::Data() {
  if (external_data_ptr != nullptr) {
    return external_data_ptr;
  }
  return buffer_;
}

const void *FDTensor::Data() const {
  if (external_data_ptr != nullptr) {
    return external_data_ptr;
  }
  return buffer_;
}

void FDTensor::StopSharing() {
  if (IsShared()) {
    ReallocFn(Nbytes());
    CopyBuffer(buffer_, external_data_ptr, Nbytes());
    external_data_ptr = nullptr;
  }
}

const void *FDTensor::CpuData() const {
  if (device == Device::GPU) {
#ifdef WITH_GPU
    auto *cpu_ptr = const_cast<std::vector<int8_t> *>(&temporary_cpu_buffer);
    cpu_ptr->resize(Nbytes());
    // need to copy cuda mem to cpu first
    if (external_data_ptr != nullptr) {
      FDASSERT(cudaMemcpy(cpu_ptr->data(), external_data_ptr, Nbytes(),
                          cudaMemcpyDeviceToHost) == 0,
               "[ERROR] Error occurs while copy memory from GPU to CPU");

    } else {
      FDASSERT(cudaMemcpy(cpu_ptr->data(), buffer_, Nbytes(),
                          cudaMemcpyDeviceToHost) == 0,
               "[ERROR] Error occurs while buffer copy memory from GPU to CPU");
    }
    return cpu_ptr->data();
#else
    FDASSERT(false,
             "The UltraInfer didn't compile under -DWITH_GPU=ON, so this is "
             "an unexpected problem happened.");
#endif
  }
  return Data();
}

void FDTensor::SetExternalData(const std::vector<int64_t> &new_shape,
                               const FDDataType &data_type, void *data_buffer,
                               const Device &new_device, int new_device_id) {
  dtype = data_type;
  shape.assign(new_shape.begin(), new_shape.end());
  external_data_ptr = data_buffer;
  device = new_device;
  device_id = new_device_id;
}

void FDTensor::ExpandDim(int64_t axis) {
  size_t ndim = shape.size();
  FDASSERT(axis >= 0 && axis <= ndim,
           "The allowed 'axis' must be in range of (0, %lu)!", ndim);
  shape.insert(shape.begin() + axis, 1);
}

void FDTensor::Squeeze(int64_t axis) {
  size_t ndim = shape.size();
  FDASSERT(axis >= 0 && axis < ndim,
           "The allowed 'axis' must be in range of (0, %lu)!", ndim);
  FDASSERT(shape[axis] == 1,
           "The No.%ld dimension of shape should be 1, but it is %ld!",
           (long)axis, (long)shape[axis]);
  shape.erase(shape.begin() + axis);
}

void FDTensor::Allocate(const std::vector<int64_t> &new_shape,
                        const FDDataType &data_type,
                        const std::string &tensor_name,
                        const Device &new_device) {
  dtype = data_type;
  name = tensor_name;
  shape.assign(new_shape.begin(), new_shape.end());
  device = new_device;
  size_t nbytes = Nbytes();
  FDASSERT(ReallocFn(nbytes),
           "The UltraInfer FDTensor allocate cpu memory error");
}

int FDTensor::Nbytes() const { return Numel() * FDDataTypeSize(dtype); }

int FDTensor::Numel() const {
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

void FDTensor::Resize(size_t new_nbytes) { ReallocFn(new_nbytes); }

void FDTensor::Resize(const std::vector<int64_t> &new_shape) {
  int numel = Numel();
  int new_numel = std::accumulate(new_shape.begin(), new_shape.end(), 1,
                                  std::multiplies<int>());
  if (new_numel > numel || external_data_ptr != nullptr) {
    size_t nbytes = new_numel * FDDataTypeSize(dtype);
    ReallocFn(nbytes);
  }
  shape.assign(new_shape.begin(), new_shape.end());
  external_data_ptr = nullptr;
}

void FDTensor::Resize(const std::vector<int64_t> &new_shape,
                      const FDDataType &data_type,
                      const std::string &tensor_name,
                      const Device &new_device) {
  if (device != new_device) {
    FreeFn();
  }
  external_data_ptr = nullptr;
  name = tensor_name;
  device = new_device;
  dtype = data_type;
  int new_nbytes = std::accumulate(new_shape.begin(), new_shape.end(), 1,
                                   std::multiplies<int>()) *
                   FDDataTypeSize(data_type);
  ReallocFn(new_nbytes);
  shape.assign(new_shape.begin(), new_shape.end());
}

bool FDTensor::Reshape(const std::vector<int64_t> &new_shape) {
  int numel = Numel();
  const int64_t unk_dim_val = -1;
  const int64_t copy_dim_val = 0;

  std::vector<int64_t> output_shape(new_shape.size(), 0);
  int64_t capacity = 1;
  int unk_dim_idx = -1;
  for (size_t i = 0; i < new_shape.size(); ++i) {
    if (new_shape[i] == unk_dim_val) {
      FDASSERT(unk_dim_idx == -1,
               "Only one dimension value of 'shape' in ReshapeOp can "
               "be -1. But received shape = [%s], shape[%d] is also -1.",
               Str(new_shape).c_str(), i);
      unk_dim_idx = i;
    } else if (new_shape[i] == copy_dim_val) {
      FDASSERT(i < shape.size(),
               "The index of 0 in `shape` must be less than "
               "the input tensor X's dimensions. "
               "But received shape = [%s], shape[%d] = 0, X's shape = [%s], "
               "X's dimensions = %d.",
               Str(new_shape).c_str(), i, Str(shape).c_str(), shape.size());
    } else {
      FDASSERT(new_shape[i] > 0,
               "Each dimension value of 'shape' in ReshapeOp must not "
               "be negative except one unknown dimension. "
               "But received  shape = [%s], shape[%d] = %d.",
               Str(new_shape).c_str(), i, new_shape[i]);
    }
    capacity *= (new_shape[i] ? new_shape[i] : shape[i]);
    output_shape[i] = (new_shape[i] ? new_shape[i] : shape[i]);
  }
  if (unk_dim_idx != -1) {
    output_shape[unk_dim_idx] = -numel / capacity;
    FDASSERT(output_shape[unk_dim_idx] * capacity == -numel,
             "The 'shape' attribute in ReshapeOp is invalid. "
             "The input tensor X'size must be divisible by known "
             "capacity of 'shape'. "
             "But received X's shape = [%s], X's size = %d, "
             "'shape' is [%s], known capacity of 'shape' is %d.",
             Str(shape).c_str(), numel, Str(new_shape).c_str(), capacity);
  } else {
    FDASSERT(numel == capacity,
             "The 'shape' in ReshapeOp is invalid. "
             "The input tensor X'size must be equal to the capacity of "
             "'shape'. "
             "But received X's shape = [%s], X's size = %d, 'shape' is "
             "[%s], the capacity of 'shape' is %d.",
             Str(shape).c_str(), numel, Str(shape).c_str(), capacity);
  }
  shape = output_shape;
  return true;
}

void FDTensor::PrintInfo(const std::string &prefix) const {
  std::cout << prefix << ": name=" << name << ", shape=";
  for (int i = 0; i < shape.size(); ++i) {
    std::cout << shape[i] << " ";
  }
  std::cout << ", buffer_=" << buffer_
            << ", external_data_ptr=" << external_data_ptr;
  double mean = 0;
  double max = -99999999;
  double min = 99999999;
  if (dtype == FDDataType::FP32) {
    CalculateStatisInfo<float>(CpuData(), Numel(), &mean, &max, &min);
  } else if (dtype == FDDataType::FP64) {
    CalculateStatisInfo<double>(CpuData(), Numel(), &mean, &max, &min);
  } else if (dtype == FDDataType::INT8) {
    CalculateStatisInfo<int8_t>(CpuData(), Numel(), &mean, &max, &min);
  } else if (dtype == FDDataType::UINT8) {
    CalculateStatisInfo<uint8_t>(CpuData(), Numel(), &mean, &max, &min);
  } else if (dtype == FDDataType::INT32) {
    CalculateStatisInfo<int32_t>(CpuData(), Numel(), &mean, &max, &min);
  } else if (dtype == FDDataType::INT64) {
    CalculateStatisInfo<int64_t>(CpuData(), Numel(), &mean, &max, &min);
  } else if (dtype == FDDataType::FP16) {
    CalculateStatisInfo<float16>(CpuData(), Numel(), &mean, &max, &min);
  } else {
    FDASSERT(false,
             "PrintInfo function doesn't support current situation, maybe you "
             "need enhance this function now.");
  }
  std::cout << ", dtype=" << Str(dtype) << ", mean=" << mean << ", max=" << max
            << ", min=" << min << std::endl;
}

bool FDTensor::ReallocFn(size_t nbytes) {
  if (device == Device::GPU) {
#ifdef WITH_GPU
    size_t original_nbytes = nbytes_allocated;
    if (nbytes > original_nbytes) {
      if (buffer_ != nullptr) {
        FDDeviceFree()(buffer_);
      }
      FDDeviceAllocator()(&buffer_, nbytes);
      nbytes_allocated = nbytes;
    }
    return buffer_ != nullptr;
#else
    FDASSERT(false, "The UltraInfer FDTensor allocator didn't compile under "
                    "-DWITH_GPU=ON,"
                    "so this is an unexpected problem happened.");
#endif
  } else {
    if (is_pinned_memory) {
#ifdef WITH_GPU
      size_t original_nbytes = nbytes_allocated;
      if (nbytes > original_nbytes) {
        if (buffer_ != nullptr) {
          FDDeviceHostFree()(buffer_);
        }
        FDDeviceHostAllocator()(&buffer_, nbytes);
        nbytes_allocated = nbytes;
      }
      return buffer_ != nullptr;
#else
      FDASSERT(false, "The UltraInfer FDTensor allocator didn't compile under "
                      "-DWITH_GPU=ON,"
                      "so this is an unexpected problem happened.");
#endif
    }
    buffer_ = realloc(buffer_, nbytes);
    nbytes_allocated = nbytes;
    return buffer_ != nullptr;
  }
}

void FDTensor::FreeFn() {
  if (external_data_ptr != nullptr)
    external_data_ptr = nullptr;
  if (buffer_ != nullptr) {
    if (device == Device::GPU) {
#ifdef WITH_GPU
      FDDeviceFree()(buffer_);
#endif
    } else {
      if (is_pinned_memory) {
#ifdef WITH_GPU
        FDDeviceHostFree()(buffer_);
#endif
      } else {
        FDHostFree()(buffer_);
      }
    }
    buffer_ = nullptr;
    nbytes_allocated = 0;
  }
}

// TODO(liqi): no src_device and dst_device
// should support copy from cpu or gpu  to cpu or gpu
void FDTensor::CopyBuffer(void *dst, const void *src, size_t nbytes,
                          const Device &device, bool is_pinned_memory) {
  if (device == Device::GPU) {
#ifdef WITH_GPU
    FDASSERT(cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToDevice) == 0,
             "[ERROR] Error occurs while copy memory from GPU to GPU");
#else
    FDASSERT(false,
             "The UltraInfer didn't compile under -DWITH_GPU=ON, so copying "
             "gpu buffer is "
             "an unexpected problem happened.");
#endif
  } else {
    if (is_pinned_memory) {
#ifdef WITH_GPU
      FDASSERT(cudaMemcpy(dst, src, nbytes, cudaMemcpyHostToHost) == 0,
               "[ERROR] Error occurs while copy memory from host to host");
#else
      FDASSERT(false,
               "The UltraInfer didn't compile under -DWITH_GPU=ON, so copying "
               "gpu buffer is "
               "an unexpected problem happened.");
#endif
    } else {
      std::memcpy(dst, src, nbytes);
    }
  }
}

FDTensor::FDTensor(const std::string &tensor_name) { name = tensor_name; }
FDTensor::FDTensor(const char *tensor_name) { name = tensor_name; }

FDTensor::FDTensor(const Scalar &scalar) {
  Allocate({1}, scalar.dtype());
  switch (scalar.dtype()) {
  case FDDataType::BOOL:
    (reinterpret_cast<bool *>(Data()))[0] = scalar.to<bool>();
    break;
  case FDDataType::UINT8:
    (reinterpret_cast<uint8_t *>(Data()))[0] = scalar.to<uint8_t>();
    break;
  case FDDataType::INT8:
    (reinterpret_cast<int8_t *>(Data()))[0] = scalar.to<int8_t>();
    break;
  case FDDataType::INT16:
    (reinterpret_cast<int16_t *>(Data()))[0] = scalar.to<int16_t>();
    break;
  case FDDataType::INT32:
    (reinterpret_cast<int *>(Data()))[0] = scalar.to<int>();
    break;
  case FDDataType::INT64:
    (reinterpret_cast<int64_t *>(Data()))[0] = scalar.to<int64_t>();
    break;
  case FDDataType::FP16:
    (reinterpret_cast<float16 *>(Data()))[0] = scalar.to<float16>();
    break;
  case FDDataType::FP32:
    (reinterpret_cast<float *>(Data()))[0] = scalar.to<float>();
    break;
  case FDDataType::FP64:
    (reinterpret_cast<double *>(Data()))[0] = scalar.to<double>();
    break;
  default:
    break;
  }
}

FDTensor::FDTensor(const FDTensor &other)
    : shape(other.shape), name(other.name), dtype(other.dtype),
      device(other.device), device_id(other.device_id) {
  // Copy buffer
  if (other.buffer_ == nullptr) {
    FreeFn();
  } else {
    size_t nbytes = Nbytes();
    FDASSERT(ReallocFn(nbytes),
             "The UltraInfer FDTensor allocate memory error");
    CopyBuffer(buffer_, other.buffer_, nbytes, device, is_pinned_memory);
  }
  external_data_ptr = other.external_data_ptr;
}

FDTensor::FDTensor(FDTensor &&other)
    : buffer_(other.buffer_), shape(std::move(other.shape)),
      name(std::move(other.name)), dtype(other.dtype),
      external_data_ptr(other.external_data_ptr), device(other.device),
      device_id(other.device_id), nbytes_allocated(other.nbytes_allocated) {
  other.name = "";
  // Note(zhoushunjie): Avoid double free.
  other.buffer_ = nullptr;
  other.external_data_ptr = nullptr;
}

FDTensor &FDTensor::operator=(const FDTensor &other) {
  if (&other != this) {
    // Copy buffer
    device_id = other.device_id;
    if (other.buffer_ == nullptr) {
      FreeFn();
      buffer_ = nullptr;
      shape = other.shape;
      name = other.name;
      dtype = other.dtype;
      device = other.device;
    } else {
      Resize(other.shape, other.dtype, other.name, other.device);
      size_t nbytes = Nbytes();
      CopyBuffer(buffer_, other.buffer_, nbytes, device, is_pinned_memory);
    }
    external_data_ptr = other.external_data_ptr;
  }
  return *this;
}

FDTensor &FDTensor::operator=(FDTensor &&other) {
  if (&other != this) {
    FreeFn();
    buffer_ = other.buffer_;
    external_data_ptr = other.external_data_ptr;

    shape = std::move(other.shape);
    name = std::move(other.name);
    dtype = other.dtype;
    device = other.device;
    device_id = other.device_id;
    nbytes_allocated = other.nbytes_allocated;

    other.name = "";
    // Note(zhoushunjie): Avoid double free.
    other.buffer_ = nullptr;
    other.external_data_ptr = nullptr;
  }
  return *this;
}

} // namespace ultra_infer
