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
#include <numeric>
#include <string>
#include <vector>

#include "ultra_infer/core/allocate.h"
#include "ultra_infer/core/fd_scalar.h"
#include "ultra_infer/core/fd_type.h"
#include "ultra_infer/runtime/enum_variables.h"

namespace ultra_infer {

/*! @brief FDTensor object used to represent data matrix
 *
 */
struct ULTRAINFER_DECL FDTensor {
  /** \brief Set data buffer for a FDTensor, e.g
   *  ```
   *  std::vector<float> buffer(1 * 3 * 224 * 224, 0);
   *  FDTensor tensor;
   *  tensor.SetData({1, 3, 224, 224}, FDDataType::FLOAT, buffer.data());
   *  ```
   * \param[in] tensor_shape The shape of tensor
   * \param[in] data_type The data type of tensor
   * \param[in] data_buffer The pointer of data buffer memory
   * \param[in] copy Whether to copy memory from data_buffer to tensor, if
   * false, this tensor will share memory with data_buffer, and the data is
   * managed by userself \param[in] data_device The device of data_buffer, e.g
   * if data_buffer is a pointer to GPU data, the device should be Device::GPU
   * \param[in] data_device_id The device id of data_buffer
   */
  void SetData(const std::vector<int64_t> &tensor_shape,
               const FDDataType &data_type, void *data_buffer,
               bool copy = false, const Device &data_device = Device::CPU,
               int data_device_id = -1) {
    SetExternalData(tensor_shape, data_type, data_buffer, data_device,
                    data_device_id);
    if (copy) {
      StopSharing();
    }
  }

  /// Get data pointer of tensor
  void *GetData() { return MutableData(); }
  /// Get data pointer of tensor
  const void *GetData() const { return Data(); }

  /// Expand the shape of tensor, it will not change the data memory, just
  /// modify its attribute `shape`
  void ExpandDim(int64_t axis = 0);

  /// Squeeze the shape of tensor, it will not change the data memory, just
  /// modify its attribute `shape`
  void Squeeze(int64_t axis = 0);

  /// Reshape the tensor, it will not change the data memory, just modify its
  /// attribute `shape`
  bool Reshape(const std::vector<int64_t> &new_shape);

  /// Total size of tensor memory buffer in bytes
  int Nbytes() const;

  /// Total number of elements in tensor
  int Numel() const;

  /// Get shape of tensor
  std::vector<int64_t> Shape() const { return shape; }

  /// Get dtype of tensor
  FDDataType Dtype() const { return dtype; }

  /** \brief Allocate cpu data buffer for a FDTensor, e.g
   *  ```
   *  FDTensor tensor;
   *  tensor.Allocate(FDDataType::FLOAT, {1, 3, 224, 224};
   *  ```
   * \param[in] data_type The data type of tensor
   * \param[in] tensor_shape The shape of tensor
   */
  void Allocate(const FDDataType &data_type,
                const std::vector<int64_t> &data_shape) {
    Allocate(data_shape, data_type, name);
  }

  /// Debug function, print shape, dtype, mean, max, min of tensor
  void PrintInfo(const std::string &prefix = "Debug TensorInfo: ") const;

  /// Name of tensor, while feed to runtime, this need be defined
  std::string name = "";

  /// Whether the tensor is owned the data buffer or share the data buffer from
  /// outside
  bool IsShared() { return external_data_ptr != nullptr; }
  /// If the tensor is share the data buffer from outside, `StopSharing` will
  /// copy to its own structure; Otherwise, do nothing
  void StopSharing();

  // ******************************************************
  // The following member and function only used by inside UltraInfer, maybe
  // removed in next version

  void *buffer_ = nullptr;
  std::vector<int64_t> shape = {0};
  FDDataType dtype = FDDataType::INT8;

  // This use to skip memory copy step
  // the external_data_ptr will point to the user allocated memory
  // user has to maintain the memory, allocate and release
  void *external_data_ptr = nullptr;
  // The internal data will be on CPU
  // Some times, the external data is on the GPU, and we are going to use
  // GPU to inference the model
  // so we can skip data transfer, which may improve the efficiency
  Device device = Device::CPU;
  // By default the device id of FDTensor is -1, which means this value is
  // invalid, and FDTensor is using the same device id as Runtime.
  int device_id = -1;

  // Whether the data buffer is in pinned memory, which is allocated
  // with cudaMallocHost()
  bool is_pinned_memory = false;

  // if the external data is not on CPU, we use this temporary buffer
  // to transfer data to CPU at some cases we need to visit the
  // other devices' data
  std::vector<int8_t> temporary_cpu_buffer;

  // The number of bytes allocated so far.
  // When resizing GPU memory, we will free and realloc the memory only if the
  // required size is larger than this value.
  size_t nbytes_allocated = 0;

  // Get data buffer pointer
  void *MutableData();

  void *Data();

  const void *Data() const;

  // Use this data to get the tensor data to process
  // Since the most scenario is process data in CPU
  // this function will return a pointer to cpu memory
  // buffer.
  // If the original data is on other device, the data
  // will copy to cpu store in `temporary_cpu_buffer`
  const void *CpuData() const;

  // void SetDataBuffer(const std::vector<int64_t>& new_shape, const FDDataType&
  // data_type, void* data_buffer, bool copy = false, const Device& new_device =
  // Device::CPU, int new_device_id = -1); Set user memory buffer for Tensor,
  // the memory is managed by the user it self, but the Tensor will share the
  // memory with user So take care with the user buffer
  void SetExternalData(const std::vector<int64_t> &new_shape,
                       const FDDataType &data_type, void *data_buffer,
                       const Device &new_device = Device::CPU,
                       int new_device_id = -1);
  // Initialize Tensor
  // Include setting attribute for tensor
  // and allocate cpu memory buffer
  void Allocate(const std::vector<int64_t> &new_shape,
                const FDDataType &data_type,
                const std::string &tensor_name = "",
                const Device &new_device = Device::CPU);

  void Resize(size_t nbytes);

  void Resize(const std::vector<int64_t> &new_shape);

  void Resize(const std::vector<int64_t> &new_shape,
              const FDDataType &data_type, const std::string &tensor_name = "",
              const Device &new_device = Device::CPU);

  bool ReallocFn(size_t nbytes);

  void FreeFn();

  FDTensor() {}
  explicit FDTensor(const std::string &tensor_name);
  explicit FDTensor(const char *tensor_name);

  // Deep copy
  FDTensor(const FDTensor &other);
  // Move constructor
  FDTensor(FDTensor &&other);

  // Deep copy assignment
  FDTensor &operator=(const FDTensor &other);
  // Move assignment
  FDTensor &operator=(FDTensor &&other);

  // Scalar to FDTensor
  explicit FDTensor(const Scalar &scalar);

  ~FDTensor() { FreeFn(); }

  static void CopyBuffer(void *dst, const void *src, size_t nbytes,
                         const Device &device = Device::CPU,
                         bool is_pinned_memory = false);
};

} // namespace ultra_infer
