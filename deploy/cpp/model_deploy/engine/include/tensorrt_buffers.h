//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#ifndef DEPLOYKIT_CPP_INCLUDE_DEPLOY_ENGINE_TENSORRT_BUFFERS_H_
#define DEPLOYKIT_CPP_INCLUDE_DEPLOY_ENGINE_TENSORRT_BUFFERS_H_


#include <cassert>
#include <iostream>
#include <iterator>
#include <memory>
#include <new>
#include <numeric>
#include <string>
#include <vector>
#include <utility>
#include <functional>

#include "NvInfer.h"
#include "./cuda_runtime_api.h"



namespace PaddleDeploy {
namespace TensorRT {
inline void setCudaDevice(int device)
{
    cudaCheck(cudaSetDevice(device));
}

//!
//! \brief  The GenericBuffer class is a templated class for buffers.
//!
//! \details This templated RAII (Resource Acquisition Is Initialization) class
//! handles the allocation,
//!          deallocation, querying of buffers on both the device and the host.
//!          It can handle data of arbitrary types because it stores byte
//!          buffers. The template parameters AllocFunc and FreeFunc are used
//!          for the allocation and deallocation of the buffer. AllocFunc must
//!          be a functor that takes in (void** ptr, size_t size) and returns
//!          bool. ptr is a pointer to where the allocated buffer address should
//!          be stored. size is the amount of memory in bytes to allocate. The
//!          boolean indicates whether or not the memory allocation was
//!          successful. FreeFunc must be a functor that takes in (void* ptr)
//!          and returns void. ptr is the allocated buffer address. It must work
//!          with nullptr input.
//!

template <typename A, typename B> inline A divUp(A x, B n) {
  return (x + n - 1) / n;
}

inline unsigned int getElementSize(nvinfer1::DataType t) {
  switch (t) {
  case nvinfer1::DataType::kINT32:
    return 4;
  case nvinfer1::DataType::kFLOAT:
    return 4;
  case nvinfer1::DataType::kHALF:
    return 2;
  case nvinfer1::DataType::kBOOL:
  case nvinfer1::DataType::kINT8:
    return 1;
  }
  throw std::runtime_error("Invalid DataType.");
  return 0;
}

inline int64_t volume(const nvinfer1::Dims &d) {
  return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

template <typename AllocFunc, typename FreeFunc> class GenericBuffer {
 public:
  //!
  //! \brief Construct an empty buffer.
  //!
  explicit GenericBuffer(nvinfer1::DataType type = nvinfer1::DataType::kFLOAT)
     : mSize(0), mCapacity(0), mType(type), mBuffer(nullptr) {}

  //!
  //! \brief Construct a buffer with the specified allocation size in bytes.
  //!
  GenericBuffer(size_t size, nvinfer1::DataType type)
      : mSize(size), mCapacity(size), mType(type) {
    if (!allocFn(&mBuffer, this->nbBytes())) {
      throw std::bad_alloc();
    }
  }

  GenericBuffer(GenericBuffer &&buf)
      : mSize(buf.mSize), mCapacity(buf.mCapacity), mType(buf.mType),
        mBuffer(buf.mBuffer) {
    buf.mSize = 0;
    buf.mCapacity = 0;
    buf.mType = nvinfer1::DataType::kFLOAT;
    buf.mBuffer = nullptr;
  }

  GenericBuffer &operator=(GenericBuffer &&buf) {
    if (this != &buf) {
      freeFn(mBuffer);
      mSize = buf.mSize;
      mCapacity = buf.mCapacity;
      mType = buf.mType;
      mBuffer = buf.mBuffer;
      // Reset buf.
      buf.mSize = 0;
      buf.mCapacity = 0;
      buf.mBuffer = nullptr;
    }
    return *this;
  }

  //!
  //! \brief Returns pointer to underlying array.
  //!
  void *data() { return mBuffer; }

  //!
  //! \brief Returns pointer to underlying array.
  //!
  const void *data() const { return mBuffer; }

  //!
  //! \brief Returns the size (in number of elements) of the buffer.
  //!
  size_t size() const { return mSize; }

  //!
  //! \brief Returns the size (in bytes) of the buffer.
  //!
  size_t nbBytes() const { return this->size() * getElementSize(mType); }

  //!
  //! \brief Resizes the buffer. This is a no-op if the new size is smaller than
  //! or equal to the current capacity.
  //!
  void resize(size_t newSize) {
    mSize = newSize;
    if (mCapacity < newSize) {
      freeFn(mBuffer);
      if (!allocFn(&mBuffer, this->nbBytes())) {
        throw std::bad_alloc{};
      }
      mCapacity = newSize;
    }
  }

  //!
  //! \brief Overload of resize that accepts Dims
  //!
  void resize(const nvinfer1::Dims &dims) { return this->resize(volume(dims)); }

  ~GenericBuffer() { freeFn(mBuffer); }

 private:
  size_t mSize{0}, mCapacity{0};
  nvinfer1::DataType mType;
  void *mBuffer;
  AllocFunc allocFn;
  FreeFunc freeFn;
};

class DeviceAllocator {
 public:
  bool operator()(void **ptr, size_t size) const {
    return cudaMalloc(ptr, size) == cudaSuccess;
  }
};

class DeviceFree {
 public:
  void operator()(void *ptr) const { cudaFree(ptr); }
};

class HostAllocator {
 public:
  bool operator()(void **ptr, size_t size) const {
    *ptr = malloc(size);
    return *ptr != nullptr;
  }
};

class HostFree {
 public:
  void operator()(void *ptr) const { free(ptr); }
};

using DeviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree>;
using HostBuffer = GenericBuffer<HostAllocator, HostFree>;

//!
//! \brief  The ManagedBuffer class groups together a pair of corresponding
//! device and host buffers.
//!
class ManagedBuffer {
 public:
  DeviceBuffer deviceBuffer;
  HostBuffer hostBuffer;
};

//!
//! \brief  The BufferManager class handles host and device buffer allocation
//! and deallocation.
//!
//! \details This RAII class handles host and device buffer allocation and
//! deallocation,
//!          memcpy between host and device buffers to aid with inference,
//!          and debugging dumps to validate inference. The BufferManager class
//!          is meant to be used to simplify buffer management and any
//!          interactions between buffers and the engine.
//!
class BufferManager {
 public:
  static const size_t kINVALID_SIZE_VALUE = ~size_t(0);

  //!
  //! \brief Create a BufferManager for handling buffer interactions with
  //! engine.
  //!
  BufferManager(std::shared_ptr<nvinfer1::ICudaEngine> engine,
                const int batchSize = 0,
                const nvinfer1::IExecutionContext *context = nullptr)
      : mEngine(engine), mBatchSize(batchSize) {
    // Full Dims implies no batch size.
    assert(engine->hasImplicitBatchDimension() || mBatchSize == 0);
    // Create host and device buffers
    for (int i = 0; i < mEngine->getNbBindings(); i++) {
      auto dims = context ? context->getBindingDimensions(i)
                          : mEngine->getBindingDimensions(i);
      size_t vol = context || !mBatchSize ? 1 : static_cast<size_t>(mBatchSize);
      nvinfer1::DataType type = mEngine->getBindingDataType(i);
      int vecDim = mEngine->getBindingVectorizedDim(i);
      if (-1 != vecDim) {  // i.e., 0 != lgScalarsPerVector
        int scalarsPerVec = mEngine->getBindingComponentsPerElement(i);
        dims.d[vecDim] = divUp(dims.d[vecDim], scalarsPerVec);
        vol *= scalarsPerVec;
      }
      vol *= volume(dims);
      std::cout << "input-" << i << " initial byteSize:" << vol << std::endl;

      std::unique_ptr<ManagedBuffer> manBuf{new ManagedBuffer()};
      manBuf->deviceBuffer = DeviceBuffer(vol, type);
      manBuf->hostBuffer = HostBuffer(vol, type);
      mDeviceBindings.emplace_back(manBuf->deviceBuffer.data());
      mManagedBuffers.emplace_back(std::move(manBuf));
      // std::cout <<  "buffer-"<< i << " initial byteSize:" <<
      // manBuf->hostBuffer.nbBytes() << std::endl;
    }
  }

  //!
  //! \brief Returns a vector of device buffers that you can use directly as
  //!        bindings for the execute and enqueue methods of IExecutionContext.
  //!
  std::vector<void *> &getDeviceBindings() { return mDeviceBindings; }

  //!
  //! \brief Returns a vector of device buffers.
  //!
  const std::vector<void *> &getDeviceBindings() const {
    return mDeviceBindings;
  }

  //!
  //! \brief Returns the device buffer corresponding to tensorName.
  //!        Returns nullptr if no such tensor can be found.
  //!
  void *getDeviceBuffer(const std::string &tensorName) const {
    return getBuffer(false, tensorName);
  }

  //!
  //! \brief Returns the host buffer corresponding to tensorName.
  //!        Returns nullptr if no such tensor can be found.
  //!
  void *getHostBuffer(const std::string &tensorName) const {
    return getBuffer(true, tensorName);
  }

  //!
  //! \brief Returns the size of the host and device buffers that correspond to
  //! tensorName.
  //!        Returns kINVALID_SIZE_VALUE if no such tensor can be found.
  //!
  size_t size(const std::string &tensorName) const {
    int index = mEngine->getBindingIndex(tensorName.c_str());
    if (index == -1)
      return kINVALID_SIZE_VALUE;
    return mManagedBuffers[index]->hostBuffer.nbBytes();
  }

  //!
  //! \brief Copy the contents of input host buffers to input device buffers
  //! synchronously.
  //!
  void copyInputToDevice() { memcpyBuffers(true, false, false); }

  //!
  //! \brief Copy the contents of output device buffers to output host buffers
  //! synchronously.
  //!
  void copyOutputToHost() { memcpyBuffers(false, true, false); }

  //!
  //! \brief Copy the contents of input host buffers to input device buffers
  //! asynchronously.
  //!
  void copyInputToDeviceAsync(const cudaStream_t &stream = 0) {
    memcpyBuffers(true, false, true, stream);
  }

  //!
  //! \brief Copy the contents of output device buffers to output host buffers
  //! asynchronously.
  //!
  void copyOutputToHostAsync(const cudaStream_t &stream = 0) {
    memcpyBuffers(false, true, true, stream);
  }

  ~BufferManager() = default;

 private:
  void *getBuffer(const bool isHost, const std::string &tensorName) const {
    int index = mEngine->getBindingIndex(tensorName.c_str());
    if (index == -1)
      return nullptr;
    return (isHost ? mManagedBuffers[index]->hostBuffer.data()
                   : mManagedBuffers[index]->deviceBuffer.data());
  }

  void memcpyBuffers(const bool copyInput, const bool deviceToHost,
                     const bool async, const cudaStream_t &stream = 0) {
    for (int i = 0; i < mEngine->getNbBindings(); i++) {
      void *dstPtr = deviceToHost ? mManagedBuffers[i]->hostBuffer.data()
                                  : mManagedBuffers[i]->deviceBuffer.data();
      const void *srcPtr = deviceToHost
                               ? mManagedBuffers[i]->deviceBuffer.data()
                               : mManagedBuffers[i]->hostBuffer.data();

      const size_t byteSize = mManagedBuffers[i]->hostBuffer.nbBytes();
      std::cout << "input-host-" << i << " runtime byteSize:"
                << mManagedBuffers[i]->hostBuffer.nbBytes() << std::endl;
      std::cout << "input-device-" << i << " runtime byteSize:"
                << mManagedBuffers[i]->deviceBuffer.nbBytes() << std::endl;

      const cudaMemcpyKind memcpyType =
          deviceToHost ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
      if ((copyInput && mEngine->bindingIsInput(i)) ||
          (!copyInput && !mEngine->bindingIsInput(i))) {
        if (async)
          // CHECK(cudaMemcpyAsync(dstPtr, srcPtr, byteSize, memcpyType, stream));
          cudaMemcpyAsync(dstPtr, srcPtr, byteSize, memcpyType, stream)
        else
          // CHECK(cudaMemcpy(dstPtr, srcPtr, byteSize, memcpyType));
          cudaMemcpy(dstPtr, srcPtr, byteSize, memcpyType)
      }
    }
  }

  // !< The pointer to the engine
  std::shared_ptr<nvinfer1::ICudaEngine> mEngine;

  // !< The batch size for legacy networks, 0 otherwise
  int mBatchSize;

  // !< The vector of pointers to managed buffers
  std::vector<std::unique_ptr<ManagedBuffer>> mManagedBuffers;

  // !< The vector of device buffers needed
  // !< for engine execution
  std::vector<void *> mDeviceBindings;
};
}  // namespace TensorRT
}  // namespace PaddleDeploy

#endif  // DEPLOYKIT_CPP_INCLUDE_DEPLOY_ENGINE_TENSORRT_BUFFERS_H_
