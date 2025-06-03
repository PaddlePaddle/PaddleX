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

#include <dlpack/dlpack.h>

#include "ultra_infer/core/fd_type.h"
#include "ultra_infer/pybind/main.h"
#include "ultra_infer/ultra_infer_model.h"
#include "ultra_infer/utils/utils.h"

namespace ultra_infer {

DLDataType FDToDlpackType(FDDataType fd_dtype) {
  DLDataType dl_dtype;
  DLDataTypeCode dl_code;

  // Number of bits required for the data type.
  size_t dt_size = 0;

  dl_dtype.lanes = 1;
  switch (fd_dtype) {
  case FDDataType::BOOL:
    dl_code = DLDataTypeCode::kDLInt;
    dt_size = 1;
    break;
  case FDDataType::UINT8:
    dl_code = DLDataTypeCode::kDLUInt;
    dt_size = 8;
    break;
  case FDDataType::INT8:
    dl_code = DLDataTypeCode::kDLInt;
    dt_size = 8;
    break;
  case FDDataType::INT16:
    dl_code = DLDataTypeCode::kDLInt;
    dt_size = 16;
    break;
  case FDDataType::INT32:
    dl_code = DLDataTypeCode::kDLInt;
    dt_size = 32;
    break;
  case FDDataType::INT64:
    dl_code = DLDataTypeCode::kDLInt;
    dt_size = 64;
    break;
  case FDDataType::FP16:
    dl_code = DLDataTypeCode::kDLFloat;
    dt_size = 16;
    break;
  case FDDataType::FP32:
    dl_code = DLDataTypeCode::kDLFloat;
    dt_size = 32;
    break;
  case FDDataType::FP64:
    dl_code = DLDataTypeCode::kDLFloat;
    dt_size = 64;
    break;

  default:
    FDASSERT(false, "Convert to DlPack, FDType \"%s\" is not supported.",
             Str(fd_dtype).c_str());
  }

  dl_dtype.code = dl_code;
  dl_dtype.bits = dt_size;
  return dl_dtype;
}

FDDataType DlpackToFDType(const DLDataType &data_type) {
  FDASSERT(data_type.lanes == 1, "FDTensor does not support dlpack lanes != 1")

  if (data_type.code == DLDataTypeCode::kDLFloat) {
    if (data_type.bits == 16) {
      return FDDataType::FP16;
    } else if (data_type.bits == 32) {
      return FDDataType::FP32;
    } else if (data_type.bits == 64) {
      return FDDataType::FP64;
    }
  }

  if (data_type.code == DLDataTypeCode::kDLInt) {
    if (data_type.bits == 8) {
      return FDDataType::INT8;
    } else if (data_type.bits == 16) {
      return FDDataType::INT16;
    } else if (data_type.bits == 32) {
      return FDDataType::INT32;
    } else if (data_type.bits == 64) {
      return FDDataType::INT64;
    } else if (data_type.bits == 1) {
      return FDDataType::BOOL;
    }
  }

  if (data_type.code == DLDataTypeCode::kDLUInt) {
    if (data_type.bits == 8) {
      return FDDataType::UINT8;
    }
  }

  return FDDataType::UNKNOWN1;
}

void DeleteUnusedDltensor(PyObject *dlp) {
  if (PyCapsule_IsValid(dlp, "dltensor")) {
    DLManagedTensor *dl_managed_tensor =
        static_cast<DLManagedTensor *>(PyCapsule_GetPointer(dlp, "dltensor"));
    dl_managed_tensor->deleter(dl_managed_tensor);
  }
}

pybind11::capsule FDTensorToDLPack(FDTensor &fd_tensor) {
  DLManagedTensor *dlpack_tensor = new DLManagedTensor;
  dlpack_tensor->dl_tensor.ndim = fd_tensor.shape.size();
  dlpack_tensor->dl_tensor.byte_offset = 0;
  dlpack_tensor->dl_tensor.data = fd_tensor.MutableData();
  dlpack_tensor->dl_tensor.shape = &(fd_tensor.shape[0]);
  dlpack_tensor->dl_tensor.strides = nullptr;
  dlpack_tensor->manager_ctx = &fd_tensor;
  dlpack_tensor->deleter = [](DLManagedTensor *m) {
    if (m->manager_ctx == nullptr) {
      return;
    }

    FDTensor *tensor_ptr = reinterpret_cast<FDTensor *>(m->manager_ctx);
    pybind11::handle tensor_handle = pybind11::cast(tensor_ptr);
    tensor_handle.dec_ref();
    free(m);
  };

  pybind11::handle tensor_handle = pybind11::cast(&fd_tensor);

  // Increase the reference count by one to make sure that the DLPack
  // representation doesn't become invalid when the tensor object goes out of
  // scope.
  tensor_handle.inc_ref();

  dlpack_tensor->dl_tensor.dtype = FDToDlpackType(fd_tensor.dtype);

  dlpack_tensor->dl_tensor.device.device_id = fd_tensor.device_id;
  if (fd_tensor.device == Device::GPU) {
    if (fd_tensor.is_pinned_memory) {
      dlpack_tensor->dl_tensor.device.device_type = DLDeviceType::kDLCUDAHost;
    } else {
      dlpack_tensor->dl_tensor.device.device_type = DLDeviceType::kDLCUDA;
    }
  } else {
    dlpack_tensor->dl_tensor.device.device_type = DLDeviceType::kDLCPU;
  }

  return pybind11::capsule(static_cast<void *>(dlpack_tensor), "dltensor",
                           &DeleteUnusedDltensor);
}

FDTensor FDTensorFromDLPack(const std::string &name,
                            const pybind11::capsule &dlpack_tensor) {
  DLManagedTensor *dl_managed_tensor =
      static_cast<DLManagedTensor *>(dlpack_tensor.get_pointer());

  void *memory_ptr = dl_managed_tensor->dl_tensor.data;
  memory_ptr = reinterpret_cast<char *>(memory_ptr) +
               dl_managed_tensor->dl_tensor.byte_offset;

  int64_t *strides = dl_managed_tensor->dl_tensor.strides;

  int ndim = dl_managed_tensor->dl_tensor.ndim;
  std::vector<int64_t> dims(dl_managed_tensor->dl_tensor.shape,
                            dl_managed_tensor->dl_tensor.shape + ndim);

  // Check if the input is contiguous and in C order
  if (strides != nullptr) {
    int64_t calculated_stride{1};
    bool is_contiguous_c_order = true;
    for (size_t i = 1; i < dims.size(); i++) {
      if (strides[ndim - i] != calculated_stride) {
        is_contiguous_c_order = false;
        break;
      }

      calculated_stride *= dims[ndim - i];
    }

    FDASSERT(is_contiguous_c_order,
             "DLPack tensor is not contiguous. Only contiguous DLPack "
             "tensors that are stored in C-Order are supported.");
  }

  Device device;
  int32_t device_id = -1;
  bool is_pinned_memory = false;
  switch (dl_managed_tensor->dl_tensor.device.device_type) {
  case DLDeviceType::kDLCUDA:
    device = Device::GPU;
    device_id = dl_managed_tensor->dl_tensor.device.device_id;
    break;
  case DLDeviceType::kDLCPU:
    device = Device::CPU;
    break;
  case DLDeviceType::kDLCUDAHost:
    device = Device::CPU;
    is_pinned_memory = true;
    break;
  default:
    FDASSERT(false,
             ("DLDevice type " +
              std::to_string(dl_managed_tensor->dl_tensor.device.device_type) +
              " is not support by Python backend.")
                 .c_str());
    break;
  }

  FDDataType dtype = DlpackToFDType(dl_managed_tensor->dl_tensor.dtype);

  PyCapsule_SetName(dlpack_tensor.ptr(), "used_dlpack");
  FDTensor fd_tensor(name);
  fd_tensor.SetExternalData(dims, dtype, memory_ptr, device, device_id);
  fd_tensor.is_pinned_memory = is_pinned_memory;
  return fd_tensor;
}

void BindFDTensor(pybind11::module &m) {
  pybind11::class_<FDTensor>(m, "FDTensor")
      .def(pybind11::init<>(), "Default Constructor")
      .def_readwrite("name", &FDTensor::name)
      .def_readonly("shape", &FDTensor::shape)
      .def_readonly("dtype", &FDTensor::dtype)
      .def_readonly("device", &FDTensor::device)
      .def("numpy", [](FDTensor &self) { return TensorToPyArray(self); })
      .def("data", &FDTensor::MutableData)
      .def("from_numpy",
           [](FDTensor &self, pybind11::array &pyarray,
              bool share_buffer = false) {
             PyArrayToTensor(pyarray, &self, share_buffer);
           })
      .def("from_external_data",
           [](const std::string &name, size_t data_addr,
              const std::vector<int64_t> &shape, const std::string &data_type,
              const std::string &data_place, int device_id) {
             auto fd_data_type = FDDataType::UNKNOWN1;
             if (data_type == "FP32") {
               fd_data_type = FDDataType::FP32;
             } else if (data_type == "FP16") {
               fd_data_type = FDDataType::FP16;
             } else if (data_type == "INT32") {
               fd_data_type = FDDataType::INT32;
             } else if (data_type == "INT64") {
               fd_data_type = FDDataType::INT64;
             } else {
               FDASSERT(false,
                        "FDTensor.from_external_data, datatype \"%s\" is not "
                        "supported.",
                        data_type.c_str());
             }

             Device fd_data_place;
             bool copy = false;
             if (data_place.find("gpu") != data_place.npos) {
               fd_data_place = Device::GPU;
             } else if (data_place.find("cpu") != data_place.npos) {
               copy = true;
               fd_data_place = Device::CPU;
             } else {
               FDASSERT(false,
                        ("Device type " + data_place +
                         " is not support by FDTensor.from_external_data.")
                            .c_str());
             }
             void *data_ptr = nullptr;
             data_ptr = reinterpret_cast<void *>(data_addr);
             FDTensor fd_tensor(name);
             fd_tensor.SetData(shape, fd_data_type,
                               static_cast<void *>(data_ptr), copy,
                               fd_data_place, device_id);
             return fd_tensor;
           })
      .def("to_dlpack", &FDTensorToDLPack)
      .def("from_dlpack", &FDTensorFromDLPack)
      .def("print_info", &FDTensor::PrintInfo);
}

} // namespace ultra_infer
