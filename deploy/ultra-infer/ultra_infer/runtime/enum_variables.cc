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

#include "ultra_infer/runtime/enum_variables.h"

namespace ultra_infer {
std::ostream &operator<<(std::ostream &out, const Backend &backend) {
  if (backend == Backend::ORT) {
    out << "Backend::ORT";
  } else if (backend == Backend::TRT) {
    out << "Backend::TRT";
  } else if (backend == Backend::PDINFER) {
    out << "Backend::PDINFER";
  } else if (backend == Backend::OPENVINO) {
    out << "Backend::OPENVINO";
  } else if (backend == Backend::RKNPU2) {
    out << "Backend::RKNPU2";
  } else if (backend == Backend::SOPHGOTPU) {
    out << "Backend::SOPHGOTPU";
  } else if (backend == Backend::POROS) {
    out << "Backend::POROS";
  } else if (backend == Backend::LITE) {
    out << "Backend::PDLITE";
  } else if (backend == Backend::HORIZONNPU) {
    out << "Backend::HORIZONNPU";
  } else if (backend == Backend::TVM) {
    out << "Backend::TVM";
  } else if (backend == Backend::OMONNPU) {
    out << "Backend::OMONNPU";
  } else {
    out << "UNKNOWN-Backend";
  }
  return out;
}

std::ostream &operator<<(std::ostream &out, const Device &d) {
  switch (d) {
  case Device::CPU:
    out << "Device::CPU";
    break;
  case Device::GPU:
    out << "Device::GPU";
    break;
  case Device::RKNPU:
    out << "Device::RKNPU";
    break;
  case Device::SUNRISENPU:
    out << "Device::SUNRISENPU";
    break;
  case Device::SOPHGOTPUD:
    out << "Device::SOPHGOTPUD";
    break;
  case Device::TIMVX:
    out << "Device::TIMVX";
    break;
  case Device::KUNLUNXIN:
    out << "Device::KUNLUNXIN";
    break;
  case Device::ASCEND:
    out << "Device::ASCEND";
    break;
  case Device::DIRECTML:
    out << "Device::DIRECTML";
    break;
  default:
    out << "Device::UNKOWN";
  }
  return out;
}

std::ostream &operator<<(std::ostream &out, const ModelFormat &format) {
  if (format == ModelFormat::PADDLE) {
    out << "ModelFormat::PADDLE";
  } else if (format == ModelFormat::ONNX) {
    out << "ModelFormat::ONNX";
  } else if (format == ModelFormat::RKNN) {
    out << "ModelFormat::RKNN";
  } else if (format == ModelFormat::SOPHGO) {
    out << "ModelFormat::SOPHGO";
  } else if (format == ModelFormat::TORCHSCRIPT) {
    out << "ModelFormat::TORCHSCRIPT";
  } else if (format == ModelFormat::HORIZON) {
    out << "ModelFormat::HORIZON";
  } else if (format == ModelFormat::TVMFormat) {
    out << "ModelFormat::TVMFormat";
  } else if (format == ModelFormat::OM) {
    out << "ModelFormat::OM";
  } else {
    out << "UNKNOWN-ModelFormat";
  }
  return out;
}

std::vector<Backend> GetAvailableBackends() {
  std::vector<Backend> backends;
#ifdef ENABLE_ORT_BACKEND
  backends.push_back(Backend::ORT);
#endif
#ifdef ENABLE_TRT_BACKEND
  backends.push_back(Backend::TRT);
#endif
#ifdef ENABLE_PADDLE_BACKEND
  backends.push_back(Backend::PDINFER);
#endif
#ifdef ENABLE_POROS_BACKEND
  backends.push_back(Backend::POROS);
#endif
#ifdef ENABLE_OPENVINO_BACKEND
  backends.push_back(Backend::OPENVINO);
#endif
#ifdef ENABLE_LITE_BACKEND
  backends.push_back(Backend::LITE);
#endif
#ifdef ENABLE_RKNPU2_BACKEND
  backends.push_back(Backend::RKNPU2);
#endif
#ifdef ENABLE_HORIZON_BACKEND
  backends.push_back(Backend::HORIZONNPU);
#endif
#ifdef ENABLE_SOPHGO_BACKEND
  backends.push_back(Backend::SOPHGOTPU);
#endif
#ifdef ENABLE_TVM_BACKEND
  backends.push_back(Backend::TVM);
#endif
#ifdef ENABLE_OM_BACKEND
  backends.push_back(Backend::OMONNPU);
#endif
  return backends;
}

bool IsBackendAvailable(const Backend &backend) {
  std::vector<Backend> backends = GetAvailableBackends();
  for (size_t i = 0; i < backends.size(); ++i) {
    if (backend == backends[i]) {
      return true;
    }
  }
  return false;
}
} // namespace ultra_infer
