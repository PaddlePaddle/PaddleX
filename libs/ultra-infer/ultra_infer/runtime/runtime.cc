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

#include "ultra_infer/runtime/runtime.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>

#include "ultra_infer/utils/unique_ptr.h"
#include "ultra_infer/utils/utils.h"
#include "yaml-cpp/yaml.h"

#ifdef ENABLE_ORT_BACKEND
#include "ultra_infer/runtime/backends/ort/ort_backend.h"
#endif

#ifdef ENABLE_TRT_BACKEND
#include "ultra_infer/runtime/backends/tensorrt/trt_backend.h"
#endif

#ifdef ENABLE_PADDLE_BACKEND
#include "ultra_infer/runtime/backends/paddle/paddle_backend.h"
#endif

#ifdef ENABLE_POROS_BACKEND
#include "ultra_infer/runtime/backends/poros/poros_backend.h"
#endif

#ifdef ENABLE_OPENVINO_BACKEND
#include "ultra_infer/runtime/backends/openvino/ov_backend.h"
#endif

#ifdef ENABLE_LITE_BACKEND
#include "ultra_infer/runtime/backends/lite/lite_backend.h"
#endif

#ifdef ENABLE_RKNPU2_BACKEND
#include "ultra_infer/runtime/backends/rknpu2/rknpu2_backend.h"
#endif

#ifdef ENABLE_SOPHGO_BACKEND
#include "ultra_infer/runtime/backends/sophgo/sophgo_backend.h"
#endif

#ifdef ENABLE_HORIZON_BACKEND
#include "ultra_infer/runtime/backends/horizon/horizon_backend.h"
#endif

#ifdef ENABLE_TVM_BACKEND
#include "ultra_infer/runtime/backends/tvm/tvm_backend.h"
#endif

namespace ultra_infer {

bool AutoSelectBackend(RuntimeOption &option) {
  auto iter0 = s_default_backends_by_format.find(option.model_format);
  if (iter0 == s_default_backends_by_format.end()) {
    FDERROR << "Cannot found a default backend for model format: "
            << option.model_format
            << ", please define the inference backend in RuntimeOption."
            << std::endl;
    return false;
  }

  auto iter1 = s_default_backends_by_device.find(option.device);
  if (iter1 == s_default_backends_by_device.end()) {
    FDERROR << "Cannot found a default backend for device: " << option.device
            << ", please define the inference backend in RuntimeOption."
            << std::endl;
    return false;
  }

  std::vector<Backend> candidates;
  for (const auto &b0 : iter0->second) {
    for (const auto &b1 : iter1->second) {
      if (b0 == b1) {
        candidates.push_back(b0);
      }
    }
  }

  if (candidates.size() == 0) {
    FDERROR << "Cannot found availabel inference backends by model format: "
            << option.model_format << " with device: " << option.device
            << std::endl;
    return false;
  }

  for (const auto &b : candidates) {
    if (IsBackendAvailable(b)) {
      option.backend = b;
      FDINFO << "UltraInfer will choose " << b << " to inference this model."
             << std::endl;
      return true;
    }
  }
  std::string debug_message = Str(candidates);
  FDERROR << "The candiate backends for " << option.model_format << " & "
          << option.device << " are " << debug_message
          << ", but both of them have not been compiled with current "
             "UltraInfer yet."
          << std::endl;
  return false;
}

bool Runtime::Init(const RuntimeOption &_option) {
  option = _option;

  // Choose default backend by model format and device if backend is not
  // specified
  if (option.backend == Backend::UNKNOWN) {
    if (!AutoSelectBackend(option)) {
      return false;
    }
  }

  if (option.backend == Backend::ORT) {
    CreateOrtBackend();
  } else if (option.backend == Backend::TRT) {
    CreateTrtBackend();
  } else if (option.backend == Backend::PDINFER) {
    CreatePaddleBackend();
  } else if (option.backend == Backend::OPENVINO) {
    CreateOpenVINOBackend();
  } else if (option.backend == Backend::LITE) {
    CreateLiteBackend();
  } else if (option.backend == Backend::RKNPU2) {
    CreateRKNPU2Backend();
  } else if (option.backend == Backend::SOPHGOTPU) {
    CreateSophgoNPUBackend();
  } else if (option.backend == Backend::POROS) {
    CreatePorosBackend();
  } else if (option.backend == Backend::HORIZONNPU) {
    CreateHorizonBackend();
  } else if (option.backend == Backend::TVM) {
    CreateTVMBackend();
  } else {
    std::string msg = Str(GetAvailableBackends());
    FDERROR << "The compiled UltraInfer only supports " << msg << ", "
            << option.backend << " is not supported now." << std::endl;
    return false;
  }
  backend_->benchmark_option_ = option.benchmark_option;
  return true;
}

TensorInfo Runtime::GetInputInfo(int index) {
  return backend_->GetInputInfo(index);
}

TensorInfo Runtime::GetOutputInfo(int index) {
  return backend_->GetOutputInfo(index);
}

std::vector<TensorInfo> Runtime::GetInputInfos() {
  return backend_->GetInputInfos();
}

std::vector<TensorInfo> Runtime::GetOutputInfos() {
  return backend_->GetOutputInfos();
}

bool Runtime::Infer(std::vector<FDTensor> &input_tensors,
                    std::vector<FDTensor> *output_tensors) {
  for (auto &tensor : input_tensors) {
    FDASSERT(tensor.device_id < 0 || tensor.device_id == option.device_id,
             "Device id of input tensor(%d) and runtime(%d) are not same.",
             tensor.device_id, option.device_id);
  }
  return backend_->Infer(input_tensors, output_tensors);
}

bool Runtime::Infer() {
  bool result = false;
  if (option.device == Device::KUNLUNXIN) {
    // FDTensor SetExternalData is not support for Device::KUNLUNXIN
    // now, so, we need to set copy_to_fd as 'true'.
    result = backend_->Infer(input_tensors_, &output_tensors_, true);
  } else {
    result = backend_->Infer(input_tensors_, &output_tensors_, false);
  }

  for (auto &tensor : output_tensors_) {
    tensor.device_id = option.device_id;
  }
  return result;
}

void Runtime::BindInputTensor(const std::string &name, FDTensor &input) {
  bool is_exist = false;
  for (auto &t : input_tensors_) {
    if (t.name == name) {
      is_exist = true;
      t.SetExternalData(input.shape, input.dtype, input.MutableData(),
                        input.device, input.device_id);
      break;
    }
  }
  if (!is_exist) {
    FDTensor new_tensor(name);
    new_tensor.SetExternalData(input.shape, input.dtype, input.MutableData(),
                               input.device, input.device_id);
    input_tensors_.emplace_back(std::move(new_tensor));
  }
}

void Runtime::BindOutputTensor(const std::string &name, FDTensor &output) {
  bool is_exist = false;
  for (auto &t : output_tensors_) {
    if (t.name == name) {
      is_exist = true;
      t.SetExternalData(output.shape, output.dtype, output.MutableData(),
                        output.device, output.device_id);
      break;
    }
  }
  if (!is_exist) {
    FDTensor new_tensor(name);
    new_tensor.SetExternalData(output.shape, output.dtype, output.MutableData(),
                               output.device, output.device_id);
    output_tensors_.emplace_back(std::move(new_tensor));
  }
}
FDTensor *Runtime::GetOutputTensor(const std::string &name) {
  for (auto &t : output_tensors_) {
    if (t.name == name) {
      return &t;
    }
  }
  FDWARNING << "The output name [" << name << "] don't exist." << std::endl;
  return nullptr;
}

void Runtime::ReleaseModelMemoryBuffer() {
  if (option.model_from_memory_) {
    option.model_file.clear();
    option.model_file.shrink_to_fit();
    option.params_file.clear();
    option.params_file.shrink_to_fit();
  }
}

void Runtime::CreatePaddleBackend() {
#ifdef ENABLE_PADDLE_BACKEND
  backend_ = utils::make_unique<PaddleBackend>();
  FDASSERT(backend_->Init(option),
           "Failed to initialized Paddle Inference backend.");
#else
  FDASSERT(false, "PaddleBackend is not available, please compiled with "
                  "ENABLE_PADDLE_BACKEND=ON.");
#endif
  FDINFO << "Runtime initialized with Backend::PDINFER in " << option.device
         << "." << std::endl;
}

void Runtime::CreateOpenVINOBackend() {
#ifdef ENABLE_OPENVINO_BACKEND
  backend_ = utils::make_unique<OpenVINOBackend>();
  FDASSERT(backend_->Init(option), "Failed to initialize OpenVINOBackend.");
#else
  FDASSERT(false, "OpenVINOBackend is not available, please compiled with "
                  "ENABLE_OPENVINO_BACKEND=ON.");
#endif
  FDINFO << "Runtime initialized with Backend::OPENVINO in " << option.device
         << "." << std::endl;
}

void Runtime::CreateTVMBackend() {
#ifdef ENABLE_TVM_BACKEND
  backend_ = utils::make_unique<TVMBackend>();
  FDASSERT(backend_->Init(option), "Failed to initialize TVM backend.");
#else
  FDASSERT(false, "TVMBackend is not available, please compiled with "
                  "ENABLE_TVM_BACKEND=ON.");
#endif
  FDINFO << "Runtime initialized with Backend::TVM in " << option.device << "."
         << std::endl;
}

void Runtime::CreateOrtBackend() {
#ifdef ENABLE_ORT_BACKEND
  backend_ = utils::make_unique<OrtBackend>();

  FDASSERT(backend_->Init(option), "Failed to initialize Backend::ORT.");
#else
  FDASSERT(false, "OrtBackend is not available, please compiled with "
                  "ENABLE_ORT_BACKEND=ON.");
#endif
  FDINFO << "Runtime initialized with Backend::ORT in " << option.device << "."
         << std::endl;
}

void Runtime::CreateTrtBackend() {
#ifdef ENABLE_TRT_BACKEND
  backend_ = utils::make_unique<TrtBackend>();
  FDASSERT(backend_->Init(option), "Failed to initialize TensorRT backend.");
#else
  FDASSERT(false, "TrtBackend is not available, please compiled with "
                  "ENABLE_TRT_BACKEND=ON.");
#endif
  FDINFO << "Runtime initialized with Backend::TRT in " << option.device << "."
         << std::endl;
}

void Runtime::CreateLiteBackend() {
#ifdef ENABLE_LITE_BACKEND
  backend_ = utils::make_unique<LiteBackend>();

  FDASSERT(backend_->Init(option),
           "Load model from nb file failed while initializing LiteBackend.");
#else
  FDASSERT(false, "LiteBackend is not available, please compiled with "
                  "ENABLE_LITE_BACKEND=ON.");
#endif
  FDINFO << "Runtime initialized with Backend::PDLITE in " << option.device
         << "." << std::endl;
}

void Runtime::CreateRKNPU2Backend() {
#ifdef ENABLE_RKNPU2_BACKEND
  backend_ = utils::make_unique<RKNPU2Backend>();
  FDASSERT(backend_->Init(option), "Failed to initialize RKNPU2 backend.");
#else
  FDASSERT(false, "RKNPU2Backend is not available, please compiled with "
                  "ENABLE_RKNPU2_BACKEND=ON.");
#endif
  FDINFO << "Runtime initialized with Backend::RKNPU2 in " << option.device
         << "." << std::endl;
}

void Runtime::CreateHorizonBackend() {
#ifdef ENABLE_HORIZON_BACKEND
  backend_ = utils::make_unique<HorizonBackend>();
  FDASSERT(backend_->Init(option), "Failed to initialize Horizon backend.");
#else
  FDASSERT(false, "HorizonBackend is not available, please compiled with ",
           " ENABLE_HORIZON_BACKEND=ON.");
#endif
  FDINFO << "Runtime initialized with Backend::HORIZONNPU in " << option.device
         << "." << std::endl;
}

void Runtime::CreateSophgoNPUBackend() {
#ifdef ENABLE_SOPHGO_BACKEND
  backend_ = utils::make_unique<SophgoBackend>();
  FDASSERT(backend_->Init(option), "Failed to initialize Sophgo backend.");
#else
  FDASSERT(false, "SophgoBackend is not available, please compiled with "
                  "ENABLE_SOPHGO_BACKEND=ON.");
#endif
  FDINFO << "Runtime initialized with Backend::SOPHGO in " << option.device
         << "." << std::endl;
}

Runtime *Runtime::Clone(void *stream, int device_id) {
  Runtime *runtime = new Runtime();
  if (option.backend != Backend::OPENVINO &&
      option.backend != Backend::PDINFER) {
    runtime->Init(option);
    FDWARNING << "Only OpenVINO/Paddle Inference support \
                  clone engine to  reduce CPU/GPU memory usage now. For "
              << option.backend
              << ", UltraInfer will create a new engine which \
                  will not share memory  with the current runtime."
              << std::endl;
    return runtime;
  }
  FDINFO << "Runtime Clone with Backend:: " << option.backend << " in "
         << option.device << "." << std::endl;
  runtime->option = option;
  runtime->backend_ = backend_->Clone(option, stream, device_id);
  return runtime;
}

void Runtime::CreatePorosBackend() {
#ifdef ENABLE_POROS_BACKEND
  backend_ = utils::make_unique<PorosBackend>();
  FDASSERT(backend_->Init(option), "Failed to initialize Poros backend.");
#else
  FDASSERT(false, "PorosBackend is not available, please compiled with "
                  "ENABLE_POROS_BACKEND=ON.");
#endif
  FDINFO << "Runtime initialized with Backend::POROS in " << option.device
         << "." << std::endl;
}

// only for poros backend
bool Runtime::Compile(std::vector<std::vector<FDTensor>> &prewarm_tensors) {
#ifdef ENABLE_POROS_BACKEND
  option.poros_option.device = option.device;
  option.poros_option.device_id = option.device_id;
  option.poros_option.enable_fp16 = option.trt_option.enable_fp16;
  option.poros_option.max_batch_size = option.trt_option.max_batch_size;
  option.poros_option.max_workspace_size = option.trt_option.max_workspace_size;

  auto casted_backend = dynamic_cast<PorosBackend *>(backend_.get());
  FDASSERT(
      casted_backend->Compile(option.model_file, prewarm_tensors,
                              option.poros_option),
      "Load model from Torchscript failed while initliazing PorosBackend.");
#else
  FDASSERT(false, "PorosBackend is not available, please compiled with "
                  "ENABLE_POROS_BACKEND=ON.");
#endif
  return true;
}

} // namespace ultra_infer
