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
#include "ultra_infer/ultra_infer_model.h"

#include "ultra_infer/utils/utils.h"

namespace ultra_infer {

std::string Str(const std::vector<Backend> &backends) {
  std::ostringstream oss;
  if (backends.size() == 0) {
    oss << "[]";
    return oss.str();
  }
  oss << "[ " << backends[0];
  for (int i = 1; i < backends.size(); ++i) {
    oss << " ," << backends[i];
  }
  oss << " ]";
  return oss.str();
}

bool CheckBackendSupported(const std::vector<Backend> &backends,
                           Backend backend) {
  for (size_t i = 0; i < backends.size(); ++i) {
    if (backends[i] == backend) {
      return true;
    }
  }
  return false;
}

bool UltraInferModel::IsSupported(const std::vector<Backend> &backends,
                                  Backend backend) {
#ifdef ENABLE_BENCHMARK
  if (runtime_option.benchmark_option.enable_profile) {
    FDWARNING << "In benchmark mode, we don't check to see if "
              << "the backend [" << backend
              << "] is supported for current model!" << std::endl;
    return true;
  } else if (!runtime_option.enable_valid_backend_check) {
    FDWARNING << "Checking for valid backend is disable, we don't"
              << " check to see if the backend [" << backend
              << "] is supported for current model!" << std::endl;
    return true;
  }
  return CheckBackendSupported(backends, backend);
#else
  if (!runtime_option.enable_valid_backend_check) {
    FDWARNING << "Checking for valid backend is disable, we don't"
              << " check to see if the backend [" << backend
              << "] is supported for current model!" << std::endl;
    return true;
  }
  return CheckBackendSupported(backends, backend);
#endif
}

bool UltraInferModel::InitRuntimeWithSpecifiedBackend() {
  if (!IsBackendAvailable(runtime_option.backend)) {
    FDERROR << runtime_option.backend
            << " is not compiled with current UltraInfer library." << std::endl;
    return false;
  }

  bool use_gpu = (runtime_option.device == Device::GPU);
  bool use_ipu = (runtime_option.device == Device::IPU);
  bool use_rknpu = (runtime_option.device == Device::RKNPU);
  bool use_horizon = (runtime_option.device == Device::SUNRISENPU);
  bool use_sophgotpu = (runtime_option.device == Device::SOPHGOTPUD);
  bool use_timvx = (runtime_option.device == Device::TIMVX);
  bool use_ascend = (runtime_option.device == Device::ASCEND);
  bool use_directml = (runtime_option.device == Device::DIRECTML);
  bool use_kunlunxin = (runtime_option.device == Device::KUNLUNXIN);

  if (use_gpu) {
    if (!IsSupported(valid_gpu_backends, runtime_option.backend)) {
      FDERROR << "The valid gpu backends of model " << ModelName() << " are "
              << Str(valid_gpu_backends) << ", " << runtime_option.backend
              << " is not supported." << std::endl;
      return false;
    }
  } else if (use_rknpu) {
    if (!IsSupported(valid_rknpu_backends, runtime_option.backend)) {
      FDERROR << "The valid rknpu backends of model " << ModelName() << " are "
              << Str(valid_rknpu_backends) << ", " << runtime_option.backend
              << " is not supported." << std::endl;
      return false;
    }
  } else if (use_horizon) {
    if (!IsSupported(valid_horizon_backends, runtime_option.backend)) {
      FDERROR << "The valid horizon backends of model " << ModelName()
              << " are " << Str(valid_horizon_backends) << ", "
              << runtime_option.backend << " is not supported." << std::endl;
      return false;
    }
  } else if (use_sophgotpu) {
    if (!IsSupported(valid_sophgonpu_backends, runtime_option.backend)) {
      FDERROR << "The valid sophgo backends of model " << ModelName() << " are "
              << Str(valid_sophgonpu_backends) << ", " << runtime_option.backend
              << " is not supported." << std::endl;
      return false;
    }
  } else if (use_timvx) {
    if (!IsSupported(valid_timvx_backends, runtime_option.backend)) {
      FDERROR << "The valid timvx backends of model " << ModelName() << " are "
              << Str(valid_timvx_backends) << ", " << runtime_option.backend
              << " is not supported." << std::endl;
      return false;
    }
  } else if (use_ascend) {
    if (!IsSupported(valid_ascend_backends, runtime_option.backend)) {
      FDERROR << "The valid ascend backends of model " << ModelName() << " are "
              << Str(valid_ascend_backends) << ", " << runtime_option.backend
              << " is not supported." << std::endl;
      return false;
    }
  } else if (use_directml) {
    if (!IsSupported(valid_directml_backends, runtime_option.backend)) {
      FDERROR << "The valid directml backends of model " << ModelName()
              << " are " << Str(valid_directml_backends) << ", "
              << runtime_option.backend << " is not supported." << std::endl;
      return false;
    }
  } else if (use_kunlunxin) {
    if (!IsSupported(valid_kunlunxin_backends, runtime_option.backend)) {
      FDERROR << "The valid kunlunxin backends of model " << ModelName()
              << " are " << Str(valid_kunlunxin_backends) << ", "
              << runtime_option.backend << " is not supported." << std::endl;
      return false;
    }
  } else if (use_ipu) {
    if (!IsSupported(valid_ipu_backends, runtime_option.backend)) {
      FDERROR << "The valid ipu backends of model " << ModelName() << " are "
              << Str(valid_ipu_backends) << ", " << runtime_option.backend
              << " is not supported." << std::endl;
      return false;
    }
  } else {
    if (!IsSupported(valid_cpu_backends, runtime_option.backend)) {
      FDERROR << "The valid cpu backends of model " << ModelName() << " are "
              << Str(valid_cpu_backends) << ", " << runtime_option.backend
              << " is not supported." << std::endl;
      return false;
    }
  }

  runtime_ = std::shared_ptr<Runtime>(new Runtime());
  if (!runtime_->Init(runtime_option)) {
    return false;
  }
  runtime_initialized_ = true;
  return true;
}

bool UltraInferModel::InitRuntimeWithSpecifiedDevice() {
  if (runtime_option.device == Device::CPU) {
    return CreateCpuBackend();
  } else if (runtime_option.device == Device::GPU) {
#ifdef WITH_GPU
    return CreateGpuBackend();
#else
    FDERROR << "The compiled UltraInfer library doesn't support GPU now."
            << std::endl;
    return false;
#endif
  } else if (runtime_option.device == Device::RKNPU) {
    return CreateRKNPUBackend();
  } else if (runtime_option.device == Device::SUNRISENPU) {
    return CreateHorizonBackend();
  } else if (runtime_option.device == Device::TIMVX) {
    return CreateTimVXBackend();
  } else if (runtime_option.device == Device::ASCEND) {
    return CreateASCENDBackend();
  } else if (runtime_option.device == Device::DIRECTML) {
    return CreateDirectMLBackend();
  } else if (runtime_option.device == Device::KUNLUNXIN) {
    return CreateKunlunXinBackend();
  } else if (runtime_option.device == Device::SOPHGOTPUD) {
    return CreateSophgoNPUBackend();
  } else if (runtime_option.device == Device::IPU) {
#ifdef WITH_IPU
    return CreateIpuBackend();
#else
    FDERROR << "The compiled UltraInfer library doesn't support IPU now."
            << std::endl;
    return false;
#endif
  }
  FDERROR << "Only support "
             "CPU/GPU/IPU/RKNPU/HORIZONNPU/TIMVX/KunlunXin/ASCEND/DirectML now."
          << std::endl;
  return false;
}

bool UltraInferModel::InitRuntime() {
  if (runtime_initialized_) {
    FDERROR << "The model is already initialized, cannot be initialized again."
            << std::endl;
    return false;
  }
  if (runtime_option.backend != Backend::UNKNOWN) {
    return InitRuntimeWithSpecifiedBackend();
  }

  return InitRuntimeWithSpecifiedDevice();
}

bool UltraInferModel::CreateCpuBackend() {
  if (valid_cpu_backends.size() == 0) {
    FDERROR << "There's no valid cpu backends for model: " << ModelName()
            << std::endl;
    return false;
  }

  for (size_t i = 0; i < valid_cpu_backends.size(); ++i) {
    if (!IsBackendAvailable(valid_cpu_backends[i])) {
      continue;
    }
    runtime_option.backend = valid_cpu_backends[i];
    runtime_ = std::shared_ptr<Runtime>(new Runtime());
    if (!runtime_->Init(runtime_option)) {
      return false;
    }
    runtime_initialized_ = true;
    return true;
  }
  FDERROR << "Found no valid backend for model: " << ModelName() << std::endl;
  return false;
}

bool UltraInferModel::CreateGpuBackend() {
  if (valid_gpu_backends.empty()) {
    FDERROR << "There's no valid gpu backends for model: " << ModelName()
            << std::endl;
    return false;
  }

  for (size_t i = 0; i < valid_gpu_backends.size(); ++i) {
    if (!IsBackendAvailable(valid_gpu_backends[i])) {
      continue;
    }
    runtime_option.backend = valid_gpu_backends[i];
    runtime_ = std::shared_ptr<Runtime>(new Runtime());
    if (!runtime_->Init(runtime_option)) {
      return false;
    }
    runtime_initialized_ = true;
    return true;
  }
  FDERROR << "Cannot find an available gpu backend to load this model."
          << std::endl;
  return false;
}

bool UltraInferModel::CreateRKNPUBackend() {
  if (valid_rknpu_backends.empty()) {
    FDERROR << "There's no valid npu backends for model: " << ModelName()
            << std::endl;
    return false;
  }

  for (size_t i = 0; i < valid_rknpu_backends.size(); ++i) {
    if (!IsBackendAvailable(valid_rknpu_backends[i])) {
      continue;
    }
    runtime_option.backend = valid_rknpu_backends[i];
    runtime_ = std::unique_ptr<Runtime>(new Runtime());
    if (!runtime_->Init(runtime_option)) {
      return false;
    }
    runtime_initialized_ = true;
    return true;
  }
  FDERROR << "Cannot find an available npu backend to load this model."
          << std::endl;
  return false;
}

bool UltraInferModel::CreateHorizonBackend() {
  if (valid_horizon_backends.empty()) {
    FDERROR << "There's no valid npu backends for model: " << ModelName()
            << std::endl;
    return false;
  }
  for (size_t i = 0; i < valid_horizon_backends.size(); ++i) {
    if (!IsBackendAvailable(valid_horizon_backends[i])) {
      continue;
    }
    runtime_option.backend = valid_horizon_backends[i];
    runtime_ = std::unique_ptr<Runtime>(new Runtime());
    if (!runtime_->Init(runtime_option)) {
      return false;
    }
    runtime_initialized_ = true;
    return true;
  }
  FDERROR << "Cannot find an available npu backend to load this model."
          << std::endl;
  return false;
}
bool UltraInferModel::CreateSophgoNPUBackend() {
  if (valid_sophgonpu_backends.empty()) {
    FDERROR << "There's no valid npu backends for model: " << ModelName()
            << std::endl;
    return false;
  }

  for (size_t i = 0; i < valid_sophgonpu_backends.size(); ++i) {
    if (!IsBackendAvailable(valid_sophgonpu_backends[i])) {
      continue;
    }
    runtime_option.backend = valid_sophgonpu_backends[i];
    runtime_ = std::unique_ptr<Runtime>(new Runtime());
    if (!runtime_->Init(runtime_option)) {
      return false;
    }
    runtime_initialized_ = true;
    return true;
  }
  FDERROR << "Cannot find an available npu backend to load this model."
          << std::endl;
  return false;
}

bool UltraInferModel::CreateTimVXBackend() {
  if (valid_timvx_backends.size() == 0) {
    FDERROR << "There's no valid timvx backends for model: " << ModelName()
            << std::endl;
    return false;
  }

  for (size_t i = 0; i < valid_timvx_backends.size(); ++i) {
    if (!IsBackendAvailable(valid_timvx_backends[i])) {
      continue;
    }
    runtime_option.backend = valid_timvx_backends[i];
    runtime_ = std::unique_ptr<Runtime>(new Runtime());
    if (!runtime_->Init(runtime_option)) {
      return false;
    }
    runtime_initialized_ = true;
    return true;
  }
  FDERROR << "Found no valid backend for model: " << ModelName() << std::endl;
  return false;
}

bool UltraInferModel::CreateKunlunXinBackend() {
  if (valid_kunlunxin_backends.size() == 0) {
    FDERROR << "There's no valid KunlunXin backends for model: " << ModelName()
            << std::endl;
    return false;
  }

  for (size_t i = 0; i < valid_kunlunxin_backends.size(); ++i) {
    if (!IsBackendAvailable(valid_kunlunxin_backends[i])) {
      continue;
    }
    runtime_option.backend = valid_kunlunxin_backends[i];
    runtime_ = std::unique_ptr<Runtime>(new Runtime());
    if (!runtime_->Init(runtime_option)) {
      return false;
    }
    runtime_initialized_ = true;
    return true;
  }
  FDERROR << "Found no valid backend for model: " << ModelName() << std::endl;
  return false;
}

bool UltraInferModel::CreateASCENDBackend() {
  if (valid_ascend_backends.size() == 0) {
    FDERROR << "There's no valid ascend backends for model: " << ModelName()
            << std::endl;
    return false;
  }

  for (size_t i = 0; i < valid_ascend_backends.size(); ++i) {
    if (!IsBackendAvailable(valid_ascend_backends[i])) {
      continue;
    }
    runtime_option.backend = valid_ascend_backends[i];
    runtime_ = std::unique_ptr<Runtime>(new Runtime());
    if (!runtime_->Init(runtime_option)) {
      return false;
    }
    runtime_initialized_ = true;
    return true;
  }
  FDERROR << "Found no valid backend for model: " << ModelName() << std::endl;
  return false;
}

bool UltraInferModel::CreateDirectMLBackend() {
  if (valid_directml_backends.size() == 0) {
    FDERROR << "There's no valid directml backends for model: " << ModelName()
            << std::endl;
    return false;
  }

  for (size_t i = 0; i < valid_directml_backends.size(); ++i) {
    if (!IsBackendAvailable(valid_directml_backends[i])) {
      continue;
    }
    runtime_option.backend = valid_directml_backends[i];
    runtime_ = std::unique_ptr<Runtime>(new Runtime());
    if (!runtime_->Init(runtime_option)) {
      return false;
    }
    runtime_initialized_ = true;
    return true;
  }
  FDERROR << "Found no valid directml backend for model: " << ModelName()
          << std::endl;
  return false;
}

bool UltraInferModel::CreateIpuBackend() {
  if (valid_ipu_backends.size() == 0) {
    FDERROR << "There's no valid ipu backends for model: " << ModelName()
            << std::endl;
    return false;
  }

  for (size_t i = 0; i < valid_ipu_backends.size(); ++i) {
    if (!IsBackendAvailable(valid_ipu_backends[i])) {
      continue;
    }
    runtime_option.backend = valid_ipu_backends[i];
    runtime_ = std::unique_ptr<Runtime>(new Runtime());
    if (!runtime_->Init(runtime_option)) {
      return false;
    }
    runtime_initialized_ = true;
    return true;
  }
  FDERROR << "Found no valid backend for model: " << ModelName() << std::endl;
  return false;
}

bool UltraInferModel::Infer(std::vector<FDTensor> &input_tensors,
                            std::vector<FDTensor> *output_tensors) {
  TimeCounter tc;
  if (enable_record_time_of_runtime_) {
    tc.Start();
  }
  auto ret = runtime_->Infer(input_tensors, output_tensors);
  if (enable_record_time_of_runtime_) {
    tc.End();
    if (time_of_runtime_.size() > 50000) {
      FDWARNING << "There are already 50000 records of runtime, will force to "
                   "disable record time of runtime now."
                << std::endl;
      enable_record_time_of_runtime_ = false;
    }
    time_of_runtime_.push_back(tc.Duration());
  }

  return ret;
}

bool UltraInferModel::Infer() {
  return Infer(reused_input_tensors_, &reused_output_tensors_);
}

std::map<std::string, float> UltraInferModel::PrintStatisInfoOfRuntime() {
  std::map<std::string, float> statis_info_of_runtime_dict;

  if (time_of_runtime_.size() < 10) {
    FDWARNING << "PrintStatisInfoOfRuntime require the runtime ran 10 times at "
                 "least, but now you only ran "
              << time_of_runtime_.size() << " times." << std::endl;
  }
  double warmup_time = 0.0;
  double remain_time = 0.0;
  int warmup_iter = time_of_runtime_.size() / 5;
  for (size_t i = 0; i < time_of_runtime_.size(); ++i) {
    if (i < warmup_iter) {
      warmup_time += time_of_runtime_[i];
    } else {
      remain_time += time_of_runtime_[i];
    }
  }
  double avg_time = remain_time / (time_of_runtime_.size() - warmup_iter);
  std::cout << "============= Runtime Statis Info(" << ModelName()
            << ") =============" << std::endl;
  std::cout << "Total iterations: " << time_of_runtime_.size() << std::endl;
  std::cout << "Total time of runtime: " << warmup_time + remain_time << "s."
            << std::endl;
  std::cout << "Warmup iterations: " << warmup_iter << std::endl;
  std::cout << "Total time of runtime in warmup step: " << warmup_time << "s."
            << std::endl;
  std::cout << "Average time of runtime exclude warmup step: "
            << avg_time * 1000 << "ms." << std::endl;

  statis_info_of_runtime_dict["total_time"] = warmup_time + remain_time;
  statis_info_of_runtime_dict["warmup_time"] = warmup_time;
  statis_info_of_runtime_dict["remain_time"] = remain_time;
  statis_info_of_runtime_dict["warmup_iter"] = warmup_iter;
  statis_info_of_runtime_dict["avg_time"] = avg_time;
  statis_info_of_runtime_dict["iterations"] = time_of_runtime_.size();

  return statis_info_of_runtime_dict;
}
} // namespace ultra_infer
