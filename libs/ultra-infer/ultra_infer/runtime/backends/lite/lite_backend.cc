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

#include "ultra_infer/runtime/backends/lite/lite_backend.h"
// https://github.com/PaddlePaddle/Paddle-Lite/issues/8290
// When compiling the UltraInfer dynamic library, namely,
// WITH_STATIC_LIB=OFF, and depending on the Paddle Lite
// static library, you need to include the fake registration
// codes of Paddle Lite. When you compile the UltraInfer static
// library and depends on the Paddle Lite static library,
// WITH_STATIC_LIB=ON, you do not need to include the fake
// registration codes for Paddle Lite, but wait until you
// use the UltraInfer static library.
#if (defined(WITH_LITE_STATIC) && (!defined(WITH_STATIC_LIB)))
#warning You are compiling the UltraInfer dynamic library with \
Paddle Lite static lib We will automatically add some registration \
codes for ops, kernels and passes for Paddle Lite.
#include "paddle_use_kernels.h" // NOLINT
#include "paddle_use_ops.h"     // NOLINT
#include "paddle_use_passes.h"  // NOLINT
#endif

#include <cstring>

namespace ultra_infer {

void LiteBackend::BuildOption(const LiteBackendOption &option) {
  option_ = option;

  if (option_.device == Device::CPU) {
    ConfigureCpu(option_);
  } else if (option_.device == Device::GPU) {
    ConfigureGpu(option_);
  } else if (option_.device == Device::TIMVX) {
    ConfigureTimvx(option_);
  } else if (option_.device == Device::KUNLUNXIN) {
    ConfigureKunlunXin(option_);
  } else if (option_.device == Device::ASCEND) {
    ConfigureAscend(option_);
  }
  if (option_.cpu_threads > 0) {
    config_.set_threads(option_.cpu_threads);
  }
  if (option_.power_mode > 0) {
    config_.set_power_mode(
        static_cast<paddle::lite_api::PowerMode>(option_.power_mode));
  }
}

bool LiteBackend::Init(const RuntimeOption &runtime_option) {
  if (initialized_) {
    FDERROR << "LiteBackend is already initialized, cannot initialize again."
            << std::endl;
    return false;
  }

  if (runtime_option.model_format != ModelFormat::PADDLE) {
    FDERROR
        << "PaddleLiteBackend only supports model format PADDLE, but now it's "
        << runtime_option.model_format << "." << std::endl;
    return false;
  }
  if (runtime_option.device != Device::CPU &&
      runtime_option.device != Device::GPU &&
      runtime_option.device != Device::KUNLUNXIN &&
      runtime_option.device != Device::ASCEND &&
      runtime_option.device != Device::TIMVX) {
    FDERROR << "PaddleLiteBackend only supports "
               "Device::CPU/Device::GPU/Device::TIMVX/Device::KUNLUNXIN/"
               "Device::ASCEND, "
               "but now it's "
            << runtime_option.device << "." << std::endl;
    return false;
  }
  if (runtime_option.device == Device::GPU &&
      !paddle::lite_api::IsOpenCLBackendValid()) {
    FDERROR << "PaddleLiteBackend GPU (OpenCL) is not supported by the current "
               "device."
            << std::endl;
  }
  if (runtime_option.model_from_memory_) {
    FDERROR << "PaddleLiteBackend doesn't support load model from memory, "
               "please load model from disk."
            << std::endl;
    return false;
  }

  if (runtime_option.params_file == "") {
    // Use light api for Arm CPU via MobileConfig.
    FDASSERT(
        runtime_option.device == Device::CPU,
        "In UltraInfer, Paddle Lite light API is only support for Arm CPU now!")
    mobile_config_.set_model_from_file(runtime_option.model_file);
    mobile_config_.set_threads(runtime_option.paddle_lite_option.cpu_threads);
    mobile_config_.set_power_mode(static_cast<paddle::lite_api::PowerMode>(
        runtime_option.paddle_lite_option.power_mode));
    // TODO(qiuyanjun): Add OpenCL support for mobile gpu.
    // Paddle-Lite/blob/develop/lite/api/tools/benchmark/benchmark.h#L265
    // mobile_config_.set_opencl_tune(
    //    tune_mode, opencl_cache_dir, opencl_tuned_file);
    // mobile_config_.set_opencl_precision(gpu_precision);
    predictor_ =
        paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(
            mobile_config_);
  } else {
    // Use full api for many hardwares via CxxConfig.
    config_.set_model_file(runtime_option.model_file);
    config_.set_param_file(runtime_option.params_file);
    BuildOption(runtime_option.paddle_lite_option);
    predictor_ =
        paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::CxxConfig>(
            config_);
    if (option_.optimized_model_dir != "") {
      FDINFO
          << "Optimized model dir is not empty, will save optimized model to: "
          << option_.optimized_model_dir << std::endl;
      predictor_->SaveOptimizedModel(
          option_.optimized_model_dir,
          paddle::lite_api::LiteModelType::kNaiveBuffer);
    }
  }

  inputs_desc_.clear();
  outputs_desc_.clear();
  inputs_order_.clear();
  std::vector<std::string> input_names = predictor_->GetInputNames();
  std::vector<std::string> output_names = predictor_->GetOutputNames();
  for (size_t i = 0; i < input_names.size(); ++i) {
    inputs_order_[input_names[i]] = i;
    TensorInfo info;
    auto tensor = predictor_->GetInput(i);
    auto shape = tensor->shape();
    info.shape.assign(shape.begin(), shape.end());
    info.name = input_names[i];
    info.dtype = LiteDataTypeToFD(tensor->precision());
    inputs_desc_.emplace_back(info);
  }
  for (size_t i = 0; i < output_names.size(); ++i) {
    TensorInfo info;
    auto tensor = predictor_->GetOutput(i);
    auto shape = tensor->shape();
    info.shape.assign(shape.begin(), shape.end());
    info.name = output_names[i];
    if (option_.device != Device::KUNLUNXIN) {
      info.dtype = LiteDataTypeToFD(tensor->precision());
    }
    outputs_desc_.emplace_back(info);
  }

  initialized_ = true;
  return true;
}

TensorInfo LiteBackend::GetInputInfo(int index) {
  FDASSERT(index < NumInputs(),
           "The index: %d should less than the number of inputs: %d.", index,
           NumInputs());
  return inputs_desc_[index];
}

std::vector<TensorInfo> LiteBackend::GetInputInfos() { return inputs_desc_; }

TensorInfo LiteBackend::GetOutputInfo(int index) {
  FDASSERT(index < NumOutputs(),
           "The index: %d should less than the number of outputs %d.", index,
           NumOutputs());
  return outputs_desc_[index];
}

std::vector<TensorInfo> LiteBackend::GetOutputInfos() { return outputs_desc_; }

bool LiteBackend::Infer(std::vector<FDTensor> &inputs,
                        std::vector<FDTensor> *outputs, bool copy_to_fd) {
  if (inputs.size() != inputs_desc_.size()) {
    FDERROR << "[LiteBackend] Size of inputs(" << inputs.size()
            << ") should keep same with the inputs of this model("
            << inputs_desc_.size() << ")." << std::endl;
    return false;
  }

  RUNTIME_PROFILE_LOOP_H2D_D2H_BEGIN
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto iter = inputs_order_.find(inputs[i].name);
    if (iter == inputs_order_.end()) {
      FDERROR << "Cannot find input with name:" << inputs[i].name
              << " in loaded model." << std::endl;
      return false;
    }

    auto tensor = predictor_->GetInput(iter->second);
    // Adjust dims only, allocate lazy.
    tensor->Resize(inputs[i].shape);
    if (inputs[i].dtype == FDDataType::FP32) {
      tensor->CopyFromCpu<float, paddle::lite_api::TargetType::kHost>(
          reinterpret_cast<const float *>(
              const_cast<void *>(inputs[i].CpuData())));
    } else if (inputs[i].dtype == FDDataType::INT32) {
      tensor->CopyFromCpu<int, paddle::lite_api::TargetType::kHost>(
          reinterpret_cast<const int *>(
              const_cast<void *>(inputs[i].CpuData())));
    } else if (inputs[i].dtype == FDDataType::INT8) {
      tensor->CopyFromCpu<int8_t, paddle::lite_api::TargetType::kHost>(
          reinterpret_cast<const int8_t *>(
              const_cast<void *>(inputs[i].CpuData())));
    } else if (inputs[i].dtype == FDDataType::UINT8) {
      tensor->CopyFromCpu<uint8_t, paddle::lite_api::TargetType::kHost>(
          reinterpret_cast<const uint8_t *>(
              const_cast<void *>(inputs[i].CpuData())));
    } else if (inputs[i].dtype == FDDataType::INT64) {
#if (defined(__aarch64__) || defined(__x86_64__) || defined(_M_X64) ||         \
     defined(_M_ARM64))
      tensor->CopyFromCpu<int64_t, paddle::lite_api::TargetType::kHost>(
          reinterpret_cast<const int64_t *>(
              const_cast<void *>(inputs[i].CpuData())));
#else
      FDASSERT(false, "FDDataType::INT64 is not support for x86/armv7 now!");
#endif
    } else {
      FDASSERT(false, "Unexpected data type of %d.", inputs[i].dtype);
    }
  }

  RUNTIME_PROFILE_LOOP_BEGIN(1)
  predictor_->Run();
  RUNTIME_PROFILE_LOOP_END

  outputs->resize(outputs_desc_.size());
  for (size_t i = 0; i < outputs_desc_.size(); ++i) {
    auto tensor = predictor_->GetOutput(i);
    if (outputs_desc_[i].dtype != LiteDataTypeToFD(tensor->precision())) {
      outputs_desc_[i].dtype = LiteDataTypeToFD(tensor->precision());
    }
    (*outputs)[i].Resize(tensor->shape(), outputs_desc_[i].dtype,
                         outputs_desc_[i].name);
    memcpy((*outputs)[i].MutableData(), tensor->data<void>(),
           (*outputs)[i].Nbytes());
  }
  RUNTIME_PROFILE_LOOP_H2D_D2H_END
  return true;
}

bool ReadFile(const std::string &filename, std::vector<char> *contents,
              bool binary) {
  FILE *fp = fopen(filename.c_str(), binary ? "rb" : "r");
  if (!fp) {
    FDERROR << "Cannot open file " << filename << "." << std::endl;
    return false;
  }
  fseek(fp, 0, SEEK_END);
  size_t size = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  contents->clear();
  contents->resize(size);
  size_t offset = 0;
  char *ptr = reinterpret_cast<char *>(&(contents->at(0)));
  while (offset < size) {
    size_t already_read = fread(ptr, 1, size - offset, fp);
    offset += already_read;
    ptr += already_read;
  }
  fclose(fp);
  return true;
}

// Convert data type from paddle lite to ultra_infer
FDDataType LiteDataTypeToFD(const paddle::lite_api::PrecisionType &dtype) {
  if (dtype == paddle::lite_api::PrecisionType::kFloat) {
    return FDDataType::FP32;
  } else if (dtype == paddle::lite_api::PrecisionType::kInt8) {
    return FDDataType::INT8;
  } else if (dtype == paddle::lite_api::PrecisionType::kInt32) {
    return FDDataType::INT32;
  } else if (dtype == paddle::lite_api::PrecisionType::kInt64) {
    return FDDataType::INT64;
  } else if (dtype == paddle::lite_api::PrecisionType::kInt16) {
    return FDDataType::INT16;
  } else if (dtype == paddle::lite_api::PrecisionType::kUInt8) {
    return FDDataType::UINT8;
  } else if (dtype == paddle::lite_api::PrecisionType::kFP64) {
    return FDDataType::FP64;
  }
  FDASSERT(false, "Unexpected data type of %s.",
           paddle::lite_api::PrecisionToStr(dtype).c_str());
  return FDDataType::FP32;
}

} // namespace ultra_infer
