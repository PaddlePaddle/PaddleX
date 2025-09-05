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

#include <cstring>

namespace ultra_infer {

#if defined(__arm__) || defined(__aarch64__)
#define FD_LITE_HOST TARGET(kARM)
#elif defined(__x86_64__)
#define FD_LITE_HOST TARGET(kX86)
#endif

std::vector<paddle::lite_api::Place>
GetPlacesForCpu(const LiteBackendOption &option) {
  std::vector<paddle::lite_api::Place> valid_places;
  valid_places.push_back(
      paddle::lite_api::Place{FD_LITE_HOST, PRECISION(kInt8)});
  if (option.enable_fp16) {
    paddle::lite_api::MobileConfig check_fp16_config;
    if (check_fp16_config.check_fp16_valid()) {
      valid_places.push_back(
          paddle::lite_api::Place{FD_LITE_HOST, PRECISION(kFP16)});
    } else {
      FDWARNING << "Current CPU doesn't support float16 precision, will "
                   "fallback to float32."
                << std::endl;
    }
  }
  valid_places.push_back(
      paddle::lite_api::Place{FD_LITE_HOST, PRECISION(kFloat)});
  return valid_places;
}

void LiteBackend::ConfigureCpu(const LiteBackendOption &option) {
  config_.set_valid_places(GetPlacesForCpu(option));
}

void LiteBackend::ConfigureGpu(const LiteBackendOption &option) {
  std::vector<paddle::lite_api::Place> valid_places;
  if (option.enable_fp16) {
    valid_places.emplace_back(paddle::lite_api::Place{
        TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageDefault)});
    valid_places.emplace_back(paddle::lite_api::Place{
        TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageFolder)});
  }
  valid_places.emplace_back(
      paddle::lite_api::Place{TARGET(kOpenCL), PRECISION(kFloat)});
  valid_places.emplace_back(paddle::lite_api::Place{
      TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kImageDefault)});
  valid_places.emplace_back(paddle::lite_api::Place{
      TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kImageFolder)});
  valid_places.emplace_back(
      paddle::lite_api::Place{TARGET(kOpenCL), PRECISION(kAny)});
  valid_places.emplace_back(
      paddle::lite_api::Place{TARGET(kOpenCL), PRECISION(kInt32)});
  valid_places.emplace_back(
      paddle::lite_api::Place{TARGET(kARM), PRECISION(kInt8)});
  valid_places.emplace_back(
      paddle::lite_api::Place{TARGET(kARM), PRECISION(kFloat)});
  config_.set_valid_places(valid_places);
}

void LiteBackend::ConfigureKunlunXin(const LiteBackendOption &option) {
  std::vector<paddle::lite_api::Place> valid_places;
  // TODO(yeliang): Placing kInt8 first may cause accuracy issues of some model
  // valid_places.push_back(
  //     paddle::lite_api::Place{TARGET(kXPU), PRECISION(kInt8)});
  if (option.enable_fp16) {
    valid_places.push_back(
        paddle::lite_api::Place{TARGET(kXPU), PRECISION(kFP16)});
  }
  valid_places.push_back(
      paddle::lite_api::Place{TARGET(kXPU), PRECISION(kFloat)});

  config_.set_xpu_dev_per_thread(option.device_id);
  config_.set_xpu_workspace_l3_size_per_thread(
      option.kunlunxin_l3_workspace_size);
  config_.set_xpu_l3_cache_method(option.kunlunxin_l3_workspace_size,
                                  option.kunlunxin_locked);
  config_.set_xpu_l3_cache_autotune(option.kunlunxin_autotune);
  config_.set_xpu_conv_autotune(option.kunlunxin_autotune,
                                option.kunlunxin_autotune_file);
  config_.set_xpu_multi_encoder_method(option.kunlunxin_precision,
                                       option.kunlunxin_adaptive_seqlen);
  config_.set_xpu_gm_workspace_method(option.kunlunxin_gm_default_size);
  if (option.kunlunxin_enable_multi_stream) {
    config_.enable_xpu_multi_stream();
  }
  auto cpu_places = GetPlacesForCpu(option);
  valid_places.insert(valid_places.end(), cpu_places.begin(), cpu_places.end());
  config_.set_valid_places(valid_places);
}

void LiteBackend::ConfigureTimvx(const LiteBackendOption &option) {
  config_.set_nnadapter_device_names({"verisilicon_timvx"});
  std::vector<paddle::lite_api::Place> valid_places;
  valid_places.push_back(
      paddle::lite_api::Place{TARGET(kNNAdapter), PRECISION(kInt8)});
  valid_places.push_back(
      paddle::lite_api::Place{TARGET(kNNAdapter), PRECISION(kFloat)});
  auto cpu_places = GetPlacesForCpu(option);
  valid_places.insert(valid_places.end(), cpu_places.begin(), cpu_places.end());
  config_.set_valid_places(valid_places);
  ConfigureNNAdapter(option);
}

void LiteBackend::ConfigureAscend(const LiteBackendOption &option) {
  config_.set_nnadapter_device_names({"huawei_ascend_npu"});
  std::vector<paddle::lite_api::Place> valid_places;
  valid_places.push_back(
      paddle::lite_api::Place{TARGET(kNNAdapter), PRECISION(kInt8)});
  valid_places.push_back(
      paddle::lite_api::Place{TARGET(kNNAdapter), PRECISION(kFloat)});
  auto cpu_places = GetPlacesForCpu(option);
  valid_places.insert(valid_places.end(), cpu_places.begin(), cpu_places.end());
  config_.set_valid_places(valid_places);
  ConfigureNNAdapter(option);
}

void LiteBackend::ConfigureNNAdapter(const LiteBackendOption &option) {
  if (!option.nnadapter_subgraph_partition_config_path.empty()) {
    std::vector<char> nnadapter_subgraph_partition_config_buffer;
    if (ReadFile(option.nnadapter_subgraph_partition_config_path,
                 &nnadapter_subgraph_partition_config_buffer, false)) {
      if (!nnadapter_subgraph_partition_config_buffer.empty()) {
        std::string nnadapter_subgraph_partition_config_string(
            nnadapter_subgraph_partition_config_buffer.data(),
            nnadapter_subgraph_partition_config_buffer.size());
        config_.set_nnadapter_subgraph_partition_config_buffer(
            nnadapter_subgraph_partition_config_string);
      }
    }
  }

  if (!option.nnadapter_context_properties.empty()) {
    config_.set_nnadapter_context_properties(
        option.nnadapter_context_properties);
  }

  if (!option.nnadapter_model_cache_dir.empty()) {
    config_.set_nnadapter_model_cache_dir(option.nnadapter_model_cache_dir);
  }

  if (!option.nnadapter_mixed_precision_quantization_config_path.empty()) {
    config_.set_nnadapter_mixed_precision_quantization_config_path(
        option.nnadapter_mixed_precision_quantization_config_path);
  }

  if (!option.nnadapter_subgraph_partition_config_path.empty()) {
    config_.set_nnadapter_subgraph_partition_config_path(
        option.nnadapter_subgraph_partition_config_path);
  }

  config_.set_nnadapter_dynamic_shape_info(option.nnadapter_dynamic_shape_info);
}

} // namespace ultra_infer
