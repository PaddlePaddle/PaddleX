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

#include "ultra_infer/core/fd_type.h"
// https://github.com/PaddlePaddle/Paddle-Lite/issues/8290
#if (defined(WITH_LITE_STATIC) && defined(WITH_STATIC_LIB))
// Whether to output some warning messages when using the
// FastDepoy static library, default OFF. These messages
// are only reserve for debugging.
#if defined(WITH_STATIC_WARNING)
#warning You are using the UltraInfer static library. We will automatically add some registration codes for ops, kernels and passes for Paddle Lite. // NOLINT
#endif
#if !defined(WITH_STATIC_LIB_AT_COMPILING)
#include "paddle_use_kernels.h" // NOLINT
#include "paddle_use_ops.h"     // NOLINT
#include "paddle_use_passes.h"  // NOLINT
#endif
#endif

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ultra_infer {

/*! Paddle Lite power mode for mobile device. */
enum LitePowerMode {
  LITE_POWER_HIGH = 0,      ///< Use Lite Backend with high power mode
  LITE_POWER_LOW = 1,       ///< Use Lite Backend with low power mode
  LITE_POWER_FULL = 2,      ///< Use Lite Backend with full power mode
  LITE_POWER_NO_BIND = 3,   ///< Use Lite Backend with no bind power mode
  LITE_POWER_RAND_HIGH = 4, ///< Use Lite Backend with rand high mode
  LITE_POWER_RAND_LOW = 5   ///< Use Lite Backend with rand low power mode
};

/*! @brief Option object to configure Paddle Lite backend
 */
struct LiteBackendOption {
  /// Paddle Lite power mode for mobile device.
  int power_mode = 3;
  // Number of threads while use CPU
  int cpu_threads = 1;
  /// Enable use half precision
  bool enable_fp16 = false;
  // Inference device, Paddle Lite support CPU/KUNLUNXIN/TIMVX/ASCEND
  Device device = Device::CPU;
  // Index of inference device
  int device_id = 0;
  // TODO(qiuyanjun): add opencl binary path and cache settings.
  std::string opencl_cache_dir = "/data/local/tmp/";
  std::string opencl_tuned_file = "/data/local/tmp/opencl_tuned_kernels.bin";

  /// kunlunxin_l3_workspace_size
  int kunlunxin_l3_workspace_size = 0xfffc00;
  /// kunlunxin_locked
  bool kunlunxin_locked = false;
  /// kunlunxin_autotune
  bool kunlunxin_autotune = true;
  /// kunlunxin_autotune_file
  std::string kunlunxin_autotune_file = "";
  /// kunlunxin_precision
  std::string kunlunxin_precision = "int16";
  /// kunlunxin_adaptive_seqlen
  bool kunlunxin_adaptive_seqlen = false;
  /// kunlunxin_enable_multi_stream
  bool kunlunxin_enable_multi_stream = false;
  /// kunlunxin_gm_default_size
  int64_t kunlunxin_gm_default_size = 0;

  /// Optimized model dir for CxxConfig
  std::string optimized_model_dir = "";
  /// nnadapter_subgraph_partition_config_path
  std::string nnadapter_subgraph_partition_config_path = "";
  /// nnadapter_subgraph_partition_config_buffer
  std::string nnadapter_subgraph_partition_config_buffer = "";
  /// nnadapter_context_properties
  std::string nnadapter_context_properties = "";
  /// nnadapter_model_cache_dir
  std::string nnadapter_model_cache_dir = "";
  /// nnadapter_mixed_precision_quantization_config_path
  std::string nnadapter_mixed_precision_quantization_config_path = "";
  /// nnadapter_dynamic_shape_info
  std::map<std::string, std::vector<std::vector<int64_t>>>
      nnadapter_dynamic_shape_info = {{"", {{0}}}};
  /// nnadapter_device_names
  std::vector<std::string> nnadapter_device_names = {};
};
} // namespace ultra_infer
