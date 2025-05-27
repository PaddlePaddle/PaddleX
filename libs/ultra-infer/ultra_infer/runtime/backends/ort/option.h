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
#include "ultra_infer/runtime/enum_variables.h"
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
namespace ultra_infer {

/*! @brief Option object to configure ONNX Runtime backend
 */
struct OrtBackendOption {
  /// Level of graph optimization,
  ///         /-1: mean default(Enable all the optimization strategy)
  ///         /0: disable all the optimization strategy/1: enable basic strategy
  ///         /2:enable extend strategy/99: enable all
  int graph_optimization_level = -1;
  /// Number of threads to execute the operator, -1: default
  int intra_op_num_threads = -1;
  /// Number of threads to execute the graph,
  ///         -1: default. This parameter only will bring effects
  ///         while the `OrtBackendOption::execution_mode` set to 1.
  int inter_op_num_threads = -1;
  /// Execution mode for the graph, -1: default(Sequential mode)
  ///         /0: Sequential mode, execute the operators in graph one by one.
  ///         /1: Parallel mode, execute the operators in graph parallelly.
  int execution_mode = -1;
  /// Inference device, OrtBackend supports CPU/GPU
  Device device = Device::CPU;
  /// Inference device id
  int device_id = 0;
  void *external_stream_ = nullptr;
  /// Use fp16 to infer
  bool enable_fp16 = false;

  std::vector<std::string> ort_disabled_ops_{};
  void DisableOrtFP16OpTypes(const std::vector<std::string> &ops) {
    ort_disabled_ops_.insert(ort_disabled_ops_.end(), ops.begin(), ops.end());
  }
};
} // namespace ultra_infer
