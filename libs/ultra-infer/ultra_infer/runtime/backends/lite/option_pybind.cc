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

#include "ultra_infer/pybind/main.h"
#include "ultra_infer/runtime/backends/lite/option.h"

namespace ultra_infer {

void BindLiteOption(pybind11::module &m) {
  pybind11::class_<LiteBackendOption>(m, "LiteBackendOption")
      .def(pybind11::init())
      .def_readwrite("power_mode", &LiteBackendOption::power_mode)
      .def_readwrite("cpu_threads", &LiteBackendOption::cpu_threads)
      .def_readwrite("enable_fp16", &LiteBackendOption::enable_fp16)
      .def_readwrite("device", &LiteBackendOption::device)
      .def_readwrite("optimized_model_dir",
                     &LiteBackendOption::optimized_model_dir)
      .def_readwrite(
          "nnadapter_subgraph_partition_config_path",
          &LiteBackendOption::nnadapter_subgraph_partition_config_path)
      .def_readwrite(
          "nnadapter_subgraph_partition_config_buffer",
          &LiteBackendOption::nnadapter_subgraph_partition_config_buffer)
      .def_readwrite("nnadapter_context_properties",
                     &LiteBackendOption::nnadapter_context_properties)
      .def_readwrite("nnadapter_model_cache_dir",
                     &LiteBackendOption::nnadapter_model_cache_dir)
      .def_readwrite("nnadapter_mixed_precision_quantization_config_path",
                     &LiteBackendOption::
                         nnadapter_mixed_precision_quantization_config_path)
      .def_readwrite("nnadapter_dynamic_shape_info",
                     &LiteBackendOption::nnadapter_dynamic_shape_info)
      .def_readwrite("nnadapter_device_names",
                     &LiteBackendOption::nnadapter_device_names)
      .def_readwrite("device_id", &LiteBackendOption::device_id)
      .def_readwrite("kunlunxin_l3_workspace_size",
                     &LiteBackendOption::kunlunxin_l3_workspace_size)
      .def_readwrite("kunlunxin_locked", &LiteBackendOption::kunlunxin_locked)
      .def_readwrite("kunlunxin_autotune",
                     &LiteBackendOption::kunlunxin_autotune)
      .def_readwrite("kunlunxin_autotune_file",
                     &LiteBackendOption::kunlunxin_autotune_file)
      .def_readwrite("kunlunxin_precision",
                     &LiteBackendOption::kunlunxin_precision)
      .def_readwrite("kunlunxin_gm_default_size",
                     &LiteBackendOption::kunlunxin_gm_default_size)
      .def_readwrite("kunlunxin_adaptive_seqlen",
                     &LiteBackendOption::kunlunxin_adaptive_seqlen)
      .def_readwrite("kunlunxin_enable_multi_stream",
                     &LiteBackendOption::kunlunxin_enable_multi_stream);
}

} // namespace ultra_infer
