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
#include "ultra_infer/runtime/backends/paddle/option.h"

namespace ultra_infer {

void BindIpuOption(pybind11::module &m) {
  pybind11::class_<IpuOption>(m, "IpuOption")
      .def(pybind11::init())
      .def_readwrite("ipu_device_num", &IpuOption::ipu_device_num)
      .def_readwrite("ipu_micro_batch_size", &IpuOption::ipu_micro_batch_size)
      .def_readwrite("ipu_enable_pipelining", &IpuOption::ipu_enable_pipelining)
      .def_readwrite("ipu_batches_per_step", &IpuOption::ipu_batches_per_step)
      .def_readwrite("ipu_enable_fp16", &IpuOption::ipu_enable_fp16)
      .def_readwrite("ipu_replica_num", &IpuOption::ipu_replica_num)
      .def_readwrite("ipu_available_memory_proportion",
                     &IpuOption::ipu_available_memory_proportion)
      .def_readwrite("ipu_enable_half_partial",
                     &IpuOption::ipu_enable_half_partial);
}

void BindPaddleOption(pybind11::module &m) {
  BindIpuOption(m);
  pybind11::class_<PaddleBackendOption>(m, "PaddleBackendOption")
      .def(pybind11::init())
      .def_readwrite("enable_fixed_size_opt",
                     &PaddleBackendOption::enable_fixed_size_opt)
      .def_readwrite("enable_log_info", &PaddleBackendOption::enable_log_info)
      .def_readwrite("enable_mkldnn", &PaddleBackendOption::enable_mkldnn)
      .def_readwrite("enable_trt", &PaddleBackendOption::enable_trt)
      .def_readwrite("enable_memory_optimize",
                     &PaddleBackendOption::enable_memory_optimize)
      .def_readwrite("switch_ir_debug", &PaddleBackendOption::switch_ir_debug)
      .def_readwrite("ipu_option", &PaddleBackendOption::ipu_option)
      .def_readwrite("xpu_option", &PaddleBackendOption::xpu_option)
      .def_readwrite("trt_option", &PaddleBackendOption::trt_option)
      .def_readwrite("allow_build_trt_at_runtime",
                     &PaddleBackendOption::allow_build_trt_at_runtime)
      .def_readwrite("collect_trt_shape",
                     &PaddleBackendOption::collect_trt_shape)
      .def_readwrite("collect_trt_shape_by_device",
                     &PaddleBackendOption::collect_trt_shape_by_device)
      .def_readwrite("mkldnn_cache_size",
                     &PaddleBackendOption::mkldnn_cache_size)
      .def_readwrite("gpu_mem_init_size",
                     &PaddleBackendOption::gpu_mem_init_size)
      .def_readwrite("is_quantize_model",
                     &PaddleBackendOption::is_quantize_model)
      .def_readwrite("inference_precision",
                     &PaddleBackendOption::inference_precision)
      .def_readwrite("enable_inference_cutlass",
                     &PaddleBackendOption::enable_inference_cutlass)
      .def_readwrite("trt_min_subgraph_size",
                     &PaddleBackendOption::trt_min_subgraph_size)
      .def_readwrite("enable_new_ir", &PaddleBackendOption::enable_new_ir)
      .def("disable_trt_ops", &PaddleBackendOption::DisableTrtOps)
      .def("delete_pass", &PaddleBackendOption::DeletePass)
      .def("set_ipu_config", &PaddleBackendOption::SetIpuConfig);
}

} // namespace ultra_infer
