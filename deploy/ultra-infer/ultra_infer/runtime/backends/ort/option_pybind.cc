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
#include "ultra_infer/runtime/backends/ort/option.h"

namespace ultra_infer {

void BindOrtOption(pybind11::module &m) {
  pybind11::class_<OrtBackendOption>(m, "OrtBackendOption")
      .def(pybind11::init())
      .def_readwrite("graph_optimization_level",
                     &OrtBackendOption::graph_optimization_level)
      .def_readwrite("intra_op_num_threads",
                     &OrtBackendOption::intra_op_num_threads)
      .def_readwrite("inter_op_num_threads",
                     &OrtBackendOption::inter_op_num_threads)
      .def_readwrite("execution_mode", &OrtBackendOption::execution_mode)
      .def_readwrite("device", &OrtBackendOption::device)
      .def_readwrite("device_id", &OrtBackendOption::device_id)
      .def_readwrite("enable_fp16", &OrtBackendOption::enable_fp16)
      .def("disable_ort_fp16_op_types",
           &OrtBackendOption::DisableOrtFP16OpTypes);
}

} // namespace ultra_infer
