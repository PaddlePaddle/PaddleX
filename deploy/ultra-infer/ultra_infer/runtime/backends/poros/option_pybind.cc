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
#include "ultra_infer/runtime/backends/poros/option.h"

namespace ultra_infer {

void BindPorosOption(pybind11::module &m) {
  pybind11::class_<PorosBackendOption>(m, "PorosBackendOption")
      .def(pybind11::init())
      .def_readwrite("long_to_int", &PorosBackendOption::long_to_int)
      .def_readwrite("use_nvidia_tf32", &PorosBackendOption::use_nvidia_tf32)
      .def_readwrite("unconst_ops_thres",
                     &PorosBackendOption::unconst_ops_thres)
      .def_readwrite("prewarm_datatypes",
                     &PorosBackendOption::prewarm_datatypes)
      .def_readwrite("enable_fp16", &PorosBackendOption::enable_fp16)
      .def_readwrite("enable_int8", &PorosBackendOption::enable_int8)
      .def_readwrite("is_dynamic", &PorosBackendOption::is_dynamic)
      .def_readwrite("max_batch_size", &PorosBackendOption::max_batch_size)
      .def_readwrite("max_workspace_size",
                     &PorosBackendOption::max_workspace_size);
}

} // namespace ultra_infer
