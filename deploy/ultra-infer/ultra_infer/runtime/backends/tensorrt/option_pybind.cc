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
#include "ultra_infer/runtime/backends/tensorrt/option.h"

namespace ultra_infer {

void BindTrtOption(pybind11::module &m) {
  pybind11::class_<TrtBackendOption>(m, "TrtBackendOption")
      .def(pybind11::init())
      .def_readwrite("enable_fp16", &TrtBackendOption::enable_fp16)
      .def_readwrite("enable_log_info", &TrtBackendOption::enable_log_info)
      .def_readwrite("max_batch_size", &TrtBackendOption::max_batch_size)
      .def_readwrite("max_workspace_size",
                     &TrtBackendOption::max_workspace_size)
      .def_readwrite("serialize_file", &TrtBackendOption::serialize_file)
      .def("set_shape", &TrtBackendOption::SetShape)
      .def("set_input_data", &TrtBackendOption::SetInputData);
}

} // namespace ultra_infer
