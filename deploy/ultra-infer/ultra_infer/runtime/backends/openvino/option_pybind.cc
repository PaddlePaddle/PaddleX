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
#include "ultra_infer/runtime/backends/openvino/option.h"

namespace ultra_infer {

void BindOpenVINOOption(pybind11::module &m) {
  pybind11::class_<OpenVINOBackendOption>(m, "OpenVINOBackendOption")
      .def(pybind11::init())
      .def_readwrite("cpu_thread_num", &OpenVINOBackendOption::cpu_thread_num)
      .def_readwrite("num_streams", &OpenVINOBackendOption::num_streams)
      .def_readwrite("affinity", &OpenVINOBackendOption::affinity)
      .def_readwrite("hint", &OpenVINOBackendOption::hint)
      .def("set_device", &OpenVINOBackendOption::SetDevice)
      .def("set_shape_info", &OpenVINOBackendOption::SetShapeInfo)
      .def("set_cpu_operators", &OpenVINOBackendOption::SetCpuOperators)
      .def("set_affinity", &OpenVINOBackendOption::SetAffinity)
      .def("set_performance_hint", &OpenVINOBackendOption::SetPerformanceHint)
      .def("set_stream_num", &OpenVINOBackendOption::SetStreamNum);
}

} // namespace ultra_infer
