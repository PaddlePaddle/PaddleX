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
#include "ultra_infer/ultra_infer_model.h"

namespace ultra_infer {

void BindFDModel(pybind11::module &m) {
  pybind11::class_<UltraInferModel>(m, "UltraInferModel")
      .def(pybind11::init<>(), "Default Constructor")
      .def("model_name", &UltraInferModel::ModelName)
      .def("num_inputs_of_runtime", &UltraInferModel::NumInputsOfRuntime)
      .def("num_outputs_of_runtime", &UltraInferModel::NumOutputsOfRuntime)
      .def("input_info_of_runtime", &UltraInferModel::InputInfoOfRuntime)
      .def("output_info_of_runtime", &UltraInferModel::OutputInfoOfRuntime)
      .def("enable_record_time_of_runtime",
           &UltraInferModel::EnableRecordTimeOfRuntime)
      .def("disable_record_time_of_runtime",
           &UltraInferModel::DisableRecordTimeOfRuntime)
      .def("print_statis_info_of_runtime",
           &UltraInferModel::PrintStatisInfoOfRuntime)
      .def("get_profile_time", &UltraInferModel::GetProfileTime)
      .def("initialized", &UltraInferModel::Initialized)
      .def_readwrite("runtime_option", &UltraInferModel::runtime_option)
      .def_readwrite("valid_cpu_backends", &UltraInferModel::valid_cpu_backends)
      .def_readwrite("valid_gpu_backends",
                     &UltraInferModel::valid_gpu_backends);
}

} // namespace ultra_infer
