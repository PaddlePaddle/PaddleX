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

namespace ultra_infer {
void BindFDMatBatch(pybind11::module &m) {
  pybind11::class_<vision::FDMatBatch>(m, "FDMatBatch")
      .def(pybind11::init<>(), "Default constructor")
      .def_readwrite("input_cache", &vision::FDMatBatch::input_cache)
      .def_readwrite("output_cache", &vision::FDMatBatch::output_cache)
      .def_readwrite("mats", &vision::FDMatBatch::mats)
      .def("from_mats",
           [](vision::FDMatBatch &self, std::vector<vision::FDMat> &_mats) {
             self.mats_holder = _mats;
             self.mats = &(self.mats_holder);
           });
}

} // namespace ultra_infer
