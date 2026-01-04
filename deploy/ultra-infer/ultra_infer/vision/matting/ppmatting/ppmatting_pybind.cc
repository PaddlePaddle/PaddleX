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
void BindPPMatting(pybind11::module &m) {
  pybind11::class_<vision::matting::PPMatting, UltraInferModel>(m, "PPMatting")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def("predict",
           [](vision::matting::PPMatting &self, pybind11::array &data) {
             auto mat = PyArrayToCvMat(data);
             vision::MattingResult res;
             self.Predict(&mat, &res);
             return res;
           });
}
} // namespace ultra_infer
