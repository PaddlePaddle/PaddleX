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
// namespace should be  `ultra_infer`
namespace ultra_infer {
// the name of Pybind function should be Bind${model_name}
void BindResNet(pybind11::module &m) {
  // the constructor and the predict function are necessary
  // the constructor is used to initialize the python model class.
  // the necessary public functions and variables like `size`, `mean_vals`
  // should also be binded.
  pybind11::class_<vision::classification::ResNet, UltraInferModel>(m, "ResNet")
      .def(pybind11::init<std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def("predict",
           [](vision::classification::ResNet &self, pybind11::array &data,
              int topk = 1) {
             auto mat = PyArrayToCvMat(data);
             vision::ClassifyResult res;
             self.Predict(&mat, &res, topk);
             return res;
           })
      .def_readwrite("size", &vision::classification::ResNet::size)
      .def_readwrite("mean_vals", &vision::classification::ResNet::mean_vals)
      .def_readwrite("std_vals", &vision::classification::ResNet::std_vals);
}
} // namespace ultra_infer
