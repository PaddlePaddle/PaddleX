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
void BindYOLOv7End2EndORT(pybind11::module &m) {
  pybind11::class_<vision::detection::YOLOv7End2EndORT, UltraInferModel>(
      m, "YOLOv7End2EndORT")
      .def(pybind11::init<std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def("predict",
           [](vision::detection::YOLOv7End2EndORT &self, pybind11::array &data,
              float conf_threshold) {
             auto mat = PyArrayToCvMat(data);
             vision::DetectionResult res;
             self.Predict(&mat, &res, conf_threshold);
             return res;
           })
      .def_readwrite("size", &vision::detection::YOLOv7End2EndORT::size)
      .def_readwrite("padding_value",
                     &vision::detection::YOLOv7End2EndORT::padding_value)
      .def_readwrite("is_mini_pad",
                     &vision::detection::YOLOv7End2EndORT::is_mini_pad)
      .def_readwrite("is_no_pad",
                     &vision::detection::YOLOv7End2EndORT::is_no_pad)
      .def_readwrite("is_scale_up",
                     &vision::detection::YOLOv7End2EndORT::is_scale_up)
      .def_readwrite("stride", &vision::detection::YOLOv7End2EndORT::stride);
}
} // namespace ultra_infer
