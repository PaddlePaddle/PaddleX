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
void BindYOLOv5Lite(pybind11::module &m) {
  pybind11::class_<vision::detection::YOLOv5Lite, UltraInferModel>(m,
                                                                   "YOLOv5Lite")
      .def(pybind11::init<std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def("predict",
           [](vision::detection::YOLOv5Lite &self, pybind11::array &data,
              float conf_threshold, float nms_iou_threshold) {
             auto mat = PyArrayToCvMat(data);
             vision::DetectionResult res;
             self.Predict(&mat, &res, conf_threshold, nms_iou_threshold);
             return res;
           })
      .def("use_cuda_preprocessing",
           [](vision::detection::YOLOv5Lite &self, int max_image_size) {
             self.UseCudaPreprocessing(max_image_size);
           })
      .def_readwrite("size", &vision::detection::YOLOv5Lite::size)
      .def_readwrite("padding_value",
                     &vision::detection::YOLOv5Lite::padding_value)
      .def_readwrite("downsample_strides",
                     &vision::detection::YOLOv5Lite::downsample_strides)
      .def_readwrite("is_mini_pad", &vision::detection::YOLOv5Lite::is_mini_pad)
      .def_readwrite("is_no_pad", &vision::detection::YOLOv5Lite::is_no_pad)
      .def_readwrite("is_scale_up", &vision::detection::YOLOv5Lite::is_scale_up)
      .def_readwrite("stride", &vision::detection::YOLOv5Lite::stride)
      .def_readwrite("max_wh", &vision::detection::YOLOv5Lite::max_wh)
      .def_readwrite("anchor_config",
                     &vision::detection::YOLOv5Lite::anchor_config)
      .def_readwrite("is_decode_exported",
                     &vision::detection::YOLOv5Lite::is_decode_exported);
}
} // namespace ultra_infer
