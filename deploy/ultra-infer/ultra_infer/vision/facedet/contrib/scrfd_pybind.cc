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
void BindSCRFD(pybind11::module &m) {
  // Bind SCRFD
  pybind11::class_<vision::facedet::SCRFD, UltraInferModel>(m, "SCRFD")
      .def(pybind11::init<std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def("predict",
           [](vision::facedet::SCRFD &self, pybind11::array &data,
              float conf_threshold, float nms_iou_threshold) {
             auto mat = PyArrayToCvMat(data);
             vision::FaceDetectionResult res;
             self.Predict(&mat, &res, conf_threshold, nms_iou_threshold);
             return res;
           })
      .def("disable_normalize", &vision::facedet::SCRFD::DisableNormalize)
      .def("disable_permute", &vision::facedet::SCRFD::DisablePermute)
      .def_readwrite("size", &vision::facedet::SCRFD::size)
      .def_readwrite("padding_value", &vision::facedet::SCRFD::padding_value)
      .def_readwrite("is_mini_pad", &vision::facedet::SCRFD::is_mini_pad)
      .def_readwrite("is_no_pad", &vision::facedet::SCRFD::is_no_pad)
      .def_readwrite("is_scale_up", &vision::facedet::SCRFD::is_scale_up)
      .def_readwrite("stride", &vision::facedet::SCRFD::stride)
      .def_readwrite("use_kps", &vision::facedet::SCRFD::use_kps)
      .def_readwrite("max_nms", &vision::facedet::SCRFD::max_nms)
      .def_readwrite("downsample_strides",
                     &vision::facedet::SCRFD::downsample_strides)
      .def_readwrite("num_anchors", &vision::facedet::SCRFD::num_anchors)
      .def_readwrite("landmarks_per_face",
                     &vision::facedet::SCRFD::landmarks_per_face);
}

} // namespace ultra_infer
