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
#include <pybind11/stl.h>

namespace ultra_infer {
void BindPPTinyPosePipeline(pybind11::module &m) {
  pybind11::class_<pipeline::PPTinyPose>(m, "PPTinyPose")
      .def(pybind11::init<
           ultra_infer::vision::detection::PicoDet *,
           ultra_infer::vision::keypointdetection::PPTinyPose *>())
      .def("predict",
           [](pipeline::PPTinyPose &self, pybind11::array &data) {
             auto mat = PyArrayToCvMat(data);
             vision::KeyPointDetectionResult res;
             self.Predict(&mat, &res);
             return res;
           })

      .def_readwrite("detection_model_score_threshold",
                     &pipeline::PPTinyPose::detection_model_score_threshold);
}

} // namespace ultra_infer
