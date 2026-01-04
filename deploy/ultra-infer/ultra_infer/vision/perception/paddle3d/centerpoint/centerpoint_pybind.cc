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
void BindCenterpoint(pybind11::module &m) {
  pybind11::class_<vision::perception::CenterpointPreprocessor,
                   vision::ProcessorManager>(m, "CenterpointPreprocessor")
      .def(pybind11::init<std::string>())
      .def("run", [](vision::perception::CenterpointPreprocessor &self,
                     std::vector<std::string> points_dir,
                     const int64_t num_point_dim, const int with_timelag) {
        std::vector<FDTensor> outputs;
        if (!self.Run(points_dir, num_point_dim, with_timelag, outputs)) {
          throw std::runtime_error("Failed to preprocess the input data in "
                                   "CenterpointPreprocessor.");
        }

        return outputs;
      });

  pybind11::class_<vision::perception::Centerpoint, UltraInferModel>(
      m, "Centerpoint")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def("predict",
           [](vision::perception::Centerpoint &self, std::string point_dir) {
             vision::PerceptionResult result;
             self.Predict(point_dir, &result);
             return result;
           })
      .def("batch_predict",
           [](vision::perception::Centerpoint &self,
              std::vector<std::string> &points_dir) {
             std::vector<vision::PerceptionResult> results;
             self.BatchPredict(points_dir, &results);
             return results;
           })
      .def_property_readonly("preprocessor",
                             &vision::perception::Centerpoint::GetPreprocessor)
      .def_property_readonly(
          "postprocessor", &vision::perception::Centerpoint::GetPostprocessor);
}
} // namespace ultra_infer
