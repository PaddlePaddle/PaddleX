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
void BindSmoke(pybind11::module &m) {
  pybind11::class_<vision::perception::SmokePreprocessor,
                   vision::ProcessorManager>(m, "SmokePreprocessor")
      .def(pybind11::init<std::string>())
      .def("run", [](vision::perception::SmokePreprocessor &self,
                     std::vector<pybind11::array> &im_list) {
        std::vector<vision::FDMat> images;
        for (size_t i = 0; i < im_list.size(); ++i) {
          images.push_back(vision::WrapMat(PyArrayToCvMat(im_list[i])));
        }
        std::vector<FDTensor> outputs;
        if (!self.Run(&images, &outputs)) {
          throw std::runtime_error(
              "Failed to preprocess the input data in SmokePreprocessor.");
        }
        for (size_t i = 0; i < outputs.size(); ++i) {
          outputs[i].StopSharing();
        }
        return outputs;
      });

  pybind11::class_<vision::perception::SmokePostprocessor>(m,
                                                           "SmokePostprocessor")
      .def(pybind11::init<>())
      .def("run",
           [](vision::perception::SmokePostprocessor &self,
              std::vector<FDTensor> &inputs) {
             std::vector<vision::PerceptionResult> results;
             if (!self.Run(inputs, &results)) {
               throw std::runtime_error(
                   "Failed to postprocess the runtime result in "
                   "SmokePostprocessor.");
             }
             return results;
           })
      .def("run", [](vision::perception::SmokePostprocessor &self,
                     std::vector<pybind11::array> &input_array) {
        std::vector<vision::PerceptionResult> results;
        std::vector<FDTensor> inputs;
        PyArrayToTensorList(input_array, &inputs, /*share_buffer=*/true);
        if (!self.Run(inputs, &results)) {
          throw std::runtime_error(
              "Failed to postprocess the runtime result in "
              "SmokePostprocessor.");
        }
        return results;
      });

  pybind11::class_<vision::perception::Smoke, UltraInferModel>(m, "Smoke")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def("predict",
           [](vision::perception::Smoke &self, pybind11::array &data) {
             auto mat = PyArrayToCvMat(data);
             vision::PerceptionResult res;
             self.Predict(mat, &res);
             return res;
           })
      .def("batch_predict",
           [](vision::perception::Smoke &self,
              std::vector<pybind11::array> &data) {
             std::vector<cv::Mat> images;
             for (size_t i = 0; i < data.size(); ++i) {
               images.push_back(PyArrayToCvMat(data[i]));
             }
             std::vector<vision::PerceptionResult> results;
             self.BatchPredict(images, &results);
             return results;
           })
      .def_property_readonly("preprocessor",
                             &vision::perception::Smoke::GetPreprocessor)
      .def_property_readonly("postprocessor",
                             &vision::perception::Smoke::GetPostprocessor);
}
} // namespace ultra_infer
