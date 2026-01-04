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
void BindPaddleClas(pybind11::module &m) {
  pybind11::class_<vision::classification::PaddleClasPreprocessor,
                   vision::ProcessorManager>(m, "PaddleClasPreprocessor")
      .def(pybind11::init<std::string>())
      .def("disable_normalize",
           [](vision::classification::PaddleClasPreprocessor &self) {
             self.DisableNormalize();
           })
      .def("disable_permute",
           [](vision::classification::PaddleClasPreprocessor &self) {
             self.DisablePermute();
           })
      .def("initial_resize_on_cpu",
           [](vision::classification::PaddleClasPreprocessor &self, bool v) {
             self.InitialResizeOnCpu(v);
           });

  pybind11::class_<vision::classification::PaddleClasPostprocessor>(
      m, "PaddleClasPostprocessor")
      .def(pybind11::init<int>())
      .def("run",
           [](vision::classification::PaddleClasPostprocessor &self,
              std::vector<FDTensor> &inputs) {
             std::vector<vision::ClassifyResult> results;
             if (!self.Run(inputs, &results)) {
               throw std::runtime_error(
                   "Failed to postprocess the runtime result in "
                   "PaddleClasPostprocessor.");
             }
             return results;
           })
      .def("run",
           [](vision::classification::PaddleClasPostprocessor &self,
              std::vector<pybind11::array> &input_array) {
             std::vector<vision::ClassifyResult> results;
             std::vector<FDTensor> inputs;
             PyArrayToTensorList(input_array, &inputs, /*share_buffer=*/true);
             if (!self.Run(inputs, &results)) {
               throw std::runtime_error(
                   "Failed to postprocess the runtime result in "
                   "PaddleClasPostprocessor.");
             }
             return results;
           })
      .def_property("topk",
                    &vision::classification::PaddleClasPostprocessor::GetTopk,
                    &vision::classification::PaddleClasPostprocessor::SetTopk);

  pybind11::class_<vision::classification::PaddleClasModel, UltraInferModel>(
      m, "PaddleClasModel")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def("clone",
           [](vision::classification::PaddleClasModel &self) {
             return self.Clone();
           })
      .def("predict",
           [](vision::classification::PaddleClasModel &self,
              pybind11::array &data) {
             cv::Mat im = PyArrayToCvMat(data);
             vision::ClassifyResult result;
             self.Predict(im, &result);
             return result;
           })
      .def("batch_predict",
           [](vision::classification::PaddleClasModel &self,
              std::vector<pybind11::array> &data) {
             std::vector<cv::Mat> images;
             for (size_t i = 0; i < data.size(); ++i) {
               images.push_back(PyArrayToCvMat(data[i]));
             }
             std::vector<vision::ClassifyResult> results;
             self.BatchPredict(images, &results);
             return results;
           })
      .def_property_readonly(
          "preprocessor",
          &vision::classification::PaddleClasModel::GetPreprocessor)
      .def_property_readonly(
          "postprocessor",
          &vision::classification::PaddleClasModel::GetPostprocessor);
}
} // namespace ultra_infer
