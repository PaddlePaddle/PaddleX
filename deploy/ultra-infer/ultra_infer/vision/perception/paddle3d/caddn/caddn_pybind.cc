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
void BindCaddn(pybind11::module &m) {
  pybind11::class_<vision::perception::CaddnPreprocessor,
                   vision::ProcessorManager>(m, "CaddnPreprocessor")
      .def(pybind11::init<std::string>())
      .def("run",
           [](vision::perception::CaddnPreprocessor &self,
              std::vector<pybind11::array> &im_list,
              std::vector<float> &cam_data, std::vector<float> &lidar_data) {
             std::vector<vision::FDMat> images;
             for (size_t i = 0; i < im_list.size(); ++i) {
               images.push_back(vision::WrapMat(PyArrayToCvMat(im_list[i])));
             }
             std::vector<FDTensor> outputs;
             if (!self.Run(&images, cam_data, lidar_data, &outputs)) {
               throw std::runtime_error(
                   "Failed to preprocess the input data in CaddnPreprocessor.");
             }
             for (size_t i = 0; i < outputs.size(); ++i) {
               outputs[i].StopSharing();
             }
             return outputs;
           });

  pybind11::class_<vision::perception::CaddnPostprocessor>(m,
                                                           "CaddnPostprocessor")
      .def(pybind11::init<>())
      .def("run",
           [](vision::perception::CaddnPostprocessor &self,
              std::vector<FDTensor> &inputs) {
             std::vector<vision::PerceptionResult> results;
             if (!self.Run(inputs, &results)) {
               throw std::runtime_error(
                   "Failed to postprocess the runtime result in "
                   "CaddnPostprocessor.");
             }
             return results;
           })
      .def("run", [](vision::perception::CaddnPostprocessor &self,
                     std::vector<pybind11::array> &input_array) {
        std::vector<vision::PerceptionResult> results;
        std::vector<FDTensor> inputs;
        PyArrayToTensorList(input_array, &inputs, /*share_buffer=*/true);
        if (!self.Run(inputs, &results)) {
          throw std::runtime_error(
              "Failed to postprocess the runtime result in "
              "CaddnPostprocessor.");
        }
        return results;
      });

  pybind11::class_<vision::perception::Caddn, UltraInferModel>(m, "Caddn")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def("predict",
           [](vision::perception::Caddn &self, pybind11::array &data,
              std::vector<float> &cam_data, std::vector<float> &lidar_data) {
             auto mat = PyArrayToCvMat(data);
             vision::PerceptionResult res;
             self.Predict(mat, cam_data, lidar_data, &res);
             return res;
           })
      .def("batch_predict",
           [](vision::perception::Caddn &self,
              std::vector<pybind11::array> &data, std::vector<float> &cam_data,
              std::vector<float> &lidar_data) {
             std::vector<cv::Mat> images;
             for (size_t i = 0; i < data.size(); ++i) {
               images.push_back(PyArrayToCvMat(data[i]));
             }
             std::vector<vision::PerceptionResult> results;
             self.BatchPredict(images, cam_data, lidar_data, &results);
             return results;
           })
      .def_property_readonly("preprocessor",
                             &vision::perception::Caddn::GetPreprocessor)
      .def_property_readonly("postprocessor",
                             &vision::perception::Caddn::GetPostprocessor);
}
} // namespace ultra_infer
