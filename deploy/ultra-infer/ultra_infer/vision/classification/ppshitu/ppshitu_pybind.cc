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
void BindPPShiTuV2(pybind11::module &m) {
  pybind11::class_<vision::classification::PPShiTuV2RecognizerPreprocessor,
                   vision::ProcessorManager>(m,
                                             "PPShiTuV2RecognizerPreprocessor")
      .def(pybind11::init<std::string>())
      .def("disable_normalize",
           [](vision::classification::PPShiTuV2RecognizerPreprocessor &self) {
             self.DisableNormalize();
           })
      .def("disable_permute",
           [](vision::classification::PPShiTuV2RecognizerPreprocessor &self) {
             self.DisablePermute();
           })
      .def("initial_resize_on_cpu",
           [](vision::classification::PPShiTuV2RecognizerPreprocessor &self,
              bool v) { self.InitialResizeOnCpu(v); });

  pybind11::class_<vision::classification::PPShiTuV2RecognizerPostprocessor>(
      m, "PPShiTuV2RecognizerPostprocessor")
      .def(pybind11::init<>())
      .def("run",
           [](vision::classification::PPShiTuV2RecognizerPostprocessor &self,
              std::vector<FDTensor> &inputs) {
             std::vector<vision::ClassifyResult> results;
             if (!self.Run(inputs, &results)) {
               throw std::runtime_error(
                   "Failed to postprocess the runtime result in "
                   "PPShiTuV2RecognizerPostprocessor.");
             }
             return results;
           })
      .def("run",
           [](vision::classification::PPShiTuV2RecognizerPostprocessor &self,
              std::vector<pybind11::array> &input_array) {
             std::vector<vision::ClassifyResult> results;
             std::vector<FDTensor> inputs;
             PyArrayToTensorList(input_array, &inputs, /*share_buffer=*/true);
             if (!self.Run(inputs, &results)) {
               throw std::runtime_error(
                   "Failed to postprocess the runtime result in "
                   "PPShiTuV2RecognizerPostprocessor.");
             }
             return results;
           })
      .def_property("feature_norm",
                    &vision::classification::PPShiTuV2RecognizerPostprocessor::
                        GetFeatureNorm,
                    &vision::classification::PPShiTuV2RecognizerPostprocessor::
                        SetFeatureNorm);

  pybind11::class_<vision::classification::PPShiTuV2Recognizer,
                   UltraInferModel>(m, "PPShiTuV2Recognizer")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def("clone",
           [](vision::classification::PPShiTuV2Recognizer &self) {
             return self.Clone();
           })
      .def("predict",
           [](vision::classification::PPShiTuV2Recognizer &self,
              pybind11::array &data) {
             cv::Mat im = PyArrayToCvMat(data);
             vision::ClassifyResult result;
             self.Predict(im, &result);
             return result;
           })
      .def("batch_predict",
           [](vision::classification::PPShiTuV2Recognizer &self,
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
          &vision::classification::PPShiTuV2Recognizer::GetPreprocessor)
      .def_property_readonly(
          "postprocessor",
          &vision::classification::PPShiTuV2Recognizer::GetPostprocessor);
}
} // namespace ultra_infer
