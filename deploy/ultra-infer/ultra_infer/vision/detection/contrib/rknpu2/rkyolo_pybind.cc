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
void BindRKYOLO(pybind11::module &m) {
  pybind11::class_<vision::detection::RKYOLOPreprocessor>(m,
                                                          "RKYOLOPreprocessor")
      .def(pybind11::init<>())
      .def("run",
           [](vision::detection::RKYOLOPreprocessor &self,
              std::vector<pybind11::array> &im_list) {
             std::vector<vision::FDMat> images;
             for (size_t i = 0; i < im_list.size(); ++i) {
               images.push_back(vision::WrapMat(PyArrayToCvMat(im_list[i])));
             }
             std::vector<FDTensor> outputs;
             if (!self.Run(&images, &outputs)) {
               throw std::runtime_error(
                   "Failed to preprocess the input data in "
                   "PaddleClasPreprocessor.");
             }
             for (size_t i = 0; i < outputs.size(); ++i) {
               outputs[i].StopSharing();
             }
             return outputs;
           })
      .def_property("size", &vision::detection::RKYOLOPreprocessor::GetSize,
                    &vision::detection::RKYOLOPreprocessor::SetSize)
      .def_property("padding_value",
                    &vision::detection::RKYOLOPreprocessor::GetPaddingValue,
                    &vision::detection::RKYOLOPreprocessor::SetPaddingValue)
      .def_property("is_scale_up",
                    &vision::detection::RKYOLOPreprocessor::GetScaleUp,
                    &vision::detection::RKYOLOPreprocessor::SetScaleUp);

  pybind11::class_<vision::detection::RKYOLOPostprocessor>(
      m, "RKYOLOPostprocessor")
      .def(pybind11::init<>())
      .def("run",
           [](vision::detection::RKYOLOPostprocessor &self,
              std::vector<FDTensor> &inputs) {
             std::vector<vision::DetectionResult> results;
             if (!self.Run(inputs, &results)) {
               throw std::runtime_error(
                   "Failed to postprocess the runtime result in "
                   "RKYOLOV5Postprocessor.");
             }
             return results;
           })
      .def("run",
           [](vision::detection::RKYOLOPostprocessor &self,
              std::vector<pybind11::array> &input_array) {
             std::vector<vision::DetectionResult> results;
             std::vector<FDTensor> inputs;
             PyArrayToTensorList(input_array, &inputs, /*share_buffer=*/true);
             if (!self.Run(inputs, &results)) {
               throw std::runtime_error(
                   "Failed to postprocess the runtime result in "
                   "RKYOLOV5Postprocessor.");
             }
             return results;
           })
      .def("set_anchor", [](vision::detection::RKYOLOPostprocessor &self,
                            std::vector<int> &data) { self.SetAnchor(data); })
      .def_property("conf_threshold",
                    &vision::detection::RKYOLOPostprocessor::GetConfThreshold,
                    &vision::detection::RKYOLOPostprocessor::SetConfThreshold)
      .def_property("nms_threshold",
                    &vision::detection::RKYOLOPostprocessor::GetNMSThreshold,
                    &vision::detection::RKYOLOPostprocessor::SetNMSThreshold)
      .def_property("class_num",
                    &vision::detection::RKYOLOPostprocessor::GetClassNum,
                    &vision::detection::RKYOLOPostprocessor::SetClassNum);

  pybind11::class_<vision::detection::RKYOLOV5, UltraInferModel>(m, "RKYOLOV5")
      .def(pybind11::init<std::string, RuntimeOption, ModelFormat>())
      .def("predict",
           [](vision::detection::RKYOLOV5 &self, pybind11::array &data) {
             auto mat = PyArrayToCvMat(data);
             vision::DetectionResult res;
             self.Predict(mat, &res);
             return res;
           })
      .def("batch_predict",
           [](vision::detection::RKYOLOV5 &self,
              std::vector<pybind11::array> &data) {
             std::vector<cv::Mat> images;
             for (size_t i = 0; i < data.size(); ++i) {
               images.push_back(PyArrayToCvMat(data[i]));
             }
             std::vector<vision::DetectionResult> results;
             self.BatchPredict(images, &results);
             return results;
           })
      .def_property_readonly("preprocessor",
                             &vision::detection::RKYOLOV5::GetPreprocessor)
      .def_property_readonly("postprocessor",
                             &vision::detection::RKYOLOV5::GetPostprocessor);

  pybind11::class_<vision::detection::RKYOLOX, UltraInferModel>(m, "RKYOLOX")
      .def(pybind11::init<std::string, RuntimeOption, ModelFormat>())
      .def("predict",
           [](vision::detection::RKYOLOX &self, pybind11::array &data) {
             auto mat = PyArrayToCvMat(data);
             vision::DetectionResult res;
             self.Predict(mat, &res);
             return res;
           })
      .def("batch_predict",
           [](vision::detection::RKYOLOX &self,
              std::vector<pybind11::array> &data) {
             std::vector<cv::Mat> images;
             for (size_t i = 0; i < data.size(); ++i) {
               images.push_back(PyArrayToCvMat(data[i]));
             }
             std::vector<vision::DetectionResult> results;
             self.BatchPredict(images, &results);
             return results;
           })
      .def_property_readonly("preprocessor",
                             &vision::detection::RKYOLOX::GetPreprocessor)
      .def_property_readonly("postprocessor",
                             &vision::detection::RKYOLOX::GetPostprocessor);

  pybind11::class_<vision::detection::RKYOLOV7, UltraInferModel>(m, "RKYOLOV7")
      .def(pybind11::init<std::string, RuntimeOption, ModelFormat>())
      .def("predict",
           [](vision::detection::RKYOLOV7 &self, pybind11::array &data) {
             auto mat = PyArrayToCvMat(data);
             vision::DetectionResult res;
             self.Predict(mat, &res);
             return res;
           })
      .def("batch_predict",
           [](vision::detection::RKYOLOV7 &self,
              std::vector<pybind11::array> &data) {
             std::vector<cv::Mat> images;
             for (size_t i = 0; i < data.size(); ++i) {
               images.push_back(PyArrayToCvMat(data[i]));
             }
             std::vector<vision::DetectionResult> results;
             self.BatchPredict(images, &results);
             return results;
           })
      .def_property_readonly("preprocessor",
                             &vision::detection::RKYOLOV7::GetPreprocessor)
      .def_property_readonly("postprocessor",
                             &vision::detection::RKYOLOV7::GetPostprocessor);
}
} // namespace ultra_infer
