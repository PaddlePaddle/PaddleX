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
void BindInsightFace(pybind11::module &m) {
  pybind11::class_<vision::faceid::InsightFaceRecognitionPreprocessor>(
      m, "InsightFaceRecognitionPreprocessor")
      .def(pybind11::init())
      .def("run",
           [](vision::faceid::InsightFaceRecognitionPreprocessor &self,
              std::vector<pybind11::array> &im_list) {
             std::vector<vision::FDMat> images;
             for (size_t i = 0; i < im_list.size(); ++i) {
               images.push_back(vision::WrapMat(PyArrayToCvMat(im_list[i])));
             }
             std::vector<FDTensor> outputs;
             if (!self.Run(&images, &outputs)) {
               throw std::runtime_error(
                   "Failed to preprocess the input data in "
                   "InsightFaceRecognitionPreprocessor.");
             }
             for (size_t i = 0; i < outputs.size(); ++i) {
               outputs[i].StopSharing();
             }
             return outputs;
           })
      .def(
          "disable_normalize",
          &vision::faceid::InsightFaceRecognitionPreprocessor::DisableNormalize)
      .def("disable_permute",
           &vision::faceid::InsightFaceRecognitionPreprocessor::DisablePermute)
      .def_property(
          "alpha",
          &vision::faceid::InsightFaceRecognitionPreprocessor::GetAlpha,
          &vision::faceid::InsightFaceRecognitionPreprocessor::SetAlpha)
      .def_property(
          "beta", &vision::faceid::InsightFaceRecognitionPreprocessor::GetBeta,
          &vision::faceid::InsightFaceRecognitionPreprocessor::SetBeta)
      .def_property(
          "size", &vision::faceid::InsightFaceRecognitionPreprocessor::GetSize,
          &vision::faceid::InsightFaceRecognitionPreprocessor::SetSize);

  pybind11::class_<vision::faceid::InsightFaceRecognitionPostprocessor>(
      m, "InsightFaceRecognitionPostprocessor")
      .def(pybind11::init())
      .def("run",
           [](vision::faceid::InsightFaceRecognitionPostprocessor &self,
              std::vector<FDTensor> &inputs) {
             std::vector<vision::FaceRecognitionResult> results;
             if (!self.Run(inputs, &results)) {
               throw std::runtime_error(
                   "Failed to postprocess the runtime result in "
                   "InsightFaceRecognitionPostprocessor.");
             }
             return results;
           })
      .def("run",
           [](vision::faceid::InsightFaceRecognitionPostprocessor &self,
              std::vector<pybind11::array> &input_array) {
             std::vector<vision::FaceRecognitionResult> results;
             std::vector<FDTensor> inputs;
             PyArrayToTensorList(input_array, &inputs, /*share_buffer=*/true);
             if (!self.Run(inputs, &results)) {
               throw std::runtime_error(
                   "Failed to postprocess the runtime result in "
                   "InsightFaceRecognitionPostprocessor.");
             }
             return results;
           })
      .def_property(
          "l2_normalize",
          &vision::faceid::InsightFaceRecognitionPostprocessor::GetL2Normalize,
          &vision::faceid::InsightFaceRecognitionPostprocessor::SetL2Normalize);

  pybind11::class_<vision::faceid::InsightFaceRecognitionBase, UltraInferModel>(
      m, "InsightFaceRecognitionBase")
      .def(pybind11::init<std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def("predict",
           [](vision::faceid::InsightFaceRecognitionBase &self,
              pybind11::array &data) {
             cv::Mat im = PyArrayToCvMat(data);
             vision::FaceRecognitionResult result;
             self.Predict(im, &result);
             return result;
           })
      .def("batch_predict",
           [](vision::faceid::InsightFaceRecognitionBase &self,
              std::vector<pybind11::array> &data) {
             std::vector<cv::Mat> images;
             for (size_t i = 0; i < data.size(); ++i) {
               images.push_back(PyArrayToCvMat(data[i]));
             }
             std::vector<vision::FaceRecognitionResult> results;
             self.BatchPredict(images, &results);
             return results;
           })
      .def_property_readonly(
          "preprocessor",
          &vision::faceid::InsightFaceRecognitionBase::GetPreprocessor)
      .def_property_readonly(
          "postprocessor",
          &vision::faceid::InsightFaceRecognitionBase::GetPostprocessor);

  pybind11::class_<vision::faceid::ArcFace,
                   vision::faceid::InsightFaceRecognitionBase>(m, "ArcFace")
      .def(pybind11::init<std::string, std::string, RuntimeOption,
                          ModelFormat>());

  pybind11::class_<vision::faceid::CosFace,
                   vision::faceid::InsightFaceRecognitionBase>(m, "CosFace")
      .def(pybind11::init<std::string, std::string, RuntimeOption,
                          ModelFormat>());

  pybind11::class_<vision::faceid::PartialFC,
                   vision::faceid::InsightFaceRecognitionBase>(m, "PartialFC")
      .def(pybind11::init<std::string, std::string, RuntimeOption,
                          ModelFormat>());

  pybind11::class_<vision::faceid::VPL,
                   vision::faceid::InsightFaceRecognitionBase>(m, "VPL")
      .def(pybind11::init<std::string, std::string, RuntimeOption,
                          ModelFormat>());
}
} // namespace ultra_infer
