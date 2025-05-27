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
void BindCenterFace(pybind11::module &m) {
  pybind11::class_<vision::facedet::CenterFacePreprocessor>(
      m, "CenterFacePreprocessor")
      .def(pybind11::init<>())
      .def("run",
           [](vision::facedet::CenterFacePreprocessor &self,
              std::vector<pybind11::array> &im_list) {
             std::vector<vision::FDMat> images;
             for (size_t i = 0; i < im_list.size(); ++i) {
               images.push_back(vision::WrapMat(PyArrayToCvMat(im_list[i])));
             }
             std::vector<FDTensor> outputs;
             std::vector<std::map<std::string, std::array<float, 2>>> ims_info;
             if (!self.Run(&images, &outputs, &ims_info)) {
               throw std::runtime_error("Failed to preprocess the input data "
                                        "in CenterFacePreprocessor.");
             }
             for (size_t i = 0; i < outputs.size(); ++i) {
               outputs[i].StopSharing();
             }
             return make_pair(outputs, ims_info);
           })
      .def_property("size", &vision::facedet::CenterFacePreprocessor::GetSize,
                    &vision::facedet::CenterFacePreprocessor::SetSize);

  pybind11::class_<vision::facedet::CenterFacePostprocessor>(
      m, "CenterFacePostprocessor")
      .def(pybind11::init<>())
      .def("run",
           [](vision::facedet::CenterFacePostprocessor &self,
              std::vector<FDTensor> &inputs,
              const std::vector<std::map<std::string, std::array<float, 2>>>
                  &ims_info) {
             std::vector<vision::FaceDetectionResult> results;
             if (!self.Run(inputs, &results, ims_info)) {
               throw std::runtime_error("Failed to postprocess the runtime "
                                        "result in CenterFacePostprocessor.");
             }
             return results;
           })
      .def("run",
           [](vision::facedet::CenterFacePostprocessor &self,
              std::vector<pybind11::array> &input_array,
              const std::vector<std::map<std::string, std::array<float, 2>>>
                  &ims_info) {
             std::vector<vision::FaceDetectionResult> results;
             std::vector<FDTensor> inputs;
             PyArrayToTensorList(input_array, &inputs, /*share_buffer=*/true);
             if (!self.Run(inputs, &results, ims_info)) {
               throw std::runtime_error("Failed to postprocess the runtime "
                                        "result in CenterFacePostprocessor.");
             }
             return results;
           })
      .def_property("conf_threshold",
                    &vision::facedet::CenterFacePostprocessor::GetConfThreshold,
                    &vision::facedet::CenterFacePostprocessor::SetConfThreshold)
      .def_property("nms_threshold",
                    &vision::facedet::CenterFacePostprocessor::GetNMSThreshold,
                    &vision::facedet::CenterFacePostprocessor::SetNMSThreshold);

  pybind11::class_<vision::facedet::CenterFace, UltraInferModel>(m,
                                                                 "CenterFace")
      .def(pybind11::init<std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def("predict",
           [](vision::facedet::CenterFace &self, pybind11::array &data) {
             auto mat = PyArrayToCvMat(data);
             vision::FaceDetectionResult res;
             self.Predict(mat, &res);
             return res;
           })
      .def("batch_predict",
           [](vision::facedet::CenterFace &self,
              std::vector<pybind11::array> &data) {
             std::vector<cv::Mat> images;
             for (size_t i = 0; i < data.size(); ++i) {
               images.push_back(PyArrayToCvMat(data[i]));
             }
             std::vector<vision::FaceDetectionResult> results;
             self.BatchPredict(images, &results);
             return results;
           })
      .def_property_readonly("preprocessor",
                             &vision::facedet::CenterFace::GetPreprocessor)
      .def_property_readonly("postprocessor",
                             &vision::facedet::CenterFace::GetPostprocessor);
}
} // namespace ultra_infer
