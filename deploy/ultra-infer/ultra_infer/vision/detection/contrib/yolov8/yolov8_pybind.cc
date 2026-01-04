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
void BindYOLOv8(pybind11::module &m) {
  pybind11::class_<vision::detection::YOLOv8Preprocessor>(m,
                                                          "YOLOv8Preprocessor")
      .def(pybind11::init<>())
      .def(
          "run",
          [](vision::detection::YOLOv8Preprocessor &self,
             std::vector<pybind11::array> &im_list) {
            std::vector<vision::FDMat> images;
            for (size_t i = 0; i < im_list.size(); ++i) {
              images.push_back(vision::WrapMat(PyArrayToCvMat(im_list[i])));
            }
            std::vector<FDTensor> outputs;
            std::vector<std::map<std::string, std::array<float, 2>>> ims_info;
            if (!self.Run(&images, &outputs, &ims_info)) {
              throw std::runtime_error(
                  "Failed to preprocess the input data in YOLOv8Preprocessor.");
            }
            for (size_t i = 0; i < outputs.size(); ++i) {
              outputs[i].StopSharing();
            }
            return make_pair(outputs, ims_info);
          })
      .def_property("size", &vision::detection::YOLOv8Preprocessor::GetSize,
                    &vision::detection::YOLOv8Preprocessor::SetSize)
      .def_property("padding_value",
                    &vision::detection::YOLOv8Preprocessor::GetPaddingValue,
                    &vision::detection::YOLOv8Preprocessor::SetPaddingValue)
      .def_property("is_scale_up",
                    &vision::detection::YOLOv8Preprocessor::GetScaleUp,
                    &vision::detection::YOLOv8Preprocessor::SetScaleUp)
      .def_property("is_mini_pad",
                    &vision::detection::YOLOv8Preprocessor::GetMiniPad,
                    &vision::detection::YOLOv8Preprocessor::SetMiniPad)
      .def_property("stride", &vision::detection::YOLOv8Preprocessor::GetStride,
                    &vision::detection::YOLOv8Preprocessor::SetStride);

  pybind11::class_<vision::detection::YOLOv8Postprocessor>(
      m, "YOLOv8Postprocessor")
      .def(pybind11::init<>())
      .def("run",
           [](vision::detection::YOLOv8Postprocessor &self,
              std::vector<FDTensor> &inputs,
              const std::vector<std::map<std::string, std::array<float, 2>>>
                  &ims_info) {
             std::vector<vision::DetectionResult> results;
             if (!self.Run(inputs, &results, ims_info)) {
               throw std::runtime_error(
                   "Failed to postprocess the runtime result in "
                   "YOLOv8Postprocessor.");
             }
             return results;
           })
      .def("run",
           [](vision::detection::YOLOv8Postprocessor &self,
              std::vector<pybind11::array> &input_array,
              const std::vector<std::map<std::string, std::array<float, 2>>>
                  &ims_info) {
             std::vector<vision::DetectionResult> results;
             std::vector<FDTensor> inputs;
             PyArrayToTensorList(input_array, &inputs, /*share_buffer=*/true);
             if (!self.Run(inputs, &results, ims_info)) {
               throw std::runtime_error(
                   "Failed to postprocess the runtime result in "
                   "YOLOv8Postprocessor.");
             }
             return results;
           })
      .def_property("conf_threshold",
                    &vision::detection::YOLOv8Postprocessor::GetConfThreshold,
                    &vision::detection::YOLOv8Postprocessor::SetConfThreshold)
      .def_property("nms_threshold",
                    &vision::detection::YOLOv8Postprocessor::GetNMSThreshold,
                    &vision::detection::YOLOv8Postprocessor::SetNMSThreshold)
      .def_property("multi_label",
                    &vision::detection::YOLOv8Postprocessor::GetMultiLabel,
                    &vision::detection::YOLOv8Postprocessor::SetMultiLabel);

  pybind11::class_<vision::detection::YOLOv8, UltraInferModel>(m, "YOLOv8")
      .def(pybind11::init<std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def("predict",
           [](vision::detection::YOLOv8 &self, pybind11::array &data) {
             auto mat = PyArrayToCvMat(data);
             vision::DetectionResult res;
             self.Predict(mat, &res);
             return res;
           })
      .def("batch_predict",
           [](vision::detection::YOLOv8 &self,
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
                             &vision::detection::YOLOv8::GetPreprocessor)
      .def_property_readonly("postprocessor",
                             &vision::detection::YOLOv8::GetPostprocessor);
}
} // namespace ultra_infer
