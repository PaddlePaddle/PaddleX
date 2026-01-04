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
void BindYOLOv5Seg(pybind11::module &m) {
  pybind11::class_<vision::detection::YOLOv5SegPreprocessor>(
      m, "YOLOv5SegPreprocessor")
      .def(pybind11::init<>())
      .def("run",
           [](vision::detection::YOLOv5SegPreprocessor &self,
              std::vector<pybind11::array> &im_list) {
             std::vector<vision::FDMat> images;
             for (size_t i = 0; i < im_list.size(); ++i) {
               images.push_back(vision::WrapMat(PyArrayToCvMat(im_list[i])));
             }
             std::vector<FDTensor> outputs;
             std::vector<std::map<std::string, std::array<float, 2>>> ims_info;
             if (!self.Run(&images, &outputs, &ims_info)) {
               throw std::runtime_error("Failed to preprocess the input data "
                                        "in PaddleClasPreprocessor.");
             }
             for (size_t i = 0; i < outputs.size(); ++i) {
               outputs[i].StopSharing();
             }
             return make_pair(outputs, ims_info);
           })
      .def_property("size", &vision::detection::YOLOv5SegPreprocessor::GetSize,
                    &vision::detection::YOLOv5SegPreprocessor::SetSize)
      .def_property("padding_value",
                    &vision::detection::YOLOv5SegPreprocessor::GetPaddingValue,
                    &vision::detection::YOLOv5SegPreprocessor::SetPaddingValue)
      .def_property("is_scale_up",
                    &vision::detection::YOLOv5SegPreprocessor::GetScaleUp,
                    &vision::detection::YOLOv5SegPreprocessor::SetScaleUp)
      .def_property("is_mini_pad",
                    &vision::detection::YOLOv5SegPreprocessor::GetMiniPad,
                    &vision::detection::YOLOv5SegPreprocessor::SetMiniPad)
      .def_property("stride",
                    &vision::detection::YOLOv5SegPreprocessor::GetStride,
                    &vision::detection::YOLOv5SegPreprocessor::SetStride);

  pybind11::class_<vision::detection::YOLOv5SegPostprocessor>(
      m, "YOLOv5SegPostprocessor")
      .def(pybind11::init<>())
      .def("run",
           [](vision::detection::YOLOv5SegPostprocessor &self,
              std::vector<FDTensor> &inputs,
              const std::vector<std::map<std::string, std::array<float, 2>>>
                  &ims_info) {
             std::vector<vision::DetectionResult> results;
             if (!self.Run(inputs, &results, ims_info)) {
               throw std::runtime_error("Failed to postprocess the runtime "
                                        "result in YOLOv5SegPostprocessor.");
             }
             return results;
           })
      .def("run",
           [](vision::detection::YOLOv5SegPostprocessor &self,
              std::vector<pybind11::array> &input_array,
              const std::vector<std::map<std::string, std::array<float, 2>>>
                  &ims_info) {
             std::vector<vision::DetectionResult> results;
             std::vector<FDTensor> inputs;
             PyArrayToTensorList(input_array, &inputs, /*share_buffer=*/true);
             if (!self.Run(inputs, &results, ims_info)) {
               throw std::runtime_error("Failed to postprocess the runtime "
                                        "result in YOLOv5SegPostprocessor.");
             }
             return results;
           })
      .def_property(
          "conf_threshold",
          &vision::detection::YOLOv5SegPostprocessor::GetConfThreshold,
          &vision::detection::YOLOv5SegPostprocessor::SetConfThreshold)
      .def_property("nms_threshold",
                    &vision::detection::YOLOv5SegPostprocessor::GetNMSThreshold,
                    &vision::detection::YOLOv5SegPostprocessor::SetNMSThreshold)
      .def_property("multi_label",
                    &vision::detection::YOLOv5SegPostprocessor::GetMultiLabel,
                    &vision::detection::YOLOv5SegPostprocessor::SetMultiLabel);

  pybind11::class_<vision::detection::YOLOv5Seg, UltraInferModel>(m,
                                                                  "YOLOv5Seg")
      .def(pybind11::init<std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def("predict",
           [](vision::detection::YOLOv5Seg &self, pybind11::array &data) {
             auto mat = PyArrayToCvMat(data);
             vision::DetectionResult res;
             self.Predict(mat, &res);
             return res;
           })
      .def("batch_predict",
           [](vision::detection::YOLOv5Seg &self,
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
                             &vision::detection::YOLOv5Seg::GetPreprocessor)
      .def_property_readonly("postprocessor",
                             &vision::detection::YOLOv5Seg::GetPostprocessor);
}
} // namespace ultra_infer
