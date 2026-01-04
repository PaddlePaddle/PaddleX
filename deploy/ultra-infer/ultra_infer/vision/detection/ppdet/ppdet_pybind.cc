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
void BindPPDet(pybind11::module &m) {
  pybind11::class_<vision::detection::PaddleDetPreprocessor,
                   vision::ProcessorManager>(m, "PaddleDetPreprocessor")
      .def(pybind11::init<std::string>())
      .def("run",
           [](vision::detection::PaddleDetPreprocessor &self,
              std::vector<pybind11::array> &im_list) {
             std::vector<vision::FDMat> images;
             for (size_t i = 0; i < im_list.size(); ++i) {
               images.push_back(vision::WrapMat(PyArrayToCvMat(im_list[i])));
             }
             std::vector<FDTensor> outputs;
             if (!self.Run(&images, &outputs)) {
               throw std::runtime_error(
                   "Failed to preprocess the input data in "
                   "PaddleDetPreprocessor.");
             }
             for (size_t i = 0; i < outputs.size(); ++i) {
               outputs[i].StopSharing();
             }
             return outputs;
           })
      .def("disable_normalize",
           [](vision::detection::PaddleDetPreprocessor &self) {
             self.DisableNormalize();
           })
      .def("disable_permute",
           [](vision::detection::PaddleDetPreprocessor &self) {
             self.DisablePermute();
           });

  pybind11::class_<vision::detection::NMSOption>(m, "NMSOption")
      .def(pybind11::init())
      .def_readwrite("background_label",
                     &vision::detection::NMSOption::background_label)
      .def_readwrite("keep_top_k", &vision::detection::NMSOption::keep_top_k)
      .def_readwrite("nms_eta", &vision::detection::NMSOption::nms_eta)
      .def_readwrite("nms_threshold",
                     &vision::detection::NMSOption::nms_threshold)
      .def_readwrite("nms_top_k", &vision::detection::NMSOption::nms_top_k)
      .def_readwrite("normalized", &vision::detection::NMSOption::normalized)
      .def_readwrite("score_threshold",
                     &vision::detection::NMSOption::score_threshold);

  pybind11::class_<vision::detection::PaddleDetPostprocessor>(
      m, "PaddleDetPostprocessor")
      .def(pybind11::init<>())
      .def(pybind11::init<std::string>())
      .def("run",
           [](vision::detection::PaddleDetPostprocessor &self,
              std::vector<FDTensor> &inputs) {
             std::vector<vision::DetectionResult> results;
             if (!self.Run(inputs, &results)) {
               throw std::runtime_error(
                   "Failed to postprocess the runtime result in "
                   "PaddleDetPostprocessor.");
             }
             return results;
           })
      .def("set_nms_option",
           [](vision::detection::PaddleDetPostprocessor &self,
              vision::detection::NMSOption option) {
             self.SetNMSOption(option);
           })
      .def("set_nms_rotated_option",
           [](vision::detection::PaddleDetPostprocessor &self,
              vision::detection::NMSRotatedOption option) {
             self.SetNMSRotatedOption(option);
           })
      .def("apply_nms",
           [](vision::detection::PaddleDetPostprocessor &self) {
             self.ApplyNMS();
           })
      .def("run", [](vision::detection::PaddleDetPostprocessor &self,
                     std::vector<pybind11::array> &input_array) {
        std::vector<vision::DetectionResult> results;
        std::vector<FDTensor> inputs;
        PyArrayToTensorList(input_array, &inputs, /*share_buffer=*/true);
        if (!self.Run(inputs, &results)) {
          throw std::runtime_error(
              "Failed to postprocess the runtime result in "
              "PaddleDetPostprocessor.");
        }
        return results;
      });

  pybind11::class_<vision::detection::PPDetBase, UltraInferModel>(m,
                                                                  "PPDetBase")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def("predict",
           [](vision::detection::PPDetBase &self, pybind11::array &data) {
             auto mat = PyArrayToCvMat(data);
             vision::DetectionResult res;
             self.Predict(&mat, &res);
             return res;
           })
      .def("batch_predict",
           [](vision::detection::PPDetBase &self,
              std::vector<pybind11::array> &data) {
             std::vector<cv::Mat> images;
             for (size_t i = 0; i < data.size(); ++i) {
               images.push_back(PyArrayToCvMat(data[i]));
             }
             std::vector<vision::DetectionResult> results;
             self.BatchPredict(images, &results);
             return results;
           })
      .def("clone",
           [](vision::detection::PPDetBase &self) { return self.Clone(); })
      .def_property_readonly("preprocessor",
                             &vision::detection::PPDetBase::GetPreprocessor)
      .def_property_readonly("postprocessor",
                             &vision::detection::PPDetBase::GetPostprocessor);

  pybind11::class_<vision::detection::PPYOLO, vision::detection::PPDetBase>(
      m, "PPYOLO")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>());

  pybind11::class_<vision::detection::PPYOLOE, vision::detection::PPDetBase>(
      m, "PPYOLOE")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>());

  pybind11::class_<vision::detection::PicoDet, vision::detection::PPDetBase>(
      m, "PicoDet")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>());

  pybind11::class_<vision::detection::PaddleYOLOX,
                   vision::detection::PPDetBase>(m, "PaddleYOLOX")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>());

  pybind11::class_<vision::detection::FasterRCNN, vision::detection::PPDetBase>(
      m, "FasterRCNN")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>());

  pybind11::class_<vision::detection::YOLOv3, vision::detection::PPDetBase>(
      m, "YOLOv3")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>());

  pybind11::class_<vision::detection::MaskRCNN, vision::detection::PPDetBase>(
      m, "MaskRCNN")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>());

  pybind11::class_<vision::detection::SSD, vision::detection::PPDetBase>(m,
                                                                         "SSD")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>());

  pybind11::class_<vision::detection::PaddleYOLOv5,
                   vision::detection::PPDetBase>(m, "PaddleYOLOv5")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>());

  pybind11::class_<vision::detection::PaddleYOLOv6,
                   vision::detection::PPDetBase>(m, "PaddleYOLOv6")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>());

  pybind11::class_<vision::detection::PaddleYOLOv7,
                   vision::detection::PPDetBase>(m, "PaddleYOLOv7")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>());

  pybind11::class_<vision::detection::PaddleYOLOv8,
                   vision::detection::PPDetBase>(m, "PaddleYOLOv8")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>());

  pybind11::class_<vision::detection::RTMDet, vision::detection::PPDetBase>(
      m, "RTMDet")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>());

  pybind11::class_<vision::detection::CascadeRCNN,
                   vision::detection::PPDetBase>(m, "CascadeRCNN")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>());

  pybind11::class_<vision::detection::PSSDet, vision::detection::PPDetBase>(
      m, "PSSDet")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>());

  pybind11::class_<vision::detection::RetinaNet, vision::detection::PPDetBase>(
      m, "RetinaNet")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>());

  pybind11::class_<vision::detection::PPYOLOESOD, vision::detection::PPDetBase>(
      m, "PPYOLOESOD")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>());

  pybind11::class_<vision::detection::FCOS, vision::detection::PPDetBase>(
      m, "FCOS")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>());

  pybind11::class_<vision::detection::TTFNet, vision::detection::PPDetBase>(
      m, "TTFNet")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>());

  pybind11::class_<vision::detection::TOOD, vision::detection::PPDetBase>(
      m, "TOOD")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>());

  pybind11::class_<vision::detection::GFL, vision::detection::PPDetBase>(m,
                                                                         "GFL")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>());

  pybind11::class_<vision::detection::SOLOv2, vision::detection::PPDetBase>(
      m, "SOLOv2")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>());

  pybind11::class_<vision::detection::PaddleDetectionModel,
                   vision::detection::PPDetBase>(m, "PaddleDetectionModel")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>());

  pybind11::class_<vision::detection::PPYOLOER, vision::detection::PPDetBase>(
      m, "PPYOLOER")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>());

  pybind11::class_<vision::detection::NMSRotatedOption>(m, "NMSRotatedOption")
      .def(pybind11::init())
      .def_readwrite("background_label",
                     &vision::detection::NMSRotatedOption::background_label)
      .def_readwrite("keep_top_k",
                     &vision::detection::NMSRotatedOption::keep_top_k)
      .def_readwrite("nms_eta", &vision::detection::NMSRotatedOption::nms_eta)
      .def_readwrite("nms_threshold",
                     &vision::detection::NMSRotatedOption::nms_threshold)
      .def_readwrite("nms_top_k",
                     &vision::detection::NMSRotatedOption::nms_top_k)
      .def_readwrite("normalized",
                     &vision::detection::NMSRotatedOption::normalized)
      .def_readwrite("score_threshold",
                     &vision::detection::NMSRotatedOption::score_threshold);
}
} // namespace ultra_infer
